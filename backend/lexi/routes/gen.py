# Lexi/lexi/routes/gen.py
from __future__ import annotations
from typing import Dict, Optional, Any, List
import uuid
import base64
import logging
import os
import time
import shutil
import secrets
from pathlib import Path
import requests
from fastapi import APIRouter, Body, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..config.config import (
    AVATAR_DIR,
    AVATAR_URL_PREFIX,
    STATIC_ROOT,
    STATIC_URL_PREFIX,
)
from ..sd.sd_pipeline import generate_avatar_pipeline, probe_prompt_images
from ..sd.flux_prompt_builder import build_flux_avatar_prompt_bundle
from ..utils.ip_seed import basename_for_ip, ip_to_seed
from ..utils.request_ip import request_ip
from .lexi_persona import _load_traits as load_traits, _save_traits as save_traits
from ..alpha.session_manager import SessionRegistry
from ..alpha.settings import AlphaSettings
from ..alpha.tour import preview_placeholder_url
from ..session_logging import log_event
from ..core.session_store import session_store

logger = logging.getLogger(__name__)
router = APIRouter()
STATIC_FILES_ROOT = STATIC_ROOT.resolve()
AVATAR_DIR.mkdir(parents=True, exist_ok=True)

SESSION_COOKIE_MAX_AGE = int(os.getenv("SESSION_COOKIE_MAX_AGE", "2592000"))
AV_PUBLIC_DIR = Path(os.getenv("AVATARS_PUBLIC_DIR") or AVATAR_DIR)
AV_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
AV_PUBLIC_URL = os.getenv("AVATARS_PUBLIC_URL") or AVATAR_URL_PREFIX
AVATARS_PUBLIC_DIR = AV_PUBLIC_DIR
AVATARS_PUBLIC_URL = AV_PUBLIC_URL
DEFAULT_SEED = Path(__file__).resolve().parents[3] / "assets" / "default.png"
LEGACY_BASENAME = "lexi_base.png"
LEGACY_BASE_PATH = AV_PUBLIC_DIR / LEGACY_BASENAME
DEFAULT_AVATAR_BASE_PROMPT = (
    "cinematic full-body portrait of Lexi, studio background, soft flattering light, "
    "realistic skin texture, natural pose, shallow depth of field, high detail, "
    "instagram editorial aesthetic"
)
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted face, extra limbs, deformed hands, watermark, logo, text, duplicate body parts"
)


def _ensure_default_symlink() -> None:
    """Backstop default avatar + symlink for first paint."""
    try:
        AV_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("Failed to ensure avatars dir %s: %s", AV_PUBLIC_DIR, exc)
    base = AV_PUBLIC_DIR / LEGACY_BASENAME
    if not base.exists():
        for candidate in (DEFAULT_SEED, AVATAR_DIR / "default.png"):
            try:
                if candidate.exists():
                    shutil.copy2(candidate, base)
                    break
            except Exception as exc:
                logger.debug("Failed to seed base avatar from %s: %s", candidate, exc)
    default_link = AV_PUBLIC_DIR / "default.png"
    try:
        default_link.unlink(missing_ok=True)
    except TypeError:
        if default_link.exists():
            default_link.unlink()
    try:
        os.symlink(LEGACY_BASENAME, default_link)
    except FileExistsError:
        pass
    except OSError as exc:
        logger.debug("default.png symlink update failed: %s", exc)


_ensure_default_symlink()


def _finalize_prompt_image(prompt_id: str, base_path: Path) -> Optional[str]:
    """One-shot check for a finished Comfy prompt and return a public URL if ready."""
    images = probe_prompt_images(prompt_id)
    if not images:
        return None

    first = images[0]
    filename = first.get("filename", "")
    subfolder = first.get("subfolder", "")
    ftype = first.get("type", "output")

    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    vr = requests.get(
        f"{os.getenv('COMFY_URL', 'http://comfy-sd:8188')}/view",
        params=params,
        stream=True,
        timeout=180,
    )
    vr.raise_for_status()
    safe_name = Path(filename).name or f"lexi_{uuid.uuid4().hex[:8]}.png"
    tmp = AVATAR_DIR / safe_name
    with tmp.open("wb") as f:
        for chunk in vr.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    # Normalize/relocate: mimic sd_pipeline finalize behavior
    try:
        from ..sd.sd_pipeline import normalize_portrait_image, _finalize_generated_image
    except Exception:
        normalize_portrait_image = None  # type: ignore
        _finalize_generated_image = None  # type: ignore
    if normalize_portrait_image:
        normalize_portrait_image(tmp)
    if _finalize_generated_image:
        final_out = _finalize_generated_image(tmp, base_path, force_output_to_base=True)
    else:
        final_out = tmp

    rel = final_out.relative_to(STATIC_FILES_ROOT).as_posix()
    public = f"{STATIC_URL_PREFIX}/{rel}"
    sep = "&" if "?" in public else "?"
    return f"{public}{sep}v={int(time.time())}"


def _to_static_url(fp: Path) -> str:
    # compute URL like /static/avatars/filename.png
    rel = fp.relative_to(STATIC_FILES_ROOT).as_posix()
    return f"{STATIC_URL_PREFIX}/{rel}"


def _write_b64_png(b64: str) -> str:
    fname = f"lexi_avatar_{os.urandom(3).hex()}.png"
    out = AVATAR_DIR / fname
    out.write_bytes(base64.b64decode(b64))
    return _to_static_url(out)


def _copy_local(src: str) -> Optional[str]:
    p = Path(src)
    if not p.exists():
        return None
    dest = AVATAR_DIR / p.name
    if str(p.resolve()) != str(dest.resolve()):
        dest.write_bytes(p.read_bytes())
    return _to_static_url(dest)


def _download_to_static(url: str) -> Optional[str]:
    try:
        import requests

        r = requests.get(url, timeout=20)
        r.raise_for_status()
        fname = f"lexi_avatar_{os.urandom(3).hex()}.png"
        out = AVATAR_DIR / fname
        out.write_bytes(r.content)
        return _to_static_url(out)
    except Exception as e:
        logger.warning("download failed: %s", e)
        return None


def _client_ip(request: Request) -> str:
    return request_ip(request)


def _avatar_basename_for_ip(ip: str) -> str:
    return basename_for_ip(ip)


def _wrap_avatar_response(payload: Dict[str, Any], ip: str, traits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "avatar_url": payload.get("image") or payload.get("url"),
        "image": payload.get("image"),
        "url": payload.get("url"),
        "filename": payload.get("filename"),
        "ip": ip,
        "traits": traits or {},
        "ok": True,
        "archived": payload.get("archived"),
    }


class AvatarGenRequest(BaseModel):
    prompt: Optional[str] = None
    persona_mode: Optional[str] = None  # <-- rename: previously "mode"
    sd_mode: Optional[str] = Field(default="txt2img", pattern="^(txt2img|img2img|inpaint)$")
    model: Optional[str] = Field(default="flux", pattern="^flux$")
    variant: Optional[str] = None
    preset: Optional[str] = None
    size: Optional[str] = None
    guidance: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    flux_pipeline: Optional[str] = Field(default="flux_v1", pattern="^(flux_v1|flux_v2)$")
    lexiverse_style: Optional[str] = Field(default="promo")
    traits: Dict[str, Any] = Field(default_factory=dict)
    extra_details: Optional[str] = None
    seed: Optional[int] = None
    strength: Optional[float] = Field(default=0.65, ge=0.0, le=1.0)
    steps: Optional[int] = Field(default=28, ge=1, le=200)
    width: Optional[int] = Field(default=1080, ge=64, le=2048)
    height: Optional[int] = Field(default=1352, ge=64, le=2048)
    upscale: Optional[bool] = False
    refine: Optional[bool] = True
    negative_prompt: Optional[str] = None
    loras: Optional[List[str]] = None
    source_path: Optional[str] = None  # for img2img/inpaint (explicit only)
    change_intensity: Optional[str] = None  # soft | medium | bold
    pose_bucket: Optional[str] = None
    camera_bucket: Optional[str] = None
    pose_feel: Optional[str] = None
    pose_id: Optional[str] = None
    pose_strength: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    pose_control_start: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pose_control_end: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    disable_controlnet: Optional[bool] = None
    controlnet_enabled: Optional[bool] = None
    controlnet_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# CANONICAL: Avatar Tools Modal endpoint. Do not remove.
# CANONICAL: Avatar Tools Modal endpoint. Do not remove.
@router.post("/gen/avatar")
async def gen_avatar(
    request: Request,
    response: Response,
    payload: AvatarGenRequest = Body(...),
    session_id: Optional[str] = Header(default=None, alias="X-Lexi-Session"),
):
    logger.info(
        "[Lexi Avatar] request prompt=%r sd_mode=%s model=%s flux_pipeline=%s lexiverse_style=%s seed=%s source=%s",
        (payload.prompt or "")[:80],
        payload.sd_mode,
        payload.model,
        payload.flux_pipeline,
        payload.lexiverse_style,
        payload.seed,
        payload.source_path,
    )
    registry = getattr(request.app.state, "alpha_sessions", None)
    header_session = session_id
    cookie_session = request.cookies.get("lex_session")
    active_session = header_session or cookie_session or getattr(request.state, "session_id", None)
    settings = AlphaSettings()

    created_session = False
    if not session_store.exists(active_session):
        active_session = session_store.create()
        created_session = True
    session_store.touch(active_session)

    if isinstance(registry, SessionRegistry):
        try:
            registry.get(active_session)
        except KeyError:
            registry.create_session(session_id=active_session)

    request.state.session_id = active_session

    response.headers["X-Lexi-Session"] = active_session
    response.set_cookie(
        "lex_session",
        active_session,
        httponly=True,
        samesite="lax",
        max_age=SESSION_COOKIE_MAX_AGE,
    )
    logger.info(
        "[Lexi SD] /gen/avatar request prompt=%r neg=%r mode=%s",
        (payload.prompt or "")[:200],
        (payload.negative_prompt or "")[:200],
        payload.sd_mode,
    )
    logger.debug("FLUX gen: header=%s cookie=%s resolved=%s created=%s",
                 header_session,
                 cookie_session,
                 active_session,
                 created_session)

    def _respond(body: Dict[str, Any]) -> JSONResponse:
        resp = JSONResponse(body)
        resp.headers["X-Lexi-Session"] = active_session
        resp.set_cookie(
            "lex_session",
            active_session,
            httponly=True,
            samesite="lax",
            max_age=SESSION_COOKIE_MAX_AGE,
        )
        return resp

    if created_session and isinstance(registry, SessionRegistry):
        try:
            registry.append_memory(
                active_session,
                {
                    "role": "system",
                    "event": "session_auto_created",
                },
            )
        except Exception:
            pass

    log_event(
        request,
        "user",
        (payload.prompt or "").strip() or "[avatar request]",
        event="gen_avatar_request",
        sd_mode=payload.sd_mode,
        persona_mode=payload.persona_mode,
        upscale=bool(payload.upscale),
    )

    if settings.alpha_strict:
        placeholder = preview_placeholder_url(settings)
        if isinstance(registry, SessionRegistry) and active_session:
            registry.append_memory(
                active_session,
                {
                    "role": "assistant",
                    "event": "alpha_strict_avatar_stub",
                    "source": "gen_endpoint",
                    "upscale": bool(payload.upscale),
                },
            )
            registry.record_metric(
                active_session,
                {"event": "alpha_strict_stub", "feature": "avatar_generate"},
            )
        log_event(
            request,
            "assistant",
            "alpha strict mode — placeholder preview.",
            event="gen_avatar_placeholder",
            url=placeholder,
        )
        return _respond(
            {
                "image": placeholder,
                "filename": placeholder.split("/")[-1],
                "narration": "alpha strict mode — placeholder preview.",
                "traits": payload.traits,
            }
        )

    if payload.upscale and isinstance(registry, SessionRegistry) and active_session:
        if not registry.increment_counter(active_session, "avatar_upscale", limit=1):
            registry.record_metric(
                active_session,
                {"event": "rate_limited", "feature": "avatar_upscale"},
            )
            log_event(
                request,
                "error",
                "upscale limit reached for this session",
                event="gen_avatar_rate_limited",
            )
            raise HTTPException(
                status_code=429,
                detail="upscale limit reached for this session.",
            )
        registry.append_memory(
            active_session,
            {
                "role": "user",
                "event": "avatar_upscale_requested",
                "traits": payload.traits,
            },
        )

    ip = _client_ip(request)
    sd_mode = (payload.sd_mode or "txt2img").strip().lower()
    base_name = _avatar_basename_for_ip(ip)
    seed_default = ip_to_seed(ip)
    if sd_mode == "txt2img" and payload.seed is None:
        seed_default = secrets.randbits(31)

    raw_prompt = (payload.prompt or "").strip()
    trait_payload = payload.traits or {}
    prompt_bundle = build_flux_avatar_prompt_bundle(
        traits={
            "hair": trait_payload.get("hair"),
            "hair_style": trait_payload.get("hair_style"),
            "skin_tone": trait_payload.get("skin_tone"),
            "eyes": trait_payload.get("eyes"),
            "outfit": trait_payload.get("outfit"),
            "vibe": trait_payload.get("vibe"),
        },
        style_flags={"lexiverse_style": payload.lexiverse_style or "promo", "lexiverse_enabled": True},
        pose_meta=None,
        user_text=payload.extra_details or raw_prompt,
    )
    prompt_text = prompt_bundle["positive"]
    negative_text = (payload.negative_prompt or prompt_bundle.get("negative") or DEFAULT_NEGATIVE_PROMPT).strip()

    # change intensity -> denoise for img2img
    change_map = {"soft": 0.35, "medium": 0.5, "bold": 0.65}
    denoise_val = min(max(payload.strength if payload.strength is not None else change_map.get(payload.change_intensity or "medium", 0.5), 0.10), 0.75)

    # default size for avatars
    width = payload.width or 832
    height = payload.height or 1024

    # Unique base name for txt2img so we don't overwrite IP base
    if sd_mode == "txt2img":
        base_name = f"lexi_{uuid.uuid4().hex[:8]}"

    # If we already composed a prompt from traits above, avoid re-applying traits downstream.
    pipeline_traits = {} if prompt_text else trait_payload

    try:
        lexiverse_style = payload.lexiverse_style or "promo"

        generated = await run_in_threadpool(
            generate_avatar_pipeline,
            prompt=prompt_text,
            negative=negative_text,
            width=width,
            height=height,
            steps=payload.steps,
            cfg_scale=4.5,
            traits=pipeline_traits or None,
            mode=sd_mode,
            source_path=payload.source_path if sd_mode != "txt2img" else None,
            denoise=denoise_val if sd_mode != "txt2img" else None,
            seed=payload.seed if payload.seed is not None else seed_default,
            changes=None,
            refiner=bool(payload.refine and payload.sd_mode == "txt2img"),
            refiner_strength=0.28,
            upscale_factor=(1.25 if payload.upscale else 1.0),
            backend=payload.model or "flux",
            variant=payload.variant,
            flux_variant=payload.variant,
            flux_preset=payload.preset,
            flux_size=payload.size,
            flux_guidance=payload.guidance,
            flux_cfg=payload.cfg,
            flux_sampler=payload.sampler,
            flux_scheduler=payload.scheduler,
            flux_denoise=payload.strength if payload.sd_mode != "txt2img" else None,
            base_name=base_name,
            flux_pipeline=payload.flux_pipeline,
            lexiverse_style=lexiverse_style,
            pose_bucket=payload.pose_bucket,
            camera_bucket=payload.camera_bucket,
            pose_feel=payload.pose_feel,
            pose_id=payload.pose_id,
            pose_strength=payload.pose_strength,
            pose_control_start=payload.pose_control_start,
            pose_control_end=payload.pose_control_end,
            disable_controlnet=bool(payload.disable_controlnet) if payload.disable_controlnet is not None else False,
            controlnet_enabled=payload.controlnet_enabled,
            controlnet_strength=payload.controlnet_strength,
            return_on_submit=True,
        )
        logger.info(
            "[Lexi Avatar] pipeline result ok=%s code=%s file=%s url=%s prompt_id=%s error=%s",
            isinstance(generated, dict) and generated.get("ok"),
            isinstance(generated, dict) and generated.get("code"),
            isinstance(generated, dict) and generated.get("file"),
            isinstance(generated, dict) and generated.get("url"),
            isinstance(generated, dict) and generated.get("prompt_id"),
            isinstance(generated, dict) and generated.get("error"),
        )
    except TimeoutError as exc:
        logger.warning("Avatar generation timed out: %s", exc)
        log_event(request, "error", f"avatar generation timed out: {exc}", event="gen_avatar_timeout")
        _ensure_default_symlink()
        ts = int(time.time())
        fallback_url = f"{AV_PUBLIC_URL}/default.png?v={ts}"
        publish_payload = {"image": fallback_url, "url": fallback_url, "filename": "default.png"}
        response_payload = _wrap_avatar_response(publish_payload, ip, payload.traits)
        return _respond(response_payload)
    except Exception as exc:
        log_event(request, "error", f"avatar generation failed: {exc}", event="gen_avatar_error")
        raise HTTPException(status_code=500, detail=f"avatar generation failed: {exc}") from exc

    def _default_payload() -> Dict[str, str]:
        _ensure_default_symlink()
        ts = int(time.time())
        url = f"{AV_PUBLIC_URL}/default.png?v={ts}"
        return {"image": url, "url": url, "filename": "default.png"}

    # If pipeline explicitly failed, return error without fallback.
    if isinstance(generated, dict) and not generated.get("ok") and generated.get("code"):
        error_resp = {
            "ok": False,
            "error": generated.get("error"),
            "code": generated.get("code"),
            "prompt_id": generated.get("prompt_id"),
        }
        return _respond(error_resp)

    publish_payload = None
    if isinstance(generated, dict) and generated.get("status") == "running" and generated.get("prompt_id"):
        # Fast-return path: prompt accepted, let frontend poll status endpoint.
        return _respond({"status": "running", "prompt_id": generated.get("prompt_id"), "ok": True})
    if isinstance(generated, dict) and generated.get("code") == "COMFY_TIMEOUT" and generated.get("prompt_id"):
        return _respond({"status": "running", "prompt_id": generated.get("prompt_id")})
    if isinstance(generated, dict) and generated.get("ok"):
        avatar_url = generated.get("avatar_url") or generated.get("url")
        if avatar_url:
            filename = None
            file_path = generated.get("file")
            if isinstance(file_path, str) and file_path:
                filename = Path(file_path).name
            elif base_name:
                filename = f"{base_name}.png" if not base_name.endswith(".png") else base_name
            if not filename:
                filename = "avatar.png"
            publish_payload = {
                "image": avatar_url,
                "url": avatar_url,
                "filename": filename,
            }

    if publish_payload is None:
        publish_payload = _default_payload()

    response_payload = _wrap_avatar_response(publish_payload, ip, payload.traits)
    if isinstance(generated, dict):
        response_payload["ok"] = bool(generated.get("ok", response_payload.get("ok", True)))
        if "error" in generated:
            response_payload["error"] = generated.get("error")
        if "code" in generated:
            response_payload["code"] = generated.get("code")

    if isinstance(generated, dict):
        for key in ("meta", "prompt_id", "ok", "code", "error", "status"):
            if key in generated and generated[key] is not None:
                response_payload.setdefault(key, generated[key])

    if isinstance(registry, SessionRegistry) and active_session:
        registry.append_memory(
            active_session,
            {
                "role": "assistant",
                "event": "avatar_generate_response",
                "upscale": bool(payload.upscale),
            },
        )
        registry.record_metric(
            active_session,
            {"event": "avatar_generate", "upscale": bool(payload.upscale)},
        )
    log_event(
        request,
        "assistant",
        "Avatar generated",
        event="gen_avatar_result",
        url=response_payload.get("image"),
    )
    return _respond(response_payload)


@router.post("/lexi/gen/generate")
async def generate_avatar(
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(None),
) -> JSONResponse:
    """
    Generate an avatar based on provided traits or last saved state.
    Returns: {"image": "/static/avatars/....png", "filename": "...", "narration": "...", "traits": {...}}
    """
    registry = getattr(request.app.state, "alpha_sessions", None)
    session_id = getattr(request.state, "session_id", None) or request.headers.get("X-Lexi-Session")
    settings = AlphaSettings()

    # Traits: from payload or persisted
    traits: Dict[str, str] = {}
    if payload and isinstance(payload.get("traits"), dict):
        traits = dict(payload["traits"])
    else:
        traits = load_traits()
    if not traits:
        raise HTTPException(status_code=400, detail="No traits available for avatar generation.")

    if settings.alpha_strict:
        placeholder = preview_placeholder_url(settings)
        if isinstance(registry, SessionRegistry) and session_id:
            registry.append_memory(
                session_id,
                {
                    "role": "assistant",
                    "event": "alpha_strict_avatar_stub",
                    "source": "legacy_generate",
                },
            )
            registry.record_metric(
                session_id,
                {"event": "alpha_strict_stub", "feature": "avatar_generate"},
            )
        log_event(
            request,
            "assistant",
            "alpha strict mode — placeholder preview.",
            event="legacy_avatar_placeholder",
            url=placeholder,
        )
        return JSONResponse(
            {
                "image": placeholder,
                "filename": placeholder.split("/")[-1],
                "narration": "alpha strict mode — placeholder preview.",
                "traits": traits,
            }
        )

    # Run pipeline (ALWAYS use keywords so traits actually reach the pipeline)
    try:
        import inspect, asyncio

        if inspect.iscoroutinefunction(generate_avatar_pipeline):
            result: Dict[str, Any] = await generate_avatar_pipeline(traits=traits, mode="txt2img")
        else:
            result = await asyncio.to_thread(
                generate_avatar_pipeline, traits=traits, mode="txt2img"
            )
    except Exception as e:
        logger.exception("Pipeline crashed: %s", e)
        log_event(request, "error", f"pipeline_crash: {e}", event="legacy_avatar_error")
        raise HTTPException(status_code=500, detail=f"pipeline_crash: {e}")

    # Normalize outputs (unchanged) ...
    b64 = (result or {}).get("image_b64") or (result or {}).get("b64")
    if isinstance(b64, str) and len(b64) > 100:
        web = _write_b64_png(b64)
        save_traits(traits, avatar_path=web)
        resp = JSONResponse(
            {
                "image": web,
                "filename": web.split("/")[-1],
                "narration": result.get("narration", ""),
                "traits": traits,
            }
        )
        log_event(
            request,
            "assistant",
            "Avatar generated",
            event="legacy_avatar_result",
            url=web,
            traits=list(traits.keys()),
        )
        if isinstance(registry, SessionRegistry) and session_id:
            registry.append_memory(
                session_id,
                {
                    "role": "assistant",
                    "event": "avatar_generate_response",
                    "source": "legacy_generate",
                },
            )
            registry.record_metric(
                session_id,
                {"event": "avatar_generate"},
            )
        return resp

    for key in ("image", "image_url", "path", "url", "file"):
        p = (result or {}).get(key)
        if not p or not isinstance(p, str):
            continue
        if p.startswith("/static/"):
            save_traits(traits, avatar_path=p)
            resp = JSONResponse(
                {
                    "image": p,
                    "filename": p.split("/")[-1],
                    "narration": result.get("narration", ""),
                    "traits": traits,
                }
            )
            log_event(
                request,
                "assistant",
                "Avatar generated",
                event="legacy_avatar_result",
                url=p,
                traits=list(traits.keys()),
            )
            return resp
        if p.startswith(("http://", "https://")):
            web = _download_to_static(p)
            if web:
                save_traits(traits, avatar_path=web)
                resp = JSONResponse(
                    {
                        "image": web,
                        "filename": web.split("/")[-1],
                        "narration": result.get("narration", ""),
                        "traits": traits,
                    }
                )
                if isinstance(registry, SessionRegistry) and session_id:
                    registry.append_memory(
                        session_id,
                        {
                            "role": "assistant",
                            "event": "avatar_generate_response",
                            "source": "legacy_generate",
                        },
                    )
                    registry.record_metric(
                        session_id,
                        {"event": "avatar_generate"},
                    )
                log_event(
                    request,
                    "assistant",
                    "Avatar generated",
                    event="legacy_avatar_result",
                    url=web,
                    traits=list(traits.keys()),
                )
                return resp
        else:
            web = _copy_local(p)
            if web:
                save_traits(traits, avatar_path=web)
                resp = JSONResponse(
                    {
                        "image": web,
                        "filename": web.split("/")[-1],
                        "narration": result.get("narration", ""),
                        "traits": traits,
                    }
                )
                if isinstance(registry, SessionRegistry) and session_id:
                    registry.append_memory(
                        session_id,
                        {
                            "role": "assistant",
                            "event": "avatar_generate_response",
                            "source": "legacy_generate",
                        },
                    )
                    registry.record_metric(
                        session_id,
                        {"event": "avatar_generate"},
                    )
                log_event(
                    request,
                    "assistant",
                    "Avatar generated",
                    event="legacy_avatar_result",
                    url=web,
                    traits=list(traits.keys()),
                )
                return resp

    logger.error("[Avatar Gen Error] No usable image returned. Raw result=%r", result)
    raise HTTPException(status_code=500, detail={"error": "no_image_from_pipeline", "raw": result})


@router.get("/gen/avatar/status/{prompt_id}")
async def gen_avatar_status(prompt_id: str, request: Request):
    """Lightweight status check for long-running Comfy prompts."""
    ip = _client_ip(request)
    base_name = _avatar_basename_for_ip(ip)
    base_path = AV_PUBLIC_DIR / (base_name if base_name.endswith(".png") else f"{base_name}.png")
    try:
        url = _finalize_prompt_image(prompt_id, base_path)
        if url:
            return {"status": "done", "avatar_url": url, "url": url, "prompt_id": prompt_id}
        return {"status": "running", "prompt_id": prompt_id}
    except Exception as exc:
        logger.warning("[Lexi Avatar] status probe failed for %s: %s", prompt_id, exc)
        return {"status": "error", "error": str(exc), "prompt_id": prompt_id}
