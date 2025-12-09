# Lexi/lexi/routes/gen.py
from __future__ import annotations
from typing import Dict, Optional, Any, List
import base64
import glob
import logging
import os
import time
import shutil
import fcntl
from pathlib import Path
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
from ..sd.sd_pipeline import generate_avatar_pipeline
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
AV_PUBLIC_DIR = Path(os.getenv("AVATARS_PUBLIC_DIR") or "/app/frontend/public/avatars")
AV_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
AV_PUBLIC_URL = os.getenv("AVATARS_PUBLIC_URL") or "/lexi/static/avatars"
AVATARS_PUBLIC_DIR = AV_PUBLIC_DIR
AVATARS_PUBLIC_URL = AV_PUBLIC_URL
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/mnt/data/comfy/output"))
DEFAULT_SEED = Path(__file__).resolve().parents[3] / "assets" / "default.png"
LEGACY_BASENAME = "lexi_base.png"
LEGACY_BASE_PATH = AV_PUBLIC_DIR / LEGACY_BASENAME


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
    for header in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
        raw = request.headers.get(header)
        if raw:
            return raw.split(",")[0].strip()
    client = getattr(request, "client", None)
    return (getattr(client, "host", None) or "unknown").strip()


def _avatar_basename_for_ip(ip: str) -> str:
    cleaned = (ip or "unknown").strip().replace(":", "_").replace("/", "_")
    return f"ip_{cleaned or 'unknown'}"


def _latest_output() -> Optional[Path]:
    pattern = str(COMFY_OUTPUT_DIR / "*.png")
    try:
        files = sorted(
            glob.glob(pattern),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
    except (FileNotFoundError, OSError):
        return None
    return Path(files[0]) if files else None


def _publish(src: Path, dest_base: str) -> Dict[str, str]:
    """Copy Comfy output into the shared avatars volume + archive snapshots."""
    AV_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    per_ip = AV_PUBLIC_DIR / f"{dest_base}.png"
    suffix = dest_base[-6:] if dest_base else "unknown"
    archived = AV_PUBLIC_DIR / f"lexi_{ts:010d}_{suffix}.png"
    lock = AV_PUBLIC_DIR / ".publish.lock"
    with open(lock, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not per_ip.exists() or src.resolve() != per_ip.resolve():
                shutil.copy2(src, per_ip)
            shutil.copy2(src, archived)
            if not LEGACY_BASE_PATH.exists() or src.resolve() != LEGACY_BASE_PATH.resolve():
                shutil.copy2(src, LEGACY_BASE_PATH)
            _ensure_default_symlink()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    q = f"?v={ts}"
    return {
        "image": f"{AV_PUBLIC_URL}/{per_ip.name}{q}",
        "url": f"{AV_PUBLIC_URL}/{per_ip.name}{q}",
        "filename": per_ip.name,
        "archived": f"{AV_PUBLIC_URL}/{archived.name}{q}",
    }


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


def _coerce_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    try:
        candidate = Path(str(value))
    except (TypeError, ValueError):
        return None
    return candidate if candidate.exists() else None


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
    traits: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    strength: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
    steps: Optional[int] = Field(default=28, ge=1, le=200)
    width: Optional[int] = Field(default=768, ge=64, le=2048)
    height: Optional[int] = Field(default=1024, ge=64, le=2048)
    upscale: Optional[bool] = False
    refine: Optional[bool] = True
    negative_prompt: Optional[str] = None
    loras: Optional[List[str]] = None
    source_path: Optional[str] = None  # for img2img/inpaint (explicit only)


@router.post("/lexi/gen/avatar")
async def gen_avatar(
    request: Request,
    response: Response,
    payload: AvatarGenRequest = Body(...),
    session_id: Optional[str] = Header(default=None, alias="X-Lexi-Session"),
):
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
    base_name = _avatar_basename_for_ip(ip)
    prompt_text = (payload.prompt or "").strip()
    if not prompt_text:
        raise HTTPException(status_code=422, detail="prompt is required")

    try:
        # Use a stronger img2img push by default to escape the stuck base.
        img2img_strength = payload.strength if payload.strength is not None else 0.7
        img2img_strength = min(max(img2img_strength, 0.10), 0.75)
        generated = await run_in_threadpool(
            generate_avatar_pipeline,
            prompt=prompt_text,
            negative=payload.negative_prompt or "low quality, blurry, deformed",
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            cfg_scale=4.5,
            traits=payload.traits or None,
            mode=payload.sd_mode,
            source_path=payload.source_path if payload.sd_mode != "txt2img" else None,
            denoise=img2img_strength,
            seed=payload.seed,
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
            flux_denoise=img2img_strength if payload.sd_mode != "txt2img" else None,
            base_name=base_name,
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

    output_path = None
    if isinstance(generated, dict):
        output_path = _coerce_path(
            generated.get("file")
            or generated.get("path")
            or generated.get("image_path")
            or generated.get("output")
        )
    elif isinstance(generated, str):
        output_path = _coerce_path(generated)
    if output_path is None or not output_path.exists():
        latest = _latest_output()
        if latest and latest.exists():
            output_path = latest

    if output_path and output_path.exists():
        publish_payload = _publish(output_path, dest_base=base_name)
    else:
        _ensure_default_symlink()
        ts = int(time.time())
        url = f"{AV_PUBLIC_URL}/default.png?v={ts}"
        publish_payload = {"image": url, "url": url, "filename": "default.png"}

    response_payload = _wrap_avatar_response(publish_payload, ip, payload.traits)

    if isinstance(generated, dict):
        for key in ("meta", "prompt_id", "ok", "code", "error"):
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
