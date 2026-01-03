# Lexi/lexi/routes/lexi_persona.py
from __future__ import annotations

"""
lexi_persona.py

Persona + conversational avatar flow endpoints used by the frontend.
- GET    /lexi/persona
- POST   /lexi/persona/add_trait              JSON { text }
- POST   /lexi/persona/generate_avatar        JSON {
      prompt?, mode? ("txt2img"|"img2img"|"inpaint"),
      changes?, denoise?, steps?, cfg?, width?, height?,
      seed?, refiner?, refiner_strength?, upscale_factor?,
      source_path?, mask_path?, invert_mask?, allow_feedback_loop?
  }
"""

import asyncio
import os
import json
import base64
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config.config import (
    AVATAR_DIR,
    AVATAR_URL_PREFIX,
    STATIC_ROOT,
    STATIC_URL_PREFIX,
)
from ..persona.persona_core import lexi_persona
from ..session_logging import log_event
from ..memory.memory_core import resolve_memory_root
from ..utils.ip_seed import (
    avatars_public_url_base,
    avatars_static_dir,
    basename_for_ip,
    filename_for_ip,
    ip_to_seed,
)
from ..utils.request_ip import request_ip
from ..utils.publish_static import latest_output_png, publish_as
from ..user_identity import request_user_id
from ..utils.user_identity import normalize_user_id, user_bucket
from ..utils.avatar_manifest import (
    record_avatar_event,
    latest_avatar_path,
    first_avatar_path,
    avatar_manifest_enabled,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/persona", tags=["persona"])
AV_PUBLIC_URL = avatars_public_url_base()
PER_IP_PROMPT = os.getenv(
    "LEXI_PER_IP_AVATAR_PROMPT",
    "cozy cyberpunk librarian Lexi, soft rim light, cinematic portrait, photoreal, editorial lighting",
)
PER_IP_CFG = float(os.getenv("LEXI_PER_IP_CFG", "3.25"))
PER_IP_STEPS = int(os.getenv("LEXI_PER_IP_STEPS", "24"))
PER_IP_WIDTH = int(os.getenv("LEXI_PER_IP_WIDTH", "832"))
PER_IP_HEIGHT = int(os.getenv("LEXI_PER_IP_HEIGHT", "1152"))
AVATAR_JOB_TTL = int(os.getenv("LEXI_AVATAR_JOB_TTL", "600"))

_AVATAR_JOBS: Dict[str, Dict[str, Any]] = {}
_IP_JOB_INDEX: Dict[str, str] = {}
_AVATAR_JOB_LOCK = asyncio.Lock()
_AVATAR_TASKS: set[asyncio.Task] = set()


def _external_base(request: Request) -> str:
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    raw_host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.hostname
        or ""
    ).split(",")[0].strip()
    host = raw_host
    hostname_only = host.split(":")[0]
    if hostname_only.endswith("lexicompanion.com"):
        proto = "https"
    if not host:
        return str(request.base_url).rstrip("/")
    return f"{proto}://{host}".rstrip("/")


def _absolute_url(request: Request, path: str) -> str:
    if isinstance(path, str) and path.startswith("http"):
        return path
    base = _external_base(request)
    target = path or ""
    return f"{base}/{target.lstrip('/')}"


def _client_ip(request: Request) -> str:
    return request_ip(request)


def _avatar_basename_for_ip(ip: str) -> str:
    return basename_for_ip(ip)


def _public_avatar_url(name: str, cache_path: Optional[Path] = None) -> str:
    base = (AV_PUBLIC_URL or AVATAR_URL_PREFIX).rstrip("/")
    url = f"{base}/{name}"
    if cache_path and cache_path.exists():
        try:
            ts = int(cache_path.stat().st_mtime)
            return f"{url}?v={ts}"
        except Exception:
            return url
    return url


def _per_ip_avatar_path(request: Request) -> Optional[str]:
    ip = _client_ip(request)
    filename = filename_for_ip(ip)
    path = AVATARS_PUBLIC_DIR / filename
    if path.exists():
        return _public_avatar_url(filename, path)
    return None


def _fallback_avatar_url(request: Request) -> str:
    candidate = _per_ip_avatar_path(request)
    if not candidate:
        candidate = _public_avatar_url(LEGACY_BASE_NAME, AVATARS_PUBLIC_DIR / LEGACY_BASE_NAME)
    public_base = os.getenv("LEX_PUBLIC_BASE")
    if public_base:
        return f"{public_base.rstrip('/')}{candidate}"
    return _absolute_url(request, candidate)


def _resolve_avatar_url(request: Request, candidate: Optional[str]) -> str:
    if isinstance(candidate, str) and candidate.strip():
        return _absolute_url(request, candidate)
    return _fallback_avatar_url(request)


async def _cleanup_avatar_jobs(now: Optional[float] = None) -> None:
    """Remove stale finished jobs to avoid unbounded growth."""
    now = now or time.time()
    stale: List[str] = []
    for job_id, info in _AVATAR_JOBS.items():
        state = info.get("state")
        updated = info.get("updated", now)
        if state in ("done", "error") and (now - updated) > AVATAR_JOB_TTL:
            stale.append(job_id)
    for job_id in stale:
        job = _AVATAR_JOBS.pop(job_id, None)
        if job:
            ip = job.get("ip")
            if ip and _IP_JOB_INDEX.get(ip) == job_id:
                _IP_JOB_INDEX.pop(ip, None)


async def _run_avatar_job(job_id: str, ip: str) -> None:
    async with _AVATAR_JOB_LOCK:
        job = _AVATAR_JOBS.get(job_id)
        if not job:
            return
        job["state"] = "running"
        job["updated"] = time.time()
    try:
        avatar_path = await ensure_per_ip_avatar(ip)
    except Exception as exc:
        async with _AVATAR_JOB_LOCK:
            job = _AVATAR_JOBS.get(job_id)
            if job:
                job.update(state="error", error=str(exc), updated=time.time())
            if _IP_JOB_INDEX.get(ip) == job_id:
                _IP_JOB_INDEX.pop(ip, None)
        logger.warning("Per-IP avatar job %s failed for %s: %s", job_id, ip, exc)
        return

    rel_url = _public_avatar_url(avatar_path.name, avatar_path)
    async with _AVATAR_JOB_LOCK:
        job = _AVATAR_JOBS.get(job_id)
        if job:
            job.update(
                state="done",
                relative_url=rel_url,
                filename=avatar_path.name,
                updated=time.time(),
            )
        if _IP_JOB_INDEX.get(ip) == job_id:
            _IP_JOB_INDEX.pop(ip, None)

# -------------------- Pipeline import (robust fallback) --------------------

# Prefer Comfy-backed pipeline; fallback to legacy wrapper if present.
_generate_fn = None  # type: ignore[assignment]
try:
    from ..sd.sd_pipeline import generate_avatar_pipeline as _generate_fn  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Legacy wrapper exposes same-ish signature in our refactor
        from ..sd.generate import generate_avatar as _generate_fn  # type: ignore
    except Exception:
        _generate_fn = None  # type: ignore

STATIC_FILES_ROOT = STATIC_ROOT
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
AVATARS_PUBLIC_DIR = avatars_static_dir()
LEGACY_BASE_NAME = "lexi_base.png"
_AVATARS_DIR = AVATAR_DIR  # used for saving generated avatars/base64 fallbacks


def _fs_to_web(fp: Path) -> str:
    """Map a file under STATIC to a /static URL."""
    fp = fp.resolve()
    try:
        rel = fp.relative_to(STATIC_FILES_ROOT.resolve()).as_posix()
    except ValueError:
        # If it's not under STATIC, copy it into avatars and return URL there
        dst = (AVATAR_DIR / fp.name).resolve()
        if dst != fp:
            dst.write_bytes(fp.read_bytes())
        rel = dst.relative_to(STATIC_FILES_ROOT.resolve()).as_posix()
    return f"{STATIC_URL_PREFIX}/{rel}"


def _web_to_fs(static_web_path: str) -> Optional[Path]:
    """Map a /static/... URL to a filesystem path if it exists; else None."""
    if not isinstance(static_web_path, str) or not static_web_path.startswith(
        f"{STATIC_URL_PREFIX}/"
    ):
        return None
    rel = static_web_path[len(STATIC_URL_PREFIX) + 1 :]
    fs = (STATIC_FILES_ROOT / rel).resolve()
    return fs if fs.exists() else None


# -------------------- Trait state helpers --------------------

PERSONA_STATE_NAME = "persona_state.json"

def _persona_state_path(user_id: Optional[str]) -> Path:
    base = resolve_memory_root()
    bucket = user_bucket(base, user_id or "shared")
    if bucket:
        return bucket / PERSONA_STATE_NAME
    fallback = base / "users" / "shared" / PERSONA_STATE_NAME
    try:
        fallback.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return fallback


def _require_user_id(request: Request) -> Optional[str]:
    if getattr(request.state, "needs_disambiguation", False):
        raise HTTPException(status_code=409, detail="identity collision")
    return request_user_id(request)


def _load_traits(user_id: Optional[str] = None) -> Dict[str, str]:
    try:
        path = _persona_state_path(user_id)
        if path.exists():
            data = json.loads(path.read_text())
            traits = data.get("traits", {}) or {}
            if isinstance(traits, dict):
                return {str(k): str(v) for k, v in traits.items()}
    except Exception as exc:
        logger.warning("Failed to load traits: %s", exc)
    return {}


def _save_state(
    traits: Dict[str, str],
    avatar_path: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
) -> None:
    state: Dict[str, Any] = {"traits": traits}
    if avatar_path:
        state["avatar_path"] = avatar_path
    try:
        path = _persona_state_path(user_id)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("Failed to save traits/state: %s", exc)


# --- Back-compat shim for routes.lexi expecting _save_traits() ---
def _save_traits(
    traits: Dict[str, str],
    avatar_path: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    _save_state(traits, avatar_path, user_id=user_id)


def _get_saved_avatar_web(user_id: Optional[str] = None) -> Optional[str]:
    try:
        path = _persona_state_path(user_id)
        if path.exists():
            data = json.loads(path.read_text())
            p = data.get("avatar_path")
            if isinstance(p, str):
                base = p.split("?")[0]
                if base.startswith(f"{STATIC_URL_PREFIX}/"):
                    return base
                # Legacy paths under /static/lex/... → normalize to new prefix
                if base.startswith("/static/lex/avatars") or base.startswith(
                    "/static/lexi/avatars"
                ):
                    name = Path(base).name
                    return f"{AVATAR_URL_PREFIX}/{name}"
                if base.startswith("http://") or base.startswith("https://"):
                    try:
                        parsed = urlparse(base)
                        rel_path = parsed.path or ""
                        if rel_path.startswith("/static/avatars"):
                            name = Path(rel_path).name
                            return f"{AVATAR_URL_PREFIX}/{name}"
                    except Exception:
                        pass
    except Exception:
        pass
    # fallback to persona object if it exposes one
    try:
        p = getattr(lexi_persona, "get_avatar_path", lambda: None)()
        if isinstance(p, str) and p.startswith("/static/"):
            return p
    except Exception:
        pass
    return None


# -------------------- Prompt assembly --------------------


def _get_missing_fields(traits: Dict[str, str]) -> List[str]:
    required_fields = ["hair", "eyes", "outfit", "style", "vibe"]  # minimal core
    return [f for f in required_fields if not traits.get(f)]


def _field_prompt(field: str) -> str:
    """Render a user-facing prompt for a missing persona field, in Lexi-first voice."""
    prompts = {
        "hair": "describe my hair (color/length)",
        "eyes": "what color are my eyes?",
        "outfit": "what am I wearing?",
        "style": "what's my style or aesthetic?",
        "vibe": "what's my vibe or energy?",
    }
    return prompts.get(field, f"describe my {field}")


def _assemble_prompt(traits: Dict[str, str]) -> str:
    defaults = {
        "hair": "medium-length blonde hair with natural highlights",
        "eyes": "blue eyes with lively catchlights",
        "outfit": "casual top",
        "style": "modern editorial",
        "vibe": "confident, playful energy",
        "pose": "relaxed pose, soft smile, natural posture",
        "lighting": "soft directional key light, subtle rim, gentle fill",
        "background": "neutral studio backdrop",
    }
    parts = [
        "photorealistic, natural skin microtexture, gentle film grain, studio-grade color science, depth of field"
    ]
    for k, default in defaults.items():
        parts.append(traits.get(k, default))
    # de-dup simple
    seen, clean = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            clean.append(p)
    return ", ".join(clean)


def _negative_prompt_base() -> str:
    return (
        "anime, cartoon, cgi, plastic skin, over-smooth, waxy, lowres, blurry, noisy, "
        "harsh sharpening, banding, jpeg artifacts, blown highlights, oversaturated skin, "
        "extra limbs, watermark, text"
    )


async def ensure_per_ip_avatar(ip: str, prompt: Optional[str] = None) -> Path:
    """
    Guarantee that a deterministic per-IP avatar exists on disk and return its path.
    """
    target = avatars_static_dir() / filename_for_ip(ip)
    if target.exists():
        return target
    if _generate_fn is None:
        raise RuntimeError("Avatar pipeline is unavailable")

    call_args: Dict[str, Any] = dict(
        prompt=(prompt or PER_IP_PROMPT),
        negative=_negative_prompt_base(),
        width=PER_IP_WIDTH,
        height=PER_IP_HEIGHT,
        steps=PER_IP_STEPS,
        cfg_scale=PER_IP_CFG,
        traits=None,
        seed=ip_to_seed(ip),
        refiner=False,
        refiner_strength=0.25,
        upscale_factor=1.0,
        backend="flux",
        mode="txt2img",
        fresh_base=True,
        base_name=basename_for_ip(ip),
    )
    try:
        import inspect
        import asyncio

        if inspect.iscoroutinefunction(_generate_fn):  # type: ignore[arg-type]
            await _generate_fn(**call_args)  # type: ignore[misc]
        else:
            await asyncio.to_thread(_generate_fn, **call_args)  # type: ignore[misc]
    except Exception as exc:
        logger.exception("Per-IP avatar generation failed for %s: %s", ip, exc)
        raise RuntimeError(f"avatar generation failed: {exc}") from exc

    if not target.exists():
        latest = latest_output_png()
        if latest:
            try:
                publish_as(latest, target.name, avatars_static_dir())
            except Exception as pub_exc:
                logger.debug("Publish fallback failed for %s: %s", ip, pub_exc)
    if not target.exists():
        raise RuntimeError("per-ip avatar render missing")
    return target


# -------------------- Models --------------------


class PersonaOut(BaseModel):
    mode: Optional[str] = None
    traits: Dict[str, str] = Field(default_factory=dict)
    image_path: Optional[str] = None
    certainty: Optional[float] = None


class TraitIn(BaseModel):
    text: str


class AvatarIn(BaseModel):
    prompt: Optional[str] = None
    mode: Optional[Literal["txt2img", "img2img"]] = "txt2img"
    changes: Optional[str] = None
    denoise: Optional[float] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    refiner: Optional[bool] = None
    refiner_strength: Optional[float] = None
    upscale_factor: Optional[float] = None
    source_path: Optional[str] = None  # can be /static/... or absolute FS path
    mask_path: Optional[str] = None  # legacy: ignored in flux-only pipeline
    invert_mask: Optional[bool] = None
    allow_feedback_loop: Optional[bool] = None
    fresh_base: Optional[bool] = None  # force new base via txt2img (ignore existing base)


# -------------------- Endpoints --------------------


@router.api_route("/avatar", methods=["GET", "POST"])
async def persona_avatar(request: Request) -> Dict[str, Any]:
    ip = _client_ip(request)
    filename = filename_for_ip(ip)
    avatar_path = AVATARS_PUBLIC_DIR / filename

    if request.method.upper() == "POST":
        try:
            ensured = await ensure_per_ip_avatar(ip)
        except RuntimeError as exc:
            logger.warning("Per-IP avatar generation failed for %s: %s", ip, exc)
            raise HTTPException(status_code=502, detail=str(exc))
        url = _absolute_url(request, _public_avatar_url(ensured.name, ensured))
        return {"status": "done", "avatar_url": url, "url": url}

    if avatar_path.exists():
        url = _absolute_url(request, _public_avatar_url(avatar_path.name, avatar_path))
        return {"status": "done", "avatar_url": url, "url": url}

    async with _AVATAR_JOB_LOCK:
        await _cleanup_avatar_jobs()
        existing_job_id = _IP_JOB_INDEX.get(ip)
        if existing_job_id:
            job = _AVATAR_JOBS.get(existing_job_id)
            if job and job.get("state") in ("queued", "running"):
                return {"status": job.get("state"), "job_id": existing_job_id}
            _IP_JOB_INDEX.pop(ip, None)

        job_id = uuid.uuid4().hex
        _AVATAR_JOBS[job_id] = {
            "job_id": job_id,
            "ip": ip,
            "state": "queued",
            "created": time.time(),
            "updated": time.time(),
            "relative_url": None,
            "filename": filename,
            "error": None,
        }
        _IP_JOB_INDEX[ip] = job_id

    task = asyncio.create_task(_run_avatar_job(job_id, ip))
    _AVATAR_TASKS.add(task)
    task.add_done_callback(_AVATAR_TASKS.discard)
    return {"status": "queued", "job_id": job_id}


@router.get("/avatar/status/{job_id}")
async def persona_avatar_status(job_id: str, request: Request) -> Dict[str, Any]:
    async with _AVATAR_JOB_LOCK:
        await _cleanup_avatar_jobs()
        job = _AVATAR_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="avatar job not found")

    state = job.get("state", "unknown")
    payload: Dict[str, Any] = {"status": state, "job_id": job_id}
    if state == "done" and job.get("relative_url"):
        absolute = _absolute_url(request, job["relative_url"])
        payload.update(avatar_url=absolute, url=absolute)
    elif state == "error":
        payload["error"] = job.get("error")
    return payload


@router.get("", response_model=PersonaOut)
@router.get("/", response_model=PersonaOut)
async def get_persona(request: Request) -> PersonaOut:
    user_id = _require_user_id(request)
    traits = _load_traits(user_id)
    avatar_web = _get_saved_avatar_web(user_id)
    image_path = _resolve_avatar_url(request, avatar_web)

    ready = not _get_missing_fields(traits)
    return PersonaOut(
        mode="default",
        traits=traits,
        image_path=image_path,
        certainty=1.0 if ready else 0.8,
    )


@router.post("/add_trait")
async def add_trait(request: Request, body: TraitIn) -> Dict[str, Any]:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    log_event(request, "user", text, event="persona_add_trait")

    user_id = _require_user_id(request)
    traits = _load_traits(user_id)
    missing = _get_missing_fields(traits)
    if missing:
        # Fill the next missing field with user-provided text
        traits[missing[0]] = text
        _save_state(traits, user_id=user_id)

    ready = not _get_missing_fields(traits)
    prompt = _assemble_prompt(traits) if ready else ""
    negative = _negative_prompt_base() if ready else ""
    narration = (
        "Here's your look! If you'd like to tweak anything, tell me what to change."
        if ready
        else (
            f"Got it. Next: { _field_prompt(_get_missing_fields(traits)[0]) }"
            if _get_missing_fields(traits)
            else ""
        )
    )

    persona = {
        "traits": traits,
        "certainty": 1.0 if ready else 0.8,
        "image_path": _resolve_avatar_url(request, _get_saved_avatar_web(user_id)),
    }
    if narration:
        log_event(
            request,
            "assistant",
            narration,
            event="persona_trait_narration",
            ready=ready,
            traits=list(traits.keys()),
        )
    return {
        "ready": ready,
        "persona": persona,
        "prompt": prompt,
        "negative": negative,
        "narration": narration,
        "added": True,
    }


@router.post("/generate_avatar")
async def generate_avatar(request: Request, body: AvatarIn) -> Dict[str, Any]:
    if _generate_fn is None:
        raise HTTPException(status_code=500, detail="Avatar pipeline is unavailable")

    user_id = _require_user_id(request)
    traits = _load_traits(user_id)
    ip = _client_ip(request)
    base_name = normalize_user_id(user_id) or _avatar_basename_for_ip(ip)

    # ---- Resolve mode
    mode = (body.mode or "txt2img").lower()
    if mode not in ("txt2img", "img2img", "inpaint"):
        mode = "txt2img"

    # ---- Resolve prompt/negative
    prompt = (body.prompt or "").strip()
    if not prompt and traits and not _get_missing_fields(traits):
        prompt = _assemble_prompt(traits)
    if not prompt:
        prompt = "portrait of Lexi, shoulders-up, engaging eye contact, soft studio lighting"

    negative = _negative_prompt_base()

    # ---- Dimensions & params (sane defaults for SDXL)
    width = body.width if isinstance(body.width, int) else 832
    height = body.height if isinstance(body.height, int) else 1152
    steps = body.steps if isinstance(body.steps, int) else 30
    cfg = body.cfg if isinstance(body.cfg, (int, float)) else 5.0
    seed = body.seed if isinstance(body.seed, int) else None
    if seed is None:
        seed = ip_to_seed(ip)
    refiner = bool(body.refiner) if isinstance(body.refiner, bool) else (mode == "txt2img")
    refiner_strength = (
        float(body.refiner_strength) if isinstance(body.refiner_strength, (int, float)) else 0.28
    )
    upscale_factor = (
        float(body.upscale_factor) if isinstance(body.upscale_factor, (int, float)) else 1.0
    )

    changes = (body.changes or "").strip() if body and body.changes else ""
    denoise = None
    if isinstance(body.denoise, (int, float)):
        denoise = float(max(0.10, min(0.75, float(body.denoise))))

    # ---- Resolve source/mask paths (accept /static or absolute FS)
    def resolve_path(maybe_path: Optional[str]) -> Optional[str]:
        if not maybe_path:
            return None
        if maybe_path.startswith("/static/"):
            p = _web_to_fs(maybe_path)
            return p.as_posix() if p else None
        # accept absolute filesystem path
        p = Path(maybe_path)
        return p.as_posix() if p.exists() else None

    src_fs: Optional[str] = None
    mask_fs: Optional[str] = None

    manifest_src: Optional[str] = None
    if avatar_manifest_enabled() and user_id:
        manifest_src = latest_avatar_path(user_id) or first_avatar_path(user_id)
        if manifest_src:
            if manifest_src.startswith("/"):
                p = _web_to_fs(manifest_src)
                manifest_src = p.as_posix() if p else manifest_src
            elif manifest_src.startswith("http"):
                manifest_src = None  # skip http source for now
            else:
                p = Path(manifest_src)
                manifest_src = p.as_posix() if p.exists() else None

    if mode in ("img2img", "inpaint"):
        # Prefer caller-provided source_path; otherwise fall back to saved avatar
        src_fs = resolve_path(body.source_path) if body.source_path else None
        if src_fs is None:
            saved_web = _get_saved_avatar_web(user_id)
            if saved_web:
                p = _web_to_fs(saved_web)
                src_fs = p.as_posix() if p else None
        if src_fs is None and manifest_src:
            src_fs = manifest_src
        if src_fs is None:
            # degrade gracefully to txt2img if no source image available
            mode = "txt2img"

    if mode == "inpaint":
        mask_fs = resolve_path(body.mask_path)
        if mask_fs is None:
            raise HTTPException(
                status_code=400,
                detail="inpaint requires mask_path (as /static/... or absolute path)",
            )

    invert_mask = bool(body.invert_mask) if isinstance(body.invert_mask, bool) else False
    allow_feedback_loop = (
        bool(body.allow_feedback_loop) if isinstance(body.allow_feedback_loop, bool) else True
    )

    # ---- Build call kwargs for pipeline
    call_args: Dict[str, Any] = dict(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg,
        traits=traits,
        seed=seed,
        refiner=refiner,
        refiner_strength=refiner_strength,
        upscale_factor=upscale_factor,
        base_name=base_name,
    )

    if changes:
        call_args["changes"] = changes

    if mode == "img2img":
        call_args.update(
            mode="img2img",
            source_path=src_fs,
        )
        if denoise is not None:
            call_args["denoise"] = denoise
        # Optional: prevent runaway self-reprocessing unless allowed
        if not allow_feedback_loop and src_fs and src_fs.startswith(_AVATARS_DIR.as_posix()):
            raise HTTPException(
                status_code=400,
                detail="Refusing to reprocess generated avatar without allow_feedback_loop=True",
            )

    elif mode == "inpaint":
        call_args.update(
            mode="inpaint",
            source_path=src_fs,
            mask_path=mask_fs,
            invert_mask=invert_mask,
        )
        if denoise is not None:
            call_args["denoise"] = denoise

    else:
        call_args["mode"] = "txt2img"
        if body.fresh_base is True:
            call_args["fresh_base"] = True

    # If caller explicitly requested fresh_base with any mode, honor it
    if body.fresh_base is True:
        call_args["fresh_base"] = True

    def _record_success(url: str, narration_text: Optional[str]) -> None:
        log_event(
            request,
            "assistant",
            narration_text or "Avatar ready",
            event="persona_generate_avatar",
            mode=mode,
            prompt=prompt,
            url=url,
            traits=list(traits.keys()),
        )
        if avatar_manifest_enabled() and user_id:
            try:
                fs_ref = None
                if url:
                    p = _web_to_fs(url) if url.startswith("/") else None
                    fs_ref = p.as_posix() if p else None
                record_avatar_event(
                    user_id,
                    image_path=fs_ref or url,
                    prompt=prompt,
                    traits=traits,
                    mode=mode,
                    seed=seed,
                    session_id=getattr(request.state, "session_id", None),
                )
            except Exception:
                logger.debug("avatar manifest record failed", exc_info=True)

    # ---- Execute pipeline (sync or async)
    try:
        import inspect, asyncio  # local import to avoid module-scope surprises

        if inspect.iscoroutinefunction(_generate_fn):  # type: ignore[arg-type]
            result: Dict[str, Any] = await _generate_fn(**call_args)  # type: ignore[misc]
        else:
            result = await asyncio.to_thread(_generate_fn, **call_args)  # type: ignore[misc]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Avatar pipeline crashed: %s", e)
        log_event(
            request,
            "error",
            f"pipeline_error: {e}",
            event="persona_generate_avatar_error",
            mode=mode,
            prompt=prompt,
        )
        raise HTTPException(status_code=500, detail=f"pipeline_error: {e}")

    # ---- Normalize output
    # Preferred: a direct URL (e.g., /static/avatars/xxx.png)
    web_url = None
    if isinstance(result, dict):
        web_url = result.get("avatar_url") or result.get("url") or result.get("image")

    if isinstance(web_url, str) and web_url.strip():
        absolute = _absolute_url(request, web_url)
        stored_path = web_url if web_url.startswith("/") else absolute
        _save_state(traits, avatar_path=stored_path, user_id=user_id)
        _record_success(absolute, result.get("narration", "Here she is!"))
        base_part = web_url.split("?")[0]
        filename = Path(base_part).name
        return {
            "ok": True,
            "image": absolute,
            "url": absolute,
            "avatar_url": absolute,
            "filename": filename,
            "narration": result.get("narration", "Here she is!"),
            "traits": traits,
        }

    # Next best: a filesystem path to an image
    file_path = None
    if isinstance(result, dict):
        file_path = result.get("file") or result.get("path")
    if isinstance(file_path, str) and Path(file_path).exists():
        web_url = _fs_to_web(Path(file_path))
        _save_state(traits, avatar_path=web_url, user_id=user_id)
        absolute = _absolute_url(request, web_url)
        _record_success(absolute, result.get("narration", "Here she is!"))
        return {
            "ok": True,
            "image": absolute,
            "url": absolute,
            "avatar_url": absolute,
            "filename": web_url.split("/")[-1],
            "narration": result.get("narration", "Here she is!"),
            "traits": traits,
        }

    # Fallback: base64
    if isinstance(result, dict):
        b64 = result.get("image_b64") or result.get("b64")
        if isinstance(b64, str) and len(b64) > 100:
            fname = f"lexi_{os.urandom(4).hex()}.png"
            out = (_AVATARS_DIR / fname).resolve()
            out.write_bytes(base64.b64decode(b64))
            web_url = _fs_to_web(out)
            _save_state(traits, avatar_path=web_url, user_id=user_id)
            absolute = _absolute_url(request, web_url)
            _record_success(absolute, result.get("narration"))
            return {
                "ok": True,
                "image": absolute,
                "url": absolute,
                "avatar_url": absolute,
                "filename": fname,
                "traits": traits,
            }

    logger.error("[Avatar Gen Error] No usable image returned. Raw result=%r", result)
    raise HTTPException(status_code=502, detail="no_image_from_pipeline")


@router.get("/debug/traits")
def debug_traits(request: Request) -> Dict[str, Any]:
    user_id = _require_user_id(request)
    file_traits = _load_traits(user_id)
    try:
        persona_traits = getattr(lexi_persona, "get_traits", lambda: {})()
    except Exception:
        persona_traits = {}
    return {"file_traits": file_traits, "persona_traits": persona_traits}
