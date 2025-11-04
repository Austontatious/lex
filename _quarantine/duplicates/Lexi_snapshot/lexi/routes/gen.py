# Lexi/lexi/routes/gen.py
from __future__ import annotations
from typing import Dict, Optional, Any, List
import os
import base64
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..sd.sd_pipeline import generate_avatar_pipeline
from .lexi_persona import _load_traits as load_traits, _save_traits as save_traits
from ..alpha.session_manager import SessionRegistry
from ..alpha.settings import AlphaSettings
from ..alpha.tour import preview_placeholder_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/gen", tags=["gen"])
# --- Resolve the same /static root the app mounted ---
# Keep this in sync with backend_core.py discovery!
from pathlib import Path as _P
_PROJECT_ROOT = _P(__file__).resolve().parents[2]         # .../Lexi
_STATIC_CANDIDATES = [
    _PROJECT_ROOT / "static",                 # Lexi/static
    _PROJECT_ROOT / "lexi" / "static",         # Lexi/lexi/static
    _P("/mnt/data/Lexi/static"),               # optional external mount
]
_STATIC_DIR = next((p for p in _STATIC_CANDIDATES if p.exists()), None)
if _STATIC_DIR is None:
    # We'll still try to write into Lexi/lexi/static as a fallback
    _STATIC_DIR = _PROJECT_ROOT / "lexi" / "static"
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)

_AVATARS_DIR = _STATIC_DIR / "lexi" / "avatars"
_AVATARS_DIR.mkdir(parents=True, exist_ok=True)

def _to_static_url(fp: Path) -> str:
    # compute URL like /static/lexi/avatars/filename.png
    rel = fp.relative_to(_STATIC_DIR).as_posix()
    return f"/static/{rel}"

def _write_b64_png(b64: str) -> str:
    fname = f"lexi_avatar_{os.urandom(3).hex()}.png"
    out = _AVATARS_DIR / fname
    out.write_bytes(base64.b64decode(b64))
    return _to_static_url(out)

def _copy_local(src: str) -> Optional[str]:
    p = Path(src)
    if not p.exists():
        return None
    dest = _AVATARS_DIR / p.name
    if str(p.resolve()) != str(dest.resolve()):
        dest.write_bytes(p.read_bytes())
    return _to_static_url(dest)

def _download_to_static(url: str) -> Optional[str]:
    try:
        import requests
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        fname = f"lexi_avatar_{os.urandom(3).hex()}.png"
        out = _AVATARS_DIR / fname
        out.write_bytes(r.content)
        return _to_static_url(out)
    except Exception as e:
        logger.warning("download failed: %s", e)
        return None

class AvatarGenRequest(BaseModel):
    prompt: Optional[str] = None
    persona_mode: Optional[str] = None        # <-- rename: previously "mode"
    sd_mode: Optional[str] = Field(default="txt2img", pattern="^(txt2img|img2img|inpaint)$")
    model: Optional[str] = Field(default=None, pattern="^(sdxl|flux)$")
    variant: Optional[str] = None
    preset: Optional[str] = None
    size: Optional[str] = None
    guidance: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    traits: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    strength: Optional[float] = Field(default=0.65, ge=0.0, le=1.0)
    steps: Optional[int] = Field(default=28, ge=1, le=200)
    width: Optional[int] = Field(default=768, ge=64, le=2048)
    height: Optional[int] = Field(default=1024, ge=64, le=2048)
    upscale: Optional[bool] = False
    refine: Optional[bool] = True
    negative_prompt: Optional[str] = None
    loras: Optional[List[str]] = None
    source_path: Optional[str] = None         # for img2img/inpaint (explicit only)

@router.post("/avatar")
async def generate_avatar_endpoint(
    request: Request,
    req: AvatarGenRequest = Body(...),
):
    from ..sd.generate import generate_avatar as _generate
    registry = getattr(request.app.state, "alpha_sessions", None)
    session_id = request.headers.get("X-Lexi-Session")
    settings = AlphaSettings()

    if settings.alpha_strict:
        placeholder = preview_placeholder_url(settings)
        if isinstance(registry, SessionRegistry) and session_id:
            registry.append_memory(
                session_id,
                {
                    "role": "assistant",
                    "event": "alpha_strict_avatar_stub",
                    "source": "gen_endpoint",
                    "upscale": bool(req.upscale),
                },
            )
            registry.record_metric(
                session_id,
                {"event": "alpha_strict_stub", "feature": "avatar_generate"},
            )
        return JSONResponse(
            {
                "image": placeholder,
                "filename": placeholder.split("/")[-1],
                "narration": "alpha strict mode — placeholder preview.",
                "traits": req.traits,
            }
        )

    if req.upscale and isinstance(registry, SessionRegistry) and session_id:
        if not registry.increment_counter(session_id, "avatar_upscale", limit=1):
            registry.record_metric(
                session_id,
                {"event": "rate_limited", "feature": "avatar_upscale"},
            )
            raise HTTPException(
                status_code=429,
                detail="upscale limit reached for this session.",
            )
        registry.append_memory(
            session_id,
            {
                "role": "user",
                "event": "avatar_upscale_requested",
                "traits": req.traits,
            },
        )

    try:
        # IMPORTANT: do NOT forward persona_mode into sd "mode"
        result = await run_in_threadpool(
            _generate,
            prompt=req.prompt or "",
            negative=req.negative_prompt or "low quality, blurry, deformed",
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=4.5,
            mode=req.sd_mode,
            source_path=req.source_path if req.sd_mode != "txt2img" else None,
            denoise=min(max(req.strength or 0.35, 0.10), 0.75),
            seed=req.seed,
            changes=None,
            refiner=bool(req.refine and req.sd_mode == "txt2img"),
            refiner_strength=0.28,
            upscale_factor=(1.25 if req.upscale else 1.0),
            backend=req.model,
            variant=req.variant,
            flux_variant=req.variant,
            flux_preset=req.preset,
            flux_size=req.size,
            flux_guidance=req.guidance,
            flux_cfg=req.cfg,
            flux_sampler=req.sampler,
            flux_scheduler=req.scheduler,
            flux_denoise=req.strength if req.sd_mode != "txt2img" else None,
        )
        if isinstance(registry, SessionRegistry) and session_id:
            registry.append_memory(
                session_id,
                {
                    "role": "assistant",
                    "event": "avatar_generate_response",
                    "upscale": bool(req.upscale),
                },
            )
            registry.record_metric(
                session_id,
                {"event": "avatar_generate", "upscale": bool(req.upscale)},
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"avatar generation failed: {e}")

@router.post("/generate")
async def generate_avatar(
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(None),
) -> JSONResponse:
    """
    Generate an avatar based on provided traits or last saved state.
    Returns: {"image": "/static/lexi/avatars/....png", "filename": "...", "narration": "...", "traits": {...}}
    """
    registry = getattr(request.app.state, "alpha_sessions", None)
    session_id = request.headers.get("X-Lexi-Session")
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
            result = await asyncio.to_thread(generate_avatar_pipeline, traits=traits, mode="txt2img")
    except Exception as e:
        logger.exception("Pipeline crashed: %s", e)
        raise HTTPException(status_code=500, detail=f"pipeline_crash: {e}")

    # Normalize outputs (unchanged) ...
    b64 = (result or {}).get("image_b64") or (result or {}).get("b64")
    if isinstance(b64, str) and len(b64) > 100:
        web = _write_b64_png(b64)
        save_traits(traits, avatar_path=web)
        resp = JSONResponse({"image": web, "filename": web.split("/")[-1], "narration": result.get("narration", ""), "traits": traits})
        if isinstance(registry, SessionRegistry) and session_id:
            registry.append_memory(
                session_id,
                {"role": "assistant", "event": "avatar_generate_response", "source": "legacy_generate"},
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
            return JSONResponse({"image": p, "filename": p.split("/")[-1], "narration": result.get("narration", ""), "traits": traits})
        if p.startswith(("http://", "https://")):
            web = _download_to_static(p)
            if web:
                save_traits(traits, avatar_path=web)
                resp = JSONResponse({"image": web, "filename": web.split("/")[-1], "narration": result.get("narration", ""), "traits": traits})
                if isinstance(registry, SessionRegistry) and session_id:
                    registry.append_memory(
                        session_id,
                        {"role": "assistant", "event": "avatar_generate_response", "source": "legacy_generate"},
                    )
                    registry.record_metric(
                        session_id,
                        {"event": "avatar_generate"},
                    )
                return resp
        else:
            web = _copy_local(p)
            if web:
                save_traits(traits, avatar_path=web)
                resp = JSONResponse({"image": web, "filename": web.split("/")[-1], "narration": result.get("narration", ""), "traits": traits})
                if isinstance(registry, SessionRegistry) and session_id:
                    registry.append_memory(
                        session_id,
                        {"role": "assistant", "event": "avatar_generate_response", "source": "legacy_generate"},
                    )
                    registry.record_metric(
                        session_id,
                        {"event": "avatar_generate"},
                    )
                return resp

    logger.error("[Avatar Gen Error] No usable image returned. Raw result=%r", result)
    raise HTTPException(status_code=500, detail={"error": "no_image_from_pipeline", "raw": result})
