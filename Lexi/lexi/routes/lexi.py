# File: Lexi/lexi/routes/lexi.py
from __future__ import annotations

"""
lexi.py

Core Lexi API routes: chat processing, persona mode management, simple intent
classification, and avatar generation endpoints wired to sd_pipeline.generate_avatar_pipeline.
"""

import json
import logging
import asyncio
import base64
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from ..utils.prompt_sifter import build_sd_prompt, extract_categories
from .lexi_persona import _load_traits, _save_traits
from ..core.config import settings
from ..config.config import STATIC_URL_PREFIX, STARTER_AVATAR_PATH

# alias public names
load_traits = _load_traits
save_traits = _save_traits
from ..persona.persona_core import lexi_persona
from ..persona.persona_config import PERSONA_MODE_REGISTRY
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..alpha.session_manager import SessionRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Lexi Core"])


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

# Paths for avatar storage and trait state persistence
AVATAR_DIR: Path = settings.AVATAR_DIR
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
TRAIT_STATE_PATH: Path = Path(__file__).resolve().parent / "lexi_persona_state.json"
STATIC_ROOT: Path = settings.STATIC_ROOT


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str


class IntentRequest(BaseModel):
    text: str


class AvatarGenRequest(BaseModel):
    prompt: Optional[str] = None  # base description (txt2img) or ignored if base exists
    changes: Optional[str] = None  # appended when doing img2img/inpaint
    traits: Optional[Dict[str, str]] = None

    mode: Optional[str] = None  # "txt2img" | "img2img" | "inpaint" (pipeline may override)
    source_path: Optional[str] = None  # for inpaint
    mask_path: Optional[str] = None  # for inpaint

    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    refiner: Optional[bool] = None
    refiner_strength: Optional[float] = None
    upscale_factor: Optional[float] = None

    # styling + control
    intent: Optional[str] = None  # "light" | "medium" | "strong"  (img2img denoise)
    nsfw: Optional[bool] = None
    style: Optional[str] = None  # "realistic" | "cinematic" | "stylized"
    negative: Optional[str] = None

    allow_feedback_loop: Optional[bool] = None
    fresh_base: Optional[bool] = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def cache_busted_url(file_path: Path) -> str:
    """Append file mtime to static URL to bust caches."""
    file_path = file_path.resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Avatar not found")
    try:
        rel_path = file_path.relative_to(STATIC_ROOT).as_posix()
    except ValueError:
        raise HTTPException(status_code=500, detail="Avatar path outside static root")
    ts = int(file_path.stat().st_mtime)
    return f"{STATIC_URL_PREFIX}/{rel_path}?v={ts}"


# ---------------------------------------------------------------------
# Health / Ready
# ---------------------------------------------------------------------
@router.get("/health")
def health():
    return {"ok": True, "service": "lexi-backend"}


@router.get("/ready")
def ready():
    return {"ready": True}


# ---------------------------------------------------------------------
# Minimal intent classifier (for FE routing hints)
# ---------------------------------------------------------------------
_AVATAR_EDIT = re.compile(
    r"\b("
    r"change|swap|update|alter|make|put on|wear|add|remove|switch|try|"
    r"dye|color|style|curl|straighten|shorten|lengthen|"
    r"skirt|stockings?|thigh[- ]?highs?|heels?|boots?|jacket|dress|"
    r"hair|bangs|ponytail|braid|"
    r"sexy|hot|cute|edgy|goth|glam|casual|lingerie|nsfw"
    r")\b",
    re.IGNORECASE,
)

_DESCRIBE = re.compile(
    r"\b(describe|what do you look like|show me your look|how do you look|what.*look.*now)\b",
    re.IGNORECASE,
)

_AVATAR_TOPIC = re.compile(
    r"\b(avatar|appearance|outfit|clothes?|dress|costume|style|wear|lingerie|look|look like)\b",
    re.IGNORECASE,
)

# Explicit request to reset/create a brand-new base image
_NEW_LOOK = re.compile(
    r"\b(new look|start over|fresh look|reset (my|the) look|new avatar|new base|start fresh)\b",
    re.IGNORECASE,
)


@router.post("/intent")
async def classify_intent(req: IntentRequest) -> Dict[str, str]:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    if _NEW_LOOK.search(text):
        return {"intent": "new_look"}
    if _DESCRIBE.search(text):
        return {"intent": "describe_avatar"}
    if _AVATAR_EDIT.search(text):
        return {"intent": "avatar_edit"}
    if _AVATAR_TOPIC.search(text):
        return {"intent": "avatar_flow"}
    return {"intent": "chat"}


# ---------------------------------------------------------------------
# Heuristics for img2img strength & nsfw
# ---------------------------------------------------------------------
_NSFW_HINTS = (
    "nsfw",
    "nude",
    "nothing else",
    "thigh-high",
    "stockings",
    "lingerie",
    "lewd",
    "topless",
)
_MEDIUM_HINTS = (
    "new look",
    "change outfit",
    "switch",
    "different hair",
    "skirt",
    "jacket",
    "dress",
)


def infer_intent_and_nsfw(prompt: Optional[str], changes: Optional[str]) -> tuple[str, bool]:
    txt = f"{prompt or ''} {changes or ''}".lower()
    if any(k in txt for k in _NSFW_HINTS):
        return "strong", True
    if any(k in txt for k in _MEDIUM_HINTS):
        return "medium", False
    return "light", False


# ---------------------------------------------------------------------
# Main chat route
# ---------------------------------------------------------------------
def _append_session_memory(request: Request, event: Dict[str, Any]) -> None:
    registry = getattr(request.app.state, "alpha_sessions", None)
    session_id = request.headers.get("X-Lexi-Session")
    if not session_id or not isinstance(registry, SessionRegistry):
        return
    try:
        registry.append_memory(session_id, event)
    except Exception:
        pass


@router.post("/process")
async def process(req: ChatRequest, request: Request):
    logger.info("🗨️ /process prompt=%r", req.prompt)
    _append_session_memory(
        request,
        {"role": "user", "event": "chat_prompt", "text": req.prompt},
    )

    # Guard against assistant loops
    if req.prompt.startswith(("Lexi:", "assistant:")):
        logger.warning("🛑 Ignoring looped assistant prompt: %r", req.prompt)
        return JSONResponse(
            {
                "cleaned": "[loop detected, halted]",
                "raw": "",
                "choices": [],
                "mode": lexi_persona.get_mode(),
            }
        )

    # Trait extraction (optional; only if wired)
    inferred = extract_traits_from_text(req.prompt)
    if inferred:
        traits = load_traits()
        traits.update(inferred)
        save_traits(traits)

        result = generate_avatar_pipeline(traits=traits)
        if not result.get("ok"):
            raise HTTPException(
                status_code=502, detail=result.get("error", "avatar generation failed")
            )

        url = result.get("url")
        if url:
            lexi_persona.set_avatar_path(url)  # type: ignore
            save_traits(traits, avatar_path=url)

        _append_session_memory(
            request,
            {
                "role": "assistant",
                "event": "chat_avatar_update",
                "traits": traits,
                "avatar_url": url,
            },
        )
        return JSONResponse(
            {
                "cleaned": "Got it, updating her look! 💄",
                "avatar_url": url,
                "traits": traits,
                "mode": lexi_persona.get_mode(),
            }
        )

    # Normal chat
    if not isinstance(req.prompt, str):
        raise HTTPException(status_code=400, detail="Prompt must be a string")

    wants_stream = "application/x-ndjson" in request.headers.get("accept", "").lower()

    def finalize_reply(raw_reply: str) -> Dict[str, Any]:
        reply_text = raw_reply or ""
        if not getattr(reply_text, "strip", None) or not reply_text.strip():
            logger.warning("❌ Empty reply for prompt: %r", req.prompt)
            reply_text = "[no response]"

        try:
            memory.remember(
                MemoryShard(
                    role="assistant", content=reply_text, meta={"tags": ["chat"], "compressed": True}
                )
            )
        except Exception as mem_err:
            logger.warning("⚠️ Memory store skipped: %s", mem_err)

        _append_session_memory(
            request,
            {"role": "assistant", "event": "chat_reply", "text": reply_text},
        )

        return {
            "cleaned": reply_text,
            "raw": reply_text,
            "choices": [{"text": reply_text}],
            "mode": lexi_persona.get_mode(),
        }

    if wants_stream:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        final_payload: Dict[str, Any] = {}
        error_holder: Dict[str, str] = {}

        def stream_callback(chunk: str) -> None:
            if not chunk:
                return
            loop.call_soon_threadsafe(queue.put_nowait, chunk)

        async def run_chat() -> None:
            try:
                reply_value = await asyncio.to_thread(
                    lexi_persona.chat, req.prompt, stream_callback=stream_callback
                )
                final_payload.update(finalize_reply(reply_value))
            except Exception as exc:
                logger.error("❌ LexiPersona.chat failed: %s", exc)
                error_holder["message"] = str(exc)
                _append_session_memory(
                    request,
                    {"role": "assistant", "event": "chat_error", "error": str(exc)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(run_chat())

        async def event_stream() -> AsyncIterator[str]:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield json.dumps({"delta": chunk}) + "\n"

            if error_holder:
                yield json.dumps({"error": error_holder["message"]}) + "\n"
            else:
                # Ensure we always emit a completion payload
                if not final_payload:
                    final_payload.update(finalize_reply(""))
                yield json.dumps({"done": True, **final_payload}) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    try:
        reply = await asyncio.to_thread(lexi_persona.chat, req.prompt)
    except Exception as exc:
        logger.error("❌ LexiPersona.chat failed: %s", exc)
        _append_session_memory(
            request,
            {"role": "assistant", "event": "chat_error", "error": str(exc)},
        )
        return JSONResponse(
            {"cleaned": f"[error] {exc}", "raw": "", "choices": [], "mode": lexi_persona.get_mode()}
        )

    payload = finalize_reply(reply)
    return JSONResponse(payload)


# ---------------------------------------------------------------------
# Set persona mode
# ---------------------------------------------------------------------
@router.post("/set_mode")
async def set_mode(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    mode = payload.get("mode")
    if mode not in PERSONA_MODE_REGISTRY:
        logger.warning("🛑 Invalid mode: %s", mode)
        raise HTTPException(status_code=400, detail="Invalid mode")
    lexi_persona.set_mode(mode)
    save_traits({}, avatar_path=None)
    logger.info("✅ Mode set to: %s", mode)
    return {"status": "ok", "mode": mode}


# ---------------------------------------------------------------------
# Persona avatar generation endpoints
# ---------------------------------------------------------------------
persona_router = APIRouter(prefix="/persona", tags=["persona"])


@persona_router.post("/generate_avatar")
async def generate_avatar(request: Request, req: AvatarGenRequest) -> JSONResponse:
    """
    Unified avatar generation endpoint.

    Pipeline policy:
      - If lexi_base.png does not exist: pipeline will create it (txt2img).
      - Otherwise: pipeline will img2img from lexi_base.png.
    """
    traits = req.traits or load_traits()
    prompt = req.prompt
    changes = req.changes

    # Derive intent/nsfw if not explicitly provided
    auto_intent, auto_nsfw = infer_intent_and_nsfw(prompt, changes)
    intent = (req.intent or "").strip().lower() or auto_intent
    nsfw = req.nsfw if req.nsfw is not None else auto_nsfw

    try:
        result = generate_avatar_pipeline(
            prompt=prompt,
            negative=req.negative,
            width=req.width or 832,
            height=req.height or 1152,
            steps=req.steps or 30,
            cfg_scale=req.cfg_scale or 5.0,
            traits=traits if traits else None,
            mode=req.mode or "txt2img",  # pipeline will override to img2img if base exists
            source_path=req.source_path,
            mask_path=req.mask_path,
            changes=changes,
            seed=req.seed,
            refiner=(
                bool(req.refiner) if req.refiner is not None else False
            ),  # refiner off by default for edits
            refiner_strength=req.refiner_strength or 0.28,
            upscale_factor=req.upscale_factor or 1.0,
            intent=intent,
            nsfw=nsfw,
            style=req.style or "realistic",
            allow_feedback_loop=bool(req.allow_feedback_loop),
            fresh_base=(bool(req.fresh_base) if req.fresh_base is not None else False)
            or bool(_NEW_LOOK.search((prompt or "") + " " + (changes or ""))),
        )
    except Exception as exc:
        logger.error("Avatar pipeline failed: %s", exc)
        return JSONResponse(
            {
                "ok": False,
                "url": _absolute_url(request, STARTER_AVATAR_PATH),
                "error": str(exc),
                "intent": intent,
                "nsfw": nsfw,
            }
        )

    if not result.get("ok"):
        return JSONResponse(
            {
                "ok": False,
                "url": _absolute_url(request, STARTER_AVATAR_PATH),
                "error": result.get("error", "Comfy pipeline error"),
                "intent": intent,
                "nsfw": nsfw,
            }
        )

    url = result["url"]
    # Return absolute URL based on incoming request host/port
    if isinstance(url, str) and url.startswith("/"):
        abs_url = _absolute_url(request, url)
    else:
        abs_url = url
    meta = result.get("meta", {})

    # Persist avatar URL to persona state
    try:
        # Store relative URL if provided to keep storage portable
        lexi_persona.set_avatar_path(url)  # type: ignore
        save_traits(traits, avatar_path=url)
    except Exception as exc:
        logger.warning("⚠️ Persona state update failed: %s", exc)

    return JSONResponse({"ok": True, "url": abs_url, "meta": meta, "intent": intent, "nsfw": nsfw})


# Legacy alias for backwards compatibility
@persona_router.post("/avatar")
def avatar_endpoint(req: ChatRequest) -> JSONResponse:
    result = generate_avatar_pipeline(prompt=req.prompt, refiner=True, style="realistic")
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "avatar generation failed"))
    return JSONResponse({"ok": True, "url": result["url"], "meta": result.get("meta", {})})


router.include_router(persona_router)


# ---------------------------------------------------------------------
# (Optional) Legacy A1111 endpoint — safe to remove if unused
# ---------------------------------------------------------------------
@router.post("/image_from_prompt")
def image_from_prompt(req: ChatRequest) -> Any:
    logger.info("🔮 Raw image_from_prompt: %s", req.prompt)
    try:
        data = build_sd_prompt(req.prompt)  # type: ignore
        prompt, negative = data["prompt"], data.get("negative")
        response = requests.post(
            "http://localhost:7860/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative,
                "steps": 20,
                "width": 256,
                "height": 512,
                "cfg_scale": 7,
                "sampler_name": "Euler a",
            },
            timeout=60,
        )
        response.raise_for_status()
        img_b64 = response.json()["images"][0]
    except Exception as exc:
        logger.error("🛑 SD API failure: %s", exc)
        return PlainTextResponse(STARTER_AVATAR_PATH, status_code=500)

    fname = f"lexi_avatar_{uuid4().hex[:6]}.png"
    file_path = AVATAR_DIR / fname
    file_path.write_bytes(base64.b64decode(img_b64))
    return {"image_url": cache_busted_url(file_path)}


# Aliases for backward compatibility
_load_traits_state = load_traits
_save_traits_state = save_traits

__all__ = ["router", "_load_traits_state", "_save_traits_state"]


def extract_traits_from_text(text: str) -> Dict[str, str]:
    """Heuristic trait extraction based on visual keywords."""
    categories = extract_categories(text or "")
    return {key: value for key, value in categories.items() if value}
