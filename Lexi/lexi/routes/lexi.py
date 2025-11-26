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
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from ..utils.prompt_sifter import build_sd_prompt, extract_categories
from .lexi_persona import _load_traits, _save_traits, ensure_per_ip_avatar
from .now import router as now_router, tools as tools_router
from ..config.config import AVATAR_URL_PREFIX, STARTER_AVATAR_PATH
load_traits = _load_traits
save_traits = _save_traits
from ..persona.persona_core import lexi_persona
from ..persona.persona_config import PERSONA_MODE_REGISTRY
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..alpha.session_manager import SessionRegistry
from ..session_logging import log_event
from ..utils.ip_seed import (
    avatars_public_url_base,
    avatars_static_dir,
    client_ip_from_headers,
    filename_for_ip,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Lexi Core"])

USE_COMFY_ONLY = os.getenv("LEX_USE_COMFY_ONLY", "0") == "1"

_DEFAULT_AVATAR_DIR = Path(__file__).resolve().parent.parent / "static" / "lexi" / "avatars"
_AVATAR_DIR_OVERRIDE = os.getenv("LEX_AVATAR_DIR")
if _AVATAR_DIR_OVERRIDE:
    try:
        AVATAR_DIR: Path = Path(_AVATAR_DIR_OVERRIDE).expanduser().resolve()
    except Exception:
        logger.warning("⚠️ Invalid LEX_AVATAR_DIR=%r; falling back to default.", _AVATAR_DIR_OVERRIDE)
        AVATAR_DIR = _DEFAULT_AVATAR_DIR
else:
    AVATAR_DIR = _DEFAULT_AVATAR_DIR

try:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("⚠️ Could not ensure avatar directory %s: %s", AVATAR_DIR, exc)

AVATARS_PUBLIC_DIR = avatars_static_dir()
LEGACY_BASE_NAME = "lexi_base.png"
AV_PUBLIC_URL = avatars_public_url_base()

def _client_ip(request: Request) -> str:
    headers = {k.lower(): v for k, v in (request.headers or {}).items()}
    fallback = getattr(getattr(request, "client", None), "host", None) or "127.0.0.1"
    return client_ip_from_headers(headers, fallback)


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
    candidate = AVATARS_PUBLIC_DIR / filename
    if candidate.exists():
        return _public_avatar_url(filename, candidate)
    return None
router.include_router(now_router)  # /now
router.include_router(tools_router)  # /tools/web_search

# Paths for avatar storage and trait state persistence
TRAIT_STATE_PATH: Path = Path(__file__).resolve().parent / "lexi_persona_state.json"


def _avatar_dir() -> Path:
    """Resolve the active avatar directory, ensuring it exists."""
    dir_path = AVATAR_DIR
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("⚠️ Could not ensure avatar directory %s: %s", dir_path, exc)
    return dir_path


def _avatar_filename(session_id: Optional[str], requested: Optional[str]) -> str:
    """Choose a safe filename for avatar outputs."""
    if isinstance(requested, str) and requested.strip():
        name = requested.strip()
    elif isinstance(session_id, str) and session_id.strip():
        name = f"{session_id.strip()}.png"
    else:
        name = f"{uuid4().hex}.png"

    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    return name


def _safe_join_avatar(filename: Optional[str], session_id: Optional[str]) -> Path:
    """Join avatar directory with filename, defaulting when missing."""
    return _avatar_dir() / _avatar_filename(session_id, filename)


def _lexi_base_path() -> Path:
    return AVATARS_PUBLIC_DIR / LEGACY_BASE_NAME


def _fallback_avatar_url(request: Request) -> str:
    """Return a safe avatar URL (public) for warn/error paths."""
    candidate = _per_ip_avatar_path(request)
    if not candidate:
        candidate = _public_avatar_url(LEGACY_BASE_NAME, _lexi_base_path())
    public_base = os.getenv("LEX_PUBLIC_BASE")
    if public_base:
        public_base = public_base.rstrip("/")
        return f"{public_base}{candidate}"
    return _absolute_url(request, candidate)


def _ensure_avatar_url(request: Request, candidate: Optional[str]) -> str:
    """Ensure avatar URL is absolute, falling back to public default."""
    if isinstance(candidate, str) and candidate.strip():
        return _absolute_url(request, candidate)
    return _fallback_avatar_url(request)


async def _per_ip_avatar_or_fallback(request: Request) -> str:
    """
    Ensure a deterministic per-IP avatar exists, falling back to the legacy base.
    """
    ip = _client_ip(request)
    try:
        avatar_path = await ensure_per_ip_avatar(ip)
        return _absolute_url(
            request,
            _public_avatar_url(avatar_path.name, avatar_path),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Per-IP avatar ensure failed for %s: %s", ip, exc)
        return _fallback_avatar_url(request)


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str
    intent: Optional[str] = None


class IntentRequest(BaseModel):
    text: str


class AvatarGenRequest(BaseModel):
    prompt: Optional[str] = None  # base description (txt2img) or ignored if base exists
    changes: Optional[str] = None  # appended when doing img2img/inpaint
    traits: Optional[Dict[str, str]] = None

    mode: Optional[str] = None  # "txt2img" | "img2img" (pipeline may override)
    source_path: Optional[str] = None  # optional override for img2img
    mask_path: Optional[str] = None  # legacy: ignored in flux-only pipeline

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
    if file_path.exists():
        ts = int(file_path.stat().st_mtime)
        rel_path = file_path.as_posix().split("/static")[-1]
        return f"/static{rel_path}?v={ts}"
    return f"/static/{file_path.name}"


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
    session_id = getattr(request.state, "session_id", None) or request.headers.get("X-Lexi-Session")
    if not session_id or not isinstance(registry, SessionRegistry):
        return
    try:
        registry.append_memory(session_id, event)
    except Exception:
        pass


@router.post("/process")
async def process(req: ChatRequest, request: Request):
    logger.info("🗨️ /process prompt=%r", req.prompt)
    log_event(request, "user", req.prompt or "", event="chat_prompt")
    _append_session_memory(
        request,
        {"role": "user", "event": "chat_prompt", "text": req.prompt},
    )

    # Guard against assistant loops
    if req.prompt.startswith(("Lexi:", "assistant:")):
        logger.warning("🛑 Ignoring looped assistant prompt: %r", req.prompt)
        log_event(
            request,
            "system",
            "[loop detected, halted]",
            event="chat_loop_guard",
        )
        return JSONResponse(
            {
                "cleaned": "[loop detected, halted]",
                "raw": "",
                "choices": [],
                "mode": lexi_persona.get_mode(),
                "session_id": getattr(request.state, "session_id", None),
            }
        )

    prompt_text = (req.prompt or "").strip()
    prompt_lower = prompt_text.lower()
    incoming_intent = (req.intent or "").strip().lower()
    if (
        incoming_intent == "avatar_edit"
        or "let's change your look" in prompt_lower
        or "lets change your look" in prompt_lower
    ):
        logger.info("Avatar edit intent disabled for prompt=%r", req.prompt)
        avatar_url = await _per_ip_avatar_or_fallback(request)
        log_event(
            request,
            "assistant",
            "Avatar edit is disabled.",
            event="avatar_edit_disabled",
        )
        return JSONResponse(
            {
                "intent": "avatar_edit_disabled",
                "message": "Avatar edit is temporarily disabled.",
                "avatar_url": avatar_url,
            }
        )

    if incoming_intent == "new_look":
        traits_state = load_traits()
        if traits_state is None:
            traits_state = {}
        has_traits = bool(traits_state)

        fresh_base = incoming_intent == "new_look" or not has_traits
        prompt_arg: Optional[str] = req.prompt if fresh_base else None
        changes_arg: Optional[str] = None
        mode_arg = "txt2img"
        intent_strength = "strong"

        try:
            result = generate_avatar_pipeline(
                prompt=prompt_arg,
                traits=traits_state if has_traits else None,
                changes=changes_arg,
                mode=mode_arg,
                intent=intent_strength,
                fresh_base=fresh_base,
                allow_feedback_loop=False,
            )
        except Exception as exc:
            if USE_COMFY_ONLY:
                logger.warning("Avatar intent pipeline failed (non-fatal): %s", exc)
                avatar_url = await _per_ip_avatar_or_fallback(request)
                return JSONResponse(
                    {
                        "status": "warn",
                        "detail": str(exc),
                        "avatar_pipeline": "skipped",
                        "mode": lexi_persona.get_mode(),
                        "session_id": getattr(request.state, "session_id", None),
                        "avatar_url": avatar_url,
                    }
                )
            logger.error("Avatar intent pipeline failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))

        if not result.get("ok"):
            error_msg = result.get("error", "avatar generation failed")
            if USE_COMFY_ONLY:
                logger.warning("Avatar intent preflight skipped (warn-only): %s", error_msg)
                avatar_url = await _per_ip_avatar_or_fallback(request)
                return JSONResponse(
                    {
                        "status": "warn",
                        "detail": error_msg,
                        "avatar_pipeline": "skipped",
                        "mode": lexi_persona.get_mode(),
                        "session_id": getattr(request.state, "session_id", None),
                        "avatar_url": avatar_url,
                    }
                )
            logger.error("Avatar intent pipeline returned error: %s", error_msg)
            raise HTTPException(status_code=502, detail=error_msg)

        url = result.get("url")
        abs_url = (
            _absolute_url(request, url)
            if isinstance(url, str) and url.startswith("/")
            else url
        )
        avatar_url = _ensure_avatar_url(request, abs_url or url)
        narration = result.get("narration") or "Got it, updating her look! 💄"

        if url:
            try:
                lexi_persona.set_avatar_path(url)  # type: ignore
                save_traits(traits_state, avatar_path=url)
            except Exception as exc:
                logger.warning("⚠️ Persona state update failed: %s", exc)

        _append_session_memory(
            request,
            {
                "role": "assistant",
                "event": "chat_avatar_update",
                "traits": traits_state,
                "avatar_url": avatar_url,
            },
        )
        log_event(
            request,
            "assistant",
            narration,
            event="chat_avatar_update",
            avatar_url=avatar_url,
        )

        response_payload: Dict[str, Any] = {
            "cleaned": narration,
            "avatar_url": avatar_url,
            "traits": traits_state,
            "mode": lexi_persona.get_mode(),
            "session_id": getattr(request.state, "session_id", None),
        }
        if result.get("meta"):
            response_payload["meta"] = result["meta"]

        return JSONResponse(response_payload)

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
        avatar_url = _ensure_avatar_url(request, url)
        if url:
            lexi_persona.set_avatar_path(url)  # type: ignore
            save_traits(traits, avatar_path=url)

        _append_session_memory(
            request,
            {
                "role": "assistant",
                "event": "chat_avatar_update",
                "traits": traits,
                "avatar_url": avatar_url,
            },
        )
        log_event(
            request,
            "assistant",
            "Got it, updating her look! 💄",
            event="chat_avatar_update",
            avatar_url=avatar_url,
        )
        return JSONResponse(
            {
                "cleaned": "Got it, updating her look! 💄",
                "avatar_url": avatar_url,
                "traits": traits,
                "mode": lexi_persona.get_mode(),
                "session_id": getattr(request.state, "session_id", None),
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
        log_event(
            request,
            "assistant",
            reply_text,
            event="chat_reply",
        )

        return {
            "cleaned": reply_text,
            "raw": reply_text,
            "choices": [{"text": reply_text}],
            "mode": lexi_persona.get_mode(),
            "session_id": getattr(request.state, "session_id", None),
        }

    if wants_stream:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        final_payload: Dict[str, Any] = {}
        error_holder: Dict[str, str] = {}

        _think_start_re = re.compile(r"<(think|thinking|scratchpad)[^>]*>", re.I)
        _think_end_re = re.compile(r"</(think|thinking|scratchpad)>", re.I)
        _strip_tokens_re = re.compile(
            r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext|object_ref_start|object_ref_end|box_start|box_end|quad_start|quad_end|vision_start|vision_end)\s*\|>",
            re.I,
        )
        _inline_think_re = re.compile(
            r"<(think|thinking|scratchpad)[^>]*>.*?(?:</(think|thinking|scratchpad)>|$)",
            re.I | re.S,
        )

        def _sanitize_chunk(raw: str, state: Dict[str, Any]) -> str:
            buf = state.get("buf", "") + raw
            in_think = state.get("in_think", False)
            out_parts: list[str] = []

            while buf:
                if in_think:
                    m_end = _think_end_re.search(buf)
                    if not m_end:
                        state["buf"] = ""
                        state["in_think"] = True
                        return ""
                    buf = buf[m_end.end() :]
                    in_think = False
                    continue

                m_start = _think_start_re.search(buf)
                if not m_start:
                    out_parts.append(buf)
                    buf = ""
                    break

                if m_start.start() > 0:
                    out_parts.append(buf[: m_start.start()])
                buf = buf[m_start.end() :]
                m_end = _think_end_re.search(buf)
                if m_end:
                    buf = buf[m_end.end() :]
                    in_think = False
                    continue
                buf = ""
                in_think = True
                break

            state["buf"] = buf
            state["in_think"] = in_think
            text = "".join(out_parts)
            if not text:
                return ""

            text = _strip_tokens_re.sub("", text)
            text = re.sub(r"</?(assistant|system|user)>", "", text, flags=re.I)
            text = re.sub(r"</?tool_call>|</?tool_response>", "", text, flags=re.I)
            text = _inline_think_re.sub("", text)
            text = _think_start_re.sub("", text)
            text = _think_end_re.sub("", text)
            return text.strip()

        def stream_callback(chunk: str) -> None:
            if not chunk:
                return
            try:
                cleaned = _sanitize_chunk(chunk, state)
                if cleaned:
                    loop.call_soon_threadsafe(queue.put_nowait, cleaned)
            except Exception as exc:
                logger.warning("stream sanitize failed: %s", exc)
                state["buf"] = ""
                state["in_think"] = False

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
        fallback_url = await _per_ip_avatar_or_fallback(request)
        return JSONResponse(
            {
                "ok": False,
                "url": fallback_url,
                "error": str(exc),
                "intent": intent,
                "nsfw": nsfw,
            }
        )

    if not result.get("ok"):
        fallback_url = await _per_ip_avatar_or_fallback(request)
        return JSONResponse(
            {
                "ok": False,
                "url": fallback_url,
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
@persona_router.api_route("/avatar", methods=["GET", "POST"])
async def avatar_endpoint(request: Request, req: Optional[ChatRequest] = None) -> JSONResponse:
    if request.method.upper() == "GET":
        avatar_url = await _per_ip_avatar_or_fallback(request)
        return JSONResponse({"ok": True, "url": avatar_url, "avatar_url": avatar_url})

    if req is None or not isinstance(req.prompt, str):
        raise HTTPException(status_code=400, detail="Prompt is required")

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
        return PlainTextResponse(f"{AVATAR_URL_PREFIX}/{LEGACY_BASE_NAME}", status_code=500)

    fname = f"lexi_avatar_{uuid4().hex[:6]}.png"
    file_path = _safe_join_avatar(fname, getattr(request.state, "session_id", None))
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
