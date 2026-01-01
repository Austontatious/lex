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
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from ..utils.prompt_sifter import build_sd_prompt, extract_categories
from .lexi_persona import _load_traits, _save_traits, ensure_per_ip_avatar
from .now import router as now_router, tools as tools_router
from ..config.config import AVATAR_URL_PREFIX, STARTER_AVATAR_PATH, AVATAR_DIR as CONFIG_AVATAR_DIR
load_traits = _load_traits
save_traits = _save_traits
from ..persona.persona_core import lexi_persona
from ..persona.prompt_templates import PromptTemplates
from ..persona.persona_config import PERSONA_MODE_REGISTRY
from ..alpha.session_manager import SessionRegistry
from ..session_logging import (
    log_event,
    log_turn,
    log_error_event,
    sanitize_for_log,
    log_safety_event,
)
from ..utils.safety import classify_safety, ensure_crisis_safety_style
from ..utils.error_responses import SOFT_ERROR_MESSAGE, soft_error_payload
from ..utils.ip_seed import (
    avatars_public_url_base,
    avatars_static_dir,
    basename_for_ip,
    filename_for_ip,
    ip_to_seed,
)
from ..utils.request_ip import request_ip
from ..user_identity import identity_payload, request_user_id
from ..prompts.tour_mode import TOUR_MODE_SHIM

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Lexi Core"])

USE_COMFY_ONLY = os.getenv("LEX_USE_COMFY_ONLY", "0") == "1"

TOUR_PROMPT_MAX_CHARS = int(os.getenv("LEXI_TOUR_PROMPT_MAX_CHARS", "600"))
TOUR_RESPONSE_MAX_TOKENS = int(os.getenv("LEXI_TOUR_RESPONSE_MAX_TOKENS", "220"))
TOUR_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("LEXI_TOUR_RATE_LIMIT_WINDOW_SECONDS", "60"))
TOUR_RATE_LIMIT_MAX = int(os.getenv("LEXI_TOUR_RATE_LIMIT_MAX", "6"))
_TOUR_RATE_LIMIT: Dict[str, List[float]] = {}
_TOUR_RATE_LIMIT_LOCK = asyncio.Lock()

AVATAR_DIR: Path = CONFIG_AVATAR_DIR
AVATARS_PUBLIC_DIR = avatars_static_dir()
LEGACY_BASE_NAME = "lexi_base.png"
AV_PUBLIC_URL = avatars_public_url_base()


def _registry(request: Request) -> Optional[SessionRegistry]:
    registry = getattr(request.app.state, "alpha_sessions", None)
    return registry if isinstance(registry, SessionRegistry) else None


def _record_metric(request: Request, event: str, detail: Optional[Dict[str, Any]] = None) -> None:
    registry = _registry(request)
    session_id = getattr(request.state, "session_id", None)
    if not registry or not session_id:
        return
    payload = {"event": event}
    if detail:
        payload.update(detail)
    try:
        registry.record_metric(session_id, payload)
    except Exception:
        return

def _client_ip(request: Request) -> str:
    return request_ip(request)


def _tour_rate_limit_key(request: Request) -> str:
    session_id = request.headers.get("x-lexi-session") or getattr(request.state, "session_id", None)
    if session_id:
        return f"session:{session_id}"
    return f"ip:{_client_ip(request)}"


async def _enforce_tour_rate_limit(request: Request) -> None:
    now = time.time()
    window_start = now - TOUR_RATE_LIMIT_WINDOW_SECONDS
    key = _tour_rate_limit_key(request)
    async with _TOUR_RATE_LIMIT_LOCK:
        history = _TOUR_RATE_LIMIT.get(key, [])
        history = [ts for ts in history if ts >= window_start]
        if len(history) >= TOUR_RATE_LIMIT_MAX:
            if history:
                _TOUR_RATE_LIMIT[key] = history
            else:
                _TOUR_RATE_LIMIT.pop(key, None)
            raise HTTPException(status_code=429, detail="Too many tour requests. Please try again soon.")
        history.append(now)
        _TOUR_RATE_LIMIT[key] = history


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


class TourPromptRequest(BaseModel):
    prompt: str
    card_id: Optional[str] = None


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
    r"\b(avatar|appearance|outfit|clothes?|dress|costume|style|wear|lingerie|look\\s+like)\b",
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


@router.post("/tour/prompt")
async def tour_prompt(req: TourPromptRequest, request: Request) -> JSONResponse:
    logger.info("🗨️ /tour/prompt card=%s", req.card_id)
    prompt_text = (req.prompt or "").strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt must be a non-empty string")
    if len(prompt_text) > TOUR_PROMPT_MAX_CHARS:
        raise HTTPException(status_code=413, detail="Prompt is too long")

    await _enforce_tour_rate_limit(request)

    if prompt_text.startswith(("Lexi:", "assistant:")):
        logger.warning("🛑 Ignoring looped assistant prompt: %r", req.prompt)
        return JSONResponse(
            {
                "cleaned": "[loop detected, halted]",
                "raw": "",
                "choices": [],
                "mode": lexi_persona.get_mode(),
            }
        )

    safety_decision = classify_safety(prompt_text)
    log_safety_event(
        request,
        turn_id=uuid4().hex,
        safety_event=safety_decision["action"],
        categories=safety_decision.get("categories", []),
    )
    if safety_decision.get("blocked"):
        refusal = "I can't help with that."
        return JSONResponse(
            {
                "cleaned": refusal,
                "raw": refusal,
                "choices": [{"text": refusal}],
                "mode": lexi_persona.get_mode(),
                "safety": safety_decision,
            },
            status_code=403,
        )

    prompt_pkg = PromptTemplates.build_prompt(
        memories_json=[],
        user_message=prompt_text,
        active_persona=lexi_persona.get_mode(),
        injections=[TOUR_MODE_SHIM],
    )
    try:
        reply = await asyncio.to_thread(
            lexi_persona._gen,
            prompt_pkg,
            sampler_overrides={"max_tokens": TOUR_RESPONSE_MAX_TOKENS},
        )
    except Exception as exc:
        logger.error("❌ LexiPersona._gen failed (tour prompt): %s", exc)
        trace = log_error_event(request, exc, context={"route": "/tour/prompt"})
        payload = soft_error_payload(error_detail=str(exc), trace_id=trace)
        payload["mode"] = lexi_persona.get_mode()
        return JSONResponse(payload)

    reply_text = ensure_crisis_safety_style(prompt_text, reply or "")
    if not reply_text.strip():
        reply_text = "[no response]"

    return JSONResponse(
        {
            "cleaned": reply_text,
            "raw": reply_text,
            "choices": [{"text": reply_text}],
            "mode": lexi_persona.get_mode(),
        }
    )


@router.post("/process")
async def process(req: ChatRequest, request: Request):
    logger.info("🗨️ /process turn")
    if getattr(request.state, "needs_disambiguation", False):
        handle_raw = request.headers.get("x-lexi-handle") or getattr(request.state, "handle_norm", None)
        handle_label = handle_raw or "that name"
        if handle_label:
            prompt = (
                f"I already know an {handle_label} — is that you or should I call you something else?"
            )
        else:
            prompt = "I already know that name — is that you or should I call you something else?"
        return JSONResponse(
            {
                "cleaned": prompt,
                "raw": "",
                "choices": [],
                "mode": lexi_persona.get_mode(),
                "session_id": getattr(request.state, "session_id", None),
                "needs_disambiguation": True,
                "candidates": getattr(request.state, "identity_candidates", []),
                "identity": identity_payload(request),
            }
        )
    resolved_user_id = request_user_id(request)
    try:
        lexi_persona.set_user(resolved_user_id)
        lexi_persona.bind_session(getattr(request.state, "session_id", None))
    except Exception:
        pass
    try:
        mem_state = lexi_persona.memory.debug_state()
        logger.info("memory_state %s", json.dumps(mem_state, ensure_ascii=True))
    except Exception:
        pass
    log_event(request, "user", req.prompt or "", event="chat_prompt")
    _append_session_memory(
        request,
        {"role": "user", "event": "chat_prompt", "text": req.prompt},
    )
    _record_metric(request, "first_message_sent")
    turn_id = getattr(request.state, "turn_id_counter", 0) or 0
    turn_id += 1
    request.state.turn_id_counter = turn_id
    start_ts = time.time()
    log_turn(
        request,
        "user",
        req.prompt or "",
        turn_id=turn_id,
        mode=lexi_persona.get_mode(),
        persona=getattr(lexi_persona, "name", "Lexi"),
        tool_calls=[],
        safety={},
        latency_ms=0,
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
    client_ip = _client_ip(request)
    base_name = basename_for_ip(client_ip)
    seed_default = ip_to_seed(client_ip)
    safety_decision = classify_safety(req.prompt or "")
    log_safety_event(
        request,
        turn_id=turn_id,
        safety_event=safety_decision["action"],
        categories=safety_decision.get("categories", []),
    )
    if safety_decision.get("blocked"):
        refusal = "I can't help with that."
        latency_ms = int((time.time() - start_ts) * 1000)
        _record_metric(request, "safety_block", {"categories": safety_decision.get("categories", [])})
        log_turn(
            request,
            "assistant",
            refusal,
            turn_id=turn_id,
            mode=lexi_persona.get_mode(),
            persona=getattr(lexi_persona, "name", "Lexi"),
            tool_calls=[],
            safety={"decision": safety_decision},
            latency_ms=latency_ms,
            model_meta=None,
        )
        return JSONResponse(
            {
                "cleaned": refusal,
                "raw": refusal,
                "choices": [{"text": refusal}],
                "mode": lexi_persona.get_mode(),
                "session_id": getattr(request.state, "session_id", None),
                "safety": safety_decision,
            },
            status_code=403,
        )

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
        traits_state = load_traits(resolved_user_id)
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
                seed=seed_default,
                base_name=base_name,
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

        url = result.get("avatar_url") or result.get("url")
        avatar_url = _ensure_avatar_url(request, url)
        narration = result.get("narration") or "Updating Lexi's look now."

        if avatar_url:
            try:
                lexi_persona.set_avatar_path(avatar_url)  # type: ignore
                save_traits(traits_state, avatar_path=avatar_url, user_id=resolved_user_id)
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
            "url": avatar_url,
            "traits": traits_state,
            "mode": lexi_persona.get_mode(),
            "session_id": getattr(request.state, "session_id", None),
        }
        if result.get("meta"):
            response_payload["meta"] = result["meta"]

        return JSONResponse(response_payload)

    # -----------------------------------------------------------------
    # DEPRECATED: legacy appearance extraction from conversation.
    # Avatar Tools Modal is the canonical path. This auto-trigger is
    # intentionally disabled to prevent silent avatar updates.
    # -----------------------------------------------------------------
    inferred = extract_traits_from_text(req.prompt)
    if inferred:
        logger.info(
            "DEPRECATED auto appearance extraction ignored (traits=%s)",
            ", ".join(f"{k}={v}" for k, v in inferred.items()),
        )
        message = (
            "Legacy auto appearance extraction has been removed. "
            "Use the Avatar Tools modal to update Lexi's look."
        )
        return JSONResponse(
            {
                "cleaned": message,
                "raw": message,
                "choices": [{"text": message}],
                "status": "ignored",
                "error": "deprecated",
                "deprecated": True,
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
        reply_text = ensure_crisis_safety_style(prompt_text, reply_text)

        gen_meta = getattr(lexi_persona, "_last_gen_meta", {}) or {}
        finish_reason = gen_meta.get("finish_reason")
        usage = gen_meta.get("usage")

        try:
            lexi_persona.memory.store_context(prompt_text, reply_text)
        except Exception as mem_err:
            logger.warning("⚠️ Memory store skipped: %s", mem_err)

        event_payload: Dict[str, Any] = {
            "role": "assistant",
            "event": "chat_reply",
            "text": reply_text,
        }
        if finish_reason is not None:
            event_payload["finish_reason"] = finish_reason
        if usage is not None:
            event_payload["usage"] = usage

        _append_session_memory(request, event_payload)
        log_event(
            request,
            "assistant",
            reply_text,
            event="chat_reply",
            finish_reason=finish_reason,
            usage=usage,
        )
        latency_ms = int((time.time() - start_ts) * 1000)
        params = gen_meta.get("params") if isinstance(gen_meta, dict) else {}
        model_meta = {
            "model_name": getattr(lexi_persona.loader, "primary_type", None),
            "tokens_in": (usage or {}).get("prompt_tokens") if isinstance(usage, dict) else None,
            "tokens_out": (usage or {}).get("completion_tokens") if isinstance(usage, dict) else None,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "repeat_penalty": params.get("repeat_penalty") or params.get("repetition_penalty"),
            "logprobs_enabled": params.get("logprobs") is True,
            "grounding_used": params.get("tool_call"),
        }
        assistant_safety = classify_safety(reply_text)
        log_safety_event(
            request,
            turn_id=turn_id,
            safety_event=assistant_safety.get("action", "observe"),
            categories=assistant_safety.get("categories", []),
        )
        log_turn(
            request,
            "assistant",
            reply_text,
            turn_id=turn_id,
            mode=lexi_persona.get_mode(),
            persona=getattr(lexi_persona, "name", "Lexi"),
            tool_calls=gen_meta.get("tool_calls") if isinstance(gen_meta, dict) else [],
            safety={"model": gen_meta.get("safety") if isinstance(gen_meta, dict) else {}, "assistant_check": assistant_safety},
            latency_ms=latency_ms,
            model_meta=model_meta,
        )
        _record_metric(request, "first_reply_rendered")

        return {
            "cleaned": reply_text,
            "raw": reply_text,
            "choices": [{"text": reply_text}],
            "mode": lexi_persona.get_mode(),
            "session_id": getattr(request.state, "session_id", None),
            "finish_reason": finish_reason,
            "usage": usage,
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
                trace = log_error_event(request, exc, context={"route": "/process", "turn_id": turn_id})
                error_holder["message"] = SOFT_ERROR_MESSAGE
                error_holder["trace_id"] = trace
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
                payload = soft_error_payload(trace_id=error_holder.get("trace_id"))
                payload["done"] = True
                yield json.dumps(payload) + "\n"
            else:
                if not final_payload:
                    final_payload.update(finalize_reply(""))
                yield json.dumps({"done": True, **final_payload}) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    try:
        reply = await asyncio.to_thread(lexi_persona.chat, req.prompt)
    except Exception as exc:
        logger.error("❌ LexiPersona.chat failed: %s", exc)
        trace = log_error_event(request, exc, context={"route": "/process", "turn_id": turn_id})
        _append_session_memory(
            request,
            {"role": "assistant", "event": "chat_error", "error": str(exc)},
        )
        payload = soft_error_payload(error_detail=str(exc), trace_id=trace)
        payload["mode"] = lexi_persona.get_mode()
        return JSONResponse(payload)

    payload = finalize_reply(reply)
    return JSONResponse(payload)


# ---------------------------------------------------------------------
# Set persona mode
# ---------------------------------------------------------------------
@router.post("/set_mode")
async def set_mode(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    mode = payload.get("mode")
    if mode not in PERSONA_MODE_REGISTRY:
        logger.warning("🛑 Invalid mode: %s", mode)
        raise HTTPException(status_code=400, detail="Invalid mode")
    lexi_persona.set_mode(mode)
    if getattr(request.state, "needs_disambiguation", False):
        raise HTTPException(status_code=409, detail="identity collision")
    user_id = request_user_id(request)
    save_traits({}, avatar_path=None, user_id=user_id)
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
    if getattr(request.state, "needs_disambiguation", False):
        raise HTTPException(status_code=409, detail="identity collision")
    user_id = request_user_id(request)
    traits = req.traits or load_traits(user_id)
    prompt = req.prompt
    changes = req.changes
    client_ip = _client_ip(request)
    base_name = basename_for_ip(client_ip)
    seed_default = ip_to_seed(client_ip)

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
            seed=req.seed if req.seed is not None else seed_default,
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
            base_name=base_name,
        )
    except Exception as exc:
        logger.error("Avatar pipeline failed: %s", exc)
        fallback_url = await _per_ip_avatar_or_fallback(request)
        return JSONResponse(
            {
                "ok": False,
                "url": fallback_url,
                "avatar_url": fallback_url,
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
                "avatar_url": fallback_url,
                "error": result.get("error", "Comfy pipeline error"),
                "intent": intent,
                "nsfw": nsfw,
            }
        )

    url = result.get("avatar_url") or result.get("url")
    abs_url = _absolute_url(request, url)
    meta = result.get("meta", {})

    # Persist avatar URL to persona state
    try:
        # Store relative URL if provided to keep storage portable
        lexi_persona.set_avatar_path(abs_url)  # type: ignore
        save_traits(traits, avatar_path=abs_url, user_id=user_id)
    except Exception as exc:
        logger.warning("⚠️ Persona state update failed: %s", exc)

    return JSONResponse(
        {
            "ok": True,
            "url": abs_url,
            "avatar_url": abs_url,
            "meta": meta,
            "intent": intent,
            "nsfw": nsfw,
        }
    )


# Alias to support older frontend calls to /lexi/generate_avatar
@router.post("/generate_avatar")
async def legacy_generate_avatar(request: Request, req: AvatarGenRequest) -> JSONResponse:
    return await generate_avatar(request, req)


# Legacy alias for backwards compatibility
@persona_router.api_route("/avatar", methods=["GET", "POST"])
async def avatar_endpoint(request: Request, req: Optional[ChatRequest] = None) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if request.method.upper() == "GET":
        avatar_url = await _per_ip_avatar_or_fallback(request)
        payload = {"ok": True, "url": avatar_url, "avatar_url": avatar_url}
        return JSONResponse(payload, headers=headers)

    if req is None or not isinstance(req.prompt, str):
        raise HTTPException(status_code=400, detail="Prompt is required")

    result = generate_avatar_pipeline(prompt=req.prompt, refiner=True, style="realistic")
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "avatar generation failed"))
    payload = {"ok": True, "url": result["url"], "meta": result.get("meta", {})}
    return JSONResponse(payload, headers=headers)


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


# DEPRECATED: legacy auto appearance extraction from chat. Avatar Tools modal is canonical.
def extract_traits_from_text(text: str) -> Dict[str, str]:
    """Heuristic trait extraction based on visual keywords."""
    categories = extract_categories(text or "")
    return {key: value for key, value in categories.items() if value}
