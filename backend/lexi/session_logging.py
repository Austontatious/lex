from __future__ import annotations

import hashlib
import re
import time
import traceback
import uuid
from typing import Any, Dict, Literal, Optional, Sequence

from fastapi import Request

from .session import write_session_event

EventRole = Literal["user", "assistant", "system", "tool", "mode", "error"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash_text(text: str) -> str:
    try:
        import xxhash  # type: ignore

        return xxhash.xxh64_hexdigest(text)
    except Exception:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _redact(text: str) -> str:
    """Lightweight redaction for emails/phones/dates; trims length."""
    if not text:
        return ""
    redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "<EMAIL>", text)
    redacted = re.sub(r"\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b", "<PHONE>", redacted)
    redacted = re.sub(r"\b\d{1,2}\/\d{1,2}\/\d{2,4}\b", "<DATE>", redacted)
    return redacted


def sanitize_for_log(text: str) -> Dict[str, Any]:
    """Return hashed/redacted representation; never includes raw."""
    safe = text or ""
    return {
        "content_hash": _hash_text(safe),
        "content_preview": _redact(safe),
        "content_len": len(safe),
    }


def _next_turn_id(request: Request) -> int:
    current = getattr(request.state, "turn_id_counter", 0) or 0
    current += 1
    request.state.turn_id_counter = current
    return current


def log_event(request: Request, role: EventRole, content: str, **extra: Any) -> None:
    """
    Append a structured event for the active session (if any).
    """
    log_path = getattr(request.state, "session_log_path", None)
    session_id = getattr(request.state, "session_id", None)
    if not log_path or not session_id:
        return

    safe = sanitize_for_log(str(content))
    payload: Dict[str, Any] = {
        "ts": _now_ms(),
        "session_id": session_id,
        "role": role,
        "content": safe["content_preview"],
        "content_hash": safe["content_hash"],
        "content_len": safe["content_len"],
        "path": extra.pop("path", request.url.path),
        "method": request.method,
        "ip": getattr(request.state, "client_ip", None),
        "ua": getattr(request.state, "user_agent", None),
    }
    if extra:
        payload.update(extra)

    write_session_event(log_path, payload)


def log_turn(
    request: Request,
    role: EventRole,
    content: str,
    *,
    turn_id: Optional[int] = None,
    mode: Optional[str] = None,
    persona: Optional[str] = None,
    tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    safety: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Per-turn logging with hashed content for privacy."""
    log_path = getattr(request.state, "session_log_path", None)
    session_id = getattr(request.state, "session_id", None)
    if not log_path or not session_id:
        return

    payload: Dict[str, Any] = {
        "ts": _now_ms(),
        "session_id": session_id,
        "role": role,
        "turn_id": turn_id or _next_turn_id(request),
        "mode": mode,
        "persona": persona,
        "tool_calls": list(tool_calls or []),
        "safety": safety or {},
        "latency_ms": latency_ms,
        "path": request.url.path,
        "ip": getattr(request.state, "client_ip", None),
    }
    payload.update(sanitize_for_log(content))
    if model_meta:
        payload["model"] = model_meta
    write_session_event(log_path, payload)


def log_safety_event(
    request: Request,
    *,
    turn_id: Optional[int],
    safety_event: str,
    categories: Sequence[str],
) -> None:
    log_path = getattr(request.state, "session_log_path", None)
    session_id = getattr(request.state, "session_id", None)
    if not log_path or not session_id:
        return
    payload: Dict[str, Any] = {
        "ts": _now_ms(),
        "session_id": session_id,
        "turn_id": turn_id,
        "safety_event": safety_event,
        "categories": list(categories),
        "path": request.url.path,
    }
    write_session_event(log_path, payload)


def log_error_event(
    request: Request,
    exc: Exception,
    *,
    trace_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    log_path = getattr(request.state, "session_log_path", None)
    session_id = getattr(request.state, "session_id", None)
    trace_id = trace_id or uuid.uuid4().hex
    if not log_path or not session_id:
        return trace_id
    payload: Dict[str, Any] = {
        "ts": _now_ms(),
        "session_id": session_id,
        "trace_id": trace_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "path": getattr(request, "url", None).path if request else None,
        "context": context or {},
    }
    write_session_event(log_path, payload)
    return trace_id


__all__ = [
    "log_event",
    "log_turn",
    "log_safety_event",
    "log_error_event",
    "sanitize_for_log",
    "EventRole",
]
