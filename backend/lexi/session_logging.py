from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional

from fastapi import Request

from .session import write_session_event

EventRole = Literal["user", "assistant", "system", "tool", "mode", "error"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def log_event(request: Request, role: EventRole, content: str, **extra: Any) -> None:
    """
    Append a structured event for the active session (if any).
    """
    log_path = getattr(request.state, "session_log_path", None)
    session_id = getattr(request.state, "session_id", None)
    if not log_path or not session_id:
        return

    payload: Dict[str, Any] = {
        "ts": _now_ms(),
        "session_id": session_id,
        "role": role,
        "content": str(content),
        "path": extra.pop("path", request.url.path),
        "method": request.method,
        "ip": getattr(request.state, "client_ip", None),
        "ua": getattr(request.state, "user_agent", None),
    }
    if extra:
        payload.update(extra)

    write_session_event(log_path, payload)


__all__ = ["log_event", "EventRole"]
