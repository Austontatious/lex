from __future__ import annotations

import ipaddress
import os
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict

import orjson
from fastapi import Request, Response

LOG_DIR = Path(os.getenv("LEX_LOG_DIR", "./logs/sessions")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_NAME = os.getenv("LEX_SESSION_COOKIE", "lex_session")
COOKIE_MAX_AGE = int(os.getenv("LEX_SESSION_COOKIE_MAX_AGE", str(60 * 60 * 24 * 30)))

VisitMeta = Dict[str, object]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    candidate = (forwarded.split(",")[0].strip() if forwarded else None) or getattr(
        request.client, "host", ""
    )
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except Exception:
        return getattr(request.client, "host", "") or "0.0.0.0"


def _session_file(session_id: str) -> Path:
    day_dir = LOG_DIR / time.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{session_id}.ndjson"


def write_session_event(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("ab") as handle:
        handle.write(orjson.dumps(payload))
        handle.write(b"\n")


def _should_use_secure_cookie(request: Request) -> bool:
    flag = os.getenv("LEX_COOKIE_SECURE", "auto").lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if flag in {"1", "true", "on", "yes"}:
        return True
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return str(proto).lower() == "https"


async def session_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    session_id = request.cookies.get(COOKIE_NAME)
    new_session = False
    if not session_id:
        session_id = uuid.uuid4().hex
        new_session = True

    log_path = _session_file(session_id)
    client_ip = _safe_ip(request)

    request.state.session_id = session_id
    request.state.session_log_path = log_path
    request.state.client_ip = client_ip
    request.state.user_agent = request.headers.get("user-agent", "")

    visit_meta: VisitMeta = {
        "ts": _now_ms(),
        "event": "visit",
        "method": request.method,
        "path": request.url.path,
        "query": request.url.query,
        "ip": client_ip,
        "ua": request.headers.get("user-agent", ""),
        "referer": request.headers.get("referer", ""),
        "session_id": session_id,
    }
    write_session_event(log_path, visit_meta)

    response = await call_next(request)

    if new_session:
        response.set_cookie(
            COOKIE_NAME,
            session_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
            secure=_should_use_secure_cookie(request),
            path="/",
        )

    return response


__all__ = ["session_middleware", "write_session_event"]
