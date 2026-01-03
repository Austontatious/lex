from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict

import orjson
from fastapi import Request, Response
from starlette.concurrency import run_in_threadpool

from .sd.generate import generate_default_avatar_for_ip
from .utils.user_profile import touch_last_seen, user_profile_feature_enabled
from .user_identity import resolve_identity

LOG_DIR = Path(os.getenv("LEX_LOG_DIR", "./logs/sessions")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_RETENTION_DAYS = int(os.getenv("LEXI_LOG_RETENTION_DAYS", "0") or "0")

COOKIE_NAME = os.getenv("LEX_SESSION_COOKIE", "lex_session")
COOKIE_MAX_AGE = int(os.getenv("LEX_SESSION_COOKIE_MAX_AGE", str(60 * 60 * 24 * 30)))
DEFAULT_AVATAR_MEDIA_DIR = Path(
    os.getenv("LEX_DEFAULT_AVATAR_DIR", "/app/media/defaults")
).expanduser()

logger = logging.getLogger("lexi.session")
_session_tasks: set[asyncio.Task] = set()

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
    default_avatar_info = None
    if new_session:
        async def _kickoff_default_avatar() -> None:
            try:
                DEFAULT_AVATAR_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
                await run_in_threadpool(
                    generate_default_avatar_for_ip,
                    client_ip,
                    str(DEFAULT_AVATAR_MEDIA_DIR),
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Default avatar bootstrap failed: %s", exc)

        try:
            task = asyncio.create_task(_kickoff_default_avatar())
            _session_tasks.add(task)
            task.add_done_callback(_session_tasks.discard)
            default_avatar_info = {"queued": True}
        except Exception:
            logger.debug("Default avatar kickoff failed", exc_info=True)
    request.state.default_avatar = default_avatar_info

    user_id, _, identity_source, needs_disambiguation, _ = resolve_identity(request)

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
        "user_id": user_id,
        "identity_source": identity_source,
        "identity_needs_disambiguation": needs_disambiguation,
    }
    write_session_event(log_path, visit_meta)

    response = await call_next(request)

    if getattr(request.state, "device_id_generated", False):
        try:
            response.headers["X-Lexi-Device"] = str(getattr(request.state, "device_id", ""))
        except Exception:
            pass

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
        # best-effort touch user profile on first visit of a session
        if user_profile_feature_enabled() and user_id:
            try:
                touch_last_seen(user_id)
            except Exception:
                logger.debug("touch_last_seen failed", exc_info=True)

    return response


__all__ = ["session_middleware", "write_session_event"]
