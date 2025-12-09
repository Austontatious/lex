from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..alpha.session_manager import SessionRegistry, SessionState
from ..utils.user_profile import AccountStore

router = APIRouter(prefix="/lexi/account", tags=["Account"])
log = logging.getLogger(__name__)
store = AccountStore()


def _registry(request: Request) -> SessionRegistry:
    registry = getattr(request.app.state, "alpha_sessions", None)
    if registry is None:
        registry = SessionRegistry()
        request.app.state.alpha_sessions = registry
    return registry


def _require_session(
    request: Request,
    registry: SessionRegistry = Depends(_registry),
) -> SessionState:
    session_id = request.headers.get("x-lexi-session") or getattr(request.state, "session_id", None)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing session")
    try:
        return registry.require(session_id)
    except KeyError:
        return registry.create_session(session_id=session_id)


def _display_name(user: dict[str, object]) -> str:
    username = user.get("username")
    email = user.get("email")
    if isinstance(username, str) and username.strip():
        return username
    if isinstance(email, str) and email.strip():
        return email
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat()


class AccountBootstrapReq(BaseModel):
    identifier: str = Field(..., min_length=1)
    entry_mode: Literal["new", "returning"]
    attempt_count: Optional[int] = None


class AccountBootstrapResp(BaseModel):
    status: Literal["CREATED_NEW", "FOUND_EXISTING", "EXISTS_CONFLICT", "NOT_FOUND"]
    user_id: Optional[str] = None
    display_name: Optional[str] = None
    has_seen_disclaimer: Optional[bool] = None


class DisclaimerAckReq(BaseModel):
    user_id: str
    accepted: bool
    version: Optional[str] = None


@router.post("/bootstrap", response_model=AccountBootstrapResp)
async def account_bootstrap(
    payload: AccountBootstrapReq,
    registry: SessionRegistry = Depends(_registry),
    session: SessionState = Depends(_require_session),
) -> AccountBootstrapResp:
    raw_identifier = (payload.identifier or "").strip()
    if not raw_identifier:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="identifier required")
    entry_mode = payload.entry_mode
    attempt_count = payload.attempt_count or 0
    session_id = session.session_id

    user = None
    username = None
    email = None
    if "@" in raw_identifier:
        email = raw_identifier.lower()
        user = store.get_by_email(email)
    else:
        username = raw_identifier
        user = store.get_by_username(username)

    # entry_mode == "new"
    if entry_mode == "new":
        if user is None:
            try:
                user = store.create_user(username=username, email=email)
            except Exception as exc:
                log.warning("account_bootstrap create_user failed: %s", exc)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="create failed")
            try:
                registry.attach_user(session_id, str(user["id"]))
            except Exception:
                log.debug("attach_user failed for session %s", session_id, exc_info=True)
            log.info(
                "account_bootstrap",
                extra={
                    "event": "account_bootstrap",
                    "session_id": session_id,
                    "identifier": raw_identifier,
                    "entry_mode": entry_mode,
                    "status": "CREATED_NEW",
                    "user_id": str(user["id"]),
                    "attempt_count": attempt_count,
                    "timestamp": _now_iso(),
                },
            )
            return AccountBootstrapResp(
                status="CREATED_NEW",
                user_id=str(user["id"]),
                display_name=_display_name(user),
                has_seen_disclaimer=bool(user.get("has_seen_disclaimer")),
            )

        log.info(
            "account_bootstrap",
            extra={
                "event": "account_bootstrap",
                "session_id": session_id,
                "identifier": raw_identifier,
                "entry_mode": entry_mode,
                "status": "EXISTS_CONFLICT",
                "user_id": str(user.get("id")),
                "attempt_count": attempt_count,
                "timestamp": _now_iso(),
            },
        )
        return AccountBootstrapResp(
            status="EXISTS_CONFLICT",
            user_id=str(user.get("id")),
            display_name=_display_name(user),
            has_seen_disclaimer=bool(user.get("has_seen_disclaimer")),
        )

    # entry_mode == "returning"
    if user is not None:
        try:
            registry.attach_user(session_id, str(user["id"]))
        except Exception:
            log.debug("attach_user failed for session %s", session_id, exc_info=True)
        log.info(
            "account_bootstrap",
            extra={
                "event": "account_bootstrap",
                "session_id": session_id,
                "identifier": raw_identifier,
                "entry_mode": entry_mode,
                "status": "FOUND_EXISTING",
                "user_id": str(user.get("id")),
                "attempt_count": attempt_count,
                "timestamp": _now_iso(),
            },
        )
        return AccountBootstrapResp(
            status="FOUND_EXISTING",
            user_id=str(user.get("id")),
            display_name=_display_name(user),
            has_seen_disclaimer=bool(user.get("has_seen_disclaimer")),
        )

    log.info(
        "account_bootstrap",
        extra={
            "event": "account_bootstrap",
            "session_id": session_id,
            "identifier": raw_identifier,
            "entry_mode": entry_mode,
            "status": "NOT_FOUND",
            "user_id": None,
            "attempt_count": attempt_count,
            "timestamp": _now_iso(),
        },
    )
    if attempt_count >= 3:
        log.warning(
            "account_bootstrap_lookup_exhausted",
            extra={
                "session_id": session_id,
                "identifier": raw_identifier,
                "entry_mode": entry_mode,
                "attempt_count": attempt_count,
                "event": "account_bootstrap_lookup_exhausted",
            },
        )
    return AccountBootstrapResp(status="NOT_FOUND")


@router.post("/disclaimer_ack")
async def disclaimer_ack(
    payload: DisclaimerAckReq,
    registry: SessionRegistry = Depends(_registry),
    session: SessionState = Depends(_require_session),
):
    user = store.get_by_id(payload.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    has_seen_before = bool(user.get("has_seen_disclaimer"))
    if payload.accepted:
        try:
            user = store.mark_disclaimer(payload.user_id, accepted=True, version=payload.version) or user
        except Exception as exc:
            log.warning("disclaimer_ack update failed: %s", exc)

    try:
        registry.attach_user(session.session_id, payload.user_id)
    except Exception:
        log.debug("attach_user failed for session %s", session.session_id, exc_info=True)

    log.info(
        "disclaimer_ack",
        extra={
            "event": "disclaimer_ack",
            "session_id": session.session_id,
            "user_id": payload.user_id,
            "accepted": payload.accepted,
            "disclaimer_version": payload.version or user.get("disclaimer_version") or "v1",
            "has_seen_disclaimer_before": has_seen_before,
            "timestamp": _now_iso(),
        },
    )
    return {"status": "OK"}
