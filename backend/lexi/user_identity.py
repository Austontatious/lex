from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .identity.identity_store import IdentityStore
from .identity.normalize import (
    is_canonical_user_id,
    is_legacy_user_id,
    normalize_handle,
)

logger = logging.getLogger("lexi.identity")
_STORE: Optional[IdentityStore] = None


def identity_store() -> IdentityStore:
    global _STORE
    if _STORE is None:
        _STORE = IdentityStore()
    return _STORE


def reset_identity_store(store: Optional[IdentityStore] = None) -> None:
    global _STORE
    _STORE = store


def resolve_identity(
    request: Any,
    *,
    store: Optional[IdentityStore] = None,
) -> Tuple[Optional[str], Optional[str], str, bool, List[Dict[str, Any]]]:
    """
    Resolve identity once per request.

    Returns (user_id, handle_norm, source, needs_disambiguation, candidates).
    """
    if request is None:
        return None, None, "missing_request", False, []

    headers = getattr(request, "headers", {})
    store = store or identity_store()

    device_id = None
    try:
        device_id = headers.get("x-lexi-device")
    except Exception:
        device_id = None

    generated_device = False
    if not device_id:
        device_id = str(uuid.uuid4())
        generated_device = True

    session_id = None
    try:
        session_id = headers.get("x-lexi-session") or getattr(request.state, "session_id", None)
    except Exception:
        session_id = None

    raw_user = None
    raw_handle = None
    try:
        raw_user = headers.get("x-lexi-user")
        raw_handle = headers.get("x-lexi-handle")
    except Exception:
        raw_user = None
        raw_handle = None

    user_id: Optional[str] = None
    handle_norm: Optional[str] = None
    needs_disambiguation = False
    candidates: List[Dict[str, Any]] = []
    source = "anon_temp"

    if raw_user:
        raw_user_norm = str(raw_user).strip().lower()
        if is_canonical_user_id(raw_user_norm):
            user_id = raw_user_norm
            store.ensure_user(user_id)
            source = "header_user"
        else:
            alias = store.get_alias(str(raw_user).strip()) or store.get_alias(raw_user_norm)
            if alias:
                user_id = alias
                source = "alias_header"

    if not user_id and device_id:
        bound = store.get_device_binding(device_id)
        if bound:
            user_id = bound
            source = "device_binding"

    if not user_id and session_id:
        bound = store.get_session_user(session_id)
        if bound:
            user_id = bound
            source = "session_binding"

    if not user_id and raw_handle:
        alias = store.get_alias(str(raw_handle).strip())
        if not alias:
            alias = store.get_alias(str(raw_handle).strip().lower())
        if alias:
            user_id = alias
            source = "alias_handle"
        else:
            handle_norm = normalize_handle(raw_handle)
            if handle_norm:
                candidates = store.list_handle_candidates(handle_norm)
                if not candidates:
                    user_id = store.create_user()
                    store.upsert_handle(handle_norm, user_id, raw_handle)
                    store.bind_device(device_id, user_id, handle_norm=handle_norm)
                    source = "handle_new"
                elif len(candidates) == 1:
                    user_id = candidates[0].get("user_id")
                    if user_id:
                        store.bind_device(device_id, user_id, handle_norm=handle_norm)
                        store.increment_handle_use(handle_norm, user_id)
                        source = "handle_single"
                else:
                    needs_disambiguation = True
                    source = "handle_collision"

    if not user_id and not needs_disambiguation:
        user_id = store.create_user()
        source = "anon_temp"
        if device_id:
            store.bind_device(device_id, user_id)

    if user_id and is_legacy_user_id(user_id):
        logger.error("Resolved legacy user id %s; expected canonical.", user_id)

    if user_id and session_id and is_canonical_user_id(user_id):
        store.upsert_session(session_id, user_id)

    try:
        request.state.device_id = device_id
        request.state.device_id_generated = generated_device
        request.state.handle_norm = handle_norm
        request.state.identity_source = source
        request.state.needs_disambiguation = needs_disambiguation
        request.state.identity_candidates = candidates
        request.state.user_id = user_id
    except Exception:
        pass

    return user_id, handle_norm, source, needs_disambiguation, candidates


def request_user_id(request: Any) -> Optional[str]:
    """Return the middleware-resolved user id (or None)."""
    if request is None:
        return None
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        logger.error("request.state.user_id missing; identity middleware may be bypassed")
    return user_id


def assert_request_user_id(request: Any, candidate: Optional[str]) -> Optional[str]:
    """Compare legacy-resolved ids with the middleware identity."""
    if request is None:
        return candidate
    current = getattr(request.state, "user_id", None)
    if candidate and current and candidate != current:
        logger.error("user_id mismatch: state=%s candidate=%s", current, candidate)
    return current or candidate


def identity_payload(request: Any) -> Dict[str, Any]:
    """Small helper for responses."""
    if request is None:
        return {}
    return {
        "user_id": getattr(request.state, "user_id", None),
        "device_id": getattr(request.state, "device_id", None),
        "session_id": getattr(request.state, "session_id", None),
        "handle_norm": getattr(request.state, "handle_norm", None),
        "source": getattr(request.state, "identity_source", None),
        "needs_disambiguation": getattr(request.state, "needs_disambiguation", False),
        "candidates": getattr(request.state, "identity_candidates", []),
    }


__all__ = [
    "resolve_identity",
    "request_user_id",
    "assert_request_user_id",
    "identity_store",
    "identity_payload",
    "reset_identity_store",
]
