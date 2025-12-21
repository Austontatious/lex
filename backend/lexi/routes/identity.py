from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..identity.identity_store import IdentityStore
from ..identity.normalize import is_canonical_user_id, normalize_handle, normalize_user_id_for_paths
from ..memory.memory_core import resolve_memory_root
from ..user_identity import identity_payload, request_user_id
from ..utils.avatar_manifest import first_avatar_path, latest_avatar_path
from ..utils.user_profile import user_data_root

logger = logging.getLogger("lexi.identity_api")
router = APIRouter(prefix="/lexi", tags=["identity"])


class IdentitySelectPayload(BaseModel):
    handle: str = Field(..., min_length=1)
    selected_user_id: str = Field(..., min_length=1)
    merge_others: bool = False


class IdentityRenamePayload(BaseModel):
    old_handle: str = Field(..., min_length=1)
    new_handle: str = Field(..., min_length=1)


_def_store = IdentityStore()


def _candidate_hints(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cand in candidates:
        user_id = cand.get("user_id")
        hint = dict(cand)
        avatar = None
        if user_id:
            avatar = latest_avatar_path(user_id) or first_avatar_path(user_id)
        if avatar:
            hint["avatar_basename"] = Path(avatar).name
        out.append(hint)
    return out


@router.get("/whoami")
def whoami(request: Request) -> Dict[str, Any]:
    payload = identity_payload(request)
    user_id = payload.get("user_id")
    memory_root = resolve_memory_root()
    memory_path = None
    user_data_path = None

    if user_id:
        safe_id = normalize_user_id_for_paths(user_id)
        if safe_id:
            memory_path = str(memory_root / "users" / safe_id)
            user_data_path = str(user_data_root() / "users" / safe_id)

    if payload.get("needs_disambiguation"):
        handle_norm = payload.get("handle_norm")
        if handle_norm:
            candidates = _def_store.list_handle_candidates(handle_norm)
            payload["candidates"] = _candidate_hints(candidates)

    payload.update(
        {
            "memory_user_path": memory_path,
            "user_data_path": user_data_path,
        }
    )
    return payload


@router.post("/identity/select")
def select_identity(payload: IdentitySelectPayload, request: Request) -> Dict[str, Any]:
    if getattr(request.state, "needs_disambiguation", False) is False:
        logger.info("identity/select called without disambiguation flag")
    handle_norm = normalize_handle(payload.handle)
    if not handle_norm:
        raise HTTPException(status_code=400, detail="invalid handle")
    if not is_canonical_user_id(payload.selected_user_id):
        raise HTTPException(status_code=400, detail="invalid selected_user_id")

    candidates = _def_store.list_handle_candidates(handle_norm)
    candidate_ids = {c.get("user_id") for c in candidates}
    if payload.selected_user_id not in candidate_ids:
        raise HTTPException(status_code=400, detail="selected_user_id not a candidate")

    device_id = getattr(request.state, "device_id", None)
    if device_id:
        _def_store.bind_device(device_id, payload.selected_user_id, handle_norm=handle_norm)

    _def_store.upsert_handle(handle_norm, payload.selected_user_id, payload.handle)
    _def_store.increment_handle_use(handle_norm, payload.selected_user_id)

    session_id = getattr(request.state, "session_id", None)
    if session_id:
        _def_store.upsert_session(session_id, payload.selected_user_id)

    if payload.merge_others:
        raise HTTPException(status_code=501, detail="merge not implemented; use tools/migrate_identity.py")

    request.state.user_id = payload.selected_user_id
    request.state.handle_norm = handle_norm
    request.state.identity_source = "handle_select"

    return {"user_id": payload.selected_user_id, "handle_norm": handle_norm}


@router.post("/identity/rename")
def rename_identity(payload: IdentityRenamePayload, request: Request) -> Dict[str, Any]:
    if getattr(request.state, "needs_disambiguation", False):
        raise HTTPException(status_code=409, detail="identity collision")

    user_id = request_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user id")

    handle_norm = normalize_handle(payload.new_handle)
    if not handle_norm:
        raise HTTPException(status_code=400, detail="invalid new handle")

    _def_store.upsert_handle(handle_norm, user_id, payload.new_handle)
    _def_store.increment_handle_use(handle_norm, user_id)

    device_id = getattr(request.state, "device_id", None)
    if device_id:
        _def_store.bind_device(device_id, user_id, handle_norm=handle_norm)

    return {"user_id": user_id, "handle_norm": handle_norm}


__all__ = ["router"]
