from __future__ import annotations

import os
import time
from typing import Optional, Deque, Dict
from collections import deque

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..user_identity import request_user_id
from ..utils.user_profile import (
    load_user_profile,
    user_profile_feature_enabled,
)
from ..utils.avatar_manifest import (
    load_avatar_manifest,
    latest_avatar_path,
    first_avatar_path,
    avatar_manifest_enabled,
)
from ..memory.vector_store import semantic_search, vector_feature_enabled
from ..session_logging import log_safety_event
from ..utils.safety import classify_safety

_RATE_LIMIT_WINDOW = float(os.getenv("LEXI_USER_API_WINDOW_SEC", "60.0"))
_RATE_LIMIT_MAX = int(os.getenv("LEXI_USER_API_MAX", "60"))
_last_hits: Dict[str, Deque[float]] = {}


def _throttle(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _last_hits.setdefault(ip, deque())
    # Drop stale hits
    cutoff = now - _RATE_LIMIT_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="too many requests")
    bucket.append(now)

router = APIRouter(prefix="/lexi/user", tags=["user"])


def _require_user(request: Request) -> str:
    if getattr(request.state, "needs_disambiguation", False):
        raise HTTPException(status_code=409, detail="identity collision")
    user = request_user_id(request)
    if not user:
        raise HTTPException(status_code=401, detail="missing user id")
    return user


@router.get("/profile")
def get_profile(request: Request):
    if not user_profile_feature_enabled():
        raise HTTPException(status_code=404, detail="user profiles disabled")
    _throttle(request)
    user_id = _require_user(request)
    profile = load_user_profile(user_id) or {"id": user_id, "exists": False}
    return profile


@router.get("/avatar/history")
def avatar_history(request: Request):
    if not user_profile_feature_enabled() or not avatar_manifest_enabled():
        raise HTTPException(status_code=404, detail="avatar manifest disabled")
    _throttle(request)
    user_id = _require_user(request)
    manifest = load_avatar_manifest(user_id)
    return {
        "user_id": user_id,
        "first": first_avatar_path(user_id),
        "latest": latest_avatar_path(user_id),
        "manifest": manifest,
    }


class VectorSearchPayload(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


@router.post("/vector/search")
def vector_search(payload: VectorSearchPayload, request: Request):
    if not vector_feature_enabled():
        raise HTTPException(status_code=404, detail="vector search disabled")
    _throttle(request)
    user_id = _require_user(request)
    safety = classify_safety(payload.query)
    log_safety_event(
        request,
        turn_id=None,
        safety_event=safety.get("action", "observe"),
        categories=safety.get("categories", []),
    )
    if safety.get("blocked"):
        raise HTTPException(status_code=403, detail="query not allowed")
    results = semantic_search(payload.query, payload.k, user_id=user_id)
    return {"results": results, "user_id": user_id}


__all__ = ["router"]
