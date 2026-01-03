from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..memory.vector_store import vector_health, vector_feature_enabled

router = APIRouter(prefix="/lexi/vector", tags=["vector"])


@router.get("/health")
def vector_health_check():
    """
    Return current vector config and readiness. 200 when enabled+healthy, 503 otherwise.
    """
    health = vector_health()
    if not vector_feature_enabled():
        raise HTTPException(status_code=503, detail=health)
    if not health.get("ok"):
        raise HTTPException(status_code=503, detail=health)
    return health


__all__ = ["router"]
