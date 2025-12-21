"""Helpers for per-user bucketing (path-safe identifiers)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ..identity.normalize import normalize_user_id_for_paths

if TYPE_CHECKING:  # pragma: no cover - typing only
    from starlette.requests import Request

logger = logging.getLogger("lexi.user_identity")


def user_id_feature_enabled() -> bool:
    """Feature flag gate for per-user persona/memory buckets."""
    flag = os.getenv("LEXI_USER_ID_ENABLED", "1")
    return str(flag).lower() in {"1", "true", "yes", "on"}


def normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    """
    Normalize a user id into a safe path fragment.
    """
    return normalize_user_id_for_paths(user_id)


def sanitize_user_id(user_id: Optional[str]) -> Optional[str]:
    """Alias for normalization (kept explicit for path safety)."""
    return normalize_user_id(user_id)


def resolve_user_id(request: "Request") -> Optional[str]:
    """Legacy shim: return request.state.user_id and log misuse."""
    if request is None:
        return None
    current = getattr(request.state, "user_id", None)
    if current:
        logger.error("resolve_user_id called; use request.state.user_id instead")
    else:
        logger.error("resolve_user_id called before identity middleware")
    return current


def user_bucket(base_dir: Path | str, user_id: Optional[str]) -> Optional[Path]:
    """
    Resolve (and create) a per-user directory under ``<base>/users/<id>``.

    Returns ``None`` when the id is absent/invalid or if the directory
    cannot be created.
    """
    normalized = normalize_user_id(user_id)
    if not normalized:
        return None

    bucket = Path(base_dir) / "users" / normalized
    try:
        bucket.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return bucket


__all__ = [
    "normalize_user_id",
    "sanitize_user_id",
    "resolve_user_id",
    "user_bucket",
    "user_id_feature_enabled",
]
