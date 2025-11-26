"""Helpers for optional per-user bucketing (email-or-name identifiers)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_INVALID_CHARS = re.compile(r"[^a-z0-9_.@-]+")


def user_id_feature_enabled() -> bool:
    """Feature flag gate for per-user persona/memory buckets."""
    flag = os.getenv("LEXI_USER_ID_ENABLED", "0")
    return str(flag).lower() in {"1", "true", "yes", "on"}


def normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    """
    Lowercase and sanitize an email-or-name into a safe path fragment.

    - Strips whitespace, swaps spaces for hyphens.
    - Collapses invalid characters to "-" and trims noisy punctuation.
    - Returns ``None`` if nothing usable remains.
    """
    if not user_id:
        return None

    text = str(user_id).strip().lower()
    if not text:
        return None

    text = text.replace(" ", "-")
    text = _INVALID_CHARS.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if not text:
        return None

    # Keep identifiers short to avoid pathological paths.
    if len(text) > 80:
        text = text[:80].rstrip("-._")

    return text or None


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


__all__ = ["normalize_user_id", "user_bucket", "user_id_feature_enabled"]
