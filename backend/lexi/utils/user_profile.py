"""
Lightweight per-user profile helpers (email-or-name, optional metadata).

All functions are safe no-ops when user_id is missing. Storage lives under
``<LEX_USER_DATA_ROOT>/users/<id>/profile.json`` and reuses the normalization
rules from `user_identity`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .user_identity import normalize_user_id, user_bucket

USER_DATA_ROOT = Path(
    os.getenv("LEX_USER_DATA_ROOT", Path(__file__).resolve().parents[2] / "memory")
).resolve()
USER_DATA_ENABLED = os.getenv("LEXI_USER_DATA_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


def user_profile_feature_enabled() -> bool:
    return USER_DATA_ENABLED


def user_data_root() -> Path:
    USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return USER_DATA_ROOT


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _profile_path(user_id: Optional[str]) -> Optional[Path]:
    normalized = normalize_user_id(user_id)
    if not normalized:
        return None
    bucket = user_bucket(user_data_root(), normalized)
    if not bucket:
        return None
    return bucket / "profile.json"


def load_user_profile(user_id: Optional[str]) -> Optional[Dict[str, object]]:
    """Return the stored profile or None if absent/invalid."""
    path = _profile_path(user_id)
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def upsert_user_profile(
    user_id: Optional[str],
    *,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
    attributes: Optional[Dict[str, object]] = None,
    touch_last_seen: bool = True,
) -> Optional[Dict[str, object]]:
    """
    Merge and persist profile fields. Returns the updated profile dict or None
    if user_id is missing/invalid.
    """
    path = _profile_path(user_id)
    if not path:
        return None

    existing = load_user_profile(user_id) or {}
    normalized = normalize_user_id(user_id)
    now = _iso_now()

    profile: Dict[str, object] = {}
    profile.update(existing)
    profile.setdefault("id", normalized)
    profile.setdefault("created_at", now)
    if touch_last_seen or "last_seen" not in profile:
        profile["last_seen"] = now
    if email is not None:
        profile["email"] = email
    if display_name is not None:
        profile["display_name"] = display_name
    if attributes:
        merged_attrs = {}
        merged_attrs.update(profile.get("attributes", {}) if isinstance(profile.get("attributes"), dict) else {})
        merged_attrs.update(attributes)
        profile["attributes"] = merged_attrs

    try:
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return None
    return profile


def touch_last_seen(user_id: Optional[str]) -> Optional[Dict[str, object]]:
    """Update last_seen while keeping other fields intact."""
    if not user_id:
        return None
    return upsert_user_profile(user_id, touch_last_seen=True)


__all__ = [
    "USER_DATA_ROOT",
    "USER_DATA_ENABLED",
    "user_profile_feature_enabled",
    "user_data_root",
    "load_user_profile",
    "upsert_user_profile",
    "touch_last_seen",
]
