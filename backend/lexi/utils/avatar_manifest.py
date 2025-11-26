"""
Per-user avatar manifest helpers (keep first + latest, with light metadata).

Design goals:
- Off by default unless you explicitly call these helpers.
- Keeps the very first avatar (anchor) and the most recent edit.
- Stores optional metadata so future img2img continuity can reuse seeds/traits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .user_identity import normalize_user_id, user_bucket
from .user_profile import user_data_root

MANIFEST_NAME = "avatars_manifest.json"
AVATAR_MANIFEST_ENABLED = os.getenv("LEXI_USER_DATA_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_path(user_id: Optional[str]) -> Optional[Path]:
    normalized = normalize_user_id(user_id)
    if not normalized:
        return None
    bucket = user_bucket(user_data_root(), normalized)
    if not bucket:
        return None
    return bucket / MANIFEST_NAME


def load_avatar_manifest(user_id: Optional[str]) -> Dict[str, object]:
    """Return manifest dict (empty if missing)."""
    if not AVATAR_MANIFEST_ENABLED:
        return {}
    path = _manifest_path(user_id)
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record_avatar_event(
    user_id: Optional[str],
    image_path: str,
    *,
    prompt: Optional[str] = None,
    traits: Optional[Dict[str, object]] = None,
    mode: str = "txt2img",
    seed: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Persist a new avatar render into the per-user manifest. Returns the manifest
    (even when user_id is None, for convenient chaining).
    """
    if not AVATAR_MANIFEST_ENABLED:
        return {}
    normalized = normalize_user_id(user_id)
    if not normalized:
        return {}

    path = _manifest_path(normalized)
    if not path:
        return {}

    manifest = load_avatar_manifest(normalized)
    now = _iso_now()
    event = {
        "path": str(image_path),
        "created_at": now,
        "mode": mode,
        "prompt": prompt,
        "traits": traits or {},
        "seed": seed,
        "session_id": session_id,
        "basename": Path(str(image_path)).name,
    }

    manifest.setdefault("user_id", normalized)
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now

    if not manifest.get("first"):
        manifest["first"] = event
    manifest["latest"] = event

    # Keep a minimal history: first + latest (unique)
    history = []
    if manifest.get("first"):
        history.append(manifest["first"])
    if manifest.get("latest") and manifest["latest"] is not manifest.get("first"):
        history.append(manifest["latest"])
    manifest["history"] = history

    try:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return manifest
    return manifest


def latest_avatar_path(user_id: Optional[str]) -> Optional[str]:
    manifest = load_avatar_manifest(user_id)
    latest = manifest.get("latest") if isinstance(manifest, dict) else None
    if isinstance(latest, dict):
        return str(latest.get("path") or latest.get("basename") or "")
    return None


def first_avatar_path(user_id: Optional[str]) -> Optional[str]:
    manifest = load_avatar_manifest(user_id)
    first = manifest.get("first") if isinstance(manifest, dict) else None
    if isinstance(first, dict):
        return str(first.get("path") or first.get("basename") or "")
    return None


__all__ = [
    "MANIFEST_NAME",
    "load_avatar_manifest",
    "record_avatar_event",
    "latest_avatar_path",
    "first_avatar_path",
    "AVATAR_MANIFEST_ENABLED",
]
