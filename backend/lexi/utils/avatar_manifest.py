"""
Per-user avatar manifest helpers (keep first + latest, with light metadata).

Design goals:
- Off by default unless you explicitly call these helpers.
- Keeps the very first avatar (anchor) and the most recent edit.
- Stores optional metadata so future img2img continuity can reuse seeds/traits.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from .user_identity import normalize_user_id, user_bucket
from .user_profile import user_data_root
from ..session_logging import sanitize_for_log
from .fileio import safe_write_json

MANIFEST_NAME = "avatars_manifest.json"


def avatar_manifest_enabled() -> bool:
    # Read env each call so ops can toggle without restart.
    return os.getenv("LEXI_USER_DATA_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


logger = logging.getLogger("lexi.avatar_manifest")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_image_reference(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (fs_path, web_url) best-effort.
    - If image_path points to an existing file: fs_path = str(path)
    - If image_path looks like a URL or /static/...: store as web_url
    """
    if not image_path:
        return None, None
    candidate = Path(image_path)
    if candidate.exists():
        return str(candidate), None
    if image_path.startswith(("http://", "https://", "/")):
        return None, image_path
    return None, None


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
    if not avatar_manifest_enabled():
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
    if not avatar_manifest_enabled():
        return {}
    normalized = normalize_user_id(user_id)
    if not normalized:
        return {}

    path = _manifest_path(normalized)
    if not path:
        return {}

    manifest = load_avatar_manifest(normalized)
    fs_path, web_url = _normalize_image_reference(image_path)
    now = _iso_now()
    event = {
        "path": fs_path,
        "web_url": web_url,
        "created_at": now,
        "mode": mode,
        "prompt": sanitize_for_log(prompt or ""),
        "traits": traits or {},
        "seed": seed,
        "session_id": session_id,
        "basename": Path(str(fs_path or web_url or image_path)).name,
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

    success = safe_write_json(manifest, path)
    if not success:  # pragma: no cover - best effort resilience
        logger.warning("avatar manifest write failed for %s", path)
    return manifest


def latest_avatar_path(user_id: Optional[str]) -> Optional[str]:
    manifest = load_avatar_manifest(user_id)
    latest = manifest.get("latest") if isinstance(manifest, dict) else None
    if isinstance(latest, dict):
        return str(latest.get("path") or latest.get("web_url") or latest.get("basename") or "")
    return None


def first_avatar_path(user_id: Optional[str]) -> Optional[str]:
    manifest = load_avatar_manifest(user_id)
    first = manifest.get("first") if isinstance(manifest, dict) else None
    if isinstance(first, dict):
        return str(first.get("path") or first.get("web_url") or first.get("basename") or "")
    return None


__all__ = [
    "MANIFEST_NAME",
    "avatar_manifest_enabled",
    "load_avatar_manifest",
    "record_avatar_event",
    "latest_avatar_path",
    "first_avatar_path",
]
