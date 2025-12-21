from __future__ import annotations

import re
from typing import Optional

_HANDLE_ALLOWED = re.compile(r"[^a-z0-9._\- ]+")
_PATH_ALLOWED = re.compile(r"[^A-Za-z0-9_.@\-]+")
_CANONICAL_RE = re.compile(r"^user_[a-f0-9-]{8,}$")


def normalize_handle(raw: Optional[str]) -> str:
    """Normalize a free-form handle into a stable lookup key."""
    if not raw:
        return ""
    text = str(raw).strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = _HANDLE_ALLOWED.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip(" _-.")
    if len(text) > 64:
        text = text[:64].rstrip(" _-.")
    return text


def is_canonical_user_id(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(_CANONICAL_RE.match(str(value).strip().lower()))


def is_legacy_user_id(value: Optional[str]) -> bool:
    """Heuristic legacy id detection for migration flows."""
    if not value:
        return False
    text = str(value).strip().lower()
    if is_canonical_user_id(text):
        return False
    if text.startswith("anon-") or text.startswith("anon_"):
        return True
    if re.match(r"^[a-z0-9._-]+[-_]\d{4,}$", text):
        return True
    if re.match(r"^[a-z0-9._-]+[-_](sess|session)[a-z0-9._-]*$", text):
        return True
    if re.match(r"^[a-z0-9._-]+[-_][a-f0-9]{6,}$", text):
        return True
    return False


def normalize_user_id_for_paths(user_id: Optional[str]) -> Optional[str]:
    """
    Normalize a canonical user id (or legacy id) into a safe path fragment.
    """
    if not user_id:
        return None
    text = str(user_id).strip()
    if not text:
        return None
    text = text.replace(" ", "-")
    text = _PATH_ALLOWED.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if not text:
        return None
    if len(text) > 80:
        text = text[:80].rstrip("-._")
    return text or None


__all__ = [
    "normalize_handle",
    "is_canonical_user_id",
    "is_legacy_user_id",
    "normalize_user_id_for_paths",
]
