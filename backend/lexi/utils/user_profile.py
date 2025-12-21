"""
Lightweight per-user profile helpers (email-or-name, optional metadata).

All functions are safe no-ops when user_id is missing. Storage lives under
``<LEX_USER_DATA_ROOT>/users/<id>/profile.json`` and reuses the normalization
rules from `user_identity`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .fileio import safe_write_json
from .user_identity import normalize_user_id, user_bucket

def _user_data_enabled() -> bool:
    return os.getenv("LEXI_USER_DATA_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(env_key: str, default_path: Path, legacy_path: Path) -> Path:
    override = os.getenv(env_key)
    if override:
        return Path(override).expanduser().resolve()
    if legacy_path.exists() and not default_path.exists():
        return legacy_path.resolve()
    return default_path.resolve()


USER_DATA_ROOT = _resolve_path(
    "LEX_USER_DATA_ROOT", _REPO_ROOT / "memory", _LEGACY_ROOT / "memory"
)
if not os.getenv("LEX_USER_DATA_ROOT"):
    memory_root = os.getenv("LEXI_MEMORY_ROOT")
    if memory_root:
        USER_DATA_ROOT = Path(memory_root).expanduser().resolve()

ACCOUNT_DB_PATH = _resolve_path(
    "LEXI_ACCOUNT_DB",
    _REPO_ROOT / "logs" / "accounts" / "accounts.sqlite3",
    _LEGACY_ROOT / "logs" / "accounts" / "accounts.sqlite3",
)


def user_profile_feature_enabled() -> bool:
    # Read env at call-time so ops can toggle without restart.
    return _user_data_enabled()


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

    success = safe_write_json(profile, path)
    return profile if success else None


def touch_last_seen(user_id: Optional[str]) -> Optional[Dict[str, object]]:
    """Update last_seen while keeping other fields intact."""
    if not user_id:
        return None
    return upsert_user_profile(user_id, touch_last_seen=True)


# ---------------------------------------------------------------------------
# Account store (expanded alpha pseudo-login)
# ---------------------------------------------------------------------------
def _normalize_email(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    cleaned = str(email).strip().lower()
    return cleaned or None


class AccountStore:
    """
    Lightweight SQLite-backed account store with uniqueness on username/email and
    disclaimer tracking.
    """

    def __init__(self, db_path: Path | str = ACCOUNT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so FastAPI workers can share the connection.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT,
                    email_normalized TEXT UNIQUE,
                    has_seen_disclaimer INTEGER DEFAULT 0,
                    disclaimer_version TEXT,
                    disclaimer_seen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_users_email_norm ON users(email_normalized);
                """
            )

    def _row_dict(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        data = dict(row)
        data["has_seen_disclaimer"] = bool(data.get("has_seen_disclaimer"))
        return data

    def _now_iso(self) -> str:
        return _iso_now()

    def create_user(self, *, username: Optional[str], email: Optional[str]) -> Dict[str, Any]:
        cleaned_username = (username or "").strip() or None
        normalized_email = _normalize_email(email)
        stored_email = normalized_email
        if not cleaned_username and not stored_email:
            raise ValueError("identifier required")
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        now = self._now_iso()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO users (id, username, email, email_normalized, created_at, updated_at, has_seen_disclaimer)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (user_id, cleaned_username, stored_email, normalized_email, now, now),
            )
        return self.require_user(user_id)

    def require_user(self, user_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone()
        result = self._row_dict(row)
        if not result:
            raise KeyError(f"user {user_id} not found")
        return result

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        cleaned = (username or "").strip()
        if not cleaned:
            return None
        row = self.conn.execute("SELECT * FROM users WHERE username = ? LIMIT 1", (cleaned,)).fetchone()
        return self._row_dict(row)

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        normalized = _normalize_email(email)
        if not normalized:
            return None
        row = self.conn.execute(
            "SELECT * FROM users WHERE email_normalized = ? LIMIT 1", (normalized,)
        ).fetchone()
        return self._row_dict(row)

    def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.require_user(user_id)
        except KeyError:
            return None

    def mark_disclaimer(
        self,
        user_id: str,
        *,
        accepted: bool,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        now = self._now_iso()
        with self.conn:
            self.conn.execute(
                """
                UPDATE users
                SET has_seen_disclaimer = ?, disclaimer_version = ?, disclaimer_seen_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if accepted else 0,
                    version or "v1",
                    now if accepted else None,
                    now,
                    user_id,
                ),
            )
        return self.get_by_id(user_id)


__all__ = [
    "USER_DATA_ROOT",
    "ACCOUNT_DB_PATH",
    "user_profile_feature_enabled",
    "user_data_root",
    "load_user_profile",
    "upsert_user_profile",
    "touch_last_seen",
    "AccountStore",
]
