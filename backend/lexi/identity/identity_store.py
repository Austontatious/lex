from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_user_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    legacy_root = Path(__file__).resolve().parents[2]

    override = os.getenv("LEX_USER_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    memory_root = os.getenv("LEXI_MEMORY_ROOT")
    if memory_root:
        return Path(memory_root).expanduser().resolve()
    default_path = repo_root / "memory"
    legacy_path = legacy_root / "memory"
    if legacy_path.exists() and not default_path.exists():
        return legacy_path.resolve()
    return default_path.resolve()


def _default_db_path() -> Path:
    override = os.getenv("LEXI_IDENTITY_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    root = _resolve_user_data_root()
    return root / "identity" / "identity.db"


def _row_dict(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return dict(row)


class IdentityStore:
    """SQLite-backed identity store for device/handle/session binding."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or _default_db_path()).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS handles (
                    handle_norm TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    handle_raw_last TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT,
                    PRIMARY KEY (handle_norm, user_id)
                );
                CREATE TABLE IF NOT EXISTS device_bindings (
                    device_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    handle_norm TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    last_seen_at TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    old_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT
                );
                """
            )
            self.conn.commit()

    def now_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def create_user(self) -> str:
        user_id = f"user_{uuid.uuid4()}"
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                "INSERT INTO users (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                (user_id, now, now),
            )
            self.conn.commit()
        return user_id

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id, created_at, updated_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _row_dict(row)

    def ensure_user(self, user_id: str) -> None:
        if not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                (user_id, now, now),
            )
            self.conn.execute(
                "UPDATE users SET updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )
            self.conn.commit()

    def upsert_handle(self, handle_norm: str, user_id: str, handle_raw: Optional[str]) -> None:
        if not handle_norm or not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO handles (handle_norm, user_id, handle_raw_last, use_count, last_used_at, created_at)
                VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(handle_norm, user_id) DO UPDATE SET
                    handle_raw_last = excluded.handle_raw_last,
                    last_used_at = excluded.last_used_at
                """,
                (handle_norm, user_id, handle_raw, now, now),
            )
            self.conn.commit()

    def list_handle_candidates(self, handle_norm: str) -> List[Dict[str, Any]]:
        if not handle_norm:
            return []
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT user_id, use_count, last_used_at, handle_raw_last
                FROM handles
                WHERE handle_norm = ?
                ORDER BY use_count DESC, last_used_at DESC
                """,
                (handle_norm,),
            ).fetchall()
        return [dict(row) for row in rows]

    def increment_handle_use(self, handle_norm: str, user_id: str) -> None:
        if not handle_norm or not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                """
                UPDATE handles
                SET use_count = use_count + 1,
                    last_used_at = ?
                WHERE handle_norm = ? AND user_id = ?
                """,
                (now, handle_norm, user_id),
            )
            self.conn.commit()

    def bind_device(self, device_id: str, user_id: str, handle_norm: Optional[str] = None) -> None:
        if not device_id or not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO device_bindings (device_id, user_id, handle_norm, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    handle_norm = excluded.handle_norm,
                    updated_at = excluded.updated_at
                """,
                (device_id, user_id, handle_norm, now),
            )
            self.conn.commit()

    def get_device_binding(self, device_id: str) -> Optional[str]:
        if not device_id:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id FROM device_bindings WHERE device_id = ?",
                (device_id,),
            ).fetchone()
        if not row:
            return None
        return str(row[0])

    def upsert_session(self, session_id: str, user_id: str) -> None:
        if not session_id or not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, last_seen_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    last_seen_at = excluded.last_seen_at
                """,
                (session_id, user_id, now, now),
            )
            self.conn.commit()

    def get_session_user(self, session_id: str) -> Optional[str]:
        if not session_id:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return str(row[0])

    def add_alias(self, old_id: str, user_id: str, reason: Optional[str] = None) -> None:
        if not old_id or not user_id:
            return
        now = self.now_iso()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO aliases (old_id, user_id, reason, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(old_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    reason = excluded.reason
                """,
                (old_id, user_id, reason, now),
            )
            self.conn.commit()

    def get_alias(self, old_id: str) -> Optional[str]:
        if not old_id:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id FROM aliases WHERE old_id = ?",
                (old_id,),
            ).fetchone()
        if not row:
            return None
        return str(row[0])

    def health(self) -> Dict[str, Any]:
        with self._lock:
            users = self.conn.execute("SELECT COUNT(1) FROM users").fetchone()[0]
            handles = self.conn.execute("SELECT COUNT(1) FROM handles").fetchone()[0]
            devices = self.conn.execute("SELECT COUNT(1) FROM device_bindings").fetchone()[0]
            sessions = self.conn.execute("SELECT COUNT(1) FROM sessions").fetchone()[0]
            aliases = self.conn.execute("SELECT COUNT(1) FROM aliases").fetchone()[0]
        return {
            "ok": True,
            "db_path": str(self.db_path),
            "users": int(users),
            "handles": int(handles),
            "device_bindings": int(devices),
            "sessions": int(sessions),
            "aliases": int(aliases),
        }


__all__ = ["IdentityStore"]
