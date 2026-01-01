from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - platform dependent
    fcntl = None

log = logging.getLogger("lexi.analytics")

ANALYTICS_DIR = os.getenv("LEXI_ANALYTICS_DIR", "/mnt/data/Lex/logs/analytics")
ANALYTICS_PATH = Path(ANALYTICS_DIR)
VISITORS_PATH = ANALYTICS_PATH / "visitors.jsonl"
HEARTBEATS_PATH = ANALYTICS_PATH / "heartbeats.jsonl"
DAILY_PATH = ANALYTICS_PATH / "daily.jsonl"
LOCK_PATH = ANALYTICS_PATH / ".lock"

LOCAL_TZ = ZoneInfo("America/Chicago")
HEARTBEAT_WINDOW_SECONDS = 120


@dataclass
class AnalyticsState:
    current_day: date | None = None
    peak_concurrent_today: int = 0
    today_visitor_ids: set[str] = field(default_factory=set)
    today_heartbeat_count: int = 0
    all_time_visitor_ids: set[str] = field(default_factory=set)
    active_map: dict[str, float] = field(default_factory=dict)
    last_rollup_written_day: date | None = None


state = AnalyticsState()
_state_lock = threading.Lock()
_dir_lock = threading.Lock()
_thread_lock = threading.Lock()
_warned_keys: set[str] = set()
_dir_ready = False
_initialized = False


def _warn_once(key: str, message: str, exc: Exception | None = None) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    if exc is not None:
        log.warning("%s: %s", message, exc)
    else:
        log.warning("%s", message)


def _ensure_dir() -> bool:
    global _dir_ready
    if _dir_ready:
        return True
    with _dir_lock:
        if _dir_ready:
            return True
        try:
            ANALYTICS_PATH.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _warn_once("analytics_dir", f"Analytics dir not writable ({ANALYTICS_PATH})", exc)
            return False
        _dir_ready = True
        return True


class _DailyFileLock:
    """Best-effort file lock for daily rollups (works across workers on Linux)."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._lock_file: Any | None = None
        self._using_thread_lock = False

    def __enter__(self) -> "_DailyFileLock":
        if fcntl is None:
            _thread_lock.acquire()
            self._using_thread_lock = True
            return self
        if not _ensure_dir():
            _thread_lock.acquire()
            self._using_thread_lock = True
            return self
        try:
            self._lock_file = self._lock_path.open("a")
            fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        except Exception as exc:  # pragma: no cover - defensive
            _warn_once("analytics_lock", "Analytics lock failed; using thread lock", exc)
            if self._lock_file is not None:
                try:
                    self._lock_file.close()
                except Exception:
                    pass
                self._lock_file = None
            _thread_lock.acquire()
            self._using_thread_lock = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None
        if self._using_thread_lock:
            _thread_lock.release()
            self._using_thread_lock = False


def _append_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    if not _ensure_dir():
        _warn_once(
            f"analytics_write:{path}",
            f"Analytics write skipped; directory not ready ({ANALYTICS_PATH})",
        )
        return False
    try:
        line = json.dumps(payload, ensure_ascii=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True
    except Exception as exc:
        _warn_once(f"analytics_write:{path}", f"Analytics write failed for {path}", exc)
        return False


def _daily_has_day(day_str: str) -> bool:
    if not DAILY_PATH.exists():
        return False
    try:
        with DAILY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("day") == day_str:
                    return True
    except Exception as exc:
        _warn_once(f"analytics_read:{DAILY_PATH}", f"Analytics read failed for {DAILY_PATH}", exc)
    return False


def _set_last_rollup(day: date) -> None:
    with _state_lock:
        state.last_rollup_written_day = day


def _write_daily_rollup(day: date, payload: dict[str, Any]) -> None:
    day_str = day.isoformat()
    with _state_lock:
        if state.last_rollup_written_day == day:
            return
    with _DailyFileLock(LOCK_PATH):
        if _daily_has_day(day_str):
            _set_last_rollup(day)
            return
        if not _ensure_dir():
            _warn_once(
                f"analytics_write:{DAILY_PATH}",
                f"Analytics write skipped; directory not ready ({ANALYTICS_PATH})",
            )
            return
        try:
            line = json.dumps(payload, ensure_ascii=True)
            with DAILY_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            _set_last_rollup(day)
        except Exception as exc:
            _warn_once(f"analytics_write:{DAILY_PATH}", f"Analytics write failed for {DAILY_PATH}", exc)


def _load_visitors() -> None:
    if not VISITORS_PATH.exists():
        return
    loaded: set[str] = set()
    try:
        with VISITORS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                visitor_id = payload.get("visitor_id")
                if isinstance(visitor_id, str) and visitor_id:
                    loaded.add(visitor_id)
    except Exception as exc:
        _warn_once(f"analytics_read:{VISITORS_PATH}", f"Analytics read failed for {VISITORS_PATH}", exc)
        return
    if loaded:
        with _state_lock:
            state.all_time_visitor_ids.update(loaded)


def initialize_state() -> None:
    global _initialized
    if _initialized:
        return
    _ensure_dir()
    _load_visitors()
    _initialized = True


def _normalize_now(now_utc: datetime | None) -> datetime:
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc


def ensure_rollover(now_local_day: date) -> None:
    rollup_payload: dict[str, Any] | None = None
    rollup_day: date | None = None
    with _state_lock:
        if state.current_day is None:
            state.current_day = now_local_day
            return
        if now_local_day == state.current_day:
            return
        rollup_day = state.current_day
        if state.today_heartbeat_count > 0 or state.today_visitor_ids:
            rollup_payload = {
                "day": rollup_day.isoformat(),
                "unique_visitors": len(state.today_visitor_ids),
                "peak_concurrent": state.peak_concurrent_today,
                "heartbeats": state.today_heartbeat_count,
            }
        state.today_visitor_ids = set()
        state.today_heartbeat_count = 0
        state.peak_concurrent_today = 0
        state.current_day = now_local_day
    if rollup_payload and rollup_day is not None:
        _write_daily_rollup(rollup_day, rollup_payload)


def record_heartbeat(visitor_id: str, now_utc: datetime | None = None) -> dict[str, Any]:
    now_utc = _normalize_now(now_utc)
    now_epoch = now_utc.timestamp()
    now_local_day = now_utc.astimezone(LOCAL_TZ).date()
    ensure_rollover(now_local_day)
    ts = now_utc.isoformat()
    new_visitor = False

    with _state_lock:
        cutoff = now_epoch - HEARTBEAT_WINDOW_SECONDS
        for key, last_seen in list(state.active_map.items()):
            if last_seen < cutoff:
                del state.active_map[key]
        state.active_map[visitor_id] = now_epoch
        state.today_visitor_ids.add(visitor_id)
        state.today_heartbeat_count += 1
        if visitor_id not in state.all_time_visitor_ids:
            state.all_time_visitor_ids.add(visitor_id)
            new_visitor = True
        concurrent_now = len(state.active_map)
        if concurrent_now > state.peak_concurrent_today:
            state.peak_concurrent_today = concurrent_now
        payload = {
            "ok": True,
            "concurrent_now": concurrent_now,
            "peak_concurrent_today": state.peak_concurrent_today,
            "unique_today": len(state.today_visitor_ids),
            "unique_all_time": len(state.all_time_visitor_ids),
        }

    if new_visitor:
        _append_jsonl(
            VISITORS_PATH,
            {"ts": ts, "visitor_id": visitor_id, "event": "first_seen"},
        )
    _append_jsonl(
        HEARTBEATS_PATH,
        {"ts": ts, "visitor_id": visitor_id, "event": "heartbeat"},
    )
    return payload


def get_summary(now_utc: datetime | None = None) -> dict[str, Any]:
    now_utc = _normalize_now(now_utc)
    now_epoch = now_utc.timestamp()
    now_local_day = now_utc.astimezone(LOCAL_TZ).date()

    with _state_lock:
        cutoff = now_epoch - HEARTBEAT_WINDOW_SECONDS
        for key, last_seen in list(state.active_map.items()):
            if last_seen < cutoff:
                del state.active_map[key]
        concurrent_now = len(state.active_map)
        current_day = state.current_day or now_local_day
        return {
            "unique_all_time": len(state.all_time_visitor_ids),
            "unique_today": len(state.today_visitor_ids),
            "concurrent_now": concurrent_now,
            "peak_concurrent_today": state.peak_concurrent_today,
            "day": current_day.isoformat(),
        }


initialize_state()
