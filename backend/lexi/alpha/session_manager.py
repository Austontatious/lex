"""Session lifecycle + logging utilities for Lexi alpha onboarding."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from .settings import AlphaSettings


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    value = dt or _utc_now()
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_ascii_json(data: Dict) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass
class SessionState:
    session_id: str
    created_at: datetime
    consent: bool
    user_id: Optional[str]
    variant: str
    session_dir: Path
    memory_path: Path
    summary_path: Path
    metrics_path: Path
    onboarding_path: Path
    disclaimer_path: Path
    counters: Dict[str, int] = field(
        default_factory=lambda: {"avatar_preview": 0, "avatar_upscale": 0}
    )
    now_topic: Optional[str] = None
    archived: bool = False
    message_count: int = 0
    last_checkpoint_count: int = 0
    disclaimer_cache: Optional[str] = None
    onboarding_text: Optional[str] = None

    def iso_created(self) -> str:
        return _iso(self.created_at)


class SessionRegistry:
    """In-memory index of active alpha sessions backed by filesystem logs."""

    CHECKPOINT_INTERVAL = 5

    def __init__(self, settings: Optional[AlphaSettings] = None):
        self._settings = settings or AlphaSettings()
        base = Path(self._settings.logs_base_dir())
        self.sessions_root = base / "sessions"
        self.archive_root = base / "archive"
        self.index_root = self.sessions_root / "_index"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionState] = {}
        self._base_counts = {
            "messages_user": 0,
            "messages_assistant": 0,
            "avatar_preview": 0,
            "avatar_upscale": 0,
            "safety_blocks": 0,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def create_session(
        self,
        *,
        consent: bool = True,
        user_id: Optional[str] = None,
        variant: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        session_id: Optional[str] = None,
    ) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            if session_id:
                existing = self._rehydrate_session(session_id)
                if existing:
                    self._sessions[session_id] = existing
                    return existing
            session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            today = _utc_now().date().isoformat()
            session_dir = self.sessions_root / today / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            memory_path = session_dir / "memory.jsonl"
            summary_path = session_dir / "summary.json"
            metrics_path = session_dir / "metrics.json"
            onboarding_path = session_dir / "onboarding.txt"
            disclaimer_path = session_dir / "disclaimer.txt"

            for path in (memory_path,):
                path.touch(exist_ok=True)

            variant_name = variant or self._pick_variant()
            state = SessionState(
                session_id=session_id,
                created_at=_utc_now(),
                consent=bool(consent),
                user_id=user_id,
                variant=variant_name,
                session_dir=session_dir,
                memory_path=memory_path,
                summary_path=summary_path,
                metrics_path=metrics_path,
                onboarding_path=onboarding_path,
                disclaimer_path=disclaimer_path,
            )

            metadata = {
                "session_id": state.session_id,
                "created_at": state.iso_created(),
                "consent": state.consent,
                "user_id": state.user_id,
                "variant": state.variant,
                "tags": list(tags or []),
                "alpha_strict": self._settings.alpha_strict,
            }
            summary_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            metrics_doc = {
                "session_id": state.session_id,
                "created_at": state.iso_created(),
                "variant": variant_name,
                "counts": {**self._base_counts, **dict(state.counters)},
                "events": [],
                "updated_at": state.iso_created(),
            }
            metrics_path.write_text(json.dumps(metrics_doc, indent=2, sort_keys=True), encoding="utf-8")

            self._sessions[session_id] = state
            self._write_index(session_id, session_dir)
            return state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                state = self._rehydrate_session(session_id)
                if state:
                    self._sessions[session_id] = state
            if not state:
                raise KeyError(f"Unknown session_id {session_id!r}")
            if state.user_id is None:
                self._refresh_state_from_disk(state)
            return state

    def require(self, session_id: Optional[str]) -> SessionState:
        if not session_id:
            raise KeyError("Session header missing")
        return self.get(session_id)

    def update_consent(self, session_id: str, consent: bool) -> None:
        with self._lock:
            state = self.get(session_id)
            if state.consent == consent:
                return
            state.consent = consent
            self._write_summary(state, {"consent": consent, "consent_updated_at": _iso()})

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def append_memory(self, session_id: str, event: Dict) -> None:
        state = self.require(session_id)
        payload = dict(event)
        payload.setdefault("ts", _iso())
        if state.consent:
            with state.memory_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            scrubbed = {k: v for k, v in payload.items() if k in {"ts", "role", "event"}}
            with state.memory_path.open("a", encoding="utf-8") as fh:
                fh.write(_ensure_ascii_json(scrubbed) + "\n")

        state.message_count += 1
        if state.message_count - state.last_checkpoint_count >= self.CHECKPOINT_INTERVAL:
            self._checkpoint(state)

    def record_metric(self, session_id: str, event: Dict) -> None:
        state = self.require(session_id)
        doc = self._load_metrics(state)
        events = doc.setdefault("events", [])
        events.append({**event, "ts": _iso()})
        doc["events"] = events
        base_counts = dict(self._base_counts)
        base_counts.update(state.counters)
        name = event.get("event")
        if name == "chat_prompt":
            base_counts["messages_user"] = base_counts.get("messages_user", 0) + 1
        if name == "chat_reply":
            base_counts["messages_assistant"] = base_counts.get("messages_assistant", 0) + 1
        if name == "avatar_preview":
            base_counts["avatar_preview"] = base_counts.get("avatar_preview", 0) + 1
        if name == "avatar_upscale":
            base_counts["avatar_upscale"] = base_counts.get("avatar_upscale", 0) + 1
        if name == "safety_block":
            base_counts["safety_blocks"] = base_counts.get("safety_blocks", 0) + 1
        doc["counts"] = base_counts
        doc["session_id"] = state.session_id
        doc["created_at"] = doc.get("created_at", state.iso_created())
        doc["variant"] = state.variant
        doc["updated_at"] = _iso()
        state.metrics_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    def set_now_topic(self, session_id: str, topic: str) -> None:
        state = self.require(session_id)
        state.now_topic = topic
        self._write_summary(
            state,
            {
                "now_topic": topic,
                "now_topic_updated_at": _iso(),
            },
        )

    def add_tags(self, session_id: str, tags: Iterable[str]) -> None:
        state = self.require(session_id)
        tag_list = set(tags)
        doc = self._load_summary(state)
        existing = set(doc.get("tags", []))
        merged = sorted(existing.union(tag_list))
        doc["tags"] = merged
        doc["updated_at"] = _iso()
        state.summary_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def increment_counter(self, session_id: str, key: str, limit: int) -> bool:
        with self._lock:
            state = self.require(session_id)
            current = state.counters.get(key, 0)
            if current >= limit:
                return False
            state.counters[key] = current + 1
            doc = self._load_metrics(state)
            doc["counts"] = dict(state.counters)
            doc["updated_at"] = _iso()
            state.metrics_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8"
            )
            return True

    # ------------------------------------------------------------------
    # User + disclaimer helpers
    # ------------------------------------------------------------------
    def attach_user(self, session_id: str, user_id: str) -> None:
        """
        Bind a user_id to the session and persist the association.
        """
        with self._lock:
            state = self.require(session_id)
            if state.user_id == user_id:
                return
            state.user_id = user_id
            self._write_summary(
                state,
                {
                    "user_id": user_id,
                    "user_attached_at": _iso(),
                },
            )

    def set_disclaimer(self, session_id: str, text: str) -> None:
        with self._lock:
            state = self.require(session_id)
            state.disclaimer_cache = text
            try:
                state.disclaimer_path.write_text(text or "", encoding="utf-8")
            except Exception:
                pass
            self._write_summary(
                state,
                {
                    "disclaimer_cached_at": _iso(),
                },
            )

    def get_disclaimer(self, session_id: str) -> Optional[str]:
        with self._lock:
            state = self.require(session_id)
            if state.disclaimer_cache:
                return state.disclaimer_cache
            try:
                cached = state.disclaimer_path.read_text(encoding="utf-8")
            except Exception:
                cached = ""
            cached = cached.strip()
            if cached:
                state.disclaimer_cache = cached
            return cached or None

    def set_onboarding_text(self, session_id: str, text: str) -> None:
        with self._lock:
            state = self.require(session_id)
            state.onboarding_text = text
            try:
                state.onboarding_path.write_text(text or "", encoding="utf-8")
            except Exception:
                pass
            self._write_summary(
                state,
                {
                    "onboarding_cached_at": _iso(),
                },
            )

    def get_onboarding_text(self, session_id: str) -> Optional[str]:
        with self._lock:
            state = self.require(session_id)
            if state.onboarding_text:
                return state.onboarding_text
            try:
                cached = state.onboarding_path.read_text(encoding="utf-8")
            except Exception:
                cached = ""
            cached = cached.strip()
            if cached:
                state.onboarding_text = cached
            return cached or None

    # ------------------------------------------------------------------
    # Archiving
    # ------------------------------------------------------------------
    def archive(self, session_id: str) -> Path:
        with self._lock:
            state = self.require(session_id)
            if state.archived:
                return self._archive_destination(state)
            destination = self._archive_destination(state)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(state.session_dir, destination, dirs_exist_ok=True)
            state.archived = True
            return destination

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pick_variant(self) -> str:
        with self._lock:
            active = len(self._sessions)
        return "tour_variant_b" if active % 2 else "tour_variant_a"

    def _checkpoint(self, state: SessionState) -> None:
        checkpoint_path = state.session_dir / "memory_checkpoint.jsonl"
        try:
            shutil.copyfile(state.memory_path, checkpoint_path)
            state.last_checkpoint_count = state.message_count
        except FileNotFoundError:
            checkpoint_path.touch(exist_ok=True)
        except Exception:
            pass

    def _write_summary(self, state: SessionState, patch: Dict) -> None:
        doc = self._load_summary(state)
        doc.update(patch)
        doc["updated_at"] = _iso()
        state.summary_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    def _load_summary(self, state: SessionState) -> Dict:
        try:
            return json.loads(state.summary_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return {}

    def _load_metrics(self, state: SessionState) -> Dict:
        try:
            return json.loads(state.metrics_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return {}

    def _archive_destination(self, state: SessionState) -> Path:
        date_dir = state.created_at.date().isoformat()
        return self.archive_root / date_dir / state.session_id

    def _refresh_state_from_disk(self, state: SessionState) -> None:
        doc = self._load_summary(state)
        if not doc:
            return
        user_id = doc.get("user_id")
        if state.user_id is None and isinstance(user_id, str) and user_id:
            state.user_id = user_id
        if "consent" in doc:
            state.consent = bool(doc.get("consent"))
        variant = doc.get("variant")
        if isinstance(variant, str) and variant:
            state.variant = variant

    def _index_path(self, session_id: str) -> Path:
        return self.index_root / session_id

    def _write_index(self, session_id: str, session_dir: Path) -> None:
        try:
            rel = session_dir.relative_to(self.sessions_root)
        except Exception:
            rel = session_dir
        try:
            self._index_path(session_id).write_text(str(rel), encoding="utf-8")
        except Exception:
            pass

    def _resolve_session_dir(self, session_id: str) -> Optional[Path]:
        index_path = self._index_path(session_id)
        try:
            rel = index_path.read_text(encoding="utf-8").strip()
        except Exception:
            rel = ""
        if rel:
            candidate = (self.sessions_root / rel).resolve()
            if candidate.exists():
                return candidate
        for candidate in self.sessions_root.glob(f"*/{session_id}"):
            if candidate.is_dir():
                self._write_index(session_id, candidate)
                return candidate
        return None

    def _parse_iso(self, value: Optional[str]) -> datetime:
        if not value:
            return _utc_now()
        try:
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except Exception:
            return _utc_now()

    def _rehydrate_session(self, session_id: str) -> Optional[SessionState]:
        session_dir = self._resolve_session_dir(session_id)
        if not session_dir:
            return None
        summary_path = session_dir / "summary.json"
        metrics_path = session_dir / "metrics.json"
        memory_path = session_dir / "memory.jsonl"
        onboarding_path = session_dir / "onboarding.txt"
        disclaimer_path = session_dir / "disclaimer.txt"
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            summary = {}
        created_at = self._parse_iso(summary.get("created_at"))
        consent = bool(summary.get("consent", True))
        user_id = summary.get("user_id")
        variant = summary.get("variant") or self._pick_variant()
        state = SessionState(
            session_id=session_id,
            created_at=created_at,
            consent=consent,
            user_id=user_id,
            variant=variant,
            session_dir=session_dir,
            memory_path=memory_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            onboarding_path=onboarding_path,
            disclaimer_path=disclaimer_path,
        )
        if not memory_path.exists():
            memory_path.touch(exist_ok=True)
        try:
            cached = disclaimer_path.read_text(encoding="utf-8").strip()
            if cached:
                state.disclaimer_cache = cached
        except Exception:
            pass
        try:
            cached = onboarding_path.read_text(encoding="utf-8").strip()
            if cached:
                state.onboarding_text = cached
        except Exception:
            pass
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8") or "{}")
            counts = metrics.get("counts")
            if isinstance(counts, dict):
                state.counters.update({k: int(v) for k, v in counts.items() if isinstance(v, int)})
        except Exception:
            pass
        return state
