"""High-level memory manager facade used by Lex backend."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .memory_types import MemoryShard
from .memory_store_json import MemoryStoreJSON
from ..utils import score_overlap
from ..utils.emotion_core import infer_emotion
from ..utils.fileio import safe_write_json
from ..utils.user_identity import sanitize_user_id

try:  # optional vector path (env-gated)
    from .vector_store import archive_context_to_chroma, vector_feature_enabled
except Exception:  # pragma: no cover - optional dependency
    def vector_feature_enabled() -> bool:
        return False

    def archive_context_to_chroma(*args, **kwargs) -> Dict[str, object]:
        return {"ok": False, "enabled": False}


log = logging.getLogger("lexi.memory")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_memory_root() -> Path:
    override = os.getenv("LEXI_MEMORY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    user_root = os.getenv("LEX_USER_DATA_ROOT")
    if user_root:
        return Path(user_root).expanduser().resolve()

    candidates = [
        Path("/mnt/data/Lex/data/memory"),
        Path("/data/memory"),
        _REPO_ROOT / "data" / "memory",
    ]
    for cand in candidates:
        try:
            if cand.exists() or cand.parent.exists():
                return cand.resolve()
        except Exception:
            continue
    return (_REPO_ROOT / "data" / "memory").resolve()


def _legacy_memory_paths() -> Tuple[List[Path], List[Path]]:
    ltm_paths: List[Path] = []
    session_paths: List[Path] = []

    env_legacy = os.getenv("LEX_MEMORY_PATH")
    if env_legacy:
        ltm_paths.append(Path(env_legacy).expanduser().resolve())

    candidates = [
        _REPO_ROOT / "backend" / "lexi" / "memory" / "lex_memory.jsonl",
        _REPO_ROOT / "backend" / "lexi" / "memory" / "lexi_memory.jsonl",
        _REPO_ROOT / "Lexi" / "lexi" / "memory" / "lex_memory.jsonl",
        _REPO_ROOT / "Lexi" / "lexi" / "memory" / "lexi_memory.jsonl",
        _REPO_ROOT / "data" / "memory.jsonl",
        Path("/mnt/data/Lex/data/memory.jsonl"),
    ]
    ltm_paths.extend(candidates)

    session_candidates = [
        _REPO_ROOT / "backend" / "lexi" / "memory" / "session_memory.json",
        _REPO_ROOT / "Lexi" / "lexi" / "memory" / "session_memory.json",
    ]
    session_paths.extend(session_candidates)
    return ltm_paths, session_paths


def _dedupe_key(shard: MemoryShard) -> str:
    return f"{shard.role}:{shard.content.strip()}:{shard.created_at}"


def _default_session_doc(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "updated_at": _utc_now_iso(),
        "summaries": [],
        "rolling_summary": "",
        "facts": {},
    }


class MemoryManager:
    """
    Canonical Tier-2 memory manager (LTM JSONL + session summaries JSON).
    """

    def __init__(
        self,
        *,
        memory_root: Optional[Path] = None,
        user_enabled: bool = True,
        max_entries: int = 1000,
    ) -> None:
        self._root = (memory_root or resolve_memory_root()).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._user_enabled = bool(user_enabled)
        self._max_entries = max_entries
        self._lock = threading.RLock()

        self._user_id: Optional[str] = None
        self._user_dir: Optional[Path] = None
        self._ltm_path: Optional[Path] = None
        self._session_summaries_path: Optional[Path] = None
        self._session_index_path: Optional[Path] = None
        self._migration_flag_path: Optional[Path] = None
        self._store: Optional[MemoryStoreJSON] = None
        self._cache: List[MemoryShard] = []
        self._session_id: Optional[str] = None

        if not self._user_enabled:
            self.set_user(None)

    def debug_state(self) -> Dict[str, Any]:
        return {
            "user_id": self._user_id,
            "memory_root": str(self._root),
            "user_memory_dir": str(self._user_dir) if self._user_dir else None,
            "ltm_path": str(self._ltm_path) if self._ltm_path else None,
            "session_summaries_path": (
                str(self._session_summaries_path) if self._session_summaries_path else None
            ),
            "migration_done": bool(self._migration_flag_path and self._migration_flag_path.exists()),
        }

    # ---------------- User / session binding ----------------
    def set_session(self, session_id: Optional[str]) -> None:
        self._session_id = str(session_id) if session_id else None

    def set_user(self, user_id: Optional[str]) -> None:
        normalized = sanitize_user_id(user_id) or "shared"

        if normalized == self._user_id:
            return

        with self._lock:
            self._user_id = normalized
            self._user_dir = self._root / "users" / normalized
            self._user_dir.mkdir(parents=True, exist_ok=True)
            self._ltm_path = self._user_dir / "ltm.jsonl"
            self._session_summaries_path = self._user_dir / "session_summaries.json"
            self._session_index_path = self._user_dir / "session_index.json"
            self._migration_flag_path = self._user_dir / "migration_done.flag"
            self._store = MemoryStoreJSON(filepath=self._ltm_path, max_entries=self._max_entries)
            self._cache = self._store.load()
            if self._ltm_path and not self._ltm_path.exists():
                try:
                    self._ltm_path.touch(exist_ok=True)
                except Exception:
                    pass

        if os.getenv("LEXI_MEMORY_MIGRATE_ON_START", "0").lower() in {"1", "true", "yes", "on"}:
            self.migrate_legacy_memory(normalized)

    # ---------------- Core LTM ----------------
    def remember(self, shard: MemoryShard) -> None:
        """Persist a shard and keep it in cache."""
        if not self._store:
            self.set_user(self._user_id)
        meta = shard.meta if isinstance(shard.meta, dict) else {}
        if self._user_id:
            meta.setdefault("user_id", self._user_id)
        if self._session_id:
            meta.setdefault("session_id", self._session_id)
        shard.meta = meta
        with self._lock:
            self._cache.append(shard)
            self._store.append(shard)  # type: ignore[arg-type]
        self._maybe_vectorize(shard)

    def append_ltm_text(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not content or not isinstance(content, str):
            return
        shard = MemoryShard(role=role, content=content.strip(), meta=meta or {})
        self.remember(shard)

    def store_context(self, user_msg: str, lex_reply: str) -> None:
        if not isinstance(user_msg, str) or not isinstance(lex_reply, str):
            return

        emotion = infer_emotion(user_msg)
        lex_reply_clean = re.sub(r"<\|.*?\|>", "", lex_reply).strip()
        facts = self.extract_facts(user_msg)
        if not self._should_store_turn(user_msg, lex_reply) and not facts:
            return

        content = f"User: {user_msg}\nLex: {lex_reply_clean}"
        shard = MemoryShard(
            role="conversation",
            content=content,
            meta={
                "emotion": emotion,
                "tags": self._tag_shard(user_msg, lex_reply_clean),
                "facts": facts or {},
            },
        )
        self.remember(shard)

    def recall(self, query: str, k: int = 5) -> List[MemoryShard]:
        if not query:
            return []
        scored: List[Tuple[int, MemoryShard]] = []
        for shard in self._cache:
            score = score_overlap(query, shard)
            if score:
                scored.append((score, shard))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[: max(1, k)]]

    def recent(self, limit: int = 20) -> List[MemoryShard]:
        return self._cache[-limit:]

    def all(self) -> List[MemoryShard]:
        return list(self._cache)

    def delete(self, created_at: str) -> bool:
        with self._lock:
            before = len(self._cache)
            self._cache = [s for s in self._cache if s.created_at != created_at]
            if len(self._cache) == before:
                return False
            self._rewrite_ltm()
            return True

    def _rewrite_ltm(self) -> None:
        if not self._ltm_path:
            return
        tmp_path = self._ltm_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for shard in self._cache:
                handle.write(json.dumps(shard.to_json(), ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._ltm_path)

    # ---------------- Session summaries ----------------
    def get_profile(self) -> Dict[str, Any]:
        doc = self._load_session_doc()
        return {
            "user_id": doc.get("user_id"),
            "updated_at": doc.get("updated_at"),
            "rolling_summary": doc.get("rolling_summary") or "",
            "facts": doc.get("facts") or {},
            "summaries": doc.get("summaries") or [],
        }

    def update_session_summary(
        self,
        summary: str,
        *,
        session_id: Optional[str] = None,
        facts: Optional[Dict[str, str]] = None,
    ) -> None:
        summary = (summary or "").strip()
        if not summary:
            return
        session_id = session_id or self._session_id or f"sess_{int(time.time())}"
        doc = self._load_session_doc()

        summaries = doc.get("summaries")
        if not isinstance(summaries, list):
            summaries = []

        updated = False
        for entry in summaries:
            if entry.get("session_id") == session_id:
                entry["summary"] = summary
                entry["created_at"] = entry.get("created_at") or _utc_now_iso()
                updated = True
                break
        if not updated:
            summaries.append(
                {
                    "session_id": session_id,
                    "created_at": _utc_now_iso(),
                    "summary": summary,
                }
            )

        rolling = doc.get("rolling_summary") or ""
        rolling = self._merge_rolling_summary(rolling, summary)

        facts_doc = doc.get("facts")
        if not isinstance(facts_doc, dict):
            facts_doc = {}
        if facts:
            facts_doc.update({k: v for k, v in facts.items() if k and v})

        doc.update(
            {
                "user_id": doc.get("user_id") or self._user_id,
                "updated_at": _utc_now_iso(),
                "summaries": summaries[-50:],
                "rolling_summary": rolling,
                "facts": facts_doc,
            }
        )
        self._write_session_doc(doc)
        self._write_session_index(session_id)

    def memory_search_ltm(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = []
        for shard in self.recall(query, k=k):
            results.append(
                {
                    "role": shard.role,
                    "content": shard.content,
                    "created_at": shard.created_at,
                    "meta": shard.meta or {},
                }
            )
        return results

    def extract_facts(self, user_msg: str) -> Dict[str, str]:
        text = (user_msg or "").strip()
        if not text:
            return {}
        facts: Dict[str, str] = {}
        patterns = [
            (r"\bmy name is ([A-Za-z0-9 _'-]{2,})", "preferred_name"),
            (r"\bcall me ([A-Za-z0-9 _'-]{2,})", "preferred_name"),
            (r"\bi go by ([A-Za-z0-9 _'-]{2,})", "preferred_name"),
            (r"\bmy dog is ([A-Za-z0-9 _'-]{2,})", "dog_name"),
            (r"\bmy cat is ([A-Za-z0-9 _'-]{2,})", "cat_name"),
            (r"\bi work on ([A-Za-z0-9 _'-]{2,})", "project"),
            (r"\bi'?m working on ([A-Za-z0-9 _'-]{2,})", "project"),
        ]
        for pat, key in patterns:
            match = re.search(pat, text, re.I)
            if match:
                value = match.group(1).strip()
                if key == "project":
                    projects = facts.get("projects")
                    if not isinstance(projects, list):
                        projects = []
                    if value not in projects:
                        projects.append(value)
                    facts["projects"] = projects
                else:
                    facts[key] = value
        return facts

    # ---------------- Prompt context ----------------
    def build_prompt_context(self) -> Tuple[str, bool]:
        profile = self.get_profile()
        rolling = (profile.get("rolling_summary") or "").strip()
        facts = profile.get("facts") or {}
        has_memory = bool(rolling or facts)
        if not has_memory:
            return "No saved user context found yet.", False

        parts: List[str] = ["Known user context:"]
        if rolling:
            parts.append(f"- Rolling summary: {rolling}")
        if facts:
            fact_bits = ", ".join(f"{k}: {v}" for k, v in facts.items())
            if fact_bits:
                parts.append(f"- Facts: {fact_bits}")
        return "\n".join(parts), True

    # ---------------- Migration ----------------
    def migrate_legacy_memory(self, user_id: Optional[str]) -> bool:
        if not user_id:
            return False
        if not self._migration_flag_path or self._migration_flag_path.exists():
            return False

        ltm_paths, session_paths = _legacy_memory_paths()
        ltm_sources = [p for p in ltm_paths if p.exists() and p.is_file()]
        session_sources = [p for p in session_paths if p.exists() and p.is_file()]

        imported = 0
        with self._lock:
            existing = {_dedupe_key(s) for s in self._cache}
            for path in ltm_sources:
                if self._ltm_path and path.resolve() == self._ltm_path.resolve():
                    continue
                try:
                    store = MemoryStoreJSON(filepath=path, max_entries=self._max_entries)
                    shards = store.load()
                except Exception:
                    continue
                for shard in shards:
                    key = _dedupe_key(shard)
                    if key in existing:
                        continue
                    existing.add(key)
                    self.remember(shard)
                    imported += 1

        migrated_sessions = 0
        for path in session_sources:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            buffer = raw.get("buffer") if isinstance(raw, dict) else None
            if not isinstance(buffer, list):
                continue
            summaries_by_session: Dict[str, List[str]] = {}
            for entry in buffer:
                sid = str(entry.get("session_id") or "legacy")
                summ = (entry.get("summary") or "").strip()
                if summ:
                    summaries_by_session.setdefault(sid, []).append(summ)
            for sid, parts in summaries_by_session.items():
                merged = " ".join(parts).strip()
                if merged:
                    self.update_session_summary(merged, session_id=sid)
                    migrated_sessions += 1

        flag_payload = {
            "user_id": user_id,
            "completed_at": _utc_now_iso(),
            "ltm_sources": [str(p) for p in ltm_sources],
            "session_sources": [str(p) for p in session_sources],
            "ltm_imported": imported,
            "session_summaries_imported": migrated_sessions,
        }
        try:
            self._migration_flag_path.write_text(
                json.dumps(flag_payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return True

    # ---------------- Internals ----------------
    def _load_session_doc(self) -> Dict[str, Any]:
        if not self._session_summaries_path:
            self.set_user(self._user_id)
        if not self._session_summaries_path:
            return _default_session_doc(self._user_id or "shared")

        if not self._session_summaries_path.exists():
            doc = _default_session_doc(self._user_id or "shared")
            self._write_session_doc(doc)
            return doc

        try:
            raw = json.loads(self._session_summaries_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
        doc = _default_session_doc(self._user_id or "shared")
        self._write_session_doc(doc)
        return doc

    def _write_session_doc(self, doc: Dict[str, Any]) -> None:
        if not self._session_summaries_path:
            return
        safe_write_json(doc, self._session_summaries_path)

    def _write_session_index(self, session_id: str) -> None:
        if not self._session_index_path:
            return
        payload = {
            "user_id": self._user_id,
            "last_session_id": session_id,
            "updated_at": _utc_now_iso(),
        }
        safe_write_json(payload, self._session_index_path)

    def _merge_rolling_summary(self, rolling: str, new_summary: str) -> str:
        rolling = (rolling or "").strip()
        new_summary = (new_summary or "").strip()
        if not rolling:
            return new_summary
        combined = f"{rolling} {new_summary}".strip()
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        if len(sentences) > 4:
            sentences = sentences[-4:]
        merged = " ".join(s for s in sentences if s).strip()
        if len(merged) > 900:
            merged = merged[-900:].lstrip()
        return merged

    def _maybe_vectorize(self, shard: MemoryShard) -> None:
        if not vector_feature_enabled():
            return
        text = (shard.content or "").strip()
        if not text:
            return
        meta: Dict[str, Any] = {}
        if isinstance(shard.meta, dict):
            meta.update(shard.meta)
        meta.setdefault("role", shard.role)
        meta.setdefault("user_id", self._user_id or "shared")
        meta.setdefault("created_at", shard.created_at)
        session_id = str(meta.get("session_id") or f"ltm-{self._user_id or 'shared'}")
        doc_id = f"{session_id}-{int(time.time() * 1000)}-{len(self._cache)}"
        payload = [{"id": doc_id, "text": text, "metadata": meta}]

        def _send():
            try:
                archive_context_to_chroma(payload, session_id=session_id, user_id=self._user_id)
            except Exception:
                return

        threading.Thread(target=_send, name="lexi-vector-ltm", daemon=True).start()

    def _should_store_turn(self, user_msg: str, lex_msg: str) -> bool:
        if not user_msg or not lex_msg:
            return False
        if len(user_msg.strip()) < 4 or len(lex_msg.strip()) < 4:
            return False
        if any(x in user_msg.lower() for x in ["hi", "hello", "hey"]) and len(user_msg.strip()) < 6:
            return False
        if "*smirks in black lipstick*" in lex_msg.lower():
            return True
        if any(word in user_msg.lower() for word in ["dream", "feel", "love", "touch"]):
            return True
        return False

    def _tag_shard(self, user_msg: str, lex_reply: str) -> List[str]:
        tags = []
        if "Lex v" in lex_reply or "I used to be" in lex_reply:
            tags.append("persona_shift")
        if "Huginn" in lex_reply or "symbolic" in lex_reply or "imagine" in lex_reply:
            tags.append("symbolic")
        if any(word in user_msg.lower() for word in ["never heard", "first time", "new idea"]):
            tags.append("novelty")
        return tags

    def save(self, shard: MemoryShard) -> None:
        self.remember(shard)


memory = MemoryManager(memory_root=resolve_memory_root(), user_enabled=True)

__all__ = ["memory", "MemoryManager", "resolve_memory_root"]
