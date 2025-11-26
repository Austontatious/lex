"""High‑level memory manager facade used by Lex backend."""

from __future__ import annotations
import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from .memory_types import MemoryShard
from .memory_store_json import MemoryStoreJSON
from ..utils.emotion_core import infer_emotion
from ..utils.user_identity import normalize_user_id, user_bucket, user_id_feature_enabled

try:  # optional vector path (env-gated)
    from .vector_store import archive_context_to_chroma, vector_feature_enabled
except Exception:  # pragma: no cover - optional dependency
    def vector_feature_enabled() -> bool:
        return False

    def archive_context_to_chroma(*args, **kwargs) -> Dict[str, object]:
        return {"ok": False, "enabled": False}


# ---- MemoryManager class ----
class MemoryManager:
    def __init__(self, store: MemoryStoreJSON, user_enabled: bool = False):
        self._default_store = store
        self.store = store
        self._cache: List[MemoryShard] = self.store.load()
        self._user_enabled = user_enabled
        self._user_id: Optional[str] = None

    def remember(self, shard: MemoryShard) -> None:
        """Persist a conversational shard and keep it in cache."""
        self._cache.append(shard)
        self.store.append(shard)
        self._maybe_vectorize(shard)

    def set_user(self, user_id: Optional[str]) -> None:
        """Swap memory store/cache based on user_id if feature flag is on."""
        if not self._user_enabled:
            return
        normalized = normalize_user_id(user_id)
        if normalized == self._user_id:
            return

        if not normalized:
            self.store = self._default_store
            self._cache = self._default_store.load()
            self._user_id = None
            return

        user_dir = user_bucket(Path(DEFAULT_MEMORY_PATH).parent, normalized)
        if not user_dir:
            return
        per_user_path = user_dir / "ltm.jsonl"
            self.store = MemoryStoreJSON(filepath=per_user_path)
            self._cache = self.store.load()
            self._user_id = normalized
            return
        # fall back to shared store if bucket creation failed
        self.store = self._default_store
        self._cache = self._default_store.load()
        self._user_id = None

    def store_context(self, user_msg: str, lex_reply: str):
        if not isinstance(user_msg, str) or not isinstance(lex_reply, str):
            return

        if not self._should_store_turn(user_msg, lex_reply):
            return

        emotion = infer_emotion(user_msg)

        # Clean out prompt tokens from lexi reply before saving
        lex_reply_clean = re.sub(r"<\|.*?\|>", "", lex_reply).strip()

        shard_data = {
            "role": "conversation",
            "content": f"User: {user_msg}\nLex: {lex_reply_clean}",
            "meta": {"emotion": emotion, "tags": self._tag_shard(user_msg, lex_reply_clean)},
        }
        shard = MemoryShard.from_json(shard_data)
        self.remember(shard)

    def _maybe_vectorize(self, shard: MemoryShard) -> None:
        """Best-effort vector ingest when enabled."""
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
        try:
            archive_context_to_chroma(
                [{"id": doc_id, "text": text, "metadata": meta}],
                session_id=session_id,
                user_id=self._user_id,
            )
        except Exception:  # pragma: no cover - defensive
            return

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

    def _tag_shard(self, user_msg: str, lex_reply: str) -> list[str]:
        tags = []
        if "Lex v" in lex_reply or "I used to be" in lex_reply:
            tags.append("persona_shift")
        if "Huginn" in lex_reply or "symbolic" in lex_reply or "imagine" in lex_reply:
            tags.append("symbolic")
        if any(word in user_msg.lower() for word in ["never heard", "first time", "new idea"]):
            tags.append("novelty")
        return tags

    def recent(self, limit: int = 20) -> List[MemoryShard]:
        """Return the most recent *limit* shards for prompt context."""
        return self._cache[-limit:]

    import math
    import time

    now = time.time()

    def score_with_decay(shard):
        age_days = max(1, (now - shard.meta.get("timestamp", now)) / 86400)
        decay = math.exp(-age_days / 30)  # 30-day half-life
        return score_overlap(msg_strip, shard) * decay

    def summarize_relevant(self, query: str, max_memories=3) -> str:
        from ..utils.summarize import summarize_pair

        all_mem = self.all()
        relevant = sorted(all_mem, key=lambda s: score_overlap(query, s), reverse=True)[
            :max_memories
        ]
        # Just jam the content together for one summarization call
        joined = "\n".join(m.content for m in relevant)
        if not joined.strip():
            return ""
        return summarize_pair(lambda p, **_: self.loader.generate(p), query, joined)

    def all(self) -> List[MemoryShard]:
        """Return the full in‑memory cache."""
        return list(self._cache)

    def save(self, shard):  # shim for older code
        return self.remember(shard)


# ---- Global instance ----
absolute_path = Path(__file__).parents[2] / "memory" / "lex_memory.jsonl"
DEFAULT_MEMORY_PATH = os.getenv("LEX_MEMORY_PATH", str(absolute_path))
store = MemoryStoreJSON(filepath=Path(DEFAULT_MEMORY_PATH))
memory = MemoryManager(store, user_enabled=user_id_feature_enabled())

__all__ = ["memory", "MemoryManager"]
