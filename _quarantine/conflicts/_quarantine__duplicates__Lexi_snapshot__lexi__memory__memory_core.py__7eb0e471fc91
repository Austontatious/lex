"""High-level memory manager facade used by the Lexi backend."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .memory_store_json import MemoryStoreJSON
from .memory_types import MemoryShard
from ..utils.emotion_core import infer_emotion

_TOKEN_STRIP_RE = re.compile(r"<\|.*?\|>")
_HALF_LIFE_SECONDS = 30 * 24 * 60 * 60  # (30 days)


class MemoryManager:
    """Thin façade that keeps a cached list of memory shards in sync with disk."""

    def __init__(self, store: MemoryStoreJSON) -> None:
        """Load existing shards from storage."""
        self.store = store
        self._cache: List[MemoryShard] = self.store.load()

    def remember(self, shard: MemoryShard) -> None:
        """Persist a conversational shard and keep it in the in-memory cache."""
        self._cache.append(shard)
        self.store.append(shard)

    def store_context(self, user_msg: str, lexi_reply: str) -> None:
        """Persist a combined user/assistant turn when heuristics signal relevance."""
        if not isinstance(user_msg, str) or not isinstance(lexi_reply, str):
            return
        if not self._should_store_turn(user_msg, lexi_reply):
            return

        emotion = infer_emotion(user_msg)
        lexi_reply_clean = _TOKEN_STRIP_RE.sub("", lexi_reply).strip()
        shard_data = {
            "role": "conversation",
            "content": f"User: {user_msg}\nLexi: {lexi_reply_clean}",
            "meta": {
                "emotion": emotion,
                "tags": self._tag_shard(user_msg, lexi_reply_clean),
            },
        }
        shard = MemoryShard.from_json(shard_data)
        self.remember(shard)

    def recent(self, limit: int = 20) -> List[MemoryShard]:
        """Return the most recent *limit* shards for prompt context."""
        if limit <= 0:
            return []
        return self._cache[-limit:]

    def all(self) -> List[MemoryShard]:
        """Return the full in-memory cache."""
        return list(self._cache)

    def delete(self, shard_id: str) -> bool:
        """Placeholder removal API (TODO: durable delete)."""
        # TODO: remove shard from persistent store.
        removed = False
        new_cache: List[MemoryShard] = []
        for shard in self._cache:
            if not removed and getattr(shard, "created_at", "") == shard_id:
                removed = True
                continue
            new_cache.append(shard)
        if removed:
            self._cache = new_cache
        return removed

    def summarize(self, thread_id: str) -> str:
        """Placeholder conversational summary hook (TODO: implement)."""
        # NOTE: Currently unused; keep signature stable for future work.
        _ = thread_id  # suppress unused warning
        return ""

    def summarize_relevant(self, query: str, max_memories: int = 3) -> str:
        """Return a lightweight concatenation of the most relevant memory contents."""
        query = (query or "").strip()
        if not query:
            return ""

        now = datetime.now(timezone.utc)
        scored: List[tuple[float, MemoryShard]] = []
        for shard in self._cache:
            age_seconds = self._age_seconds(shard, now)
            score = self.score_with_decay(shard.content, query, age_seconds)
            if score <= 0:
                continue
            scored.append((score, shard))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        top = [shard.content for _, shard in scored[:max(1, max_memories)]]
        return "\n".join(top)

    @staticmethod
    def score_with_decay(text: str, query: str, age_seconds: float, half_life: float = _HALF_LIFE_SECONDS) -> float:
        """Return overlap score with exponential time decay."""
        if not text or not query:
            return 0.0

        lower_text = text.lower()
        lower_query = query.lower()

        # Substring bonus.
        score = 1.0 if lower_query in lower_text else 0.0

        text_tokens = set(re.findall(r"[a-z0-9']{3,}", lower_text))
        query_tokens = set(re.findall(r"[a-z0-9']{3,}", lower_query))
        if text_tokens and query_tokens:
            overlap = len(text_tokens & query_tokens)
            score += overlap / max(1, len(query_tokens))

        decay = math.exp(-max(0.0, age_seconds) / max(1.0, half_life))
        return score * decay

    def save(self, shard: MemoryShard) -> None:
        """Backwards-compatible alias for remember()."""
        self.remember(shard)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _should_store_turn(self, user_msg: str, lexi_msg: str) -> bool:
        user = (user_msg or "").strip()
        lexi = (lexi_msg or "").strip()
        if not user or not lexi:
            return False
        if len(user) < 4 or len(lexi) < 4:
            return False
        if any(x in user.lower() for x in ("hi", "hello", "hey")) and len(user) < 6:
            return False
        if "*smirks in black lipstick*" in lexi.lower():
            return True
        if any(word in user.lower() for word in ("dream", "feel", "love", "touch")):
            return True
        return False

    def _tag_shard(self, user_msg: str, lexi_reply: str) -> List[str]:
        tags: List[str] = []
        lexi_lower = lexi_reply.lower()
        user_lower = user_msg.lower()
        if "lexi v" in lexi_reply or "i used to be" in lexi_reply:
            tags.append("persona_shift")
        if any(keyword in lexi_lower for keyword in ("huginn", "symbolic", "imagine")):
            tags.append("symbolic")
        if any(phrase in user_lower for phrase in ("never heard", "first time", "new idea")):
            tags.append("novelty")
        return tags

    def _age_seconds(self, shard: MemoryShard, now: datetime) -> float:
        created_at = getattr(shard, "created_at", None)
        if isinstance(created_at, str):
            try:
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                return (now - dt.replace(tzinfo=timezone.utc)).total_seconds()
            except ValueError:
                pass
        return 0.0


# ---- Global instance ---------------------------------------------------- #
absolute_path = Path(__file__).parents[2] / "memory" / "lexi_memory.jsonl"
store = MemoryStoreJSON(filepath=absolute_path)
memory = MemoryManager(store)

__all__ = ["memory", "MemoryManager"]
