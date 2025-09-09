"""High‑level memory manager facade used by Lex backend."""

from __future__ import annotations
from typing import List
from pathlib import Path
from .memory_types import MemoryShard
from .memory_store_json import MemoryStoreJSON
from ..utils.emotion_core import infer_emotion

# ---- MemoryManager class ----
class MemoryManager:
    def __init__(self, store: MemoryStoreJSON):
        self.store = store
        self._cache: List[MemoryShard] = self.store.load()

    def remember(self, shard: MemoryShard) -> None:
        """Persist a conversational shard and keep it in cache."""
        self._cache.append(shard)
        self.store.append(shard)

    def store_context(self, user_msg: str, lex_reply: str):
        if not isinstance(user_msg, str) or not isinstance(lex_reply, str):
            return

        if not self._should_store_turn(user_msg, lex_reply):
            return

        emotion = infer_emotion(user_msg)
        
        # Clean out prompt tokens from lex reply before saving
        lex_reply_clean = re.sub(r"<\|.*?\|>", "", lex_reply).strip()

        shard_data = {
            "role": "conversation",
            "content": f"User: {user_msg}\nLex: {lex_reply_clean}",
            "meta": {
                "emotion": emotion,
                "tags": self._tag_shard(user_msg, lex_reply_clean)
            }
        }
        shard = MemoryShard.from_json(shard_data)
        self.remember(shard)
        
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
        relevant = sorted(all_mem, key=lambda s: score_overlap(query, s), reverse=True)[:max_memories]
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
store = MemoryStoreJSON(filepath=absolute_path)
memory = MemoryManager(store)

__all__ = ["memory", "MemoryManager"]

