# lex/utils/__init__.py
import re
from typing import Union, Dict, Any

WordBag = set[str]

def _tokenize(text: str) -> WordBag:
    """Lower-case alpha tokens ≥ 3 chars (quick-and-dirty stop-word filter)."""
    return {
        w for w in re.findall(r"[A-Za-z']+", text.lower())
        if len(w) > 2 and w not in {
            "the", "and", "but", "for", "that", "this", "with", "you", "your",
            "are", "was", "were", "have", "has"
        }
    }

def score_overlap(query: str, shard: Union[str, Dict[str, Any]]) -> int:
    """
    Naive overlap score between *query* and a ``MemoryShard`` or plain string.
    Counting shared keywords gives us a cheap relevance proxy.
    """
    if isinstance(shard, dict):         # when we get JSON from disk
        text = shard.get("content", "")
    else:                               # when we get the object itself
        text = getattr(shard, "content", str(shard))

    return len(_tokenize(query) & _tokenize(text))

