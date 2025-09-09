"""Thread‑safe JSONL persistence backend for Lex's memory."""

from __future__ import annotations
import re
import json
import threading
from pathlib import Path
from typing import List

from .memory_types import MemoryShard

_LOCK = threading.Lock()


class MemoryStoreJSON:
    def __init__(self, filepath: Path, max_entries: int = 1000) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    # ---------------- Public API ---------------- #

    def append(self, shard: MemoryShard) -> None:
        """Append a shard to persistent storage."""
        with _LOCK, self.filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(shard.to_json(), ensure_ascii=False) + "\n")

    def load(self) -> List[MemoryShard]:
        """Robustly load JSONL memory shards.

        Features:
          - Skips blank / whitespace lines.
          - Skips and logs malformed JSON lines (without aborting entire load).
          - Ignores non-dict JSON (lists / primitives) with warning.
          - Enforces `max_entries` *after* successful parse (keeps most recent N).
          - Optional lightweight validation + de‑duplication (can be toggled).
        """
        if not self.filepath.exists():
            return []

        shards: list[MemoryShard] = []

        # Optional: keep a hash set to avoid exact duplicate consecutive entries.
        # Comment out if you want raw history including duplicates.
        seen_last_hash: str | None = None

        with _LOCK, self.filepath.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue  # skip blank
                # Fast path: malformed lines often start with '{' missing closing '}' – skip early if clearly broken
                if not line.startswith('{') or line.count('{') != line.count('}'):
                    logger.warning("[Lex] Skipping structurally bad memory line: %r", line[:120])
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logger.warning("[Lex] Skipping bad memory line (%s): %r", e, line[:120])
                    continue
                if not isinstance(obj, dict):
                    logger.warning("[Lex] Non-dict JSON in memory file skipped: %r", type(obj).__name__)
                    continue
                try:
                    shard = MemoryShard.from_json(obj)
                except Exception as e:
                    logger.warning("[Lex] Could not materialize MemoryShard: %s | data=%r", e, obj)
                    continue
                # Basic validation
                if not shard.content or len(shard.content.strip()) < 2:
                    continue
                # De‑dup consecutive identical (by content+role)
                dedupe_key = f"{shard.role}:{shard.content.strip()}"
                curr_hash = hash(dedupe_key)
                if curr_hash == seen_last_hash:
                    continue
                seen_last_hash = curr_hash
                shards.append(shard)

        # Keep only the last max_entries
        if self.max_entries and len(shards) > self.max_entries:
            shards = shards[-self.max_entries:]

        return shards

