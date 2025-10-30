# Lexi/lexi/memory/memory_store_json.py

from __future__ import annotations
import re
import json
import threading
from pathlib import Path
from typing import List
import os
import logging

from .memory_types import MemoryShard

_LOCK = threading.Lock()
logger = logging.getLogger("lexi.memory")  # ensure logger exists

class MemoryStoreJSON:
    def __init__(self, filepath: Path, max_entries: int = 1000) -> None:

        self.filepath = Path(filepath)
        parent = self.filepath.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to a writable base dir if original target is not writable.
            base = Path(os.getenv("LEX_DATA_DIR", "/mnt/data/Lexi_data")) / "memory"
            base.mkdir(parents=True, exist_ok=True)
            # Keep the same filename, rebase the directory.
            self.filepath = base / self.filepath.name
            logger.warning(
                "Memory path %s not writable; falling back to %s",
                str(parent), str(self.filepath)
            )
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
                    logger.warning("[Lexi] Skipping structurally bad memory line: %r", line[:120])
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logger.warning("[Lexi] Skipping bad memory line (%s): %r", e, line[:120])
                    continue
                if not isinstance(obj, dict):
                    logger.warning("[Lexi] Non-dict JSON in memory file skipped: %r", type(obj).__name__)
                    continue
                try:
                    shard = MemoryShard.from_json(obj)
                except Exception as e:
                    logger.warning("[Lexi] Could not materialize MemoryShard: %s | data=%r", e, obj)
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

