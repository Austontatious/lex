from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime(ISO8601)


@dataclass
class MemoryShard:
    """Represents a single conversational turn stored in memory."""

    role: str  # user | assistant
    content: str
    created_at: str = utc_now()

    # Additional arbitrary metadata (e.g. sentiment, tags)
    meta: Dict[str, Any] | None = None

    def emotion_scores(self) -> Dict[str, float]:
        return self.meta.get("emotion", {}) if self.meta else {}

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "MemoryShard":
        return MemoryShard(
            role=data.get("role", ""),
            content=data.get("content", ""),
            created_at=data.get("created_at", utc_now()),
            meta=data.get("meta"),
        )
