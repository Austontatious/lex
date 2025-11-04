# Lexi/lexi/utils/now_utils.py
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional

def to_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def freshness_decay(published_at: Optional[datetime], half_life_hours: float = 36.0) -> float:
    """Return 0..1 freshness score; newer items score closer to 1."""
    if not published_at:
        return 0.5
    pub = to_aware_utc(published_at)
    now = utcnow()
    age_h = max(0.0, (now - pub).total_seconds() / 3600.0)
    return 0.5 + 0.5 * math.exp(-age_h / half_life_hours)

def log_now(msg: str):
    print(f"[NOW] {msg}", flush=True)

def short_id(prefix: str, text: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"
