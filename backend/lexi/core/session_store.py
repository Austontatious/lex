import os
import time
import secrets
from typing import Optional

_REDIS_URL = os.getenv("REDIS_URL")
_SESSION_TTL = int(os.getenv("SESSION_TTL", "2592000"))

class _MemStore:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, float]] = {}

    def create(self) -> str:
        sid = secrets.token_hex(16)
        self._store[sid] = {"ts": time.time()}
        return sid

    def exists(self, sid: Optional[str]) -> bool:
        return bool(sid and sid in self._store)

    def touch(self, sid: str) -> None:
        self._store.setdefault(sid, {})["ts"] = time.time()

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

class _RedisStore:
    def __init__(self, url: str) -> None:
        import redis as _redis  # type: ignore

        self._redis = _redis.Redis.from_url(url, decode_responses=True)

    def create(self) -> str:
        sid = secrets.token_hex(16)
        key = f"sess:{sid}"
        self._redis.hset(key, mapping={"ts": str(time.time())})
        self._redis.expire(key, _SESSION_TTL)
        return sid

    def exists(self, sid: Optional[str]) -> bool:
        if not sid:
            return False
        return bool(self._redis.exists(f"sess:{sid}"))

    def touch(self, sid: str) -> None:
        key = f"sess:{sid}"
        self._redis.hset(key, mapping={"ts": str(time.time())})
        self._redis.expire(key, _SESSION_TTL)

def _build_store():
    if _REDIS_URL and redis is not None:
        try:
            store = _RedisStore(_REDIS_URL)
            store._redis.ping()
            return store
        except Exception:  # pragma: no cover
            pass
    return _MemStore()

session_store = _build_store()
