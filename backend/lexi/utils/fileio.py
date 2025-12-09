from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("lexi.fileio")


def safe_write_json(
    payload: Dict[str, Any],
    path: Path | str,
    *,
    retries: int = 2,
    backoff: float = 0.1,
) -> bool:
    """
    Atomically write JSON with a couple retries. Returns True on success.

    - Writes to a temporary file in the same directory, then os.replace to avoid partial files.
    - On failure, retries with backoff to handle transient fs hiccups (e.g., slow mounts).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    tmp_path: Path | None = None
    while attempt <= retries:
        try:
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=target.parent, encoding="utf-8"
            ) as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target)
            return True
        except Exception as exc:  # pragma: no cover - best effort resiliency
            attempt += 1
            logger.warning("safe_write_json failed (attempt %s/%s): %s", attempt, retries, exc)
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt <= retries:
                time.sleep(backoff * attempt)
    return False


__all__ = ["safe_write_json"]
