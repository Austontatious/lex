from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from ..utils.request_ip import request_ip

log = logging.getLogger("lexi.routes.feedback")

_DEFAULT_DIR = Path("/mnt/data/lexi_feedback")


def _is_private_dir(path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    if not path.is_dir():
        return False
    if st.st_uid != os.geteuid():
        return False
    return (st.st_mode & 0o077) == 0


def _open_feedback_file(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def _prepare_feedback_dir() -> Path:
    """
    Pick or create the directory that will store JSON feedback files.
    """
    candidates = []
    env_value = os.getenv("LEXI_FEEDBACK_DIR")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(_DEFAULT_DIR)

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_symlink():
                log.error("Refusing to use symlinked feedback directory: %s", candidate)
                continue
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(candidate, 0o700)
            except PermissionError:
                pass
            if _is_private_dir(candidate):
                return candidate
            log.error("Feedback directory is not private (mode/owner): %s", candidate)
        except OSError as exc:  # pragma: no cover - defensive logging
            log.error("Failed to prepare feedback directory %s: %s", candidate, exc)

    raise RuntimeError("Unable to prepare a feedback directory")


FEEDBACK_DIR = _prepare_feedback_dir()


class FeedbackIn(BaseModel):
    message: str
    email: EmailStr | None = None


router = APIRouter(tags=["feedback"])


@router.post("/lexi/feedback", status_code=status.HTTP_201_CREATED)
async def save_feedback(payload: FeedbackIn, request: Request):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message is required")

    try:
        client_ip = request_ip(request)
    except Exception:  # pragma: no cover - best effort only
        client_ip = None

    timestamp = datetime.now(timezone.utc).isoformat()
    safe_ts = timestamp.replace(":", "-")
    last_exc: OSError | None = None
    for _ in range(3):
        record_id = uuid4().hex
        path = FEEDBACK_DIR / f"{safe_ts}_{record_id}.json"
        record = {
            "id": record_id,
            "timestamp": timestamp,
            "message": message,
            "email": payload.email,
            "client_ip": client_ip,
        }
        try:
            with _open_feedback_file(path) as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)
            return {"status": "ok", "id": record_id}
        except FileExistsError:
            continue
        except OSError as exc:
            last_exc = exc
            break

    if last_exc is None:
        last_exc = FileExistsError("Feedback file already exists")
    log.error("Could not persist feedback at %s: %s", path, last_exc)
    raise HTTPException(status_code=500, detail="Failed to save feedback") from last_exc
