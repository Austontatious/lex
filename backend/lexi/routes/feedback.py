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


def _prepare_feedback_dir() -> Path:
    """
    Pick or create the directory that will store JSON feedback files.
    """
    candidates = []
    env_value = os.getenv("LEXI_FEEDBACK_DIR")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(_DEFAULT_DIR)
    candidates.append(Path("/tmp/lexi_feedback"))

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
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
    record = {
        "id": uuid4().hex,
        "timestamp": timestamp,
        "message": message,
        "email": payload.email,
        "client_ip": client_ip,
    }

    safe_ts = timestamp.replace(":", "-")
    path = FEEDBACK_DIR / f"{safe_ts}_{record['id']}.json"

    try:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.error("Could not persist feedback at %s: %s", path, exc)
        raise HTTPException(status_code=500, detail="Failed to save feedback") from exc

    return {"status": "ok", "id": record["id"]}
