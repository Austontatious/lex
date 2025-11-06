"""Alpha onboarding and temporary session instrumentation endpoints."""

from __future__ import annotations

from typing import Iterable, Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..alpha.intent import classify_intent
from ..alpha.session_manager import SessionRegistry, SessionState
from ..alpha.settings import AlphaSettings
from ..alpha.tour import preview_placeholder_url, tour_script

router = APIRouter(prefix="/lexi/alpha", tags=["Lexi Alpha"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _registry(request: Request) -> SessionRegistry:
    registry = getattr(request.app.state, "alpha_sessions", None)
    if registry is None:
        registry = SessionRegistry()
        request.app.state.alpha_sessions = registry
    return registry


def _require_session(
    request: Request,
    session_id: Optional[str] = Header(default=None, alias="X-Lexi-Session"),
) -> SessionState:
    registry = _registry(request)
    resolved_session = session_id or getattr(request.state, "session_id", None)
    if not resolved_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing session")
    try:
        return registry.require(resolved_session)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SessionStartPayload(BaseModel):
    consent: bool = True
    user_id: Optional[str] = Field(default=None, max_length=64)
    variant: Optional[str] = Field(default=None, max_length=64)
    tags: Iterable[str] = ("alpha",)


class ConsentPayload(BaseModel):
    consent: bool


class IntentPayload(BaseModel):
    text: str


class TourPreviewPayload(BaseModel):
    prompt: Optional[str] = None


class NowTopicPayload(BaseModel):
    topic: str


class MemoryNotePayload(BaseModel):
    note: str


class FeedbackPayload(BaseModel):
    helpful: bool
    comment: Optional[str] = None


class MetricEventPayload(BaseModel):
    event: str
    detail: Optional[dict] = None


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
@router.post("/session/start")
def start_session(payload: SessionStartPayload, request: Request) -> JSONResponse:
    registry = _registry(request)
    state = registry.create_session(
        consent=payload.consent,
        user_id=payload.user_id,
        variant=payload.variant,
        tags=payload.tags,
    )
    registry.append_memory(
        state.session_id,
        {"role": "system", "event": "session_started", "consent": state.consent},
    )
    registry.record_metric(
        state.session_id,
        {"event": "session_started", "consent": state.consent, "variant": state.variant},
    )
    settings = AlphaSettings()
    return JSONResponse(
        {
            "session_id": state.session_id,
            "consent": state.consent,
            "variant": state.variant,
            "alpha_strict": settings.alpha_strict,
        }
    )


@router.post("/session/consent")
def update_consent(
    payload: ConsentPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    registry.update_consent(session.session_id, payload.consent)
    registry.record_metric(
        session.session_id,
        {"event": "consent_updated", "consent": payload.consent},
    )
    registry.append_memory(
        session.session_id,
        {"role": "system", "event": "consent_updated", "consent": payload.consent},
    )
    return JSONResponse({"consent": payload.consent})


@router.post("/session/end")
def end_session(
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    dest = registry.archive(session.session_id)
    registry.append_memory(
        session.session_id,
        {"role": "system", "event": "session_archived", "path": str(dest)},
    )
    registry.record_metric(
        session.session_id,
        {"event": "session_archived", "archive_path": str(dest)},
    )
    return JSONResponse({"archived": True, "archive_path": str(dest)})


@router.get("/session/memory")
def download_memory(
    request: Request,
    session: SessionState = Depends(_require_session),
):
    path = session.memory_path
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory log missing")
    filename = f"{session.session_id}_memory.jsonl"
    return FileResponse(
        path,
        filename=filename,
        media_type="application/jsonl",
    )


@router.get("/tour/script")
def get_tour_script() -> JSONResponse:
    return JSONResponse({"steps": tour_script()})


# ---------------------------------------------------------------------------
# Intent + metrics
# ---------------------------------------------------------------------------
@router.post("/intent")
def detect_intent(
    payload: IntentPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    intent = classify_intent(payload.text)
    registry = _registry(request)
    registry.append_memory(
        session.session_id,
        {"role": "user", "event": "intent_probe", "text": payload.text, "intent": intent},
    )
    registry.record_metric(
        session.session_id,
        {"event": "intent_detected", "intent": intent},
    )
    return JSONResponse({"intent": intent})


@router.post("/session/metrics")
def record_metric_event(
    payload: MetricEventPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    registry.record_metric(
        session.session_id,
        {"event": payload.event, "detail": payload.detail},
    )
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Tour actions
# ---------------------------------------------------------------------------
def _rate_limit_or_raise(session: SessionState, request: Request, key: str, limit: int) -> None:
    registry = _registry(request)
    if not registry.increment_counter(session.session_id, key, limit):
        registry.record_metric(
            session.session_id,
            {"event": "rate_limited", "feature": key},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="let’s save the rest for later — tour limit hit.",
        )


@router.post("/tour/avatar-preview")
def avatar_preview(
    payload: TourPreviewPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    _rate_limit_or_raise(session, request, "avatar_preview", limit=2)
    registry = _registry(request)
    if payload.prompt:
        registry.append_memory(
            session.session_id,
            {"role": "user", "event": "tour_avatar_prompt", "text": payload.prompt},
        )
    registry.append_memory(
        session.session_id,
        {
            "role": "assistant",
            "event": "tour_avatar_preview",
            "placeholder": True,
        },
    )
    registry.record_metric(
        session.session_id,
        {"event": "tour_avatar_preview"},
    )
    settings = AlphaSettings()
    return JSONResponse(
        {
            "preview_url": preview_placeholder_url(settings),
            "alpha_strict": settings.alpha_strict,
        }
    )


@router.post("/tour/now-topic")
def set_now_topic(
    payload: NowTopicPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    registry.set_now_topic(session.session_id, payload.topic)
    registry.append_memory(
        session.session_id,
        {"role": "user", "event": "tour_now_topic", "topic": payload.topic},
    )
    registry.record_metric(
        session.session_id,
        {"event": "tour_now_topic"},
    )
    return JSONResponse({"now_topic": payload.topic})


@router.post("/tour/memory-note")
def remember_note(
    payload: MemoryNotePayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    registry.append_memory(
        session.session_id,
        {"role": "user", "event": "tour_memory_note", "note": payload.note},
    )
    registry.record_metric(
        session.session_id,
        {"event": "tour_memory_note"},
    )
    return JSONResponse({"ack": True})


@router.post("/tour/feedback")
def tour_feedback(
    payload: FeedbackPayload,
    request: Request,
    session: SessionState = Depends(_require_session),
) -> JSONResponse:
    registry = _registry(request)
    registry.record_metric(
        session.session_id,
        {"event": "tour_feedback", "helpful": payload.helpful, "comment": payload.comment},
    )
    registry.append_memory(
        session.session_id,
        {
            "role": "user",
            "event": "tour_feedback",
            "helpful": payload.helpful,
            "comment": payload.comment,
        },
    )
    return JSONResponse({"ok": True})
