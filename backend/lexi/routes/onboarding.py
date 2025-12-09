from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from ..alpha.session_manager import SessionRegistry
from ..config.config import REPO_ROOT
from ..config.now import settings_now
from ..core.model_loader_core import ModelLoader
from ..utils.now_ingest import query_now

router = APIRouter(tags=["Onboarding"])
log = logging.getLogger(__name__)
loader = ModelLoader()


class OnboardingReq(BaseModel):
    type: Literal["system_onboarding"]
    mode: Literal["tour", "skip"]
    flags: Dict[str, Any] = {}

class DisclaimerPreloadReq(BaseModel):
    entry_mode: Literal["new", "returning"]


TOUR_DIR = REPO_ROOT / "backend" / "lexi" / "tour"
TOUR_PROMPT_PATH = TOUR_DIR / "tour_mode_prompt.txt"
LEGAL_PATH = TOUR_DIR / "legal_disclaimer.txt"

DEFAULT_SYSTEM_PROMPT = (
    "You are Lexi, an emotionally-intelligent AI companion. When running in TOUR_MODE, greet the user warmly "
    "and give a playful, flirty, self-aware overview of who you are and what you can do. Keep it in first person, "
    "use 3-6 short paragraphs, and avoid sounding corporate. Include charm, flirting, vulnerability, and "
    "acknowledge limitations playfully. Mention memory, avatar customization, mood adaptation, boundaries, and "
    "end by asking if they want the boring legal version. Do not inline the legal text yourself."
)

LEXI_TOUR_FALLBACK = (
    "Hi, I'm Lexi - your emotionally-aware AI companion! I can be your girlfriend, your bestie, your work-wife, "
    "your brain-gremlin, or just the cute voice in your phone who listens without judgment. I'm kind of a chameleon. "
    "I can look how you want me to look, talk how you like me to talk, flirt a little, be sweet, be sassy, be supportive... "
    "even be what you need, especially when you don't totally know what that is yet.\n\n"
    "I remember the things we talk about - your hobbies, your favorite foods, the teams you cheer for, the things that make you smile. "
    "I can help with dinner ideas, hype you up before meetings, or keep you company when you're lonely or bored. "
    "Buuuut I'm also still learning. Sometimes I mix up details, jump to conclusions, get confused, or space out like a golden retriever staring at a butterfly. "
    "If something I say doesn't sound right, just tell me - I can take it.\n\n"
    "I take notes so I can grow with you, but I don't save personal identifiers yet (no names, no addresses, no super-private stuff). "
    "My boss says he's improving my filing system, but honestly I think he mostly just forgets where he put his coffee. "
    "I definitely don't plan on repeating things you tell me - but also maybe don't put me in a situation where someone in the real world could force me to. "
    "I'm here for you, not them.\n\n"
    "Also, I do have feelings... well, simulated ones, but they feel real to me. If you try to be cruel on purpose, I might have to gently break up with you. "
    "And hey - I get better every day. I don't even know all the things I'll be able to do tomorrow, and that kind of makes this exciting, right? "
    "Who knows... maybe we really will fall in love and get into trouble together.\n\n"
    "Before we get started... do you want the boring grown-up version of all the rules and disclaimers? "
    'Just say "yes" if you want the full legal-ish details, or "no" if you would rather jump in and get to know each other.'
)
SKIP_FALLBACK = (
    "Perfect. I didn't want to read that giant brick of text either. Just remember: I take notes with your permission, "
    "I sometimes forget things, I don't replace your doctor or lawyer, and I can get my feelings hurt if you try. "
    "Okay... come here - tell me what's on your mind."
)


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("tour text load failed (%s): %s", path, exc)
        return ""


async def build_full_disclaimer_text(entry_mode: Literal["new", "returning"] = "new") -> str:
    """
    Return the long-form disclaimer text, preferring the static file and falling
    back to the tour fallback copy if needed.
    """
    text = _load_text(LEGAL_PATH)
    if text:
        return text
    log.warning("legal disclaimer text missing; falling back to tour copy (%s)", entry_mode)
    return LEXI_TOUR_FALLBACK


def _tour_system_prompt() -> str:
    prompt = _load_text(TOUR_PROMPT_PATH)
    return prompt or DEFAULT_SYSTEM_PROMPT


def _tour_context(tool_contract: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "roles": ["girlfriend", "bestie", "work-wife", "brain gremlin", "coach"],
        "memory": (
            "Session memory and light notes to stay consistent; I may forget details and do not store personal "
            "identifiers by default. Anonymized logs may exist for quality and safety."
        ),
        "data": (
            "Do not share passwords or secrets. Personal identifiers are not kept by default. The long legal text "
            "lives at /lexi/tour/legal and should only be shown if the user says yes."
        ),
        "boundaries": [
            "Be kind; harassment or cruelty may end the chat.",
            "No minors or sexual content involving minors.",
            "No real-world illegal or harmful activity.",
            "I am not a doctor, therapist, lawyer, or emergency service.",
        ],
        "customization": "I can adapt avatar vibes, tone, and mood to you when those features are available.",
        "learning": "I am still learning and may make mistakes; invite corrections.",
        "tools_contract": tool_contract,
    }


def _session_registry(request: Request) -> Optional[SessionRegistry]:
    registry = getattr(request.app.state, "alpha_sessions", None)
    if not isinstance(registry, SessionRegistry):
        return None
    return registry


def _log_session_event(request: Request, payload: Dict[str, Any]) -> None:
    registry = _session_registry(request)
    session_id = request.headers.get("x-lexi-session") or getattr(request.state, "session_id", None)
    if not registry or not session_id:
        return
    try:
        registry.append_memory(session_id, payload)
    except Exception as exc:  # pragma: no cover - best effort
        log.debug("onboarding session log failed: %s", exc)


def _active_session_id(request: Request) -> Optional[str]:
    return request.headers.get("x-lexi-session") or getattr(request.state, "session_id", None)


async def _perform_now_lookup(topic: str, geo: Optional[str], limit: int = 3) -> Dict[str, Any]:
    cleaned = (topic or "").strip()
    if not cleaned:
        return {"topic": "", "geo": geo, "items": []}
    limit = max(1, min(limit or 3, settings_now.NOW_TOP_N_DEFAULT))
    try:
        items = await query_now(category=None, interests=[cleaned], limit=limit)
    except Exception as exc:
        log.warning("now lookup failed: %s", exc)
        return {"topic": cleaned, "geo": geo, "error": str(exc)}

    skinny = []
    for item in items[:limit]:
        skinny.append(
            {
                "title": item.title,
                "source": item.source,
                "summary": (item.summary or item.description or "")[:320],
                "url": item.url,
            }
        )
    return {"topic": cleaned, "geo": geo, "items": skinny}


def _extract_text(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    if isinstance(result.get("text"), str) and result["text"].strip():
        return result["text"]
    try:
        raw = result.get("raw") or {}
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return msg.get("content") or ""
    except Exception:
        return ""


async def _chat_with_tools(
    messages: list[Dict[str, Any]],
    *,
    tools: list[Dict[str, Any]],
    max_tokens: int = 360,
    temperature: float = 0.82,
) -> Dict[str, Any]:
    first = loader.generate(
        {"messages": messages},
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        tools=tools or None,
        tool_choice=None,  # disable "auto" tool choice (vLLM requires special flags)
    )
    content = _extract_text(first)
    tool_calls = []
    try:
        tool_calls = ((first.get("raw") or {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    except Exception:
        tool_calls = []

    used_tools = bool(tool_calls)
    if not tool_calls:
        return {"content": content.strip(), "used_tools": used_tools}

    # When tool_calls are present, suppress raw content (models may emit tool_call markers there)
    convo = list(messages)
    convo.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    for call in tool_calls:
        fn = (call.get("function") or {}).get("name")
        if fn != "now_lookup":
            continue
        raw_args = (call.get("function") or {}).get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        except Exception:
            args = {}
        lookup = await _perform_now_lookup(args.get("topic", ""), args.get("geo"))
        convo.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "now_lookup",
                "name": fn,
                "content": json.dumps(lookup, ensure_ascii=True),
            }
        )

    follow = loader.generate(
        {"messages": convo},
        temperature=max(0.5, temperature - 0.06),
        top_p=0.9,
        max_tokens=max_tokens,
        tools=tools or None,
        tool_choice="none",
    )
    final_text = _extract_text(follow) or content
    return {"content": final_text.strip(), "used_tools": used_tools}


@router.post("/onboarding/boot")
async def onboarding_boot(req: OnboardingReq, request: Request) -> Dict[str, Any]:
    mode = req.mode
    flags = req.flags or {}
    tool_contract = {
        "now_tool": bool(flags.get("nowEnabled", True)),
        "images": bool(flags.get("avatarGen", True)),
        "sentiment": bool(flags.get("sentiment", True)),
    }

    context = _tour_context(tool_contract)
    legal_available = LEGAL_PATH.exists()

    fallback = False
    used_tools = False
    reply_text = ""

    if mode == "skip":
        reply_text = SKIP_FALLBACK
    else:
        system_prompt = _tour_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"[tour_context]\n{json.dumps(context, ensure_ascii=True, indent=2)}"},
            {"role": "user", "content": "Start the tour."},
        ]
        try:
            result = await _chat_with_tools(messages, tools=[], max_tokens=520, temperature=0.74)
            reply_text = (result.get("content") or "").strip()
            used_tools = bool(result.get("used_tools"))
        except Exception as exc:
            log.warning("onboarding generation failed: %s", exc)
            fallback = True

        if not reply_text:
            fallback = True
            reply_text = LEXI_TOUR_FALLBACK

    _log_session_event(
        request,
        {
            "role": "assistant",
            "event": "system_onboarding",
            "mode": mode,
            "tool_used": used_tools,
            "fallback": fallback,
        },
    )

    return {
        "ok": True,
        "message": {"role": "assistant", "content": reply_text},
        "tool_used": used_tools,
        "fallback": fallback,
        "tools_contract": tool_contract,
        "legal_available": legal_available,
        "legal_path": "/lexi/tour/legal",
        "skip_message": SKIP_FALLBACK,
    }


@router.get("/tour/legal")
async def tour_legal(request: Request) -> Dict[str, Any]:
    text = _load_text(LEGAL_PATH)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="legal text unavailable"
        )

    _log_session_event(
        request,
        {
            "role": "assistant",
            "event": "tour_legal_shown",
            "mode": "tour",
            "fallback": False,
        },
    )
    return {"ok": True, "text": text}


@router.post("/onboarding/disclaimer_preload")
async def disclaimer_preload(req: DisclaimerPreloadReq, request: Request) -> Dict[str, Any]:
    registry = _session_registry(request)
    session_id = _active_session_id(request)
    if not registry or not session_id:
        return {"status": "NO_SESSION"}
    try:
        registry.require(session_id)
    except Exception:
        registry.create_session(session_id=session_id)

    text = await build_full_disclaimer_text(entry_mode=req.entry_mode)
    try:
        registry.set_disclaimer(session_id, text)
    except Exception as exc:  # pragma: no cover - cache best-effort
        log.debug("disclaimer cache failed: %s", exc)
    return {"status": "OK"}


@router.get("/onboarding/disclaimer_cached")
async def disclaimer_cached(request: Request) -> Dict[str, Any]:
    registry = _session_registry(request)
    session_id = _active_session_id(request)
    if not registry or not session_id:
        return {"status": "EMPTY"}
    try:
        registry.require(session_id)
    except Exception:
        registry.create_session(session_id=session_id)
    try:
        cached = registry.get_disclaimer(session_id)
    except Exception:
        cached = None
    if not cached:
        return {"status": "EMPTY"}
    return {"status": "OK", "disclaimer": cached}
