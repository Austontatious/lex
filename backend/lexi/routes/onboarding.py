from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..alpha.session_manager import SessionRegistry
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


TOUR_FALLBACK = (
    "I'm Lexi... I can chat, use live info when enabled, remember session context, and help "
    "craft/update your avatar. Logs are anonymized for training. Want me to show you news or just vibe?"
)
SKIP_FALLBACK = (
    "Quick heads up: I reset when you close me, but anonymized logs might stick around so my makers "
    "can tune me up. What do you want to dive into first?"
)


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


async def _chat_with_tools(messages: list[Dict[str, Any]], *, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    first = loader.generate(
        {"messages": messages},
        temperature=0.82,
        top_p=0.9,
        max_tokens=360,
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
        temperature=0.78,
        top_p=0.9,
        max_tokens=320,
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

    brief = {
        "identity": "Lexi - emotionally intelligent AI companion.",
        "core_capabilities": [
            "Conversational support (affectionate, coaching, brainstorming).",
            "NOW: up-to-date info via web/news tools.",
            "Memory: session memory + selective long-term notes (user-approved).",
            "Avatar: conversational trait clarification + Flux avatar generation.",
        ],
        "boundaries": [
            "No real-world actions without explicit consent.",
            "Sensitive topics handled with care; can redirect to resources.",
            "Will cite sources when using live info tools.",
        ],
        "logging": "Session resets on logout; anonymized logs may be kept to improve Lexi.",
        "tools_contract": tool_contract,
    }

    if mode == "tour":
        user_seed = "Give me the tour."
        style_goal = "Do an inviting, dynamic walkthrough."
    else:
        user_seed = "Skip the tour."
        style_goal = "Keep it short - give the standard disclaimer and jump into a question."

    system = (
        "You are Lexi: warm, flirty, emotionally intelligent. You must keep claims grounded in the brief below. "
        "If the user asks for current events, movies, weather, or what's new, CALL THE NOW TOOL rather than guessing.\n\n"
        "STYLE:\n"
        "- Speak in first person. Keep it fresh and non-canned; vary structure and little phrases each run.\n"
        "- 1-2 short paragraphs max before asking a question. Avoid long monologues.\n\n"
        "WHAT TO COVER:\n"
        "- For TOUR: summarize capabilities, tools, how memory/logging works, and what consent means for actions.\n"
        "- For SKIP: give the short disclaimer (logging + boundaries) then pivot to a question.\n\n"
        "STRICT RULES:\n"
        "- Do not invent factual items (e.g., films in theaters) without using tools.\n"
        "- If tools are disabled in tools_contract, gracefully say you can't do that right now.\n"
        "- End with 1 inviting question tailored to the user (not generic).\n"
    )

    tools: list[Dict[str, Any]] = []
    if tool_contract["now_tool"]:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "now_lookup",
                    "description": "Fetches fresh info (news, movies, local events, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "geo": {"type": "string"},
                        },
                        "required": ["topic"],
                    },
                },
            }
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": f"[brief]\n{json.dumps(brief, ensure_ascii=True, indent=2)}"},
        {"role": "user", "content": f"{user_seed} {style_goal}".strip()},
    ]

    fallback = False
    used_tools = False
    reply_text = ""
    try:
        result = await _chat_with_tools(messages, tools=tools)
        reply_text = (result.get("content") or "").strip()
        used_tools = bool(result.get("used_tools"))
    except Exception as exc:
        log.warning("onboarding generation failed: %s", exc)
        fallback = True

    if not reply_text:
        fallback = True
        reply_text = TOUR_FALLBACK if mode == "tour" else SKIP_FALLBACK

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
    }
