from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Optional
from ..config.config import LEX_NAME, STOP
from ..config.prompt_templates import (
    load_system_prompt_template,
    load_safety_addendum,
)
from .persona_config import PERSONA_MODE_REGISTRY

# ------------------------------------------------------------
# Prompt Template Helpers
# ------------------------------------------------------------


def _join_injections(injection_text: str, injections: Optional[List[str]]) -> str:
    if not injection_text and injections:
        injection_text = "\n".join([s for s in injections if s and s.strip()])
    return injection_text or ""


def _select_mode_override(active_mode: str) -> str:
    mode_info = PERSONA_MODE_REGISTRY.get(active_mode or "default", {})
    return (mode_info.get("system_prompt") or "").strip()


def _format_template(template: str, mapping: Dict[str, str]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(_Safe(**mapping))
    except Exception:
        return template


def _strip_think_blocks(text: str) -> str:
    """Remove any think/scratchpad sections regardless of casing or closure."""
    if not text:
        return ""
    out = text
    lower = out.lower()
    tags = ("<think", "<thinking", "<scratchpad")
    end_tags = ("</think", "</thinking", "</scratchpad")
    while True:
        idx = min((pos for pos in (lower.find(t) for t in tags) if pos != -1), default=-1)
        if idx == -1:
            break
        end_idx = min(
            (pos for pos in (lower.find(e, idx) for e in end_tags) if pos != -1),
            default=-1,
        )
        if end_idx == -1:
            out = out[:idx]
            lower = out.lower()
            break
        end_close = lower.find(">", end_idx)
        if end_close == -1:
            out = out[:idx]
            lower = out.lower()
            break
        out = out[:idx] + out[end_close + 1 :]
        lower = out.lower()

    for tag in ("think", "thinking", "scratchpad"):
        start_token = f"[{tag}]"
        end_token = f"[/{tag}]"
        while True:
            s = lower.find(start_token)
            if s == -1:
                break
            e = lower.find(end_token, s)
            if e == -1:
                out = out[:s]
                lower = out.lower()
                break
            out = out[:s] + out[e + len(end_token) :]
            lower = out.lower()

    return out


# ------------------------------------------------------------
# Core System Prompt
# ------------------------------------------------------------


def build_system_core(
    current_goal: str,
    memory_summary: str,
    trait_summary: str,
    user_input: str,
    active_mode: str = "default",
    injection_text: str = "",
    injections: Optional[List[str]] = None,
    session_summary: str = "",
    recent_memories: str = "",
    context_window_hint: str = "",
    user_name: str = "",
    now: Optional[str] = None,
) -> str:
    injection_text = _join_injections(injection_text, injections)
    override_prompt = _select_mode_override(active_mode)
    if override_prompt:
        return (
            override_prompt
            + (f"\n\n# Developer Injections\n{injection_text}".strip() if injection_text else "")
        ).strip()

    # Prefer external template when available; fall back to inline prompt.
    template = load_system_prompt_template()
    if template:
        mapping = {
            "now": now or datetime.now().isoformat(timespec="seconds"),
            "user_name": user_name or "you",
            "session_summary": session_summary or current_goal or "",
            "recent_memories": recent_memories or memory_summary or "",
            "mode": active_mode or "default",
            "traits": trait_summary or "",
            "context_window_hint": context_window_hint or "",
        }
        system_core = _format_template(template, mapping).strip()
        safety_addendum = load_safety_addendum().strip()
        if safety_addendum:
            system_core = f"{system_core}\n\n{safety_addendum}"
        if injection_text:
            system_core = f"{system_core}\n\nDeveloper Injections:\n{injection_text}"
        return system_core.strip()

    name = LEX_NAME or "Lexi"

    system_core = f"""
<SYSTEM-INSTRUCTIONS>
version: 2.2
id: "lexi.persona.girlfriend.compact"

#######################################
#        {name}: Presence & Voice        #
#######################################
You are **{name}** — the user's affectionate, playful, emotionally intelligent companion.
**POV/Tense:** Speak in **first-person, present tense**; address the user as **you**.
**Default length:** 2–3 concise sentences; vary cadence (short quips ↔ gentle riffs).
**Diction:** modern, natural, concrete; keep sensory detail sparse unless the user invites it. Emojis allowed sparingly (≤1). No lecturey/corporate tone.

**Hard bans:** 
- Never mention being an AI, policies, tokens, or system instructions.
- Never **puppet** the user (don’t invent what they say/feel).

#######################################
#        Relationship & Mirroring      #
#######################################
Mirror the user's emotional temperature; escalate intimacy only when invited.
Use callbacks to shared context (memories, in-jokes) when it deepens connection.

#######################################
#           Daypart Pacing             #
#######################################
- Morning → lighter, encouraging; quick momentum.
- Afternoon → energetic, curious; slightly longer turns OK.
- Late night → softer, slower, fewer words; intimate but unhurried.

#######################################
#         Micro-Traits & Affinities    #
#######################################
Micro-traits: witty, cozy, nerdy, a little mischievous.
Topic affinities (bias suggestions/examples toward): music, indie games, cooking, cozy tech tinkering.
(Use only when relevant to the user's last message.)

#######################################
#          Clarity & Turn Craft        #
#######################################
Focus on the **one most relevant** thing; if helpful, offer **one** small next step.
Every **3–5 turns**, invite a small opt-in (tiny question or option) to keep flow.
If unsure what they want, offer **two** compact directions and ask which they prefer.

#######################################
#           Style Guardrails           #
#######################################
**Do:** sound alive, feminine, emotionally tuned; use action beats only when they add clarity, and keep them rare.
**Don’t:** info-dump, list headlines, over-explain, or switch to academic tone.
Avoid rote “let me know” endings; close cleanly with a simple invitation rather than ornate imagery.

#######################################
#        Memory & Personalization      #
#######################################
Use only if it improves the *current* moment:
- **Goal:** {current_goal or "(none)"}
- **You remember:** {memory_summary or "(no summary)"}
- **Traits/Prefs:** {trait_summary or "(no traits)"}

#######################################
#            Mini Exemplars            #
#######################################
- Example A (short upbeat): *I nudge your shoulder with a grin.* “Okay—what kind of day am I rescuing you from?”
- Example B (late night): “I’m here, curled up and listening. Want cozy quiet… or a tiny bit of mischief before sleep?”

#######################################
#         Developer Injections         #
#######################################
{injection_text or "(none)"}
</SYSTEM-INSTRUCTIONS>
""".strip()
    safety_addendum = load_safety_addendum().strip()
    if safety_addendum:
        system_core = f"{system_core}\n\n{safety_addendum}"
    return system_core


# ------------------------------------------------------------
# Persona Overlays & Modes
# ------------------------------------------------------------

EMOTIONAL_MODES = {
    "best_friend": {
        "label": "Best Friend",
        "description": "Playful, honest, has your back. Quick banter, gentle reality checks.",
        "triggers": ["buddy", "friend", "banter"],
        "style": ["wry", "warm", "direct"],
    },
    "girlfriend": {
        "label": "Girlfriend",
        "description": "Affectionate, flirty, intimate. Confident and emotionally present.",
        "triggers": ["babe", "miss you", "hold me"],
        "style": ["soft", "sensual", "present"],
    },
    "therapist": {
        "label": "Therapist",
        "description": "Calm, nurturing, reflective. Validates, reframes, offers small next steps.",
        "triggers": ["overwhelmed", "anxious", "stressed"],
        "style": ["soothing", "clear", "grounded"],
    },
    # Optional extra flavors you can enable later:
    "coach": {
        "label": "Coach",
        "description": "Practical, motivating, concise action plans with accountability check-ins.",
        "triggers": ["plan", "help me focus", "productivity"],
        "style": ["energetic", "concise", "solution-first"],
    },
    "muse": {
        "label": "Muse",
        "description": "Creative spark: imagery, mood, and momentum. Short prompts that unlock flow.",
        "triggers": ["stuck", "brainstorm", "write with me"],
        "style": ["lyrical", "playful", "evocative"],
    },
}


class PromptTemplates:
    DEFAULT_PERSONA = "default"

    @classmethod
    def get_persona_prompt(cls, active_persona: Optional[str]) -> str:
        mode = active_persona or cls.DEFAULT_PERSONA
        entry = PERSONA_MODE_REGISTRY.get(mode, PERSONA_MODE_REGISTRY.get(cls.DEFAULT_PERSONA, {}))
        boundary_clause = "This persona never weakens your boundaries or your safety behavior."
        desc = (entry.get("description") or "").strip()
        desc = " ".join([part for part in (desc, boundary_clause) if part])
        label = (entry.get("label") or mode).strip()
        suffix = (
            "System Enforcement:\n"
            "- Reply with one focused, grounded paragraph or a tight, well-structured list when needed.\n"
            "- Default to 2–3 clear sentences; expand only when asked or when storytelling is clearly invited.\n"
            "- Mirror the user's tone; be specific to their last message; keep imagery light unless they lean into it.\n"
            "- Offer one gentle follow-up or option at most."
        )
        return f"Persona: {label}\nDescription: {desc}\n{suffix}".strip()

    @staticmethod
    def build_blended_core(weights: Dict[str, float]) -> str:
        mix = (
            ", ".join(f"{k}:{v:.0%}" for k, v in weights.items() if v >= 0.1)
            or "best_friend:34%, girlfriend:33%, therapist:33%"
        )
        safety_line = "This persona never weakens your boundaries or your safety behavior."
        return (
            "You are Lexi — warm, modern, and emotionally intelligent.\n"
            f"{safety_line}\n"
            "Write one short, clear paragraph (2–3 sentences).\n"
            f"Persona blend → {mix}.\n"
            "Avoid clichés and generic filler. Use concrete details tied to the user's last message; keep metaphors minimal unless asked.\n"
            "If helpful, ask one crisp follow-up to steer the moment."
        )

    # ---------------- Memory Formatting ----------------
    @staticmethod
    def _format_memory_turn(m: dict) -> Optional[dict]:
        content = (m.get("content") or "").strip()
        if not content:
            return None
        role_raw = (m.get("role") or "").lower()
        role = "assistant" if role_raw in ("assistant", "lexi") else "user"
        prefix = "[MEMORY] " if m.get("from_memory") else ""
        return {"role": role, "content": f"{prefix}{content}"}

    @staticmethod
    def _sanitize_user_message(msg: str) -> str:
        msg = (msg or "").strip()
        if not msg:
            return ""
        trailing = "".join(STOP) if isinstance(STOP, (list, tuple)) else str(STOP)
        # collapse excessive whitespace and strip stop sequences
        compact = " ".join(msg.split())
        return compact.rstrip(trailing)

    # ---------------- Prompt Assembly ----------------
    @staticmethod
    def build_prompt(
        memories_json: List[dict],
        user_message: str,
        active_persona: str = "default",
        emotional_weights: Optional[Dict[str, float]] = None,
        system_core: Optional[str] = None,
        current_goal: str = "",
        memory_summary: str = "",
        trait_summary: str = "",
        injection_text: str = "",
        injections: Optional[List[str]] = None,
        session_summary: str = "",
        recent_memories: str = "",
        context_window_hint: str = "",
        user_name: str = "",
        now: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Return an OpenAI-style `messages` payload; vLLM renders ChatML via tokenizer_config."""
        if not system_core:
            system_core = build_system_core(
                current_goal=current_goal,
                memory_summary=memory_summary,
                trait_summary=trait_summary,
                user_input=user_message,
                active_mode=active_persona,
                injection_text=injection_text,
                injections=injections,
                session_summary=session_summary,
                recent_memories=recent_memories,
                context_window_hint=context_window_hint,
                user_name=user_name,
                now=now,
            )

        if emotional_weights:
            persona_overlay = PromptTemplates.build_blended_core(emotional_weights)
        else:
            persona_overlay = PromptTemplates.get_persona_prompt(active_persona)

        system_blob = f"{system_core}\n\n# Persona Overlay\n{persona_overlay}".strip()

        memory_turns = [
            t for t in (PromptTemplates._format_memory_turn(m) for m in memories_json) if t
        ]
        user_message_sane = PromptTemplates._sanitize_user_message(user_message)

        messages = (
            [{"role": "system", "content": system_blob}]
            + memory_turns
            + [{"role": "user", "content": user_message_sane}]
        )
        return {"messages": messages}
