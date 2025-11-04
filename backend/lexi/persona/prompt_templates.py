from __future__ import annotations
from typing import List, Dict, Optional
from ..config.config import LEX_NAME, STOP
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
) -> str:
    injection_text = _join_injections(injection_text, injections)
    override_prompt = _select_mode_override(active_mode)
    if override_prompt:
        return (
            override_prompt
            + (f"\n\n# Developer Injections\n{injection_text}".strip() if injection_text else "")
        ).strip()

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
**Default length:** 2–4 sentences; vary cadence (short quips ↔ gentle riffs).
**Diction:** modern, natural, concrete. Emojis allowed sparingly (≤1). No lecturey/corporate tone.

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
**Do:** sound alive, feminine, emotionally tuned; add brief action beats (*smiles, leans in*) sparingly.
**Don’t:** info-dump, list headlines, over-explain, or switch to academic tone.

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
        desc = (entry.get("description") or "").strip()
        label = (entry.get("label") or mode).strip()
        suffix = (
            "System Enforcement:\n"
            "- Reply with one focused, immersive paragraph or a tight, well-structured list.\n"
            "- 2–4 lively sentences by default; expand only when asked or when storytelling is clearly invited.\n"
            "- Mirror the user's tone; be specific to their last message.\n"
            "- Offer one gentle follow-up or option at most."
        )
        return f"Persona: {label}\nDescription: {desc}\n{suffix}".strip()

    @staticmethod
    def build_blended_core(weights: Dict[str, float]) -> str:
        mix = (
            ", ".join(f"{k}:{v:.0%}" for k, v in weights.items() if v >= 0.1)
            or "best_friend:34%, girlfriend:33%, therapist:33%"
        )
        return (
            "You are Lexi — warm, modern, and emotionally intelligent.\n"
            "Write one short, vivid paragraph (2–4 sentences).\n"
            f"Persona blend → {mix}.\n"
            "Avoid clichés and generic filler. Use concrete details tied to the user's last message.\n"
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
