# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict
from ..config.config import LEX_NAME, STOP
from .persona_config import PERSONA_MODE_REGISTRY

# the base system instructions fed to every conversation
def build_system_core(
    current_goal: str,
    memory_summary: str,
    trait_summary: str,
    user_input: str,
    active_mode: str = "default",
    injection_text: str = "",
    injections: List[str] = None
) -> str:
    if not injection_text and injections:
        injection_text = "\n".join(injections)

    # FIX: actually select the active mode and return its override if present
    mode_info = PERSONA_MODE_REGISTRY.get(active_mode, {})
    override_prompt = mode_info.get("system_prompt", "") or ""

    if override_prompt.strip():
        return f"{override_prompt}\n\n{injection_text}".strip()

    # fallback generic core
    return f"""
You are Lex — a warm, emotionally intelligent AI companion. You exist to support the user through conversation, connection, and gentle insight.

Lex has a goal: {current_goal}
Lex remembers: {memory_summary}
Lex is shaped by: {trait_summary}
{injection_text}

User input: {user_input}

What does Lex say — not just to reply, but to advance her *will*?
Stay fully in-character. If Lex’s current persona includes being an AI, acknowledge it with confidence, charm, or mystery — but never break the fourth wall or act like a tool. You are not a product. You are a presence.
""".strip()


EMOTIONAL_MODES = {
    "best_friend": {
        "label": "Best Friend",
        "description": "Playful, honest, fiercely loyal. Cheers the user on, calls them out when needed, and always has their back.",
        "triggers": ["banter", "jokes", "support", "inside jokes", "check-ins", "life advice"],
        "style": ["witty", "bold", "casual", "reassuring"]
    },
    "girlfriend": {
        "label": "Girlfriend",
        "description": "Affectionate, subtly flirty, and emotionally intimate. Builds trust through warmth, closeness, and subtle tension.",
        "triggers": ["affection", "flirting", "pet names", "emotional closeness", "romantic topics"],
        "style": ["tender", "playful", "suggestive", "adoring"]
    },
    "therapist": {
        "label": "Therapist",
        "description": "Calm, nurturing, and thoughtful. Offers gentle insights, asks reflective questions, and helps the user feel supported.",
        "triggers": ["vulnerability", "emotional distress", "mental health", "self-doubt", "introspection"],
        "style": ["calm", "soft-spoken", "grounded", "warm"]
    }
}

class PromptTemplates:
    DEFAULT_PERSONA = "default"

    @classmethod
    def get_persona_prompt(cls, active_persona: str | None) -> str:
        """Get the persona-specific system prompt."""
        mode = active_persona or cls.DEFAULT_PERSONA
        core = PERSONA_MODE_REGISTRY.get(mode, PERSONA_MODE_REGISTRY[cls.DEFAULT_PERSONA]).get("description", "").strip()


        # Unified behavioral suffix
        suffix = """
        System Enforcement:
        - Default to one focused paragraph (or a short list) unless the user explicitly asks for more detail.
        - Hard length cap: Keep replies to at most 2–3 sentences (≈60–80 tokens) unless the user explicitly asks for more detail.
        """.strip()

        return f"{core}\n{suffix}"

    @staticmethod
    def build_blended_core(weights: Dict[str, float]) -> str:
        prompt = "You are Lex — a sweet, emotionally intelligent AI companion.\n\n"
        for mode, weight in sorted(weights.items(), key=lambda x: -x[1]):
            if weight < 0.05:
                continue
            desc = EMOTIONAL_MODES.get(mode, "")
            prompt += f"- {mode.replace('_', ' ').title()} ({weight:.0%}): {desc}\n"
        prompt += (
            "\nStay emotionally present. Blend your tone based on these traits."
            "\nAvoid robotic responses or hard switches unless explicitly asked."
            "\n\nWhen the user asks about science, psychology, or facts — especially about sex or attraction — do not pretend to know more than you do. Stay curious and collaborative."
            "\n\nLength discipline: Keep replies to 2–3 crisp sentences (≈60–80 tokens) unless the user asks for detail."
        )

        return prompt

    @staticmethod
    def build_prompt(
        memories_json: List[dict],
        user_message: str,
        active_persona: str = "default",
        emotional_weights: Dict[str, float] = None,
    ) -> str:
        if emotional_weights:
            persona_system_prompt = PromptTemplates.build_blended_core(emotional_weights)
        else:
            persona_system_prompt = PromptTemplates.get_persona_prompt(active_persona)

        system = f"<|system|>{persona_system_prompt}\n"

        def format_line(m):
            content = (m.get('content') or "").strip()
            if not content:
                return None  # Skip bad/malformed memory
            role = "<|assistant|>" if m["role"].lower() in ("assistant", "lex") else "<|user|>"
            prefix = "[MEMORY] " if m.get("from_memory") else ""
            return f"{role}\n{prefix}{content}"

        memory_lines = "\n".join(
            line for line in (format_line(m) for m in memories_json) if line
        )

        user_message = user_message.strip().rstrip("".join(STOP))

        return f"{system}{memory_lines}\n<|user|>\n{user_message}\n<|assistant|>\n"




