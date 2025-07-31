"""Lex – a lightweight, local‑first AI companion chatbot.

This package is intentionally minimal. All heavy multi‑agent
or tool orchestration from the original FRIDAY project has been
removed. Lex focuses on three core capabilities:

1. Conversational response generation via a single local model.
2. Lightweight, JSON‑backed memory for personalization.
3. A warm, emotionally‑aware persona for human interaction.
"""

__all__ = ["config", "model_loader", "memory", "persona", "prompt_templates"]
