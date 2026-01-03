"""Lex – a lightweight, local‑first AI companion chatbot.

This package is intentionally minimal. All heavy multi‑agent
or tool orchestration from the original FRIDAY project has been
removed. Lex focuses on three core capabilities:

1. Conversational response generation via a single local model.
2. Lightweight, JSON‑backed memory for personalization.
3. A warm, emotionally‑aware persona for human interaction.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["config", "model_loader", "memory", "persona", "prompt_templates"]

_BASE = f"{__name__}.Lexi.lexi"
_LAZY_SUBMODULES = {
    "config": f"{_BASE}.config",
    "model_loader": f"{_BASE}.core.model_loader_core",
    "memory": f"{_BASE}.memory",
    "persona": f"{_BASE}.persona",
    "prompt_templates": f"{_BASE}.persona.prompt_templates",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_SUBMODULES.get(name)
    if target:
        module = importlib.import_module(target)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES.keys()))
