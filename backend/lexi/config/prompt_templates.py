from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "persona"
    / "prompt_templates"
    / "lexi_system_qwen3_moe.txt"
)
SAFETY_ADDENDUM_PATH = (
    Path(__file__).resolve().parent.parent
    / "persona"
    / "prompt_templates"
    / "safety_addendum.txt"
)

_env_path = os.getenv("LEXI_SYSTEM_PROMPT_PATH")
if _env_path:
    candidate = Path(_env_path).expanduser()
    SYSTEM_PROMPT_PATH = candidate if candidate.is_absolute() else _REPO_ROOT / candidate
else:
    SYSTEM_PROMPT_PATH = DEFAULT_SYSTEM_PROMPT_PATH

SYSTEM_PROMPT_PATH = SYSTEM_PROMPT_PATH.resolve()


@lru_cache(maxsize=1)
def load_system_prompt_template() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.getLogger(__name__).warning("Failed to load system prompt template: %s", exc)
        return ""


@lru_cache(maxsize=1)
def load_safety_addendum() -> str:
    try:
        return SAFETY_ADDENDUM_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.getLogger(__name__).warning("Failed to load safety addendum: %s", exc)
        return ""


__all__ = [
    "SYSTEM_PROMPT_PATH",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "SAFETY_ADDENDUM_PATH",
    "load_system_prompt_template",
    "load_safety_addendum",
]
