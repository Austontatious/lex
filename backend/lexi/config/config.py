"""
lex/config.py  –  Central runtime & generation configuration for Lex.

All tunables can be overridden by environment variables:
  LEX_CONTEXT_LENGTH, LEX_GPU_LAYERS, LEX_TEMPERATURE, LEX_TOP_P, LEX_TOP_K,
  LEX_MIN_P, LEX_REPEAT_PENALTY, LEX_FREQUENCY_PENALTY, LEX_PRESENCE_PENALTY,
  LEX_MAX_TOKENS, LEX_MODEL_PATH, LEX_DATA_DIR, LEX_MAX_MEMORY_ENTRIES,
  LEX_STOP_EXTRA (comma separated additions to base STOP list).
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Final, Callable, TypeVar

from .paths import (
    AVATAR_DIR as PATHS_AVATAR_DIR,
    AVATAR_URL_BASE,
    STATIC_DIR,
    STATIC_URL_PREFIX as PATHS_STATIC_URL_PREFIX,
)

T = TypeVar("T")

# ------------------------------------------------------------------
# Repository paths
# ------------------------------------------------------------------

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
STATIC_URL_PREFIX: Final[str] = PATHS_STATIC_URL_PREFIX
STATIC_ROOT: Final[Path] = STATIC_DIR
AVATAR_DIR: Final[Path] = PATHS_AVATAR_DIR
AVATAR_URL_PREFIX: Final[str] = AVATAR_URL_BASE

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_env(
    name: str, default: str, cast: Callable[[str], T], clamp: tuple[T, T] | None = None
) -> T:
    raw = os.getenv(name, default)
    try:
        val = cast(raw)
    except Exception:
        val = cast(default)
    if clamp:
        lo, hi = clamp
        if val < lo:
            val = lo
        if val > hi:
            val = hi
    return val


def _parse_stop(extra: str | None) -> list[str]:
    if not extra:
        return []
    return [s for s in (tok.strip() for tok in extra.split(",")) if s]


# ------------------------------------------------------------------
# Core model + runtime
# ------------------------------------------------------------------

MODEL_PATH: Final[str] = os.getenv(
    "LEX_MODEL_PATH",
    "/workspace/ai-lab/models/Lexifun-uncensored/llama-3-8b-lexifun-uncensored-v1-q4_k_m.gguf",
)

# Context length – never exceed model rope context; 4096–8192 typical for 8B.
CONTEXT_LENGTH: Final[int] = _get_env("LEX_CONTEXT_LENGTH", "4096", int, (512, 8192))

# GPU offload: if not explicitly set, use heuristic (roughly “all” layers if small).
_default_gpu_layers = "0"
if "LEX_GPU_LAYERS" not in os.environ:
    # Heuristic: offload aggressively for latency until 60 layers cap (llama3 8B ≈ 48 layers).
    _default_gpu_layers = "60"  # capped later if model has fewer
GPU_LAYERS: Final[int] = _get_env("LEX_GPU_LAYERS", _default_gpu_layers, int, (0, 200))

# ------------------------------------------------------------------
# Sampling parameters
# ------------------------------------------------------------------

TEMPERATURE: Final[float] = _get_env("LEX_TEMPERATURE", "0.70", float, (0.05, 2.0))
TOP_P: Final[float] = _get_env("LEX_TOP_P", "0.95", float, (0.10, 1.0))
TOP_K: Final[int] = _get_env("LEX_TOP_K", "0", int, (0, 50000))  # 0 = disabled (llama.cpp default)
MIN_P: Final[float] = _get_env("LEX_MIN_P", "0.05", float, (0.0, 0.5))  # nucleus floor; optional
REPEAT_PENALTY: Final[float] = _get_env("LEX_REPEAT_PENALTY", "1.1", float, (1.0, 2.5))
FREQUENCY_PENALTY: Final[float] = _get_env("LEX_FREQUENCY_PENALTY", "0.18", float, (0.0, 1.0))
PRESENCE_PENALTY: Final[float] = _get_env("LEX_PRESENCE_PENALTY", "0.18", float, (0.0, 1.0))

# Generation budget
MAX_TOKENS: Final[int] = _get_env("LEX_MAX_TOKENS", "768", int, (32, 1024))

# ------------------------------------------------------------------
# Stop sequences
# ------------------------------------------------------------------

# Base minimal list; expand only if model starts leaking new turns or format shifts.
_BASE_STOP: list[str] = [
    "<|end_of_text|>",  # llama.cpp GGUF EOS
    "</s>",  # HF fallback
    #    "<|user|>",         # ChatML end-of-assistant boundary
]

_extra = _parse_stop(os.getenv("LEX_STOP_EXTRA"))
STOP: Final[list[str]] = _BASE_STOP + _extra

# (Optional) Conditional inclusion of assistant marker
if os.getenv("LEX_STOP_ASSISTANT") == "1":
    STOP.append("<|assistant|>")

# ------------------------------------------------------------------
# Persona & memory state
# ------------------------------------------------------------------
LEX_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ROUTES_DIR: Final[Path] = Path(__file__).resolve().parent / "routes"
MODE_STATE_PATH = Path(__file__).parent.parent / "routes" / "lex_persona_state.json"
TRAIT_STATE_PATH = LEX_ROOT / "routes" / "lex_persona_state.json"
MEMORY_PATH: Final[str] = os.getenv(
    "LEX_MEMORY_PATH", "/workspace/ai-lab/Lex/memory/lex_memory.jsonl"
)
MAX_MEMORY_ENTRIES: Final[int] = _get_env("LEX_MAX_MEMORY_ENTRIES", "1000", int, (100, 10000))

# Static assets
# point new sessions at the rolling base avatar; per-IP publishing still overrides later
STARTER_AVATAR_PATH: Final[str] = f"{AVATAR_URL_PREFIX}/lexi_base.png"

# Branding
LEX_NAME: Final[str] = os.getenv("LEX_NAME", "Lex")
LEX_VERSION: Final[str] = "0.1.0"

# ------------------------------------------------------------------
# Derived / packed config (optional convenience dict)
# ------------------------------------------------------------------

RUNTIME_CONFIG: Final[dict[str, object]] = {
    "model_path": MODEL_PATH,
    "context_length": CONTEXT_LENGTH,
    "gpu_layers": GPU_LAYERS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "min_p": MIN_P,
    "repeat_penalty": REPEAT_PENALTY,
    "frequency_penalty": FREQUENCY_PENALTY,
    "presence_penalty": PRESENCE_PENALTY,
    "max_tokens": MAX_TOKENS,
    "stop": STOP,
}
