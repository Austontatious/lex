"""Runtime feature flags and configuration helpers for Lexi alpha flows."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AlphaSettings:
    """Lightweight view over environment driven runtime toggles."""

    alpha_strict: bool = _env_flag("ALPHA_STRICT", default=False)

    @staticmethod
    def logs_base_dir() -> str:
        # keep directories explicit — mount point defined in requirements
        return os.getenv("LEX_LOG_DIR", "/mnt/data/lexi/logs")

