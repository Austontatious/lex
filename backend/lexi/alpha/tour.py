"""Server-side helpers for the alpha tour scripting and stubs."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config.config import STARTER_AVATAR_PATH

from .settings import AlphaSettings
from .tour_content import TOUR_STEPS, onboarding_copy


PLACEHOLDER_PREVIEW_URL = STARTER_AVATAR_PATH


def tour_script() -> List[Dict[str, str]]:
    """Expose a copy of the canonical tour steps."""
    return [dict(step) for step in TOUR_STEPS]


def onboarding_script() -> Dict[str, Any]:
    """Return the structured onboarding copy for AlphaWelcome."""
    return onboarding_copy()


def preview_placeholder_url(settings: AlphaSettings | None = None) -> str:
    """Return a stub preview URL respecting ALPHA_STRICT defaults."""
    _ = settings or AlphaSettings()
    return PLACEHOLDER_PREVIEW_URL
