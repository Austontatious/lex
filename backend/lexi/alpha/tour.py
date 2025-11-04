"""Server-side helpers for the alpha tour scripting and stubs."""

from __future__ import annotations

from typing import Dict, List

from ..config.config import STARTER_AVATAR_PATH

from .settings import AlphaSettings


PLACEHOLDER_PREVIEW_URL = STARTER_AVATAR_PATH


def tour_script() -> List[Dict[str, str]]:
    return [
        {
            "slug": "intro",
            "prompt": "describe a vibe, i’ll sketch a look.",
            "narration": (
                "awesome. we’ll do a quick spin: avatar vibes → ‘now’ topic → "
                "emotions → memory. ready?"
            ),
        },
        {
            "slug": "avatar_preview",
            "prompt": "give me a vibe (e.g., 'cozy cyberpunk librarian'); i’ll sketch a tiny preview.",
            "narration": (
                "i’ll run a tiny, low-stakes preview so you see how avatar vibes evolve. "
                "full renders take longer, so we keep it light here."
            ),
        },
        {
            "slug": "now_topic",
            "prompt": "throw me a topic and i’ll thread it through our chat.",
            "narration": "this sticks for the next few turns — i keep it in my short-term 'now' awareness.",
        },
        {
            "slug": "emotion_axes",
            "prompt": "watch the emotion axes pulse when i react — warmth, energy, curiosity, confidence, playfulness.",
            "narration": "i nudge these as we talk so you can tell how lexi is vibing.",
        },
        {
            "slug": "memory_explainer",
            "prompt": "tell me one thing to remember *just for this session*.",
            "narration": (
                "i’ll remember it until you log out. after that, only an anonymized diary sticks around for the dev team."
            ),
        },
        {
            "slug": "wrap",
            "prompt": "cool to keep chatting?",
            "narration": "ready when you are — want to keep riffing or bounce back to freestyle chat?",
        },
    ]


def preview_placeholder_url(settings: AlphaSettings | None = None) -> str:
    """Return a stub preview URL respecting ALPHA_STRICT defaults."""
    _ = settings or AlphaSettings()
    return PLACEHOLDER_PREVIEW_URL
