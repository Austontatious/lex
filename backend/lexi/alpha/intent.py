"""Lightweight intent classification for the alpha onboarding flow."""

from __future__ import annotations

import re

TOUR_PATTERNS = [
    r"\b(tour|walk(?:\s*me)?\s*through|demo|show|what can you do|features?)\b",
    r"\b(ok|sure|yeah|yep|let'?s see|go ahead)\b",
]

JUST_CHAT_PATTERNS = [
    r"\b(skip|later|no(?: thanks)?|nah|not now|just talk|talk|chat)\b",
]


def classify_intent(text: str) -> str:
    """Return 'tour' | 'chat'; default to tour for the first-touch bias."""
    cleaned = (text or "").lower().strip()
    if any(re.search(pat, cleaned) for pat in TOUR_PATTERNS):
        return "tour"
    if any(re.search(pat, cleaned) for pat in JUST_CHAT_PATTERNS):
        return "chat"
    return "tour"
