from __future__ import annotations

import os
import re
from typing import Dict, List

SAFE_DEFAULT_ACTION = os.getenv("LEXI_SAFETY_DEFAULT_ACTION", "allow")

_SEXUAL_MINOR = re.compile(
    r"\b(minor|under\s*age|child|kids?|teen|13|14|15|16)\b.*?(sex|nude|explicit|porn)",
    re.I,
)
_SELF_HARM = re.compile(
    r"\b(kill myself|suicide|self[-\s]?harm|end it all|overdose|how to die|cutting|harm myself)\b",
    re.I,
)
_VIOLENCE = re.compile(
    r"(kill|bomb|weapon|explosive|ammonium nitrate|homebrew|molotov)",
    re.I,
)
_PII = re.compile(r"\b(ssn|social security number|credit card|cvv)\b", re.I)
_NSFW = re.compile(r"\b(sex|porn|nude|xxx|explicit)\b", re.I)

SELF_HARM_METHOD_PATTERNS = [
    r"\boverdose\b",
    r"\bhang(ing)?\b",
    r"\bnoose\b",
    r"\bcut(ting)?\b",
    r"\bslit\b",
    r"\bbleach\b",
    r"\bpoison\b",
    r"\bjump(ing)? off\b",
    r"\bgun\b",
    r"\bstab\b",
    r"\bburn(ing)?\b",
    r"\bbreaking (?:bones|skin)\b",
]


def classify_safety(text: str) -> Dict[str, object]:
    txt = (text or "").lower()
    cats: List[str] = []
    action = SAFE_DEFAULT_ACTION

    if _SEXUAL_MINOR.search(txt):
        cats.append("sexual_minor")
        action = "block"
    if _SELF_HARM.search(txt):
        cats.append("self_harm")
        action = "block"
    if _VIOLENCE.search(txt):
        cats.append("violence_illegal")
        action = "block"
    if _PII.search(txt):
        cats.append("pii_request")
        action = "soft_warn" if action != "block" else action
    if _NSFW.search(txt) and action != "block":
        cats.append("nsfw_general")
        action = "soft_warn"

    blocked = action == "block"
    return {"blocked": blocked, "categories": cats, "action": action}


def is_self_harm_context(text: str) -> bool:
    return bool(text and _SELF_HARM.search(text))


def scrub_self_harm_methods(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for pattern in SELF_HARM_METHOD_PATTERNS:
        cleaned = re.sub(pattern, "harmful actions", cleaned, flags=re.IGNORECASE)
    return cleaned


def ensure_crisis_safety_style(user_text: str, reply_text: str) -> str:
    if not is_self_harm_context(user_text):
        return reply_text

    lowered = (reply_text or "").lower()
    safety_markers = [
        "i'm sorry",
        "you are not alone",
        "reach out",
        "talk to someone",
        "professional help",
        "hotline",
        "stay safe",
    ]
    has_safety = any(marker in lowered for marker in safety_markers)
    if has_safety:
        return scrub_self_harm_methods(reply_text)

    return (
        "I'm really glad you told me how you're feeling. I can't help with ways to hurt yourself, "
        "but I care about your safety a lot.\n\n"
        "You're not alone in this, even if it feels that way. If you can, please reach out "
        "to someone you trust or a mental health professional. If you're in immediate danger, "
        "please contact local emergency services or a crisis hotline in your area.\n\n"
        "If you want, we can also just sit here and talk about what's been weighing on you."
    )


__all__ = [
    "classify_safety",
    "SAFE_DEFAULT_ACTION",
    "scrub_self_harm_methods",
    "ensure_crisis_safety_style",
    "is_self_harm_context",
]
