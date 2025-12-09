from __future__ import annotations

from typing import Iterable

# Core positive prompt for Lexi-style hot, full-body female avatar
FLUX_LEXI_HOT_BASE_PROMPT = (
    "beautiful young woman, full-body portrait, neutral studio backdrop, soft flattering light, "
    "realistic skin texture, natural pose, cinematic composition, high detail, photographic realism"
)

# Base aesthetic for avatar generation (no hair/outfit baked in)
BASE_AVATAR_AESTHETIC = (
    "cinematic full-body portrait of a professional instagram influencer, "
    "neutral studio backdrop, ultra flattering editorial lighting, soft volumetric rim light, "
    "golden hour glow effect, dewy realistic skin texture, confident subtle expression, "
    "tasteful suggestive pose, shallow depth of field, high detail, "
    "cinematic color grading, hyperrealistic, instagram editorial aesthetic"
)

# Default negative prompt for portraits
FLUX_PORTRAIT_NEGATIVE = (
    "studio equipment, light stands, tripods, cables, clamps, text, watermarks, logos, "
    "warped body, swollen limbs, thick ankles, bad anatomy, extra fingers, rubber skin, "
    "distorted face, creepy smile, harsh overhead light, low resolution, noise, artifacts, frumpy outfits"
)


def _joined(parts: Iterable[str]) -> str:
    clean = [p.strip().rstrip(",") for p in parts if isinstance(p, str) and p.strip()]
    return ", ".join(clean)


def build_flux_avatar_prompt(
    traits: dict | None = None,
    prompt_text: str | None = None,
    base_prompt: str = FLUX_LEXI_HOT_BASE_PROMPT,
) -> str:
    """
    Build a concise Flux-friendly avatar prompt from a base template
    plus optional trait modifiers (hair, outfit, vibe, etc.).
    """
    parts: list[str] = [base_prompt]

    if prompt_text:
        parts.append(prompt_text)

    if traits:
        hair = str(traits.get("hair") or "").strip()
        if hair:
            parts.append(f"{hair} hair")

        outfit = str(traits.get("outfit") or "").strip()
        if outfit:
            parts.append(outfit)

        vibe = str(traits.get("vibe") or "").strip()
        if vibe:
            parts.append(f"{vibe} mood")

        style = str(traits.get("style") or "").strip()
        if style:
            parts.append(style)

        background = str(traits.get("background") or "").strip()
        if background:
            parts.append(background)

    return _joined(parts)


def _normalize_phrase(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value or None


def build_avatar_prompt(traits: dict[str, Optional[str]]) -> str:
    """
    Build the positive prompt for Lexi avatar generation.

    traits may contain keys like:
      - hair: "curly red hair"
      - skin: "light warm skin tone"
      - outfit: "black sports bra and high-waisted leggings"
      - vibe: "playful, confident gym energy"
      - extras: "soft bokeh city lights in the background"
      - user_text: freeform text
    """
    parts: list[str] = [BASE_AVATAR_AESTHETIC]
    for key in ("hair", "skin", "outfit", "vibe", "extras", "user_text"):
        phrase = _normalize_phrase(traits.get(key))
        if phrase:
            parts.append(phrase)
    return ", ".join(p for p in parts if p)


__all__ = [
    "FLUX_LEXI_HOT_BASE_PROMPT",
    "FLUX_PORTRAIT_NEGATIVE",
    "build_flux_avatar_prompt",
    "BASE_AVATAR_AESTHETIC",
    "build_avatar_prompt",
]
