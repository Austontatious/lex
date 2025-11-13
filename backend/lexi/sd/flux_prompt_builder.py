from __future__ import annotations

from typing import Iterable, Tuple

BASE_POS_A = (
    "full-body portrait in the Lexiverse style of Lexi — a beautiful young woman indoors; "
    "camera at waist height, vertical 4:5 composition; subject centered with mild pose asymmetry; "
    "natural hand engagement, confident and relaxed expression; cinematic realism; "
    "bright natural light, balanced color tones, vivid depth, and soft highlights."
)

BASE_POS_B = (
    "provocative editorial glamour, bold and confident sensuality; "
    "studio-lit neon wash with cinematic lighting balance, vibrant LED exposure, and soft diffusion; "
    "glossy highlights, rich tonal contrast, illuminated contours, and glowing skin; "
    "soft golden-hour lighting with warm key and subtle fill; balanced exposure and realistic HDR tone curve; "
    "balanced warm daylight, soft sun-kissed luminance, gentle specular highlights, natural skin tones; "
    "indirect soft key with gentle rim light and subtle depth shadows; "
    "minimal casual outfit with neutral palette and clean textures; timeless aesthetic; "
    "daylight-balanced color temperature, cinematic neutral tone mapping; "
    "dual-tone magenta–blue rimlight for cinematic color separation."
)

BASE_NEG_A = (
    "flat lighting, harsh shadows, dim exposure, dull or washed-out tones, deformed proportions; "
    "low contrast, underexposed or uneven illumination; "
    "oversaturated orange or yellow cast, tungsten hue, artificial bronze skin, high-contrast shadows; "
    "visible lamps, spotlights, lighting gear, lens flare, strong backlight; "
    "no formal suits, business attire, dresses, jewelry, or overtly revealing outfits; "
    "avoid monochrome or flat tone, over-orange coloration, or yellow cast."
)

BASE_NEG_B = (
    "flat lighting, harsh shadows, dim exposure, low contrast, washed-out tones; "
    "oversaturated warm hues or tungsten color cast; visible lighting equipment, glare, or strong backlight; "
    "unnatural bronze skin tone or exaggerated contrast; formalwear, jewelry, or explicit attire; "
    "monochrome or over-orange color grading."
)


def _joined(parts: Iterable[str]) -> str:
    clean = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return " ".join(clean).strip()


def build_prompts(
    *,
    traits: list[str] | None = None,
    style_delta: str = "",
    outfit: str = "",
    mood: str = "",
    env: str = "",
) -> Tuple[str, str, str, str]:
    """
    Assemble deterministic base prompts with optional additive deltas.
    """
    tail_parts: list[str] = []
    if style_delta:
        tail_parts.append(f"style: {style_delta}")
    if outfit:
        tail_parts.append(f"outfit: {outfit}")
    if mood:
        tail_parts.append(f"mood: {mood}")
    if env:
        tail_parts.append(f"environment: {env}")
    if traits:
        tail_parts.extend(traits)

    tail = _joined(tail_parts)
    positive_clip = BASE_POS_A
    positive_t5 = BASE_POS_B
    if tail:
        positive_clip = f"{BASE_POS_A} {tail}".strip()
        positive_t5 = f"{BASE_POS_B} {tail}".strip()

    return positive_clip, positive_t5, BASE_NEG_A, BASE_NEG_B


__all__ = [
    "BASE_POS_A",
    "BASE_POS_B",
    "BASE_NEG_A",
    "BASE_NEG_B",
    "build_prompts",
]
