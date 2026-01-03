from typing import Dict, List, Tuple

"""
sd_prompt_styles.py

Defines base styles, camera settings, post effects, default poses,
negative prompts, semantic tags, and mode presets for Stable Diffusion avatar generation.

Also exposes helper fns used by the pipeline:
  - _style_realistically_unreal() -> (positive_style, style_negatives)
  - _negative_prompt() -> global_negatives
"""

# --------------------------------------------------------------------------
# Positive style prompts (tight, photoreal + polished IG look)
# --------------------------------------------------------------------------
BASE_STYLE: str = (
    "ultra-detailed portrait, photoreal skin texture, soft studio key light, gentle fill, "
    "subtle film grain, clean color grading, natural expression, crisp eyes"
)

# For photoreal portraits, 85mm-ish “look” gives flattering compression; keep text concise.
CAMERA_STYLE: str = (
    "DSLR/ mirrorless portrait, 85mm look, f/1.4 shallow depth of field, creamy bokeh, "
    "center-weighted composition, editorial framing"
)

POST_EFFECTS: str = (
    "high-end retouch, realistic pores, microcontrast on eyes and lips, smooth tonal rolloff, "
    "no plastic skin, balanced warmth"
)

DEFAULT_POSE: str = (
    "relaxed pose, shoulders turned slightly, soft smile, hands visible but unobtrusive"
)

# --------------------------------------------------------------------------
# Negative style prompt to filter out unwanted elements
# Keep this lean but effective; avoid over-blocking anatomy when doing outfits.
# --------------------------------------------------------------------------
NEGATIVE_PROMPT: str = (
    "lowres, jpeg artifacts, oversharpened, over-smoothed skin, plastic/waxy skin, "
    "cartoon, anime, illustration, cel shading, 3d render, cgi, toon, "
    "harsh shadows, blown highlights, banding, chromatic aberration, lens dirt, watermark, text, logo, signature, "
    "distorted facial features, cross-eye, lazy eye, extra limbs, extra fingers, missing fingers, fused fingers, mangled hands, "
    "severe body distortion, tiling, frame, border"
)

# --------------------------------------------------------------------------
# Semantic tags for quick trait/scene lookup
# (Used to enrich prompts based on mode/persona)
# --------------------------------------------------------------------------
SEMANTIC_TAGS: Dict[str, List[str]] = {
    # vibes / personas
    "schoolgirl": ["schoolgirl outfit", "plaid skirt", "button-up shirt"],
    "goth": ["goth girl", "black lipstick", "mesh top", "fishnets"],
    "flirty": ["flirty vibe", "teasing expression"],
    "brat": ["playful scowl", "hands on hips"],
    "innocent": ["innocent look", "wide eyes", "shy posture"],
    "dominant": ["confident stance", "commanding gaze"],
    "editorial": ["editorial fashion look", "studio backdrop", "beauty lighting"],
    "beach": ["bikini", "sun-kissed skin", "ocean bokeh", "wind in hair"],
    "evening": ["evening vibe", "warm sunset lighting", "golden hour glow"],

    # physical traits
    "blonde": ["blonde hair"],
    "brunette": ["brown hair"],
    "redhead": ["red hair"],
    "black hair": ["black hair"],
    "green eyes": ["green eyes"],
    "blue eyes": ["blue eyes"],
    "hazel eyes": ["hazel eyes"],
    "tattoo": ["visible tattoo"],

    # playful add-ons
    "cat ears": ["cat ears", "nekomimi"],
    "tail": ["tail"],

    # expressions
    "smile": ["smiling"],
    "smirk": ["smirking"],
    "pout": ["pouting lips"],
}

# --------------------------------------------------------------------------
# Mode presets mapping to semantic tags
# (Feel free to expand per persona/mode)
# --------------------------------------------------------------------------
MODE_PRESETS: Dict[str, List[str]] = {
    "kitty_girl": ["cat ears", "tail", "collar"],
    "brat": ["brat", "thigh-high socks"],
    "babygirl": ["innocent", "cute outfit", "youthful appearance"],
    "domme": ["dominant", "latex outfit", "studio backdrop"],
    "sundown": ["evening", "relaxed pose"],
    "beach_day": ["beach", "playful"],
    "studio_portrait": ["editorial", "softbox lighting", "neutral gray backdrop"],
}

# --------------------------------------------------------------------------
# Helper functions consumed by the pipeline
# --------------------------------------------------------------------------
def _style_realistically_unreal() -> Tuple[str, str]:
    """
    Returns (positive_style_scaffold, style_specific_negatives)
    Positive is intentionally compact: SDXL respects concise, high-signal phrases.
    """
    positive = ", ".join([
        BASE_STYLE,
        CAMERA_STYLE,
        POST_EFFECTS,
        DEFAULT_POSE,
    ])

    # Style-level negatives tuned for “realistically unreal” (glam but real)
    style_negatives = (
        "muddy skin, heavy makeup artifacts, body liquify artifacts, oversoft face, "
        "excessive skin smoothing, plastic reflections"
    )

    return positive, style_negatives


def _negative_prompt() -> str:
    """
    Global negatives layered under style negatives (pipeline concatenates them).
    Keep strong but not so aggressive it blocks outfit edits.
    """
    return NEGATIVE_PROMPT


__all__ = [
    # constants
    "BASE_STYLE",
    "CAMERA_STYLE",
    "POST_EFFECTS",
    "DEFAULT_POSE",
    "NEGATIVE_PROMPT",
    "SEMANTIC_TAGS",
    "MODE_PRESETS",
    # helpers used by sd_pipeline
    "_style_realistically_unreal",
    "_negative_prompt",
]
