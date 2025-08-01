from typing import Dict, List

"""
sd_prompt_styles.py

Defines base styles, camera settings, post effects, default poses,
negative prompts, semantic tags, and mode presets for Stable Diffusion avatar generation.
"""

# --------------------------------------------------------------------------
# Positive style prompts
# --------------------------------------------------------------------------
BASE_STYLE: str = (
    "ultra-detailed, natural lighting, photo of a beautiful woman, soft lighting"
)

CAMERA_STYLE: str = (
    "DSLR photo, 35mm lens, shallow depth of field, cinematic framing, soft bokeh"
)

POST_EFFECTS: str = (
    "warm, high clarity, balanced tones, studio-style"
)

DEFAULT_POSE: str = (
    "relaxed pose, arms and hands visible, soft smile"
)

# --------------------------------------------------------------------------
# Negative style prompt to filter out unwanted elements
# --------------------------------------------------------------------------
NEGATIVE_PROMPT: str = (
    "anime, cartoon, cel shading, illustration, flat shading, plastic skin, overly smooth skin, "
    "3d render, fake background, lowres, blurry, stylized face, shiny clothing, latex, "
    "sketch, drawing, monochrome, sepia, deformed, ugly, blurry, distorted, cropped, watermark, text, painting, "
    "extra limbs, tracking dots, white dots, artifacts, tracking markers, glowing spots, sci-fi elements, blur, "
    "soft focus, hazy, dreamy style, soft edges"
)

# --------------------------------------------------------------------------
# Semantic tags for quick trait lookup
# --------------------------------------------------------------------------
SEMANTIC_TAGS: Dict[str, List[str]] = {
    "schoolgirl": ["schoolgirl outfit", "plaid skirt", "button-up shirt"],
    "goth": ["goth girl", "black lipstick", "fishnets"],
    "cyberpunk": ["cyberpunk style", "neon lighting", "tech accessories"],
    "flirty": ["flirty", "biting lip", "teasing expression"],
    "brat": ["bratty attitude", "playful scowl", "hands on hips"],
    "innocent": ["innocent look", "wide eyes", "shy posture"],
    "dominant": ["dominant", "confident stance", "commanding gaze"],
    "blonde": ["blonde hair"],
    "brunette": ["brown hair"],
    "green eyes": ["green eyes"],
    "blue eyes": ["blue eyes"],
    "tattoo": ["visible tattoo"],
    "cat ears": ["cat ears", "nekomimi"],
    "tail": ["tail"],
    "smile": ["smiling"],
    "smirk": ["smirking"],
    "frown": ["frowning"],
    "pout": ["pouting lips"],
    "sassy": ["sassy pose"],
    "sexy": ["sexy pose"],
    "playful": ["playful pose"],
    "model": ["model-like posture"],
}

# --------------------------------------------------------------------------
# Mode presets mapping to semantic tags
# --------------------------------------------------------------------------
MODE_PRESETS: Dict[str, List[str]] = {
    "kitty_girl": ["cat ears", "tail", "collar"],
    "brat": ["playful expression", "thigh-high socks"],
    "babygirl": ["big eyes", "cute outfit", "youthful appearance"],
    "domme": ["confident stance", "latex outfit", "whip"],
    "sundown": ["warm sunset lighting", "relaxed pose", "evening vibe"],
    "beach_day": ["bikini", "sun-kissed skin", "ocean bokeh", "wind in hair"],
    "studio_portrait": ["neutral gray backdrop", "softbox lighting", "editorial fashion shoot"],
}

__all__ = [
    "BASE_STYLE",
    "CAMERA_STYLE",
    "POST_EFFECTS",
    "DEFAULT_POSE",
    "NEGATIVE_PROMPT",
    "SEMANTIC_TAGS",
    "MODE_PRESETS",
]

