# lex_prompt_styles.py

BASE_STYLE = (
    "ultra-detailed, natural lighting, photo of a beautiful woman, soft lighting"
)

CAMERA_STYLE = (
    "DSLR photo, 35mm lens, shallow depth of field, cinematic framing, soft bokeh"
)

POST_EFFECTS = (
    "warm, high clarity, balanced tones, studio-style"
)

DEFAULT_POSE = "relaxed pose, arms and hands visible, soft smile"


NEGATIVE_PROMPT = (
    "anime, cartoon, cel shading, illustration, flat shading, plastic skin, overly smooth skin, "
    "3d render, fake background, lowres, blurry, stylized face, shiny clothing, latex, "
    "sketch, drawing, monochrome, sepia, deformed, ugly, blurry, distorted, cropped, watermark, text, painting, "
    "extra limbs, tracking dots, white dots, artifacts, tracking markers, glowing spots, sci-fi elements, blur, "
    "soft focus, hazy, dreamy style, soft edges"
)

SEMANTIC_TAGS = {
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
    "model": ["model-like posture"]
}

MODE_PRESETS = {
    "kitty_girl": ["cat ears", "tail", "collar"],
    "brat": ["playful expression", "thigh-high socks"],
    "babygirl": ["big eyes", "cute outfit", "youthful appearance"],
    "domme": ["confident stance", "latex outfit", "whip"],
    "sundown": ["warm sunset lighting", "relaxed pose", "evening vibe"],
    "beach_day": ["bikini", "sun-kissed skin", "ocean bokeh", "wind in hair"],
    "studio_portrait": ["neutral gray backdrop", "softbox lighting", "editorial fashion shoot"]
}

