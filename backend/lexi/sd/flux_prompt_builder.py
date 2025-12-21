from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, TypedDict, List


# --------- Typed trait keys ----------
Hair = Literal["brunette", "blonde", "redhead", "black"]
HairStyle = Literal["straight", "wavy", "curly", "updo"]
SkinTone = Literal["fair", "light_medium", "olive", "tan", "deep"]
EyeColor = Literal["brown", "hazel", "green", "blue"]
Outfit = Literal["lbd", "lounge", "casual", "business", "sporty"]
Vibe = Literal["soft", "playful", "elegant", "confident", "sultry"]

# Keep for backward compatibility, but we no longer “style-direct” with it.
LexiverseStyle = Literal["promo"]


class AvatarTraits(TypedDict, total=False):
    hair: Hair
    hair_style: HairStyle
    skin_tone: SkinTone
    eyes: EyeColor
    outfit: Outfit
    vibe: Vibe


class StyleFlags(TypedDict, total=False):
    # Back-compat fields (safe to ignore)
    lexiverse_style: LexiverseStyle

    # Canonical flags
    lexiverse_enabled: bool

    # NEW: treat Lexiverse as a tag/token (often your LoRA trigger token)
    # If your pipeline already injects the LoRA, leave this None.
    lexiverse_tag: str

    # NEW: if you ever want to push a *photographic* polish without “poster art”
    quality_tag: str


class PoseMeta(TypedDict, total=False):
    pose_bucket: Optional[str]
    camera_bucket: Optional[str]
    lighting_bucket: Optional[str]


# --------- Descriptor maps ----------
HAIR_MAP: Dict[Hair, str] = {
    "brunette": "rich dark brunette",
    "blonde": "soft golden blonde",
    "redhead": "warm copper red hair",
    "black": "glossy jet black hair",
}

HAIR_STYLE_MAP: Dict[HairStyle, str] = {
    "straight": "sleek straight",
    "wavy": "loose beachy waves",
    "curly": "big soft curls",
    "updo": "romantic messy updo",
}

SKIN_TONE_MAP: Dict[SkinTone, str] = {
    "fair": "fair porcelain skin",
    "light_medium": "soft light peachy skin",
    "olive": "warm olive-toned skin",
    "tan": "sun-kissed tan skin",
    "deep": "rich deep brown skin",
}

EYE_MAP: Dict[EyeColor, str] = {
    "brown": "warm brown eyes",
    "hazel": "gold-flecked hazel eyes",
    "green": "vibrant green eyes",
    "blue": "bright blue eyes",
}

OUTFIT_MAP: Dict[Outfit, str] = {
    "lbd": "sleek curve-hugging little black dress, subtle glamorous styling",
    "lounge": "soft fitted loungewear, cozy knit top and snug lounge pants",
    "casual": "fitted tee and high-waisted skinny jeans, elevated casual look",
    "business": "tailored blazer over a fitted top and slim trousers, polished and sharp",
    "sporty": "fitted sports bra and high-waisted leggings, athletic gym-ready outfit",
}

VIBE_MAP: Dict[Vibe, str] = {
    "soft": "soft relaxed body language, warm gentle smile, inviting eyes, cozy approachable energy",
    "playful": "sparkling eyes, playful smirk, slightly tilted head, teasing fun energy",
    "elegant": "poised posture, graceful hands, serene expression, refined elegant presence",
    "confident": "strong stance, shoulders back, direct eye contact, subtle knowing smile, confident energy",
    "sultry": "slow smoldering gaze, lips slightly parted, subtly arched posture, relaxed sensual energy",
}


# --------- Base prompts ----------
# IMPORTANT: do not hard-ban props or backgrounds here; user intent (scarf/background/etc) should win.
BASE_LEXIVERSE_STYLE = (
    "a single beautiful young woman as the main subject, portrait photo, "
    "studio-quality lighting, high detail, realistic skin texture, "
    "natural lens rendering, no other people, healthy feminine body proportions"
)

BASE_NEUTRAL_STYLE = (
    "a single young woman as the main subject, portrait photo, "
    "studio-quality lighting, high detail, realistic skin texture, "
    "natural lens rendering, no other people, wide framing, subject occupies no more than 70% of vertical frame, healthy feminine body proportions"
)

BASE_AVATAR_AESTHETIC = BASE_LEXIVERSE_STYLE


# --------- Negative packs ----------
# Keep this mostly “quality + artifact + forbidden overlays”
NEG_BASE = (
    "no person, empty scene, props only, "
    "text, watermark, logo, caption, signature, "
    "lowres, blurry, out of focus, motion blur, jpeg artifacts, "
    "bad anatomy, extra limbs, extra fingers, missing fingers, fused fingers, "
    "deformed hands, deformed face, asymmetrical face, "
    "overexposed, underexposed, harsh shadow, fog, haze"
)

# This is the key to stopping poster/comic drift when your LoRA pushes stylization.
NEG_STYLE_GUARD = (
    "illustration, comic, cartoon, anime, manga, cel shading, lineart, vector art, "
    "poster, graphic design, typography, title text, letters, "
    "3d render, unreal engine render, plastic skin, doll, mannequin"
)
NEG_FRAMING = "mid-shot, waist-up, thigh-up, cropped legs, cropped feet, half-body framing"
NEG_HALO = "halo, outline, glow, rimlight, backlight, cutout, sticker, overprocessed HDR, bloom"
NEG_BODY = "overdefined muscles, bodybuilder physique"

# Back-compat export for callers expecting a single string.
DEFAULT_NEGATIVE_PROMPT = f"{NEG_BASE}, {NEG_STYLE_GUARD}, {NEG_FRAMING}, {NEG_HALO}, {NEG_BODY}"


# --------- Helpers ----------
def _joined(parts: Iterable[str]) -> str:
    clean: List[str] = []
    seen = set()
    for p in parts:
        if not isinstance(p, str):
            continue
        s = p.strip().rstrip(",")
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        clean.append(s)
        seen.add(k)
    return ", ".join(clean)


def _hair_phrase(traits: AvatarTraits) -> Optional[str]:
    hair = traits.get("hair")
    style = traits.get("hair_style")
    parts: List[str] = []
    if hair and hair in HAIR_MAP:
        # HAIR_MAP already includes “hair” for redhead/black; keep it consistent
        parts.append(HAIR_MAP[hair])
    if style and style in HAIR_STYLE_MAP:
        parts.append(HAIR_STYLE_MAP[style])
    if not parts:
        return None

    # Avoid “hair hair”
    phrase = " ".join(parts).replace("hair hair", "hair")
    if "hair" not in phrase:
        phrase += " hair"
    return phrase


def _pose_hint_from_meta(pose_meta: Optional[PoseMeta]) -> Optional[str]:
    # Your meta uses POSE_# buckets; we don’t know mapping here.
    # If you later normalize meta to “playful/seductive/cozy/confident”, it will work.
    if not pose_meta:
        return None
    bucket = (pose_meta.get("pose_bucket") or "").strip().lower()
    if bucket == "playful":
        return "playful pose, relaxed dynamic body language"
    if bucket == "seductive":
        return "confident alluring pose, subtle flirtatious body language"
    if bucket == "cozy":
        return "cozy relaxed pose, comfortable intimate body language"
    if bucket == "confident":
        return "confident pose, shoulders back, assertive stance"
    return None


def _extend_negative(neg: List[str], *items: str) -> None:
    for it in items:
        s = (it or "").strip()
        if s:
            neg.append(s)


def _dynamic_trait_negatives(traits: AvatarTraits) -> List[str]:
    """
    Dynamic negatives: suppress *categorical* conflicts.
    Keep these specific + small; too much causes dead/overconstrained faces.
    """
    neg: List[str] = []

    hair = traits.get("hair")
    if hair == "redhead":
        _extend_negative(neg, "blonde hair", "black hair", "brown hair")
    elif hair == "blonde":
        _extend_negative(neg, "red hair", "ginger hair", "black hair", "brown hair")
    elif hair == "brunette":
        _extend_negative(neg, "blonde hair", "red hair", "ginger hair", "black hair")
    elif hair == "black":
        _extend_negative(neg, "blonde hair", "red hair", "ginger hair", "brown hair")

    hs = traits.get("hair_style")
    if hs == "curly":
        _extend_negative(neg, "straight hair")
    elif hs == "straight":
        _extend_negative(neg, "curly hair", "ringlets")
    elif hs == "wavy":
        _extend_negative(neg, "pin-straight hair", "tight curls")
    elif hs == "updo":
        _extend_negative(neg, "hair down")

    eyes = traits.get("eyes")
    if eyes == "blue":
        _extend_negative(neg, "brown eyes", "green eyes", "hazel eyes")
    elif eyes == "green":
        _extend_negative(neg, "brown eyes", "blue eyes", "hazel eyes")
    elif eyes == "hazel":
        _extend_negative(neg, "brown eyes", "blue eyes", "green eyes")
    elif eyes == "brown":
        _extend_negative(neg, "blue eyes", "green eyes", "hazel eyes")

    skin = traits.get("skin_tone")
    # Light-touch only
    if skin == "olive":
        _extend_negative(neg, "very pale skin", "very dark skin")
    elif skin == "fair":
        _extend_negative(neg, "very dark skin", "deep brown skin")
    elif skin == "deep":
        _extend_negative(neg, "very pale skin", "porcelain skin")

    outfit = traits.get("outfit")
    if outfit == "business":
        _extend_negative(
            neg,
            "lingerie", "bikini", "swimsuit",
            "sports bra", "gym outfit",
            "pajamas", "loungewear",
            "hoodie", "sweatpants",
        )
    elif outfit == "sporty":
        _extend_negative(neg, "blazer", "business suit", "tailored suit")
    elif outfit == "lounge":
        _extend_negative(neg, "blazer", "business suit", "corporate suit")
    elif outfit == "lbd":
        _extend_negative(neg, "blazer", "business suit", "sports bra")

    vibe = traits.get("vibe")
    if vibe == "playful":
        _extend_negative(neg, "angry expression", "sad expression", "deadpan expression")
    elif vibe == "elegant":
        _extend_negative(neg, "goofy expression", "silly face")
    elif vibe == "confident":
        _extend_negative(neg, "timid expression", "shy expression")

    return neg


def _dynamic_user_intent_negatives(user_text: Optional[str]) -> List[str]:
    t = (user_text or "").strip().lower()
    if not t:
        return []

    neg: List[str] = []

    # If user explicitly wants clean studio, suppress busy environments.
    if "neutral" in t and ("studio" in t or "backdrop" in t):
        _extend_negative(
            neg,
            "busy background", "street scene", "crowd", "nature", "plants",
            "office clutter", "room clutter", "messy room",
        )

    if "selfie" in t:
        _extend_negative(neg, "tripod", "light stand", "camera rig")

    return neg


# --------- Canonical builders ----------
def build_flux_avatar_prompt_bundle(
    traits: Optional[AvatarTraits] = None,
    style_flags: Optional[StyleFlags] = None,
    pose_meta: Optional[PoseMeta] = None,
    user_text: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return {"positive": str, "negative": str} built from canonical descriptors,
    with dynamic negatives to improve categorical adherence.
    """
    traits = traits or {}
    style_flags = style_flags or {}

    lexiverse_enabled = style_flags.get("lexiverse_enabled", True)

    # If your LoRA is already applied in the pipeline, set lexiverse_tag to "" or omit it.
    lexiverse_tag = (style_flags.get("lexiverse_tag") or "").strip() if lexiverse_enabled else ""
    quality_tag = (style_flags.get("quality_tag") or "").strip()

    base_style = BASE_LEXIVERSE_STYLE if lexiverse_enabled else BASE_NEUTRAL_STYLE

    extra = (user_text or "").strip() or None

    parts = [
        base_style,
        # User intent early (background color / scarf / accessories / “teal background” etc)
        extra,
        # Tags last-ish so they don’t become “the prompt”
        _hair_phrase(traits),
        EYE_MAP.get(traits.get("eyes", "")),
        SKIN_TONE_MAP.get(traits.get("skin_tone", "")),
        OUTFIT_MAP.get(traits.get("outfit", "")),
        VIBE_MAP.get(traits.get("vibe", "")),
        _pose_hint_from_meta(pose_meta),
        quality_tag,
        lexiverse_tag,
    ]

    positive = _joined(parts)

    neg_parts: List[str] = [NEG_BASE, NEG_STYLE_GUARD, NEG_FRAMING, NEG_HALO, NEG_BODY]
    neg_parts.extend(_dynamic_trait_negatives(traits))
    neg_parts.extend(_dynamic_user_intent_negatives(extra))

    negative = _joined(neg_parts)

    return {"positive": positive, "negative": negative}


# Backward-compat: keep original helpers as thin wrappers
def build_flux_avatar_prompt(
    traits: dict | None = None,
    prompt_text: str | None = None,
    base_prompt: str = BASE_LEXIVERSE_STYLE,
) -> str:
    bundle = build_flux_avatar_prompt_bundle(
        traits=traits or {},
        style_flags={"lexiverse_enabled": True},
        pose_meta=None,
        user_text=prompt_text,
    )
    return bundle["positive"]


def build_avatar_prompt(traits: dict[str, Optional[str]], include_base: bool = True) -> str:
    bundle = build_flux_avatar_prompt_bundle(
        traits=traits,
        style_flags={"lexiverse_enabled": include_base},
    )
    return bundle["positive"]


__all__ = [
    "Hair",
    "HairStyle",
    "SkinTone",
    "EyeColor",
    "Outfit",
    "Vibe",
    "LexiverseStyle",
    "AvatarTraits",
    "StyleFlags",
    "PoseMeta",
    "HAIR_MAP",
    "HAIR_STYLE_MAP",
    "SKIN_TONE_MAP",
    "EYE_MAP",
    "OUTFIT_MAP",
    "VIBE_MAP",
    "BASE_LEXIVERSE_STYLE",
    "BASE_NEUTRAL_STYLE",
    "DEFAULT_NEGATIVE_PROMPT",
    "build_flux_avatar_prompt_bundle",
    "build_flux_avatar_prompt",
    "build_avatar_prompt",
]
