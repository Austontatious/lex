import re
from typing import Dict, List, Any
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# ---------------------------------------------------------------------------
# Stopwords to reduce visual noise in long prompts
# ---------------------------------------------------------------------------
VISUAL_STOPWORDS = {
    "dslr", "photorealistic", "high detail", "cinematic", "35mm", "bokeh", "shallow depth of field",
    "natural lighting", "ultra-realistic", "photo", "realistic skin", "film grain", "editorial",
    "balanced tones", "vibrant color", "soft lighting", "studio", "full-length", "clarity",
    "focus", "magazine", "lighting", "background", "shot", "image", "pose", "portrait", "composition"
}

# ---------------------------------------------------------------------------
# Negative & positive quality anchors
# ---------------------------------------------------------------------------
NEGATIVE_MASTER: str = (
    "(low quality, worst quality:1.2), lowres, bad anatomy, bad hands, extra limbs, "
    "missing fingers, watermark, signature, censored, nsfw censor, auto-toon, plastic skin, 3d render"
)
QUALITY_TAGS: str = "best quality, masterpiece, 8k"
# Brand-specific style tokens for "realistic unreality"
BRAND_STYLE_TAGS: str = (
    "octane render, cinematic tone mapping, 8K VRay, volumetric rim light, filmic color grade"
)

# ---------------------------------------------------------------------------
# Token processing
# ---------------------------------------------------------------------------
def tokenize_prompt(prompt: str) -> List[str]:
    return [t.strip() for t in prompt.split(",") if t.strip()]
    
def truncate_prompt(text: str, max_tokens=77) -> str:
    tokens = tokenizer.tokenize(text)
    truncated = tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return truncated

def deduplicate(tokens: List[str], aggressive: bool = True) -> List[str]:
    seen = set()
    cleaned = []
    for t in tokens:
        key = re.sub(r"[^\w]", "", t.lower())
        key = re.sub(r"s$", "", key)
        if aggressive:
            key = re.sub(r"\b(" + "|".join(VISUAL_STOPWORDS) + r")\b", "", key)
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    return cleaned

def sift_prompt(prompt: str, token_limit: int = 75, aggressive: bool = True, log_trim: bool = False) -> str:
    tokens = tokenize_prompt(prompt)
    tokens = deduplicate(tokens, aggressive=aggressive)
    if len(tokens) > token_limit:
        trimmed = tokens[:token_limit]
        if log_trim:
            print(f"[PromptSifter] Trimmed {len(tokens) - token_limit} tokens.")
        tokens = trimmed
    return ", ".join(tokens)

# ---------------------------------------------------------------------------
# Optional: appearance trait extraction for dynamic prompts
# ---------------------------------------------------------------------------
HAIR_COLORS = (
    r"blonde|black|brown|brunette|red|auburn|ginger|silver|grey|gray|"
    r"platinum|pink|blue|green|purple|orange"
)
HAIR_LENGTH = r"long|shoulder-length|short|bob|pixie|waist-length|chin-length"
EYE_COLORS = r"green|blue|brown|hazel|amber|grey|gray|violet"

HAIR_RE = re.compile(fr"\b(({HAIR_LENGTH})\s+)?({HAIR_COLORS})\s+hair\b", re.I)
EYES_RE = re.compile(fr"\b({EYE_COLORS})\s+eyes?\b", re.I)
LIPS_RE = re.compile(r"\bred\s+lipstick\b", re.I)
OUTFIT_RE = re.compile(r"\b(short|mini|micro)\s+skirt\b", re.I)
SHOES_RE = re.compile(r"\b(stiletto|high)\s+heels?\b", re.I)

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def extract_categories(appearance_request: str) -> Dict[str, str]:
    cats: Dict[str, str] = {k: "" for k in ("hair", "eyes", "lips", "outfit", "shoes")}

    if (m := HAIR_RE.search(appearance_request)) is not None:
        cats["hair"] = _clean(" ".join(filter(None, m.groups()))) + " hair"

    if (m := EYES_RE.search(appearance_request)) is not None:
        cats["eyes"] = _clean(m.group(0))

    if LIPS_RE.search(appearance_request):
        cats["lips"] = "red lipstick"

    if (m := OUTFIT_RE.search(appearance_request)) is not None:
        cats["outfit"] = _clean(m.group(0))

    if (m := SHOES_RE.search(appearance_request)) is not None:
        cats["shoes"] = _clean(m.group(0))

    return cats

# Optional LLM backfill stub
try:
    from lex.persona import lex_persona  # type: ignore
except ModuleNotFoundError:
    lex_persona = None

def llm_fill(appearance_request: str, cats: Dict[str, str]) -> None:
    if lex_persona is None:
        return
    missing = [k for k, v in cats.items() if not v]
    if not missing:
        return
    system_prompt = (
        "Extract the following appearance fields from the text below as a comma-"
        "separated list with *no* extra words (use single adjectives only). "
        "Fields: " + ", ".join(missing) + "."
    )
    reply = lex_persona.chat(system_prompt + "\n---\n" + appearance_request).strip()
    values = [p.strip() for p in reply.split(",")]
    for field, val in zip(missing, values):
        if val:
            cats[field] = val

# Brand-specific Instagram model aesthetic tokens
INSTAGRAM_TAGS: str = (
    "Instagram aesthetic, fashion editorial, beauty retouch, high contrast, "
    "vibrant color, glossy finish, overexposed highlights, dramatic shadows"
)

# Unified public builder

def build_sd_prompt (traits: Dict[str, str], token_limit: int = 75) -> Dict[str, Any]:
    """
    Convert persona traits dict into a prompt, ensuring everything is retained.
    Also appends quality, brand, and aesthetic tags. Token-clamped.
    """
    categories = {}
    raw_tokens = []

    for k, v in traits.items():
        if not v:
            continue
        v_clean = v.strip().lower()

        if "hair" in k:
            categories["hair"] = v_clean
        elif "eye" in k:
            categories["eyes"] = v_clean
        elif "lip" in k:
            categories["lips"] = v_clean
        elif "outfit" in k:
            categories["outfit"] = v_clean
        elif "shoe" in k:
            categories["shoes"] = v_clean
        elif "body" in k:
            categories["body"] = v_clean
        elif "style" in k:
            categories["style"] = v_clean
        elif "vibe" in k:
            categories["vibe"] = v_clean
        else:
            categories[k] = v_clean

        raw_tokens.append(v_clean)

    full_prompt = ", ".join(raw_tokens + [QUALITY_TAGS, BRAND_STYLE_TAGS, INSTAGRAM_TAGS])
    cleaned = sift_prompt(full_prompt, token_limit=token_limit)

    return {
        "prompt": cleaned,
        "positive": cleaned,
        "negative": NEGATIVE_MASTER,
        "categories": categories,
    }


__all__ = [
    "sift_prompt",
    "build_sd_prompt",
    "extract_categories",
    "NEGATIVE_MASTER"
]

