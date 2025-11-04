import logging
import re
from threading import Lock
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPTokenizer

# Configure module-level logger
g = logging.getLogger(__name__)

_tokenizer: Optional["CLIPTokenizer"] = None
_tokenizer_lock = Lock()


def _get_clip_tokenizer_cls():
    from transformers import CLIPTokenizer

    return CLIPTokenizer


def _ensure_tokenizer() -> "CLIPTokenizer":
    """Lazy-load the CLIP tokenizer to avoid blocking import time."""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                tokenizer_cls = _get_clip_tokenizer_cls()
                _tokenizer = tokenizer_cls.from_pretrained("openai/clip-vit-large-patch14")
    return _tokenizer

# ---------------------------------------------------------------------------
# Stopwords to reduce visual noise in long prompts
# ---------------------------------------------------------------------------
VISUAL_STOPWORDS: frozenset[str] = frozenset({
    "dslr", "photorealistic", "high detail", "cinematic", "35mm", "bokeh", "shallow depth of field",
    "natural lighting", "ultra-realistic", "photo", "realistic skin", "film grain", "editorial",
    "balanced tones", "vibrant color", "soft lighting", "studio", "full-length", "clarity",
    "focus", "magazine", "lighting", "background", "shot", "image", "pose", "portrait", "composition"
})
STOPWORDS_PATTERN: re.Pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, VISUAL_STOPWORDS)) + r")\b"
)

# ---------------------------------------------------------------------------
# Negative & positive quality anchors
# ---------------------------------------------------------------------------
NEGATIVE_MASTER: str = (
    "(low quality, worst quality:1.2), lowres, bad anatomy, bad hands, extra limbs, "
    "missing fingers, watermark, signature, censored, nsfw censor, auto-toon, plastic skin, 3d render"
)
QUALITY_TAGS: str = "best quality, masterpiece, 8k"
BRAND_STYLE_TAGS: str = (
    "octane render, cinematic tone mapping, 8K VRay, volumetric rim light, filmic color grade"
)
INSTAGRAM_TAGS: str = (
    "Instagram aesthetic, fashion editorial, beauty retouch, high contrast, "
    "vibrant color, glossy finish, overexposed highlights, dramatic shadows"
)

# ---------------------------------------------------------------------------
# Token processing
# ---------------------------------------------------------------------------

def tokenize_prompt(prompt: str) -> List[str]:
    """
    Split a comma-delimited prompt string into individual tokens.
    """
    return [token.strip() for token in prompt.split(",") if token.strip()]
    

def truncate_prompt(text: str, max_tokens: int = 77) -> str:
    """
    Tokenize using CLIPTokenizer and truncate to max_tokens.
    """
    tokenizer = _ensure_tokenizer()
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])


def deduplicate(tokens: List[str], aggressive: bool = True) -> List[str]:
    """
    Remove duplicate tokens and optionally strip visual stopwords.
    """
    seen: set[str] = set()
    cleaned: List[str] = []

    for token in tokens:
        # Normalize token for deduplication key
        key = re.sub(r"[^\w]", "", token.lower())
        key = re.sub(r"s$", "", key)
        if aggressive:
            key = STOPWORDS_PATTERN.sub("", key)

        if key not in seen:
            seen.add(key)
            cleaned.append(token)

    return cleaned


def sift_prompt(
    prompt: str,
    token_limit: int = 75,
    aggressive: bool = True,
    log_trim: bool = False
) -> str:
    """
    Process, deduplicate, and limit tokens in a text prompt.
    """
    tokens = tokenize_prompt(prompt)
    tokens = deduplicate(tokens, aggressive=aggressive)

    if len(tokens) > token_limit:
        overflow = len(tokens) - token_limit
        tokens = tokens[:token_limit]
        if log_trim:
            g.debug(f"[PromptSifter] Trimmed {overflow} tokens.")

    return ", ".join(tokens)

# ---------------------------------------------------------------------------
# Appearance trait extraction for dynamic prompts
# ---------------------------------------------------------------------------
HAIR_COLORS = (
    r"blonde|black|brown|brunette|red|auburn|ginger|silver|grey|gray|"
    r"platinum|pink|blue|green|purple|orange"
)
HAIR_LENGTH = r"long|shoulder-length|short|bob|pixie|waist-length|chin-length"
EYE_COLORS = r"green|blue|brown|hazel|amber|grey|gray|violet"

HAIR_RE: re.Pattern = re.compile(
    fr"\b(({HAIR_LENGTH})\s+)?({HAIR_COLORS})\s+hair\b", re.IGNORECASE
)
EYES_RE: re.Pattern = re.compile(
    fr"\b({EYE_COLORS})\s+eyes?\b", re.IGNORECASE
)
LIPS_RE: re.Pattern = re.compile(r"\bred\s+lipstick\b", re.IGNORECASE)
OUTFIT_RE: re.Pattern = re.compile(r"\b(short|mini|micro)\s+skirt\b", re.IGNORECASE)
SHOES_RE: re.Pattern = re.compile(r"\b(stiletto|high)\s+heels?\b", re.IGNORECASE)


def extract_categories(appearance_request: str) -> Dict[str, str]:
    """
    Extract appearance traits (hair, eyes, lips, outfit, shoes) from the text.
    """
    categories: Dict[str, str] = {key: "" for key in ("hair", "eyes", "lips", "outfit", "shoes")}

    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    if (m := HAIR_RE.search(appearance_request)):
        categories["hair"] = _clean(" ".join(filter(None, m.groups()))) + " hair"

    if (m := EYES_RE.search(appearance_request)):
        categories["eyes"] = _clean(m.group(0))

    if LIPS_RE.search(appearance_request):
        categories["lips"] = "red lipstick"

    if (m := OUTFIT_RE.search(appearance_request)):
        categories["outfit"] = _clean(m.group(0))

    if (m := SHOES_RE.search(appearance_request)):
        categories["shoes"] = _clean(m.group(0))

    return categories

# Optional LLM backfill stub
try:
    from lexi.persona import lexi_persona  # type: ignore
except ModuleNotFoundError:
    lexi_persona = None  # type: Optional[Any]


def llm_fill(appearance_request: str, cats: Dict[str, str]) -> None:
    """
    Use lexi_persona to backfill missing appearance categories.
    """
    if not lexi_persona:
        return

    missing = [field for field, val in cats.items() if not val]
    if not missing:
        return

    system_prompt = (
        "Extract the following appearance fields from the text below as a comma-"
        f"separated list with *no* extra words (use single adjectives only). Fields: {', '.join(missing)}."
    )
    reply = lexi_persona.chat(f"{system_prompt}\n---\n{appearance_request}").strip()
    values = [v.strip() for v in reply.split(",")]

    for field, val in zip(missing, values):
        if val:
            cats[field] = val


def build_sd_prompt(
    traits: Dict[str, str],
    token_limit: int = 75
) -> Dict[str, Any]:
    """
    Build an SD prompt dict with positive, negative, and category data.
    """
    categories: Dict[str, str] = {}
    raw_tokens: List[str] = []

    for key, value in traits.items():
        if not value:
            continue

        lc_val = value.strip().lower()
        if "hair" in key:
            categories["hair"] = lc_val
        elif "eye" in key:
            categories["eyes"] = lc_val
        elif "lip" in key:
            categories["lips"] = lc_val
        elif "outfit" in key:
            categories["outfit"] = lc_val
        elif "shoe" in key:
            categories["shoes"] = lc_val
        elif "body" in key:
            categories["body"] = lc_val
        elif "style" in key:
            categories["style"] = lc_val
        elif "vibe" in key:
            categories["vibe"] = lc_val
        else:
            categories[key] = lc_val

        raw_tokens.append(lc_val)

    full_prompt = ", ".join(raw_tokens + [QUALITY_TAGS, BRAND_STYLE_TAGS, INSTAGRAM_TAGS])
    cleaned = sift_prompt(full_prompt, token_limit=token_limit)

    return {
        "prompt": cleaned,
        "positive": cleaned,
        "negative": NEGATIVE_MASTER,
        "categories": categories,
    }

__all__ = [
    "truncate_prompt",
    "sift_prompt",
    "build_sd_prompt",
    "extract_categories",
    "llm_fill",
    "NEGATIVE_MASTER",
]
