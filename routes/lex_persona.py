from __future__ import annotations

"""
lex_persona.py

Conversational avatar generation flow with step-by-step trait elicitation and intent detection.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

from ..config.config import TRAIT_STATE_PATH
from ..persona.persona_core import lex_persona

logger = logging.getLogger(__name__)
router = APIRouter(tags=["persona"])

# Request schema for intent detection
class IntentRequest(BaseModel):
    text: str

# Fields for avatar conversation flow: (field, prompt)
AVATAR_FIELDS: List[tuple[str, str]] = [
    ("body_type", "What's your body type or build? (e.g. 'petite', 'athletic', 'curvy')"),
    ("hair", "How about your hair? (e.g. 'long black', 'short blonde bob')"),
    ("eyes", "And your eye color/shape? (e.g. 'green', 'almond-shaped brown')"),
    ("style", "What's your overall style? (e.g. 'goth', 'retro glam')"),
    ("outfit", "Describe your outfit or accessories (e.g. 'lace lingerie', 'school uniform')"),
    ("vibe", "Sum up your vibe/personality (e.g. 'bratty', 'playful')"),
]

# Triggers indicating avatar-related intent
AVATAR_TRIGGERS: List[str] = [
    'wear', 'style', 'outfit', 'costume', 'dress', 'clothing', 'lingerie', 'avatar'
]

DESCRIBE_PATTERN = re.compile(r"\b(describe|what do you look like|show me your look|how do you look)\b", re.IGNORECASE)

# --- Helper functions ---
def _load_traits() -> Dict[str, str]:
    """
    Load saved traits from state file.
    """
    try:
        if TRAIT_STATE_PATH.exists():
            data = json.loads(TRAIT_STATE_PATH.read_text())
            traits = data.get('traits', {})
            if isinstance(traits, dict):
                return {str(k): str(v) for k, v in traits.items()}
    except Exception as exc:
        logger.warning("Failed to load traits: %s", exc)
    return {}


def _save_traits(traits: Dict[str, str], avatar_path: Optional[str] = None) -> None:
    """
    Persist traits (and optional avatar path) to state file.
    """
    state: dict[str, Any] = {'traits': traits}
    if avatar_path:
        state['avatar_path'] = avatar_path
    try:
        TRAIT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("Failed to save traits: %s", exc)


def get_missing_fields(traits: Dict[str, str]) -> List[str]:
    """
    Return list of required avatar fields not yet provided.
    """
    return [field for field, _ in AVATAR_FIELDS if not traits.get(field)]


def next_question(traits: Dict[str, str]) -> str:
    """
    Get the prompt for the next missing avatar field.
    """
    missing = get_missing_fields(traits)
    if missing:
        for field, question in AVATAR_FIELDS:
            if field == missing[0]:
                return question
    return ""


def assemble_prompt(traits: Dict[str, str]) -> str:
    """
    Combine trait values into a Stable Diffusion prompt string.
    """
    # Default fallbacks embedded in prompt
    defaults = {
        'hair': 'long, sunlit blonde hair with natural shine',
        'eyes': 'sparkling blue eyes with playful expression',
        'outfit': 'tight daisy duke shorts, weathered brown cowboy boots',
        'pose': 'kneeling confidently on a hay bale, one hand on her hip, smiling',
        'vibe': 'effortlessly playful, magnetic and full of life',
        'style': 'all-American, flirty, vibrant',
        'lighting': 'golden hour sunlight, warm, cinematic glow',
        'location': 'barn with bales of hay, rustic details'
    }
    # Merge provided traits with defaults for missing
    prompt_parts = []
    base = (
        "ultra-realistic photo, high detail, Lex, vibrant color, photorealistic skin texture,"
        " depth of field, studio-quality lighting"
    )
    prompt_parts.append(base)
    for key, default in defaults.items():
        prompt_parts.append(traits.get(key, default))
    return ", ".join(prompt_parts)


def assemble_negative_prompt() -> str:
    """
    Standard negative prompt to filter out unwanted styles.
    """
    return (
        "sketch, drawing, illustration, monochrome, black and white, sepia, deformed, ugly, blurry,"
        " distorted, low quality, cropped, watermark, text, painting, anime, cartoon"
    )

# --- API endpoints ---
@router.post("/intent")
async def detect_intent(req: IntentRequest) -> Dict[str, str]:
    """
    Classify user text into 'avatar_flow', 'describe_avatar', or 'chat'.
    """
    text = req.text.lower()
    if any(re.search(rf'\b{re.escape(trigger)}\b', text) for trigger in AVATAR_TRIGGERS):
        return {'intent': 'avatar_flow'}
    if DESCRIBE_PATTERN.search(text):
        return {'intent': 'describe_avatar'}
    return {'intent': 'chat'}

@router.post("/add_trait")
async def add_trait(request: Request) -> Dict[str, Any]:
    """
    Add a single trait to the next missing field in avatar flow.
    """
    payload = await request.json()
    trait = str(payload.get('trait', '')).strip()
    if not trait:
        raise HTTPException(status_code=400, detail="Missing 'trait' field")

    traits = _load_traits()
    missing = get_missing_fields(traits)
    if missing:
        traits[missing[0]] = trait
        save_traits(traits)

    ready = not get_missing_fields(traits)
    prompt = assemble_prompt(traits) if ready else ''
    negative = assemble_negative_prompt() if ready else ''
    narration = (
        "Here's your look! If you'd like to tweak anything, let me know." if ready
        else next_question(traits)
    )
    persona = {
        'traits': traits,
        'certainty': 1.0 if ready else 0.8,
        'image_path': lex_persona.get_avatar_path()
    }
    return {'ready': ready, 'persona': persona, 'prompt': prompt,
            'negative': negative, 'narration': narration, 'added': True}

@router.post("/avatar_step")
async def avatar_step(request: Request) -> Dict[str, Any]:
    """
    Stepwise conversation for avatar traits: ask missing fields until done.
    """
    payload = await request.json()
    traits: Dict[str, str] = payload.get('traits', {}) or {}
    reply: str = str(payload.get('reply', '')).strip()

    if reply and get_missing_fields(traits):
        traits[get_missing_fields(traits)[0]] = reply
        save_traits(traits)

    ready = not get_missing_fields(traits)
    prompt = assemble_prompt(traits) if ready else ''
    negative = assemble_negative_prompt() if ready else ''
    narration = (
        "Here's your look! If you'd like to tweak anything, let me know." if ready
        else next_question(traits)
    )
    persona = {
        'traits': traits,
        'certainty': 1.0 if ready else 0.8,
        'image_path': lex_persona.get_avatar_path()
    }
    return {'ready': ready, 'persona': persona, 'prompt': prompt,
            'negative': negative, 'narration': narration}

@router.get("/get")
async def get_persona() -> Dict[str, Any]:
    """
    Retrieve current persona state and avatar path.
    """
    try:
        state = json.loads(TRAIT_STATE_PATH.read_text())
        traits = state.get('traits', {})
        avatar = state.get('avatar_path', '/static/lex/avatars/default.png')
        # Validate file existence
        file_path = Path(avatar)
        if not file_path.exists():
            avatar = '/static/lex/avatars/default.png'
        ready = not get_missing_fields(traits)
        return {'traits': traits, 'certainty': 1.0 if ready else 0.8, 'image_path': avatar}
    except Exception as exc:
        logger.warning("Failed loading persona: %s", exc)
        return {'traits': {}, 'certainty': 0.0, 'image_path': '/static/lex/avatars/default.png'}

@router.get("/debug/traits")
def debug_traits() -> Dict[str, Any]:
    """
    Debug endpoint: compare file-based traits vs in-memory persona traits.
    """
    file_traits = _load_traits()
    persona_traits = getattr(lex_persona, 'get_traits', lambda: {})()
    return {'file_traits': file_traits, 'persona_traits': persona_traits}

__all__ = ['router', 'assemble_prompt', '_load_traits', '_save_traits']


