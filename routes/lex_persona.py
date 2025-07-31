# routes/lex_persona.py — conversational avatar generation flow

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import List, Set, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request
from ..config import TRAIT_STATE_PATH
from ..persona_core import lex_persona
logger = logging.getLogger(__name__)
router = APIRouter(tags=["persona"])

# 1) Define IntentRequest before you use it
class IntentRequest(BaseModel):
    text: str

# Key "appearance" fields for avatars:
AVATAR_FIELDS = [
    ("body_type", "What's your body type or build? (e.g. 'petite', 'athletic', 'curvy')"),
    ("hair", "How about your hair? (e.g. 'long black', 'short blonde bob')"),
    ("eyes", "And your eye color/shape? (e.g. 'green', 'almond-shaped brown')"),
    ("style", "What's your overall style? (e.g. 'goth', 'retro glam')"),
    ("outfit", "Describe your outfit or accessories? (e.g. 'lace lingerie', 'school uniform')"),
    ("vibe", "Sum up your vibe/personality? (e.g. 'bratty', 'playful')"),
]


def extract_traits_from_text(text: str) -> Dict[str, str]:
    """
    Use local LLM to infer avatar traits from user input. Returns a dictionary with keys like:
    - style
    - outfit
    - vibe
    - pose
    - lighting
    - accessory
    """

    prompt = f"""
You are a visual stylist AI. Given a user's request for an avatar appearance, extract a creative description
broken down into labeled traits. Use up to 6 keys: `style`, `outfit`, `vibe`, `pose`, `accessory`, `lighting`.

Only return a JSON object. Do not explain. Avoid repeating user phrasing.

User: {text}
Traits:
"""

    try:
        response = local_model.generate(prompt=prompt, max_tokens=200, stop=["\n\n"])
        json_block = extract_json_block(response)
        return json.loads(json_block)
    except Exception as e:
        print(f"[Trait Extractor]: Failed to parse traits — {e}")
        return {}

    
def _load_traits() -> Dict[str, str]:
    try:
        if TRAIT_STATE_PATH.exists():
            data = json.loads(TRAIT_STATE_PATH.read_text())
            t = data.get("traits", {})
            if isinstance(t, dict):
                return {k: str(v) for k, v in t.items()}
    except Exception as e:
        logger.warning("Failed to load traits file: %s", e)
    return {}

def _save_traits(traits: dict, avatar_path: str = None):
    try:
        state = {"traits": traits}
        if avatar_path:
            state["avatar_path"] = avatar_path
        TRAIT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("Failed saving traits + avatar_path: %s", e)


def _get_missing_fields(traits: Dict[str, str]) -> List[str]:
    return [k for k, _ in AVATAR_FIELDS if not traits.get(k)]

def _get_next_question(traits: Dict[str, str]) -> str:
    for k, q in AVATAR_FIELDS:
        if not traits.get(k):
            return q
    return ""

def _assemble_prompt(traits: Dict[str, str]) -> str:
    hair = traits.get("hair", "long, sunlit blonde hair with natural shine")
    eyes = traits.get("eyes", "sparkling blue eyes with playful expression")
    outfit = traits.get("outfit", "tight daisy duke shorts, weathered brown cowboy boots")
    pose = traits.get("pose", "kneeling confidently on a hay bale, one hand on her hip, smiling")
    vibe = traits.get("vibe", "effortlessly playful, magnetic and full of life")
    style = traits.get("style", "all-American, flirty, vibrant")
    lighting = traits.get("lighting", "golden hour sunlight, warm, cinematic glow")
    location = traits.get("location", "barn with bales of hay, rustic details")

    # You can combine these with comma separation:
    base = "ultra-realistic photo, high detail, Lex, vibrant color, photorealistic skin texture, depth of field, studio-quality lighting"
    prompt = (
        f"{base}, {style}, {hair}, {eyes}, {outfit}, {pose}, {location}, "
        f"{vibe}, {lighting}, dslr, richly colored, cinematic composition, no monochrome"
    )
    return prompt

def _assemble_negative() -> str:
    return (
        "sketch, drawing, illustration, monochrome, black and white, sepia, deformed, ugly, blurry, "
        "distorted, low quality, cropped, watermark, text, painting, anime, cartoon"
    )

    
@router.post("/intent")
async def detect_intent(req: IntentRequest):
    txt = req.text.lower()

    # 🎭 Trait/appearance triggers
    avatar_triggers = [
        "wear", "style", "outfit", "look", "costume", "dress", "pose as", "as a", "in a", "clothing", "lingerie",
        "avatar", "show me", "you look", "what do you look like", "your appearance", "your outfit"
    ]
    if any(k in txt for k in avatar_triggers):
        return {"intent": "avatar_flow"}

    # 🧠 Description without triggering avatar generation
    if re.search(r"\b(describe|what do you look like|show me your look|how do you look)\b", txt):
        return {"intent": "describe_avatar"}

    return {"intent": "chat"}

@router.post("/add_trait")
async def add_trait(request: Request):
    """
    Drop-in legacy endpoint: Accepts {trait: "..."} and integrates it into the new avatar flow.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    trait = payload.get("trait", "").strip()
    if not trait:
        raise HTTPException(status_code=400, detail="Missing 'trait' field")
    # Load traits (from file or memory)
    traits = _load_traits()
    # Find next missing field
    missing = _get_missing_fields(traits)
    if missing:
        traits[missing[0]] = trait
    _save_traits(traits)
    # Pass to the modern avatar flow for correct response
    # Simulate a modern avatar_step call for correct narration/ready/etc
    if not _get_missing_fields(traits):
        prompt = _assemble_prompt(traits)
        negative = _assemble_negative()
        narration = "Here's your look! If you want to tweak anything, use the modify feature."
        ready = True
    else:
        prompt = ""
        negative = ""
        narration = _get_next_question(traits)
        ready = False
    persona = {
        "traits": traits,
        "certainty": 1.0 if ready else 0.8,
        "image_path": lex_persona.get_avatar_path(),
    }
    return {
        "ready": ready,
        "persona": persona,
        "prompt": prompt,
        "negative": negative,
        "narration": narration,
        "added": True
    }

@router.post("/avatar_step")
async def avatar_step(request: Request):
    """
    Conversational endpoint to assemble avatar traits step by step, asking only for missing fields.
    Will automatically move to ready state when complete.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Expect a dict: {traits: {body_type, hair, ...}} or a flat text "reply"
    traits: Dict[str, str] = payload.get("traits", {}) or {}
    reply: str = payload.get("reply", "").strip()

    # If a reply is present, assign it to the next missing field:
    missing = _get_missing_fields(traits)
    if reply and missing:
        traits[missing[0]] = reply

    _save_traits(traits)  # persist

    # Determine if we're done:
    missing = _get_missing_fields(traits)
    if not missing:
        prompt = _assemble_prompt(traits)
        negative = _assemble_negative()
        # Optional: Actually generate avatar here, or trigger on frontend
        narration = "Here's your look! If you want to tweak anything, use the modify feature."
        ready = True
    else:
        prompt = ""
        negative = ""
        narration = _get_next_question(traits)
        ready = False

    persona = {
        "traits": traits,
        "certainty": 1.0 if not missing else 0.8,
        "image_path": lex_persona.get_avatar_path(),
    }

    return {
        "ready": ready,
        "persona": persona,
        "prompt": prompt,
        "negative": negative,
        "narration": narration,
    }

@router.get("/get")
async def get_persona():
    try:
        state = json.loads(TRAIT_STATE_PATH.read_text())
        traits_map = state.get("traits", {})
        avatar_path = state.get("avatar_path", "/static/lex/avatars/default.png")
        # Confirm avatar file actually exists:
        # (adjust this as needed for your env!)
        static_root = Path(__file__).resolve().parent.parent / "static" / "lex" / "avatars"
        avatar_file = static_root / Path(avatar_path).name
        if not avatar_file.exists():
            avatar_path = "/static/lex/avatars/default.png"
        return {
            "traits": traits_map,
            "certainty": 1.0 if not _get_missing_fields(traits_map) else 0.8,
            "image_path": avatar_path,
        }
    except Exception as e:
        logger.warning("Failed loading persona for /get: %s", e)
        return {
            "traits": {},
            "certainty": 0.0,
            "image_path": "/static/lex/avatars/default.png"
        }
    
    print(f"State loaded: {state}")
    print(f"avatar_path: {avatar_path}")
    print(f"Checking avatar file: {avatar_file}, exists: {avatar_file.exists()}")


@router.get("/debug/traits")
def debug_traits():
    return {
        "file_traits": _load_traits(),
        "persona_traits": getattr(lex_persona, "get_traits", lambda: {})(),
    }

