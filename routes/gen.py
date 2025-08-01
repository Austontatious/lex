# routes/gen.py — Avatar generation endpoint (Stable Diffusion)

import os
import base64
import logging
from pathlib import Path
from uuid import uuid4
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from .lex_persona import _save_traits, _load_traits
from ..config.config import TRAIT_STATE_PATH

logger = logging.getLogger(__name__)
router = APIRouter(tags=["avatar"])

# --- Constants ---
IMAGE_DIR = Path("/workspace/ai-lab/Lex/lex/static/lex/avatars")
IMAGE_URL_PREFIX = "/static/lex/avatars"

# --- Request Model ---
class AvatarGenRequest(BaseModel):
    traits: Dict[str, str]

# --- Endpoint: Generate avatar image from traits ---
@router.post("/generate")
async def generate_avatar(req: AvatarGenRequest | None = None):
    traits = req.traits if req and req.traits else _load_traits()

    try:
        result = generate_avatar_pipeline(traits)  # ✅ use dict, not str
        image_b64 = result.get("image_b64")
        narration = result.get("narration", "Here's how I look now!")


        if not image_b64:
            raise ValueError("Stable Diffusion pipeline did not return image_b64")

        # Save decoded image
        filename = f"lex_avatar_{uuid4().hex[:8]}.png"
        filepath = IMAGE_DIR / filename
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_b64))

        web_path = f"{IMAGE_URL_PREFIX}/{filename}"

        # Persist traits and avatar path to JSON state
        _save_traits(traits, avatar_path=web_path)

        return {
            "image": web_path,
            "filename": filename,
            "narration": narration,
            "traits": traits,
        }

    except Exception as e:
        logger.exception(f"[Avatar Gen Error] {e}")
        raise HTTPException(status_code=500, detail="Avatar generation failed")

