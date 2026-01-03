import base64
import logging
from pathlib import Path
from uuid import uuid4
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from .lex_persona import _load_traits as load_traits, _save_traits as save_traits
from ..config.config import TRAIT_STATE_PATH

# Initialize router and logger
router = APIRouter(tags=["avatar"])
logger = logging.getLogger(__name__)

# Constants for image storage
IMAGE_DIR: Path = Path("/workspace/ai-lab/Lex/lex/static/lex/avatars")
IMAGE_URL_PREFIX: str = "/static/lex/avatars"

# --- Request model ---
class AvatarGenRequest(BaseModel):
    traits: Dict[str, str]

# --- Endpoint: Generate avatar image from traits ---
@router.post("/generate")
async def generate_avatar(
    req: Optional[AvatarGenRequest] = None
) -> Dict[str, object]:
    """
    Generate an avatar based on provided traits or last saved state.

    - Builds and runs the SD pipeline.
    - Decodes and saves the resultant image.
    - Persists the new traits and avatar path.
    - Returns image metadata and narration.

    :param req: AvatarGenRequest containing 'traits', optional.
    :returns: Dict with 'image', 'filename', 'narration', 'traits'.
    :raises HTTPException: on any failure during generation or saving.
    """
    # Determine traits: use request or fallback to persisted state
    traits: Dict[str, str] = (
        req.traits if req and req.traits else load_traits()
    )

    try:
        # Run full avatar generation pipeline
        result: Dict[str, str] = generate_avatar_pipeline(traits)
        image_b64: Optional[str] = result.get("image_b64")
        narration: str = result.get("narration", "Here's how I look now!")

        if not image_b64:
            raise ValueError("Stable Diffusion pipeline did not return image_b64")

        # Ensure image directory exists
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # Decode and save image
        filename = f"lex_avatar_{uuid4().hex[:8]}.png"
        filepath = IMAGE_DIR / filename
        with filepath.open("wb") as f:
            f.write(base64.b64decode(image_b64))

        web_path = f"{IMAGE_URL_PREFIX}/{filename}"

        # Persist traits and new avatar path
        _save_traits(traits, avatar_path=web_path)

        return {
            "image": web_path,
            "filename": filename,
            "narration": narration,
            "traits": traits,
        }

    except Exception as exc:
        logger.exception("[Avatar Gen Error] %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Avatar generation failed."
        )

