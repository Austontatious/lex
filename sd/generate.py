# avatar/generator.py

import base64
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

import requests

from ..utils.prompt_sifter import build_sd_prompt
from .sd_prompt_styles import BASE_STYLE, CAMERA_STYLE, DEFAULT_POSE, POST_EFFECTS, NEGATIVE_PROMPT

# Directory to save generated avatar images
IMAGE_DIR: Path = Path("/workspace/ai-lab/Lex/lex/static/lex/avatars")
IMAGE_URL_PREFIX: str = "/static/lex/avatars"

logger = logging.getLogger(__name__)

def generate_avatar(
    traits: Dict[str, str],
    *,
    save_under_mode: Optional[str] = None
) -> Dict[str, Optional[object]]:
    """
    Generate an avatar image based on given traits and return metadata including
    the image URL, filename, narration, and categories.

    :param traits: A mapping of appearance trait names to values.
    :param save_under_mode: Optional mode under which to save the avatar (unused).
    :return: Dict containing 'image', 'filename', 'narration', and 'categories'.
    """
    # Construct appearance description
    appearance_text = ", ".join(
        f"{key}: {value}" for key, value in traits.items() if value
    )
    combined_input = ", ".join([
        BASE_STYLE,
        DEFAULT_POSE,
        CAMERA_STYLE,
        POST_EFFECTS,
        appearance_text,
    ])

    # Build and clamp the prompt
    prompt_data = build_sd_prompt(combined_input, token_limit=75)

    # Prepare payload for Stable Diffusion API request
    payload = {
        "prompt": prompt_data["prompt"],
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 60,
        "cfg_scale": 9.0,
        "sampler_name": "DPM++ SDE Karras",
        "width": 348,
        "height": 512,
    }

    logger.info("[Lex Avatar] Sending prompt to SD: %s", payload["prompt"])
    response = requests.post(
        "http://127.0.0.1:7860/sdapi/v1/txt2img",
        json=payload
    )
    response.raise_for_status()

    # Decode image from the response
    images = response.json().get("images", [])
    if not images:
        logger.error("No images returned from SD API")
        raise ValueError("Stable Diffusion API returned no images")

    image_b64 = images[0]
    filename = f"lex_avatar_{uuid.uuid4().hex[:8]}.png"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGE_DIR / filename

    with filepath.open("wb") as f:
        f.write(base64.b64decode(image_b64))

    return {
        "image": f"{IMAGE_URL_PREFIX}/{filename}",
        "filename": filename,
        "narration": "Here's how I look now!",
        "categories": prompt_data.get("categories"),
    }

