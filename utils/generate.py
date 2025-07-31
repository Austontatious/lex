# avatar/generator.py

import os, uuid, base64, requests, logging
from ..lex_prompt_styles import BASE_STYLE, CAMERA_STYLE, POST_EFFECTS, DEFAULT_POSE, NEGATIVE_PROMPT
from ..utils.prompt_sifter import build_sd_prompt

IMAGE_DIR = "/workspace/ai-lab/Lex/lex/static/lex/avatars"
IMAGE_URL_PREFIX = "/static/lex/avatars"

logger = logging.getLogger(__name__)

def generate_avatar(traits, *, save_under_mode=None):
    appearance_text = ", ".join(f"{k}: {v}" for k, v in traits.items() if v)
    combined_input = ", ".join([BASE_STYLE, DEFAULT_POSE, CAMERA_STYLE, POST_EFFECTS, appearance_text])
    prompt_data = build_sd_prompt(combined_input, token_limit=75)  # ✅ token limit enforced here


    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 60,
        "cfg_scale": 9.0,
        "sampler_name": "DPM++ SDE Karras",
        "width": 348,
        "height": 512,
    }

    logger.info("[Lex Avatar] Sending prompt to SD: %s", payload["prompt"])
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
    response.raise_for_status()

    image_b64 = response.json()["images"][0]
    filename = f"lex_avatar_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(IMAGE_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_b64))

    return {
        "image": f"{IMAGE_URL_PREFIX}/{filename}",
        "filename": filename,
        "narration": "Here's how I look now!",
        "categories": prompt_data["categories"],
    }

