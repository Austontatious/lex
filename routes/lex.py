from __future__ import annotations

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# New pipeline import
from ..sd.sd_pipeline import generate_avatar_pipeline
from ..utils.prompt_sifter import build_sd_prompt
from .lex_persona import _load_traits, _save_traits
from ..persona.persona_core import lex_persona
from ..persona.persona_config import PERSONA_MODES
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Lex Core"])

# Paths
AVATAR_DIR = Path(__file__).resolve().parent.parent / "static" / "lex" / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
TRAIT_STATE_PATH = Path(__file__).resolve().parent / "lex_persona_state.json"

# Request schemas
class ChatRequest(BaseModel):
    prompt: str

TaskRequest = ChatRequest

def cache_busted_url(file_path: Path) -> str:
    if file_path.exists():
        ts = int(file_path.stat().st_mtime)
        rel_path = file_path.as_posix().split("/static")[-1]
        return f"/static{rel_path}?v={ts}"
    return f"/static{file_path.name}"

# Utility: extract simple traits
def extract_traits_from_text(text: str) -> dict:
    text = text.lower()
    traits: dict = {}
    match = lambda kw: kw in text

    if match("cowgirl"):
        traits.update({"style": "cowgirl", "outfit": "cutoff shorts, vest", "vibe": "bold"})
    elif match("goth"):
        traits.update({"style": "gothic", "outfit": "corset, mesh", "vibe": "brooding"})
    # add more keyword-based trait extraction as needed

    # modifiers
    if match("sunset"):
        traits["lighting"] = "golden hour"
    if match("lingerie"):
        traits.update({"outfit": "lace lingerie", "vibe": "seductive"})

    return traits

# MAIN ROUTES
@router.post("/process")
async def process(req: ChatRequest):
    logger.info("🗨️ /process prompt=%r", req.prompt)

    # 1) Trait inference
    inferred = extract_traits_from_text(req.prompt)
    if inferred:
        traits = _load_traits_state(TRAIT_STATE_PATH)
        traits.update(inferred)
        _save_traits_state(TRAIT_STATE_PATH, traits)

        # regenerate avatar with full pipeline
        result = generate_avatar_pipeline(", ".join(f"{k}: {v}" for k, v in traits.items()))
        lex_persona.set_avatar_path(result.get("image_url", result.get("image_b64")))
        _save_traits_state(TRAIT_STATE_PATH, traits)
        return {
            "cleaned": "Got it, updating her look! 💄",
            "avatar_url": cache_busted_url(AVATAR_DIR / Path(lex_persona.get_avatar_path()).name),

            "traits": traits,
            "mode": lex_persona.get_mode(),
        }

    # 2) Normal chat
    if not isinstance(req.prompt, str):
        raise HTTPException(status_code=400, detail="Prompt must be a string")

    try:
        reply = await asyncio.to_thread(lex_persona.chat, req.prompt)
    except Exception as e:
        logger.error("❌ LexPersona.chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    if not reply or not reply.strip():
        logger.warning("❌ Empty reply for prompt: %r", req.prompt)
        reply = "[no response]"

    # 3) Memory
    try:
        if True:  # your condition here
            memory.remember(
                MemoryShard(role="assistant", content=reply,
                           meta={"tags": ["chat"], "compressed": True})
            )
    except Exception as mem_err:
        logger.warning("⚠️ Memory store skipped: %s", mem_err)

    return JSONResponse({
        "cleaned": reply,
        "raw": reply,
        "choices": [{"text": reply}],
        "mode": lex_persona.get_mode(),
    })

@router.post("/set_mode")
async def set_mode(payload: dict = Body(...)):
    mode = payload.get("mode")
    if mode not in PERSONA_MODES:
        logger.warning("🛑 Invalid mode: %s", mode)
        raise HTTPException(status_code=400, detail="Invalid mode")
    lex_persona.set_mode(mode)
    _save_traits_state(TRAIT_STATE_PATH, {})  # reset traits on mode change if desired
    logger.info("✅ Mode set to: %s", mode)
    return {"status": "ok", "mode": mode}

@router.post("/avatar")
def avatar_endpoint(req: ChatRequest):
    """Generate or refine avatar directly from prompt text"""
    # direct pipeline
    result = generate_avatar_pipeline(req.prompt)
    return JSONResponse({
        "image_b64": result["image_b64"],
        "narration": result.get("narration", "Here she is!"),
    })

@router.post("/image_from_prompt")
def image_from_prompt(req: ChatRequest):
    logger.info("🔮 Raw image_from_prompt: %s", req.prompt)
    try:
        prompt, negative, _ = build_sd_prompt(req.prompt).values()
        r = requests.post("http://localhost:7860/sdapi/v1/txt2img",
                          json={"prompt": prompt, "negative_prompt": negative,
                                "steps": 20, "width": 256, "height": 512,
                                "cfg_scale": 7, "sampler_name": "Euler a"}, timeout=60)
        r.raise_for_status()
        img_b64 = r.json()["images"][0]
    except Exception as e:
        logger.error("🛑 SD API failure: %s", e)
        return PlainTextResponse("/static/lex/avatars/default.png", status_code=500)

    fname = f"lex_avatar_{uuid.uuid4().hex[:6]}.png"
    with open(AVATAR_DIR / fname, "wb") as f:
        f.write(base64.b64decode(img_b64))
    file_path = AVATAR_DIR / fname
    return {"image_url": cache_busted_url(file_path)}


