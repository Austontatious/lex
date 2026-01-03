# File: lex/routes/lex.py

from __future__ import annotations

"""
lex.py

Core Lex API routes: chat processing, persona mode management, and direct image endpoints.
"""
import json
import logging
import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..sd.sd_pipeline import generate_avatar_pipeline
from ..utils.prompt_sifter import build_sd_prompt
from .lex_persona import _load_traits, _save_traits
# alias public names
load_traits = _load_traits
save_traits = _save_traits
from ..persona.persona_core import lex_persona
from ..persona.persona_config import PERSONA_MODE_REGISTRY
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Lex Core"])
router = APIRouter(tags=["Health"])

# Paths for avatar storage and trait state persistence
AVATAR_DIR: Path = Path(__file__).resolve().parent.parent / "static" / "lex" / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
TRAIT_STATE_PATH: Path = Path(__file__).resolve().parent / "lex_persona_state.json"

# --- Request schemas ---
class ChatRequest(BaseModel):
    prompt: str

TaskRequest = ChatRequest

# --- Utility functions ---
def cache_busted_url(file_path: Path) -> str:
    """
    Append the file's modification timestamp to its static URL to bust caches.
    """
    if file_path.exists():
        ts = int(file_path.stat().st_mtime)
        rel_path = file_path.as_posix().split("/static")[-1]
        return f"/static{rel_path}?v={ts}"
    return f"/static{file_path.name}"

# --- Main processing route ---
@router.get("/lex/health")
def health():
    return {"ok": True, "service": "lex-backend"}

@router.get("/lex/ready")
def ready():
    # you can add checks here later (model loaded, SD alive, etc.)
    return {"ready": True}
    
@router.post("/process")
async def process(req: ChatRequest) -> JSONResponse:
    """
    Handle chat input: infer appearance traits to regenerate avatar or pass through normal chat.
    """
    logger.info("🗨️ /process prompt=%r", req.prompt)

    # 🛡️ PROTECT AGAINST FEEDBACK LOOPS:
    if req.prompt.startswith("Lex:") or req.prompt.startswith("assistant:"):
        logger.warning("🛑 Ignoring looped assistant prompt: %r", req.prompt)
        return JSONResponse({"cleaned": "[loop detected, halted]", "raw": "", "choices": [], "mode": lex_persona.get_mode()})
    # 1) Trait inference
    inferred = extract_traits_from_text(req.prompt) if 'extract_traits_from_text' in globals() else {}
    if inferred:
        traits = load_traits()
        traits.update(inferred)
        save_traits(traits)

        result = generate_avatar_pipeline(
            ", ".join(f"{k}: {v}" for k, v in traits.items())
        )
        avatar_path = result.get("image", result.get("image_b64"))
        lex_persona.set_avatar_path(avatar_path)  # type: ignore
        save_traits(traits, avatar_path=avatar_path)

        file_path = AVATAR_DIR / Path(lex_persona.get_avatar_path()).name  # type: ignore
        return JSONResponse({
            "cleaned": "Got it, updating her look! 💄",
            "avatar_url": cache_busted_url(file_path),
            "traits": traits,
            "mode": lex_persona.get_mode(),
        })

    # 2) Normal chat
    if not isinstance(req.prompt, str):
        raise HTTPException(status_code=400, detail="Prompt must be a string")

    try:
        reply = await asyncio.to_thread(lex_persona.chat, req.prompt)
    except Exception as exc:
        logger.error("❌ LexPersona.chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not reply or not getattr(reply, "strip", None) or not reply.strip():
        logger.warning("❌ Empty reply for prompt: %r", req.prompt)
        reply = "[no response]"

    # 3) Memory write
    try:
        memory.remember(
            MemoryShard(role="assistant", content=reply, meta={"tags": ["chat"], "compressed": True})
        )
    except Exception as mem_err:
        logger.warning("⚠️ Memory store skipped: %s", mem_err)

    return JSONResponse({
        "cleaned": reply,
        "raw": reply,
        "choices": [{"text": reply}],
        "mode": lex_persona.get_mode(),
    })

# --- Mode setting route ---
@router.post("/set_mode")
async def set_mode(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    """
    Set the persona mode and optionally reset traits.
    """
    mode = payload.get("mode")
    if mode not in PERSONA_MODE_REGISTRY:
        logger.warning("🛑 Invalid mode: %s", mode)
        raise HTTPException(status_code=400, detail="Invalid mode")
    lex_persona.set_mode(mode)
    save_traits({}, avatar_path=None)
    logger.info("✅ Mode set to: %s", mode)
    return {"status": "ok", "mode": mode}

# --- Direct avatar endpoint ---
@router.post("/avatar")
def avatar_endpoint(req: ChatRequest) -> JSONResponse:
    """
    Generate or refine avatar directly from prompt text.
    """
    result = generate_avatar_pipeline(req.prompt)  # type: ignore
    return JSONResponse({
        "image_b64": result.get("image_b64"),
        "narration": result.get("narration", "Here she is!"),
    })

# --- Image from prompt via SD API ---
@router.post("/image_from_prompt")
def image_from_prompt(req: ChatRequest) -> Any:
    """
    Generate an image from raw prompt using external SD API.
    """
    logger.info("🔮 Raw image_from_prompt: %s", req.prompt)
    try:
        data = build_sd_prompt(req.prompt)  # type: ignore
        prompt, negative = data["prompt"], data.get("negative")
        response = requests.post(
            "http://localhost:7860/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative,
                "steps": 20,
                "width": 256,
                "height": 512,
                "cfg_scale": 7,
                "sampler_name": "Euler a",
            },
            timeout=60,
        )
        response.raise_for_status()
        img_b64 = response.json()["images"][0]
    except Exception as exc:
        logger.error("🛑 SD API failure: %s", exc)
        return PlainTextResponse("/static/lex/avatars/default.png", status_code=500)

    fname = f"lex_avatar_{uuid4().hex[:6]}.png"
    file_path = AVATAR_DIR / fname
    file_path.write_bytes(base64.b64decode(img_b64))
    return {"image_url": cache_busted_url(file_path)}


# Aliases for backward compatibility
_load_traits_state = load_traits
_save_traits_state = save_traits

__all__ = ['router', '_load_traits_state', '_save_traits_state']

