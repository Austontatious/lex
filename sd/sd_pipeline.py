import torch

import os
import io
import base64
import logging
import re
import numpy as np
from torchvision import transforms
from uuid import uuid4
from pathlib import Path
from difflib import get_close_matches
from typing import List
from PIL import Image, ImageEnhance
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from gfpgan import GFPGANer
from realesrgan import RealESRGANer as RealESRGAN
from ..utils.prompt_sifter import build_sd_prompt, NEGATIVE_MASTER

# --------------------------
# Configuration
# --------------------------
BASE_MODEL_DIR = "/workspace/ai-lab/models/stable-diffusion-webui/models/Stable-diffusion/"
LEX_AVATAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static/lex/avatars"))
os.makedirs(LEX_AVATAR_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

DEFAULTS = {
    "steps": 35,
    "sampler_index": "Euler a",
    "cfg_scale": 4,
    "width": 384,
    "height": 448,
}

# --------------------------
# Globals and Lazy Init
# --------------------------
_sd_pipe = None
_img2img_pipe = None

# --------------------------
# Pipeline Loaders
# --------------------------
def get_sd_pipe():
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    model_path = os.path.join(BASE_MODEL_DIR, "dreamshaperXL_alpha2Xl10.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _sd_pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _sd_pipe.to(device)
    _sd_pipe.enable_attention_slicing()
    _sd_pipe.enable_vae_slicing()

    # 🩹 Deep patch CLIPTextEmbeddings.forward to ensure dtype = Long
    # 🩹 Patch entire CLIPTextEmbeddings module to force Long dtype
    try:
        clip_embed = _sd_pipe.text_encoder.text_model.embeddings
        orig_forward = clip_embed.forward

        def patched_forward(self, input_ids=None, position_ids=None, **kwargs):
            if input_ids is not None:
                input_ids = input_ids.to(dtype=torch.long)
            if position_ids is not None:
                position_ids = position_ids.to(dtype=torch.long)
            return orig_forward(input_ids=input_ids, position_ids=position_ids, **kwargs)

        clip_embed.forward = patched_forward.__get__(clip_embed, type(clip_embed))
        logger.info("[Lex Avatar] Patched CLIPTextEmbeddings to enforce Long dtype on all IDs")
    except Exception as e:
        logger.warning(f"[Lex Avatar] Failed to patch CLIPTextEmbeddings: {e}")


    return _sd_pipe


def get_img2img_pipe():
    global _img2img_pipe
    if _img2img_pipe is None:
        model_path = os.path.join(BASE_MODEL_DIR, "stable-diffusion-xl-base-1.0")
        _img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        _img2img_pipe.to("cpu")
        _img2img_pipe.enable_attention_slicing()
        _img2img_pipe.enable_vae_slicing()
        _img2img_pipe.scheduler = get_sd_pipe().scheduler
    return _img2img_pipe

# --------------------------
# GFPGAN Setup
# --------------------------
weights_path = (Path(__file__).resolve().parent / "weights" / "GFPGANv1.3.pth").resolve()
assert weights_path.exists(), f"[GFPGAN] Model file not found at: {weights_path}"
gfpganer = GFPGANer(
    model_path=str(weights_path),
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
    device="cpu"
)

# --------------------------
# Core Functions
# --------------------------
def txt2img(prompt: str, negative_prompt: str, steps=30, cfg_scale=9.0):
    pipe = get_sd_pipe()
    device = pipe.device
    text_embed = torch.zeros((1, 1280), dtype=torch.float16, device=device)
    time_ids = torch.zeros((1, 6), dtype=torch.float16, device=device)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        added_cond_kwargs={"text_embeds": text_embed, "time_ids": time_ids},
        width=384,
        height=448,
    )
    return result.images[0]

def img2img_refine(
    init_img,
    prompt: str,
    negative_prompt: str,
    strength: float = 0.5,
    steps: int = 20,
    cfg_scale: float = 9.0
) -> Image.Image:
    """Refines an image using the img2img pipeline with validation and normalization."""

    if init_img is None:
        raise ValueError("[img2img_refine] init_img is None")

    if isinstance(init_img, np.ndarray):
        init_img = Image.fromarray(init_img)
    elif isinstance(init_img, torch.Tensor):
        from torchvision import transforms
        init_img = transforms.ToPILImage()(init_img.cpu().squeeze(0))
    elif isinstance(init_img, Image.Image):
        pass  # good
    else:
        raise TypeError(f"[img2img_refine] Unsupported init_img type: {type(init_img)}")

    if init_img.mode != "RGB":
        init_img = init_img.convert("RGB")

    pipe = get_img2img_pipe()

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_img,
            strength=strength,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
        )
    except Exception as e:
        logger.exception("[img2img_refine] Pipeline call failed:")
        raise RuntimeError(f"[img2img_refine] Pipeline call failed: {e}")

    if not result or not hasattr(result, "images") or not result.images:
        raise ValueError("[img2img_refine] Pipeline call failed: no images returned")

    return result.images[0]


def restore_faces(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    result = gfpganer.enhance(arr, has_aligned=False, only_center_face=False, paste_back=True)

    if isinstance(result, tuple):
        if len(result) == 2:
            _, restored = result
        elif len(result) == 3:
            _, _, restored = result
        else:
            raise ValueError(f"Unexpected return value from enhance(): {result}")
    else:
        raise TypeError(f"enhance() returned unexpected type: {type(result)}")

    return Image.fromarray(restored)

def upscale(img: Image.Image) -> Image.Image:
    return img  # stub

def polish(img: Image.Image) -> Image.Image:
    return img  # stub

# --------------------------
# Prompt Assembly
# --------------------------
#from .lex_prompt_styles import BASE_STYLE, CAMERA_STYLE, POST_EFFECTS, DEFAULT_POSE, #NEGATIVE_PROMPT

def enrich_and_validate_prompt(traits: dict, token_limit: int = 75) -> tuple[str, str]:
    result = build_sd_prompt(traits, token_limit=token_limit)
    prompt = result["prompt"]
    negative = result.get("negative", NEGATIVE_MASTER)

    assert isinstance(prompt, str) and isinstance(negative, str), f"Invalid prompts: {prompt=}, {negative=}"
    return prompt, negative

# --------------------------
# Avatar Generation Pipeline
# --------------------------
def generate_avatar_pipeline(traits: dict) -> dict:
    pos, neg = enrich_and_validate_prompt(traits)
    print(f"[Lex Avatar] Final SD prompt: {pos}")
    base = txt2img(pos, neg)

    if base is None or not isinstance(base, Image.Image):
        raise RuntimeError("[Avatar Gen] txt2img returned invalid image")

    try:
        refined = img2img_refine(base, pos, neg)
    except Exception as e:
        logger.error(f"[Avatar Gen] img2img_refine failed: {e}")
        refined = base

    # Extra fail-safe — validate again before face restore
    if not isinstance(refined, Image.Image) or refined.mode != "RGB":
        logger.warning("[Avatar Gen] Refined image is invalid — skipping face restoration")
        final = base
        
    faces = restore_faces(refined)
    upscaled = upscale(faces)
    final = polish(upscaled)

    buffer = io.BytesIO()
    final.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    if test_mode:
        return "🧪 Avatar generated (test mode, no image)"


    return {"image_b64": b64, "narration": "Here’s your new look!"}

