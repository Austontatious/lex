import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import base64
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from gfpgan import GFPGANer
from realesrgan import RealESRGANer as RealESRGAN

from ..utils.prompt_sifter import build_sd_prompt, NEGATIVE_MASTER

# Configure logging
logger = logging.getLogger(__name__)

# --------------------------
# Configuration constants
# --------------------------
BASE_MODEL_DIR = Path(
    "/workspace/ai-lab/models/stable-diffusion-webui/models/Stable-diffusion/"
)
LEX_AVATAR_DIR = Path(__file__).resolve().parent / "static" / "lex" / "avatars"
LEX_AVATAR_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS: Dict[str, Any] = {
    "steps": 35,
    "sampler_index": "Euler a",
    "cfg_scale": 4,
    "width": 384,
    "height": 448,
}

# Lazy-initialized pipelines
_sd_pipe: Optional[StableDiffusionXLPipeline] = None
_img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None

# --------------------------
# Pipeline loading helpers
# --------------------------

def get_sd_pipe() -> StableDiffusionXLPipeline:
    """
    Load or return a cached StableDiffusionXLPipeline, with patched CLIP embeddings.
    """
    global _sd_pipe
    if _sd_pipe:
        return _sd_pipe  # type: ignore

    model_path = BASE_MODEL_DIR / "dreamshaperXL_alpha2Xl10.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    _sd_pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _sd_pipe.to(device)
    _sd_pipe.enable_attention_slicing()
    _sd_pipe.enable_vae_slicing()

    # Patch CLIPTextEmbeddings.forward to enforce long dtype
    try:
        clip_embed = _sd_pipe.text_encoder.text_model.embeddings
        orig_forward = clip_embed.forward

        def patched_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> Any:
            if input_ids is not None:
                input_ids = input_ids.to(dtype=torch.long)
            if position_ids is not None:
                position_ids = position_ids.to(dtype=torch.long)
            return orig_forward(input_ids=input_ids, position_ids=position_ids, **kwargs)

        clip_embed.forward = patched_forward.__get__(clip_embed, type(clip_embed))
        logger.info("[Lex Avatar] Patched CLIPTextEmbeddings to enforce long dtype")
    except Exception as e:
        logger.warning(f"[Lex Avatar] CLIP embedding patch failed: {e}")

    return _sd_pipe  # type: ignore


def get_img2img_pipe() -> StableDiffusionXLImg2ImgPipeline:
    """
    Load or return a cached img2img pipeline, sharing scheduler with SD pipe.
    """
    global _img2img_pipe
    if _img2img_pipe:
        return _img2img_pipe  # type: ignore

    model_dir = BASE_MODEL_DIR / "stable-diffusion-xl-base-1.0"
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _img2img_pipe.to("cpu")
    _img2img_pipe.enable_attention_slicing()
    _img2img_pipe.enable_vae_slicing()
    _img2img_pipe.scheduler = get_sd_pipe().scheduler  # type: ignore

    return _img2img_pipe

# --------------------------
# GFPGAN and RealESRGAN setup
# --------------------------
weights_path = Path(__file__).resolve().parent / "weights" / "GFPGANv1.3.pth"
if not weights_path.exists():
    raise FileNotFoundError(f"[GFPGAN] Model not found: {weights_path}")

gfpganer = GFPGANer(
    model_path=str(weights_path),
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
    device="cpu",
)

# --------------------------
# Core pipeline functions
# --------------------------

def txt2img(
    prompt: str,
    negative_prompt: str,
    steps: int = DEFAULTS["steps"],
    cfg_scale: float = DEFAULTS["cfg_scale"],
) -> Image.Image:
    """
    Generate an image from text using the SD pipeline.
    """
    pipe = get_sd_pipe()
    device = pipe.device

    # Prepare empty embeddings for text_time condition
    text_embed = torch.zeros((1, 1280), dtype=torch.float16, device=device)
    time_ids = torch.zeros((1, 6), dtype=torch.float16, device=device)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        added_cond_kwargs={"text_embeds": text_embed, "time_ids": time_ids},
        width=DEFAULTS["width"],
        height=DEFAULTS["height"],
    )
    return result.images[0]


def img2img_refine(
    init_img: Any,
    prompt: str,
    negative_prompt: str,
    strength: float = 0.5,
    steps: int = 20,
    cfg_scale: float = 9.0,
) -> Image.Image:
    """
    Refine an initial image with img2img, validating inputs and outputs.
    """
    if init_img is None:
        raise ValueError("[img2img_refine] init_img is None")

    # Convert numpy or torch tensor to PIL Image
    if isinstance(init_img, np.ndarray):
        img = Image.fromarray(init_img)
    elif isinstance(init_img, torch.Tensor):
        img = transforms.ToPILImage()(init_img.cpu().squeeze(0))
    elif isinstance(init_img, Image.Image):
        img = init_img
    else:
        raise TypeError(f"[img2img_refine] Unsupported init_img type: {type(init_img)}")

    # Ensure RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    pipe = get_img2img_pipe()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=img,
            strength=strength,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
        )
    except Exception as e:
        logger.exception("[img2img_refine] Pipeline call failed")
        raise RuntimeError(f"[img2img_refine] Pipeline call failed: {e}")

    images = getattr(result, "images", None)
    if not images:
        raise ValueError("[img2img_refine] No images returned from pipeline")

    return images[0]


def restore_faces(img: Image.Image) -> Image.Image:
    """
    Enhance faces in an image using GFPGAN.
    """
    arr = np.array(img)
    result = gfpganer.enhance(arr, has_aligned=False, only_center_face=False, paste_back=True)

    # Extract restored image from GFPGAN output
    if isinstance(result, tuple):
        # GFPGANer.enhance returns (bgr_img, restored, ...) or similar
        *_, restored = result
    else:
        raise TypeError(f"[restore_faces] Unexpected GFPGAN result type: {type(result)}")

    return Image.fromarray(restored)


def upscale(img: Image.Image) -> Image.Image:
    """
    Placeholder for upscaling (RealESRGAN).
    """
    return img


def polish(img: Image.Image) -> Image.Image:
    """
    Placeholder for final image polishing effects.
    """
    return img

# --------------------------
# Prompt enrichment
# --------------------------

def enrich_and_validate_prompt(
    traits: Dict[str, str],
    token_limit: int = 75,
) -> Tuple[str, str]:
    """
    Build and validate SD positive and negative prompts from traits.
    """
    result = build_sd_prompt(traits, token_limit=token_limit)
    prompt = result.get("prompt")
    negative = result.get("negative", NEGATIVE_MASTER)

    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise AssertionError(
            f"Invalid prompt types: prompt={type(prompt)}, negative={type(negative)}"
        )
    return prompt, negative

# --------------------------
# Full avatar generation pipeline
# --------------------------

def generate_avatar_pipeline(traits: Dict[str, str]) -> Dict[str, Any]:
    """
    Complete avatar pipeline: txt2img, refine, face restore, upscale, polish, encode.
    """
    positive, negative = enrich_and_validate_prompt(traits)
    logger.info("[Avatar Pipeline] Final SD prompt: %s", positive)

    base_img = txt2img(positive, negative)
    if not isinstance(base_img, Image.Image):
        raise RuntimeError("[Avatar Pipeline] txt2img did not return a PIL Image")

    try:
        refined_img = img2img_refine(base_img, positive, negative)
    except Exception as e:
        logger.error("[Avatar Pipeline] img2img_refine failed: %s", e)
        refined_img = base_img

    # Validate before face restoration
    if not isinstance(refined_img, Image.Image) or refined_img.mode != "RGB":
        logger.warning(
            "[Avatar Pipeline] Refined image invalid; skipping face restoration"
        )
        final_img = base_img
    else:
        faces_img = restore_faces(refined_img)
        upscaled_img = upscale(faces_img)
        final_img = polish(upscaled_img)

    # Encode final image to base64
    buffer = io.BytesIO()
    final_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode()

    # Preserve test_mode logic even if undefined
    try:
        if test_mode:  # noqa: F821
            return {"message": "🧪 Avatar generated (test mode, no image)"}
    except NameError:
        pass

    return {"image_b64": b64_str, "narration": "Here’s your new look!"}


__all__ = [
    "generate_avatar_pipeline",
    "txt2img",
    "img2img_refine",
    "restore_faces",
    "upscale",
    "polish",
    "enrich_and_validate_prompt",
]

