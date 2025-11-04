# ── Lexi/lexi/sd/sd_pipeline.py ────────────────────────────────────────
"""
ComfyUI-backed SDXL helpers with a two-pass Base→Refiner flow and edit modes.

Modes:
  - txt2img (default): SDXL base → (optional) refiner → save
  - img2img: low-denoise edits to preserve identity/look
  - inpaint: masked edits

Continuity:
  - If no seed is provided, we derive a stable "identity seed" from traits so that the face/look
    stays consistent between sessions. You can still override seed per-call.

Env:
  COMFY_URL         (default: http://127.0.0.1:8188)
  COMFY_BASE_CKPT   (default: sdxl-base-1.0/sd_xl_base_1.0.safetensors)
  COMFY_REFINER_CKPT(default: sdxl-refiner-1.0/sd_xl_refiner_1.0.safetensors)
  COMFY_UPSCALE     (default: false)  -> "true" to enable simple latent upscale hop
  LEX_IMAGE_DIR     (default: <repo>/frontend/public/avatars)
"""

from __future__ import annotations
import random
import base64
import hashlib
import os
import json
import time
import shutil
import uuid
import logging

log = logging.getLogger("lexi.sd")
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from ..config.config import AVATAR_DIR, AVATAR_URL_PREFIX
from .sd_prompt_styles import (
    _style_realistically_unreal as _styles_style_realistically_unreal,
    _negative_prompt as _styles_negative_prompt,
    MODE_PRESETS,
)

# ------------------------- Config/Paths -------------------------

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

# ✅ Use FILENAMES, not folder prefixes
BASE_CKPT = os.getenv("COMFY_BASE_CKPT", "sd_xl_base_1.0.safetensors")
REFINER_CKPT = os.getenv("COMFY_REFINER_CKPT", "sd_xl_refiner_1.0.safetensors")

USE_UPSCALE = str(os.getenv("COMFY_UPSCALE", "false")).lower() in ("1", "true", "yes")
PUBLIC_BASE_URL = os.getenv("LEX_PUBLIC_BASE_URL", "").rstrip("/")

LEX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = AVATAR_DIR
IMAGE_DIR = Path(os.getenv("LEX_IMAGE_DIR", str(DEFAULT_IMAGE_DIR)))
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- Backend selection -------------------------
SD_BACKEND = os.getenv("SD_BACKEND", "sdxl").strip().lower()
DEFAULT_FLUX_VARIANT = os.getenv("FLUX_MODEL_VARIANT", "kontext-dev").strip().lower()

# Flux model paths
FLUX_MODELS_DIR = Path(os.getenv("FLUX_MODELS_DIR", "/mnt/data/comfy/models"))
FLUX_DIFFUSION_DIR = Path(
    os.getenv("FLUX_DIFFUSION_DIR", str(FLUX_MODELS_DIR / "diffusion_models"))
)
FLUX_TEXT_ENCODER_DIR = Path(
    os.getenv("FLUX_TEXT_ENCODER_DIR", str(FLUX_MODELS_DIR / "text_encoders"))
)
FLUX_VAE_PATH = Path(os.getenv("FLUX_VAE_PATH", str(FLUX_MODELS_DIR / "vae" / "ae.safetensors")))
FLUX_CLIP_L = Path(os.getenv("FLUX_CLIP_L", str(FLUX_TEXT_ENCODER_DIR / "clip_l.safetensors")))
FLUX_T5XXL = Path(
    os.getenv("FLUX_T5XXL", str(FLUX_TEXT_ENCODER_DIR / "t5xxl_fp8_e4m3fn.safetensors"))
)

FLUX_DEFAULT_GUIDANCE = float(os.getenv("FLUX_GUIDANCE_DEFAULT", "3.5"))
FLUX_DEFAULT_CFG = float(os.getenv("FLUX_CFG_DEFAULT", "1.8"))
FLUX_DEFAULT_STEPS = int(os.getenv("FLUX_STEPS_DEFAULT", "22"))
FLUX_DEFAULT_SAMPLER = os.getenv("FLUX_SAMPLER_DEFAULT", "euler")
FLUX_DEFAULT_SCHEDULER = os.getenv("FLUX_SCHEDULER_DEFAULT", "simple")

FLUX_CANVAS_PRESETS: Dict[str, Tuple[int, int]] = {
    "portrait": (1024, 1344),
    "full_body": (896, 1600),
    "tall": (832, 1664),
    "square": (1024, 1024),
}

FLUX_PRESETS = {
    "full_body_fashion": {
        "subject_prefix": "full-body portrait, head-to-toe in frame, feet visible on studio floor, center-framed, elegant posture, ",
        "style": "fashion editorial lighting, balanced contrast, 50 mm lens, 8k clarity",
    },
    "beauty_editorial": {
        "subject_prefix": "mid-length portrait, shoulder-up framing, beauty lighting, ",
        "style": "soft key light, crisp focus, magazine-grade color",
    },
}


class FluxPromptAdapter:
    SUBJECT_STYLE_FALLBACK = (
        "professional studio lighting, neutral 5500K key with gentle rim, "
        "sharp focus, f/4 depth of field, clean editorial look, realistic skin texture"
    )
    NEGATIVE = (
        "close-up, half-body, headshot, cropped, zoomed-in, "
        "cartoon, anime, 3d render, deformed body, extra limbs, "
        "bad anatomy, blurry, watermark, text, logo, harsh shadows, plastic skin, fog, haze"
    )

    @staticmethod
    def split(prompt: str, style_hint: Optional[str] = None) -> Tuple[str, str]:
        subj = prompt.strip()
        style = FluxPromptAdapter.SUBJECT_STYLE_FALLBACK
        if style_hint:
            style = f"{style}, {style_hint}"
        return subj, style.strip()

    @staticmethod
    def negatives(additional: Optional[str] = None) -> str:
        parts = [FluxPromptAdapter.NEGATIVE]
        if additional:
            parts.append(additional)
        return ", ".join(p for p in parts if p)


# Sanity log at import time (helps verify env scope)
try:
    log.info("[Lexi SD] COMFY_URL=%s", COMFY_URL)
except Exception:
    pass


# New: workflow JSONs (defaults point at your uploaded files)
def _resolve_repo_path(val: str, fallback_rel: str) -> Path:
    """
    Resolve env path; if it's relative, anchor it to LEX_ROOT so it's stable
    regardless of the container's working directory.
    """
    p = Path(val) if val else Path(fallback_rel)
    return p if p.is_absolute() else (LEX_ROOT / p)


WORKFLOW_FACE = _resolve_repo_path(
    os.getenv("LEX_WORKFLOW_FACE", ""), "sd/workflows/face_workflow.json"
)
WORKFLOW_BODY = _resolve_repo_path(
    os.getenv("LEX_WORKFLOW_BODY", ""), "sd/workflows/body_workflow.json"
)

# ---- Model registry (capability-based) -------------------------------------
# NOTE: this must come *after* LEX_ROOT is defined
try:
    from ..models.model_registry import ModelRegistry

    REG_PATH = LEX_ROOT / "models" / "registry.yaml"
    REG = ModelRegistry(REG_PATH) if REG_PATH.exists() else None
except Exception:
    REG = None

# ---- Model/LoRA selectors ----------------------------------------------------

CHECKPOINT_DIR = Path(os.getenv("SD_CKPT_DIR") or "/mnt/data/models/sd/checkpoints")
LORA_DIR = Path(os.getenv("SD_LORA_DIR") or "/mnt/data/models/sd/loras")
COMFY_ROOT = Path(os.getenv("COMFY_ROOT") or "/mnt/data/comfy")
COMFY_CKPT_ROOT = COMFY_ROOT / "models" / "checkpoints"
COMFY_LORA_ROOT = COMFY_ROOT / "models" / "loras"
COMFY_VAE_ROOT = COMFY_ROOT / "models" / "vae"


def _relname_under(root: Path, filename: str) -> str:
    """
    Return the relative name Comfy expects for a given filename.
    If the file exists directly under root, returns 'filename'.
    If it's in a subfolder (e.g., 'external/filename'), returns that.
    If multiple matches exist, prefer shallowest.
    Falls back to the original filename if nothing is found.
    """
    # direct
    if (root / filename).exists():
        return filename
    # any subdir
    candidates = list(root.rglob(filename))
    if candidates:
        # pick shortest relpath
        rp = min((c.relative_to(root) for c in candidates), key=lambda p: len(str(p)))
        return str(rp).replace("\\", "/")
    return filename


@dataclass(frozen=True)
class FluxPaths:
    ckpt: Path
    clip_l: Path
    t5xxl: Path
    ae: Path


FLUX_VARIANTS: Dict[str, FluxPaths] = {
    "kontext-dev": FluxPaths(
        ckpt=FLUX_DIFFUSION_DIR / "flux1-kontext-dev.safetensors",
        clip_l=FLUX_CLIP_L,
        t5xxl=FLUX_T5XXL,
        ae=FLUX_VAE_PATH,
    ),
    "dev": FluxPaths(
        ckpt=FLUX_DIFFUSION_DIR / "flux1-dev.safetensors",
        clip_l=FLUX_CLIP_L,
        t5xxl=FLUX_T5XXL,
        ae=FLUX_VAE_PATH,
    ),
    "schnell": FluxPaths(
        ckpt=FLUX_DIFFUSION_DIR / "flux1-schnell.safetensors",
        clip_l=FLUX_CLIP_L,
        t5xxl=FLUX_T5XXL,
        ae=FLUX_VAE_PATH,
    ),
}


def _flux_variant_paths(variant: Optional[str]) -> FluxPaths:
    key = (variant or DEFAULT_FLUX_VARIANT or "kontext-dev").strip().lower()
    paths = FLUX_VARIANTS.get(key)
    if not paths:
        raise ValueError(
            f"Unsupported Flux variant '{variant}'. Available: {', '.join(sorted(FLUX_VARIANTS))}"
        )
    # verify files exist
    missing = [p for p in (paths.ckpt, paths.clip_l, paths.t5xxl, paths.ae) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Flux assets missing: {', '.join(str(m) for m in missing)}. "
            "Check FLUX_* environment variables."
        )
    return paths


def _flux_relname(root: Path, path: Path) -> str:
    return _relname_under(root, path.name if path.is_absolute() else str(path))


def _flux_ckpt_name(path: Path) -> str:
    return _flux_relname(FLUX_DIFFUSION_DIR, path)


def _flux_clip_name(path: Path) -> str:
    return _flux_relname(FLUX_TEXT_ENCODER_DIR, path)


def _flux_vae_name(path: Path) -> str:
    return _flux_relname(FLUX_MODELS_DIR / "vae", path)


def _ckpt_name_for_comfy(filename: str) -> str:
    return _relname_under(COMFY_CKPT_ROOT, filename)


def _lora_name_for_comfy(filename: str) -> str:
    return _relname_under(COMFY_LORA_ROOT, filename)


def _vae_name_for_comfy(filename: str) -> str:
    return _relname_under(COMFY_VAE_ROOT, filename)


def _coerce_seed(seed_in, traits: Optional[Dict[str, str]], add_outfit_salt: bool) -> int:
    """
    Always return a 32-bit int seed.
    - If seed_in is None or non-int, derive from traits (hair/eyes/face/skin).
    - Optionally salt with outfit-dependent pose to vary composition across outfits.
    """
    # If caller passed a string/float/etc., try to cast
    try:
        if isinstance(seed_in, (str, float)):
            seed = int(seed_in)
        elif isinstance(seed_in, int):
            seed = seed_in
        else:
            seed = None
    except Exception:
        seed = None

    if seed is None:
        seed = _identity_seed_from_traits(traits or {})

    if add_outfit_salt and traits and traits.get("outfit"):
        seed = (int(seed) ^ _pose_seed_from_outfit(traits)) & 0xFFFFFFFF

    return int(seed) & 0xFFFFFFFF


_MAJOR_EDIT_TOKENS = {
    "hair": [
        "long hair",
        "short hair",
        "bob",
        "pixie",
        "ponytail",
        "bun",
        "bangs",
        "fringe",
        "updo",
    ],
    "body": ["curvy", "hourglass", "athletic", "slim thick"],
    "outfit": ["jean shorts", "shorts", "jacket", "dress", "gown", "hoodie", "coat", "skirt"],
}
OUTFIT_KEYWORDS = {
    "outfit",
    "shirt",
    "tshirt",
    "t-shirt",
    "tee",
    "top",
    "jeans",
    "shorts",
    "skirt",
    "dress",
    "hoodie",
    "jacket",
    "leggings",
    "stockings",
    "thigh-high",
    "boots",
    "heels",
    "sneakers",
    "jean shorts",
    "denim shorts",
    "crop top",
}
HAIR_KEYWORDS = {"ponytail", "bun", "bangs", "pigtails", "braid"}

# Your provided lists:
BASE_CHECKPOINTS = [
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "LexiIllustriousSDXL.safetensors",
    "Illustrious-XL-v2.0.safetensors",
    "bigLust_v16SDXL.safetensors",
    "realisticFreedom_omegaSDXL.safetensors",
    "ilustmix_v70Cinematic.safetensors",
    "dreamshaperXL_alpha2Xl10.safetensors",
    "sd_xl_base_1.0.safetensors",
    "sd_xl_base_1.0_0.9vae.safetensors",
]

LORA_OUTFITS_DEFAULT = [
    {"name": "Outfit_soph-HankyHemCropTop-ILXL.safetensors", "unet": 0.75, "clip": 0.35},
    {"name": "hourglassv2_SDXL.safetensors", "unet": 0.50, "clip": 0.30},
]

NSFW_MERGES = [
    "LexiIllustriousPornSDXL.safetensors",
    "novaillustrousNSFW_v20SDXL.safetensors",
    "pyrosNSFWSDXL_v05.safetensors",
    "pornworksRealPornPhoto_v04SDXL.safetensors",
]


def _exists_ckpt(name: str) -> bool:
    return (CHECKPOINT_DIR / name).is_file()


def _exists_lora(name: str) -> bool:
    return (LORA_DIR / name).is_file()


def _select_checkpoint(style: str, nsfw: bool) -> str:
    # Favor realism for the “realistically unreal” baseline
    if nsfw:
        for name in [
            "bigLust_v16SDXL.safetensors",
            "realisticFreedom_omegaSDXL.safetensors",
            *NSFW_MERGES,
            "sd_xl_base_1.0_0.9vae.safetensors",
            "sd_xl_base_1.0.safetensors",
        ]:
            if _exists_ckpt(name):
                return name

    if style in ("realistic", "cinematic"):
        for name in [
            "realisticFreedom_omegaSDXL.safetensors",
            "dreamshaperXL_alpha2Xl10.safetensors",
            "sd_xl_base_1.0_0.9vae.safetensors",
            "sd_xl_base_1.0.safetensors",
        ]:
            if _exists_ckpt(name):
                return name

    if style == "stylized":
        for name in [
            "ilustmix_v70Cinematic.safetensors",
            "Illustrious-XL-v2.0.safetensors",
            "LexiIllustriousSDXL.safetensors",
            "sd_xl_base_1.0_0.9vae.safetensors",
            "sd_xl_base_1.0.safetensors",
        ]:
            if _exists_ckpt(name):
                return name

    # Fallback
    for name in BASE_CHECKPOINTS:
        if _exists_ckpt(name):
            return name
    return "sd_xl_base_1.0.safetensors"


def _chain_loras(
    g: Dict[str, Any], model_key: list, clip_key: list, loras: list[dict], prefix: str
) -> tuple[list, list]:
    """
    Attach a chain of LoRAs, returning (model_out, clip_out).
    Each LoRA node takes (model, clip) in and outputs (model, clip) out.
    """
    current_model, current_clip = model_key, clip_key
    for i, l in enumerate(loras):
        if not _exists_lora(l["name"]):
            continue
        node_name = f"{prefix}_lora_{i}"
        g[node_name] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": current_model,
                "clip": current_clip,
                "lora_name": l["name"],
                "strength_model": float(l.get("unet", 0.6)),
                "strength_clip": float(l.get("clip", 0.3)),
            },
        }
        current_model = [node_name, 0]
        current_clip = [node_name, 1]
    return current_model, current_clip


# ------------------------- Prompt helpers -------------------------


def _style_realistically_unreal() -> Tuple[str, str]:
    """Delegate to centralized prompt styles for consistency across the app."""
    return _styles_style_realistically_unreal()


def _prompt_from_traits(traits: Dict[str, str], composition: str = "portrait") -> str:
    style_pos, _ = _style_realistically_unreal()
    comp = {
        "portrait": "portrait of Lexi, shoulders-up, engaging eye contact",
        "three_quarter": "3/4 body shot of Lexi, standing, outfit fully visible, natural stance",
        "full_body": "full-body shot of Lexi, outfit fully visible, natural stance",
    }.get(composition, "portrait of Lexi, shoulders-up, engaging eye contact")
    pieces = [comp, style_pos]

    # Pull common traits (fallbacks if missing)
    hair = traits.get("hair", "medium-length blonde hair with natural highlights")
    eyes = traits.get("eyes", "blue eyes with lively catchlights")
    vibe = traits.get("vibe", "confident, playful energy")
    outfit = traits.get("outfit", "casual top")
    style = traits.get("style", "modern editorial")
    pose = traits.get("pose", "relaxed pose, soft smile, natural posture")
    lighting = traits.get("lighting", "soft directional key light, subtle rim, gentle fill")
    background = traits.get("background", "neutral studio backdrop")

    pieces += [hair, eyes, vibe, outfit, style, pose, lighting, background]
    # De-dup & join
    seen, clean = set(), []
    for p in pieces:
        p = p.strip()
        if not p or p in seen:
            continue
        clean.append(p)
        seen.add(p)
    return ", ".join(clean)


def _augment_prompt_with_traits(prompt: str, traits: Optional[Dict[str, str]]) -> str:
    """Lightly enrich an LLM-provided prompt with persona traits if missing.
    Keeps the LLM in charge; only appends hair/eyes/outfit/vibe/lighting/background when absent.
    """
    if not traits:
        return prompt
    p = (prompt or "").lower()
    adds = []

    def maybe_add(key: str):
        val = str(traits.get(key, "")).strip()
        if val and key not in p and val.lower() not in p:
            adds.append(val)

    for k in ("hair", "eyes", "outfit", "vibe", "lighting", "background"):
        maybe_add(k)
    if adds:
        prompt = f"{prompt}, {', '.join(adds)}"

    # Apply persona/mode presets if present (e.g., traits.preset="kitty_girl")
    try:
        preset_key = None
        for k in ("preset", "mode", "persona", "style_preset"):
            v = str(traits.get(k, "")).strip().lower()
            if v:
                preset_key = v
                break
        if preset_key and preset_key in MODE_PRESETS:
            cur = prompt.lower()
            add_tags = [t for t in MODE_PRESETS[preset_key] if t.lower() not in cur]
            if add_tags:
                prompt = f"{prompt}, {', '.join(add_tags)}"
    except Exception:
        pass

    return prompt


def _negative_prompt(extra: Optional[str] = None) -> str:
    base_neg = _styles_negative_prompt()
    return (base_neg + ", " + extra) if extra else base_neg


# ------------------------- Identity/Seed -------------------------


def _identity_seed_from_traits(traits: Dict[str, str]) -> int:
    # Salt identity with hair/eyes only (face continuity)
    base_keys = {
        k: traits[k]
        for k in sorted(traits.keys())
        if any(t in k for t in ("hair", "eyes", "face", "skin"))
    }
    key = "|".join(f"{k}={base_keys[k]}" for k in sorted(base_keys.keys()))
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _pose_seed_from_outfit(traits: Dict[str, str]) -> int:
    outfit = traits.get("outfit", "")
    h = hashlib.sha256(("[pose]" + outfit).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ------------------------- Comfy API helpers -------------------------
def _get_ckpt_enum() -> list[str]:
    """Ask Comfy which checkpoint filenames it knows about."""
    try:
        r = requests.get(f"{COMFY_URL}/object_info", timeout=20)
        r.raise_for_status()
        j = r.json()
        node = j.get("CheckpointLoaderSimple", {})
        p = (node.get("input", {}) or {}).get("ckpt_name", {})
        enum = p.get("enum", []) if isinstance(p, dict) else []
        return [str(x) for x in enum]
    except Exception:
        return []


def _resolve_ckpt(name: str, enum: list[str]) -> str:
    """Return an allowed filename; fall back to close match or first available."""
    if name in enum:
        return name
    # try basename in case a path slipped in
    base = os.path.basename(name)
    if base in enum:
        return base
    # fuzzy: prefer sd_xl_* files
    for cand in enum:
        if base.split(".")[0] in cand:
            return cand
    return enum[0] if enum else base


def _post_graph(graph: Dict[str, Any]) -> str:
    r = requests.post(
        f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": "lexi"}, timeout=60
    )
    r.raise_for_status()
    j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    return j.get("prompt_id") or j.get("id") or ""


def _post_workflow(workflow: Dict[str, Any]) -> str:
    """Identical to _post_graph but named to reflect we’re sending a full JSON workflow."""
    return _post_graph(workflow)


def _wait_for_images(prompt_id: str, timeout_s: int = 240) -> List[Dict[str, Any]]:
    start = time.time()
    # Poll /history/<id> and only accept images for this exact prompt_id
    while time.time() - start < timeout_s:
        hr = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
        if hr.ok and hr.headers.get("content-type", "").startswith("application/json"):
            hist = hr.json()
            rec = hist.get(prompt_id) if isinstance(hist, dict) else None
            if isinstance(rec, dict):
                outs = rec.get("outputs") or {}
                if isinstance(outs, dict):
                    for _, node_out in outs.items():
                        imgs = node_out.get("images") or []
                        if imgs:
                            return imgs
        time.sleep(0.8)
    return []


def _download_image(
    filename: str, subfolder: str = "", ftype: str = "output", dst: Optional[Path] = None
) -> Path:
    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    vr = requests.get(f"{COMFY_URL}/view", params=params, stream=True, timeout=180)
    vr.raise_for_status()
    if dst is None:
        dst = IMAGE_DIR / f"lexi_{uuid.uuid4().hex[:8]}.png"
    with dst.open("wb") as f:
        for chunk in vr.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return dst


def _finalize_generated_image(out: Path, base_path: Path, force_output_to_base: bool) -> Path:
    """
    Apply Lexi-specific naming policy:
      - Base renders overwrite lexi_base.png
      - Iterations always produce unique lexi_XXXX.png files.
    """
    base_name = base_path.name
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if force_output_to_base:
        final_out = base_path
        if out.resolve() != final_out.resolve():
            final_out.parent.mkdir(parents=True, exist_ok=True)
            try:
                out.replace(final_out)
            except Exception:
                data = out.read_bytes()
                final_out.write_bytes(data)
                try:
                    out.unlink()
                except Exception:
                    pass
        return final_out

    # Iteration render — ensure unique filename
    if out.name == base_name or out == base_path:
        final_out = IMAGE_DIR / f"lexi_{uuid.uuid4().hex[:8]}.png"
        final_out.write_bytes(out.read_bytes())
        try:
            out.unlink()
        except Exception:
            pass
        return final_out

    if not out.name.startswith("lexi_") or out.suffix.lower() != ".png":
        final_out = IMAGE_DIR / f"lexi_{uuid.uuid4().hex[:8]}.png"
        final_out.write_bytes(out.read_bytes())
        try:
            out.unlink()
        except Exception:
            pass
        return final_out

    return out


# ------------------------- Workflow loader/patcher -------------------------


def _load_workflow_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _replace_ckpt_names(flow: Dict[str, Any], base_file: str, ref_file: Optional[str]) -> None:
    """Update checkpoint names for both editor-style and API-style payloads."""
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") == "CheckpointLoaderSimple":
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and vals:
                if vals[0] and (
                    "refiner" in str(vals[0]).lower() or "checkpoints/sd_xl_refiner" in str(vals[0])
                ):
                    if ref_file:
                        vals[0] = _ckpt_name_for_comfy(ref_file)
                else:
                    vals[0] = _ckpt_name_for_comfy(base_file)
    # API-style
    if "nodes" not in flow:
        first = True
        for node in flow.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "CheckpointLoaderSimple":
                ckpt = _ckpt_name_for_comfy(ref_file if (not first and ref_file) else base_file)
                node.setdefault("inputs", {})["ckpt_name"] = ckpt
                first = False


def _set_empty_latent_size(flow: Dict[str, Any], w: int, h: int) -> None:
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") in ("EmptyLatentImage", "EmptyLatentImageNode"):
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and len(vals) >= 2:
                vals[0], vals[1] = int(w), int(h)
    # API-style
    if "nodes" not in flow:
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") in (
                "EmptyLatentImage",
                "EmptyLatentImageNode",
            ):
                node.setdefault("inputs", {}).update({"width": int(w), "height": int(h)})


def _set_load_image(flow: Dict[str, Any], filename: str) -> None:
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") == "LoadImage":
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and vals:
                vals[0] = filename
    # API-style
    if "nodes" not in flow:
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                node.setdefault("inputs", {})["image"] = filename


def _set_text_prompts(flow: Dict[str, Any], positive: str, negative: str) -> None:
    """
    Replace all CLIPTextEncode node texts:
    - Nodes whose current text looks like a 'negative' template get `negative`
    - All others get `positive`
    """
    neg_markers = (
        "cartoon, anime",
        "bad anatomy",
        "lowres",
        "watermark",
        "3d render",
        "harsh shadow",
    )
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") == "CLIPTextEncode":
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and vals:
                old = str(vals[0])
                vals[0] = negative if any(m in old for m in neg_markers) else positive
    # API-style
    if "nodes" not in flow:
        # Heuristic: if the current text contains negative markers, treat as negative
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                old = str((node.get("inputs") or {}).get("text", ""))
                (node.setdefault("inputs", {}))["text"] = (
                    negative if any(m in old for m in neg_markers) else positive
                )


def _set_sampler_params(
    flow: Dict[str, Any], *, steps: int, cfg: float, seed: int, denoise: Optional[float] = None
) -> None:
    """
    Set seed/steps/cfg for all KSampler nodes.
    Only override 'denoise' when the node isn't a full txt2img step:
      - Editor-style: skip if existing denoise is ~1.0
      - API-style: skip if inputs.denoise is >= 0.99
    This prevents blank outputs when a workflow's first KSampler starts from EmptyLatentImage.
    """
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") == "KSampler":
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and len(vals) >= 7:
                vals[0] = int(seed)
                vals[2] = int(steps)
                vals[3] = float(cfg)
                if denoise is not None:
                    try:
                        existing = float(vals[6])
                    except Exception:
                        existing = None
                    if existing is None or existing < 0.99:
                        vals[6] = float(denoise)
    # API-style
    if "nodes" not in flow:
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                inp = node.setdefault("inputs", {})
                inp["seed"] = int(seed)
                inp["steps"] = int(steps)
                inp["cfg"] = float(cfg)
                if denoise is not None:
                    existing = inp.get("denoise")
                    try:
                        exf = float(existing) if existing is not None else None
                    except Exception:
                        exf = None
                    if exf is None or exf < 0.99:
                        inp["denoise"] = float(denoise)


def _ensure_latent_upscale(flow: Dict[str, Any], scale_by: float) -> None:
    if scale_by is None or scale_by <= 1.01:
        return
    # Editor-style
    for n in flow.get("nodes", []):
        if n.get("type") in ("LatentUpscaleBy", "ImageScaleBy"):
            vals = n.get("widgets_values", [])
            if isinstance(vals, list) and len(vals) >= 2 and isinstance(vals[1], (int, float)):
                vals[1] = float(scale_by)
    # API-style
    if "nodes" not in flow:
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") in (
                "LatentUpscaleBy",
                "ImageScaleBy",
            ):
                node.setdefault("inputs", {})["scale_by"] = float(scale_by)


# ------------------------- Aesthetic tuning -------------------------


def _tune_lora_strengths(
    flow: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    default_unet: float = 0.5,
    default_clip: float = 0.25,
) -> None:
    """
    Adjust LoraLoader strengths to favor a 'realistically unreal' aesthetic:
    - Moderate UNet influence, low CLIP influence (keeps text prompt authority).
    - Per-LoRA overrides for known names.
    Applies to both editor and API style payloads.
    """
    ov = {k.lower(): v for k, v in (overrides or {}).items()}

    def _apply(node: Dict[str, Any]) -> None:
        if node.get("class_type") != "LoraLoader":
            return
        inp = node.setdefault("inputs", {})
        name = str(inp.get("lora_name", "")).lower()
        unet, clip = ov.get(name, (default_unet, default_clip))
        inp["strength_model"] = float(unet)
        inp["strength_clip"] = float(clip)

    if "nodes" in flow and isinstance(flow.get("nodes"), list):
        for n in flow["nodes"]:
            # Editor exports sometimes have type instead of class_type
            if n.get("type") == "LoraLoader":
                # normalize into API-style dict to reuse _apply
                node = {"class_type": "LoraLoader", "inputs": {}}
                w = n.get("widgets_values") or []
                # widgets order can vary; safest to not try mapping here
                # prefer updating directly on editor style when possible
                # However, 'inputs' are not typically present; skip
                continue
    else:
        for node in flow.values():
            if isinstance(node, dict):
                _apply(node)


def _tune_controlnet_strength(flow: Dict[str, Any], *, strength: float = 0.25) -> None:
    """Set ControlNetApplyAdvanced strength to a balanced default."""
    if "nodes" in flow and isinstance(flow.get("nodes"), list):
        for n in flow["nodes"]:
            t = n.get("type") or n.get("class_type")
            if t == "ControlNetApplyAdvanced":
                vals = n.get("widgets_values", [])
                if isinstance(vals, list) and vals:
                    vals[0] = float(strength)
    else:
        for node in flow.values():
            if isinstance(node, dict) and node.get("class_type") == "ControlNetApplyAdvanced":
                node.setdefault("inputs", {})["strength"] = float(strength)


def _apply_realistically_unreal_profile(flow: Dict[str, Any], *, is_face: bool) -> None:
    """
    Apply curated weights for 'realistically unreal' Instagram model aesthetic.
    - Reduce CLIP weights on all LoRAs so the text prompt dominates composition.
    - Keep identity LoRA (ip-adapter faceid) moderately strong.
    - Keep body/shape LoRA modest.
    - Tame hair/color LoRAs.
    - Set ControlNet pose strength to a balanced value for body flow.
    """
    overrides: Dict[str, Tuple[float, float]] = {
        # Identity
        "ip-adapter-faceid-plusv2_sdxl_lora.safetensors": (0.50, 0.12),
        # Faces detail
        "ai_top_faces_1474526.safetensors": (0.42, 0.18),
        "sdxl_betterfaces-lora_v1.safetensors": (0.32, 0.18),
        "lora-sdxl-perfect-eyes.safetensors": (0.45, 0.22),
        "eyesxl_v2.safetensors": (0.22, 0.18),
        # Body/shape
        # Disable shape forcing; let prompt drive proportions
        "hourglassv2_sdxl.safetensors": (0.0, 0.0),
        # Hair/color sliders
        "colorful hair slider_alpha1.0_rank4_noxattn_last.safetensors": (0.10, 0.15),
        # Portrait helper
        "portrait_1000_girl_faces.safetensors": (0.30, 0.18),
    }
    _tune_lora_strengths(
        flow, overrides=overrides, default_unet=0.38 if is_face else 0.25, default_clip=0.16
    )
    if not is_face:
        _tune_controlnet_strength(flow, strength=0.25)


def _has_image_sink(payload: Dict[str, Any]) -> bool:
    """Return True if the payload will emit images that show up in history."""
    if not isinstance(payload, dict):
        return False
    if "nodes" in payload:
        nodes = payload.get("nodes") or []
        return any(
            (n.get("type") in ("SaveImage", "PreviewImage", "ImageSave", "ImageOutput"))
            for n in nodes
        )
    # API-prompt style
    return any(
        (v.get("class_type") in ("SaveImage", "PreviewImage", "ImageSave"))
        for v in payload.values()
        if isinstance(v, dict)
    )


def _attach_save_if_missing(flow: Dict[str, Any]) -> None:
    """
    Ensure the workflow/prompt has a SaveImage node connected to an image output.
    Works for both editor-style and API-prompt style JSON.
    """
    if not isinstance(flow, dict):
        return
    # 1) API-prompt style (no 'nodes' key)
    if "nodes" not in flow:
        has_save = any(
            isinstance(v, dict) and v.get("class_type") == "SaveImage" for v in flow.values()
        )
        if has_save:
            return
        # find a likely image source: prefer ImageScaleBy, then VAEDecode, then ImageComposite
        candidates = [
            k
            for k, v in flow.items()
            if isinstance(v, dict)
            and v.get("class_type") in ("ImageScaleBy", "VAEDecode", "ImageComposite")
        ]
        if not candidates:
            return
        try:
            last_id = sorted(candidates, key=lambda k: int(k))[-1]
        except Exception:
            last_id = candidates[-1]
        # choose a new numeric id
        numeric_keys = [int(k) for k in flow.keys() if isinstance(k, str) and k.isdigit()]
        new_id = str((max(numeric_keys) + 1) if numeric_keys else 9999)
        flow[new_id] = {
            "class_type": "SaveImage",
            "inputs": {"images": [last_id, 0], "filename_prefix": "lexi"},
        }
        return

    # 2) Editor-workflow style ('nodes'/'links')
    nodes = flow.get("nodes")
    if not isinstance(nodes, list):
        return
    if any((n.get("type") == "SaveImage" or n.get("class_type") == "SaveImage") for n in nodes):
        return

    # prefer ImageScaleBy, else VAEDecode, else ImageComposite
    last = None
    for pref in ("ImageScaleBy", "VAEDecode", "ImageComposite"):
        imgs = [n for n in nodes if (n.get("type") or n.get("class_type")) == pref]
        if imgs:
            try:
                last = sorted(imgs, key=lambda n: int(n.get("id", 0)))[-1]
            except Exception:
                last = imgs[-1]
            break
    if not last:
        return

    try:
        save_id = max(int(n.get("id", 0)) for n in nodes) + 1
    except Exception:
        save_id = 10001

    save_node = {
        "id": save_id,
        "type": "SaveImage",
        "widgets_values": ["lexi"],
        "inputs": [{"name": "images"}],
        "_meta": {"title": "Save Image"},
    }
    nodes.append(save_node)

    links = flow.setdefault("links", [])
    try:
        new_link_id = (max((l[0] for l in links), default=1000) + 1) if links else 1001
    except Exception:
        new_link_id = 1001
    # wire from last node output 0 -> save.images (slot 0)
    try:
        links.append([int(new_link_id), int(last["id"]), 0, int(save_id), 0])
    except Exception:
        pass


def _use_face_workflow(
    *,
    prompt: str,
    negative: str,
    w: int,
    h: int,
    steps: int,
    cfg: float,
    seed: int,
    refiner_strength: float,
    upscale_factor: float,
    base_file: str,
    refiner_file: Optional[str],
) -> Dict[str, Any]:
    """
    Load and patch your face/portrait workflow (base → refiner) for the first base render.
    """
    flow = _load_workflow_json(WORKFLOW_FACE)
    _replace_ckpt_names(flow, base_file, refiner_file)
    _set_empty_latent_size(flow, w, h)
    _set_text_prompts(flow, prompt, negative)
    _set_sampler_params(
        flow, steps=steps, cfg=cfg, seed=seed, denoise=None
    )  # face graph handles denoise internally
    _ensure_latent_upscale(flow, upscale_factor if USE_UPSCALE else 1.0)
    _apply_realistically_unreal_profile(flow, is_face=True)
    return flow


def _use_body_workflow(
    *,
    prompt: str,
    negative: str,
    w: int,
    h: int,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float,
    src_uploaded_name: str,
    base_file: str,
    refiner_file: Optional[str],
    upscale_factor: float,
) -> Dict[str, Any]:
    """
    Load and patch your full-body workflow (OpenPose+IP-Adapter FaceID) for outfit/pose edits.
    """
    flow = _load_workflow_json(WORKFLOW_BODY)
    _replace_ckpt_names(flow, base_file, refiner_file)
    _set_empty_latent_size(flow, w, h)
    _set_load_image(flow, src_uploaded_name)
    _set_text_prompts(flow, prompt, negative)
    _set_sampler_params(flow, steps=steps, cfg=cfg, seed=seed, denoise=denoise)
    _ensure_latent_upscale(flow, upscale_factor if USE_UPSCALE else 1.0)
    _apply_realistically_unreal_profile(flow, is_face=False)
    return flow


# ------------------------- Graph builders -------------------------


def _graph_sdxl_txt2img(
    prompt: str,
    negative: str,
    w: int,
    h: int,
    steps: int,
    cfg: float,
    seed: int,
    use_refiner: bool,
    refiner_strength: float,
    upscale_factor: float,
    base_file: str,  # <<< NEW
    refiner_file: Optional[str] = None,  # <<< NEW
    style: str = "realistic",
    nsfw: bool = False,
) -> Dict[str, Any]:
    """
    SDXL txt2img: optionally use refiner at low strength for 'polish'.
    """
    g: Dict[str, Any] = {}

    # Use registry-selected filenames (already Comfy-visible)
    base_ckpt_name = _ckpt_name_for_comfy(base_file)
    g["ckpt_base"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": base_ckpt_name},
    }

    # Encode with base CLIP by default
    clip_in = ["ckpt_base", 1]
    model_in = ["ckpt_base", 0]
    vae_in = ["ckpt_base", 2]

    g["t_enc_base"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_in, "text": prompt}}
    g["t_neg_base"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_in, "text": negative},
    }
    g["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": w, "height": h, "batch_size": 1},
    }
    g["ks_base"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_in,
            "positive": ["t_enc_base", 0],
            "negative": ["t_neg_base", 0],
            "latent_image": ["latent", 0],
            "seed": int(seed),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
        },
    }

    prev_latent = ["ks_base", 0]
    prev_vae = vae_in

    # Optional refiner pass
    if use_refiner and refiner_file:
        ref_ckpt_name = _ckpt_name_for_comfy(refiner_file)
        if _exists_ckpt(ref_ckpt_name):
            g["ckpt_ref"] = {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ref_ckpt_name},
            }
            g["t_enc_ref"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt_ref", 1], "text": prompt},
            }
            g["t_neg_ref"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt_ref", 1], "text": negative},
            }
            g["ks_ref"] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["ckpt_ref", 0],
                    "positive": ["t_enc_ref", 0],
                    "negative": ["t_neg_ref", 0],
                    "latent_image": prev_latent,
                    "seed": int(seed),
                    "steps": max(8, steps // 3),
                    "cfg": max(3.5, cfg - 0.5),
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": float(refiner_strength),
                },
            }
            prev_latent = ["ks_ref", 0]
            prev_vae = ["ckpt_ref", 2]

    if USE_UPSCALE and upscale_factor and upscale_factor > 1.01:
        g["lat_up"] = {
            "class_type": "LatentUpscale",
            "inputs": {
                "samples": prev_latent,
                "upscale_method": "nearest-exact",
                "scale_by": float(upscale_factor),
            },
        }
        prev_latent = ["lat_up", 0]

    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": prev_latent, "vae": prev_vae}}
    g["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "lexi"},
    }
    return g


def _graph_sdxl_img2img(
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float,
    source_filename: str,
    source_subfolder: str,
    base_file: str,  # <<< NEW
    style: str = "realistic",
    nsfw: bool = False,
    outfit_loras: Optional[List[Dict[str, Any]]] = None,
    reg_loras: Optional[List[Tuple[str, float, float]]] = None,  # <<< NEW
) -> Dict[str, Any]:
    """
    SDXL img2img with optional outfit LoRAs.
    Keep denoise ~0.35–0.55 for outfit/appearance edits while preserving identity.
    """
    g: Dict[str, Any] = {}

    # Pick checkpoint based on style/nsfw and map to Comfy's expected enum string
    base_ckpt_name = _ckpt_name_for_comfy(base_file)
    g["ckpt"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": base_ckpt_name},
    }

    # Load source and encode with CKPT's VAE
    g["load"] = {
        "class_type": "LoadImage",
        "inputs": {"image": source_filename, "choose file to upload": "image"},
    }
    g["enc"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["load", 0], "vae": ["ckpt", 2]},
    }

    # Start from raw checkpoint outputs
    model_handle: List[Any] = ["ckpt", 0]
    clip_handle: List[Any] = ["ckpt", 1]

    # Chain registry-provided LoRAs first (identity, style, etc.)
    for i, (fname, unet_w, clip_w) in enumerate(reg_loras or []):
        node_name = f"reg_lora_{i}"
        g[node_name] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_handle,
                "clip": clip_handle,
                "lora_name": _lora_name_for_comfy(fname),
                "strength_model": float(unet_w),
                "strength_clip": float(clip_w),
            },
        }
        model_handle = [node_name, 0]
        clip_handle = [node_name, 1]

    # Resolve list of LoRAs to apply (can be empty)
    lora_list = outfit_loras if outfit_loras is not None else LORA_OUTFITS_DEFAULT

    # Chain outfit LoRAs (if any)
    for i, lr in enumerate(lora_list or []):
        name = lr.get("name")
        if not name:
            continue
        # Map filename to the exact enum string Comfy expects (handles subfolder prefixes)
        lora_enum_name = _lora_name_for_comfy(str(name))
        node_name = f"outfit_lora_{i}"
        g[node_name] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_handle,
                "clip": clip_handle,
                "lora_name": lora_enum_name,
                "strength_model": float(lr.get("unet", 0.6)),
                "strength_clip": float(lr.get("clip", 0.3)),
            },
        }
        # LoraLoader outputs (model, clip)
        model_handle = [node_name, 0]
        clip_handle = [node_name, 1]

    # Text encodes using the (possibly LoRA-modified) CLIP
    g["t_enc"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_handle, "text": prompt}}
    g["t_neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_handle, "text": negative}}

    # Sample with the (possibly LoRA-modified) UNet
    g["ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_handle,
            "positive": ["t_enc", 0],
            "negative": ["t_neg", 0],
            "latent_image": ["enc", 0],
            "seed": int(seed),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": float(denoise),
        },
    }

    g["dec"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]},
    }
    g["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["dec", 0], "filename_prefix": "lexi"},
    }
    return g


def _graph_sdxl_inpaint(
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float,
    source_filename: str,  # uploaded to Comfy; use _upload_image_to_comfy
    mask_filename: str,  # uploaded to Comfy; white=paint, black=protect
    invert_mask: bool = False,  # if your mask is inverse, flip it here
) -> Dict[str, Any]:
    """
    SDXL inpaint:
      LoadImage(src) + LoadImageMask(mask)
      -> VAEEncode(src)
      -> SetLatentNoiseMask(mask) (optionally invert)
      -> KSampler(denoise<1, noise masked)
      -> VAEDecode -> SaveImage
    Recommended: denoise ~0.25–0.55 for targeted edits.
    """
    g: Dict[str, Any] = {}

    # Model / text encoders
    g["ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": BASE_CKPT}}
    g["t_pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": prompt}}
    g["t_neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": negative}}

    # Source image + mask
    g["img"] = {
        "class_type": "LoadImage",
        "inputs": {"image": source_filename, "choose file to upload": "image"},
    }
    g["mask"] = {
        "class_type": "LoadImageMask",
        "inputs": {"image": mask_filename, "choose file to upload": "image"},
    }

    # Encode to latent
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0], "vae": ["ckpt", 2]}}

    # Apply the mask as noise mask for inpainting
    # (white = editable area by default; invert if needed)
    g["noise_mask"] = {
        "class_type": "SetLatentNoiseMask",
        "inputs": {
            "samples": ["enc", 0],
            "mask": ["mask", 0],
            "invert": bool(invert_mask),
        },
    }

    # Sample only inside masked area
    g["ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["ckpt", 0],
            "positive": ["t_pos", 0],
            "negative": ["t_neg", 0],
            "latent_image": ["noise_mask", 0],
            "seed": int(seed),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": float(denoise),
        },
    }

    # Decode and save
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}}
    g["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["dec", 0], "filename_prefix": "lexi"},
    }

    return g


# ------------------------- Flux workflow builders -------------------------


def _flux_txt2img_graph(
    paths: FluxPaths,
    subject_prompt: str,
    style_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    guidance: float,
    seed: int,
    sampler: str,
    scheduler: str,
) -> Dict[str, Any]:
    """Build a Flux txt2img workflow graph for Comfy."""
    graph: Dict[str, Any] = {}

    graph["ckpt"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": _flux_ckpt_name(paths.ckpt)},
    }
    graph["vae"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": _flux_vae_name(paths.ae)},
    }
    graph["dual_clip"] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": _flux_clip_name(paths.clip_l),
            "clip_name2": _flux_clip_name(paths.t5xxl),
            "model": "flux",
            "weight": "default",
        },
        "widgets_values": [
            _flux_clip_name(paths.clip_l),
            _flux_clip_name(paths.t5xxl),
            "flux",
            "default",
        ],
    }
    graph["encode_positive"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": ["dual_clip", 0],
            "clip_l": subject_prompt,
            "t5xxl": style_prompt,
            "guidance": float(guidance),
        },
    }
    neg_string = negative_prompt or ""
    graph["encode_negative"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": ["dual_clip", 0],
            "clip_l": neg_string,
            "t5xxl": neg_string,
            "guidance": float(guidance),
        },
    }
    graph["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["ckpt", 0],
            "positive": ["encode_positive", 0],
            "negative": ["encode_negative", 0],
            "latent_image": ["latent", 0],
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "lexi"},
    }
    return graph


def _flux_img2img_graph(
    paths: FluxPaths,
    subject_prompt: str,
    style_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    guidance: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    source_filename: str,
) -> Dict[str, Any]:
    """Flux img2img preserving composition via VAEEncode."""
    graph: Dict[str, Any] = {}

    graph["ckpt"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": _flux_ckpt_name(paths.ckpt)},
    }
    graph["vae"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": _flux_vae_name(paths.ae)},
    }
    graph["dual_clip"] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": _flux_clip_name(paths.clip_l),
            "clip_name2": _flux_clip_name(paths.t5xxl),
            "model": "flux",
            "weight": "default",
        },
        "widgets_values": [
            _flux_clip_name(paths.clip_l),
            _flux_clip_name(paths.t5xxl),
            "flux",
            "default",
        ],
    }
    graph["encode_positive"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": ["dual_clip", 0],
            "clip_l": subject_prompt,
            "t5xxl": style_prompt,
            "guidance": float(guidance),
        },
    }
    neg_string = negative_prompt or ""
    graph["encode_negative"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": ["dual_clip", 0],
            "clip_l": neg_string,
            "t5xxl": neg_string,
            "guidance": float(guidance),
        },
    }
    graph["load"] = {
        "class_type": "LoadImage",
        "inputs": {"image": source_filename},
        "widgets_values": [source_filename, "image"],
    }
    graph["encode"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["load", 0], "vae": ["vae", 0]},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["ckpt", 0],
            "positive": ["encode_positive", 0],
            "negative": ["encode_negative", 0],
            "latent_image": ["encode", 0],
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": float(denoise),
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "lexi"},
    }
    return graph


def _run_flux_backend(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    mode: str,
    base_path: Path,
    force_output_to_base: bool,
    source_path: Optional[Path],
    variant: Optional[str],
    preset: Optional[str],
    size: Optional[str],
    guidance: Optional[float],
    sampler: Optional[str],
    scheduler: Optional[str],
    denoise: Optional[float],
    allow_feedback_loop: bool,
    public_base_url: str,
) -> Dict[str, Any]:
    paths = _flux_variant_paths(variant)

    # Canvas selection (override if preset provided)
    size_key = (size or "").strip().lower()
    canvas = FLUX_CANVAS_PRESETS.get(size_key)
    if canvas:
        width, height = canvas

    # Adjust defaults if still using SDXL parameters
    try:
        flux_steps = int(float(steps))
    except Exception:
        flux_steps = FLUX_DEFAULT_STEPS
    if flux_steps >= 28 and flux_steps in (28, 30):
        flux_steps = FLUX_DEFAULT_STEPS
    try:
        flux_cfg = float(cfg_scale)
    except Exception:
        flux_cfg = FLUX_DEFAULT_CFG
    if flux_cfg > 3.5:
        flux_cfg = FLUX_DEFAULT_CFG
    if guidance is not None:
        try:
            flux_guidance = float(guidance)
        except Exception:
            flux_guidance = FLUX_DEFAULT_GUIDANCE
    else:
        flux_guidance = FLUX_DEFAULT_GUIDANCE
    k_sampler = (sampler or FLUX_DEFAULT_SAMPLER) or FLUX_DEFAULT_SAMPLER
    k_scheduler = (scheduler or FLUX_DEFAULT_SCHEDULER) or FLUX_DEFAULT_SCHEDULER

    preset_cfg = FLUX_PRESETS.get((preset or "").strip().lower())
    subject_prompt = prompt
    style_hint = None
    if preset_cfg:
        subject_prompt = f"{preset_cfg.get('subject_prefix', '')}{subject_prompt}"
        style_hint = preset_cfg.get("style")

    subject_prompt, style_prompt = FluxPromptAdapter.split(subject_prompt, style_hint)
    negative_flux = FluxPromptAdapter.negatives(negative)

    if mode == "img2img":
        if source_path is None or not source_path.exists():
            raise ValueError(
                "img2img requested but no valid source image was provided for Flux backend."
            )
        if (
            _is_in(IMAGE_DIR, source_path)
            and source_path.name != base_path.name
            and not allow_feedback_loop
        ):
            raise ValueError(
                "Flux img2img refuses to reprocess a fresh avatar (non-base). "
                "Use the fixed base image or pass allow_feedback_loop=True."
            )

    # Build graph
    if mode == "txt2img":
        graph = _flux_txt2img_graph(
            paths=paths,
            subject_prompt=subject_prompt,
            style_prompt=style_prompt,
            negative_prompt=negative_flux,
            width=width,
            height=height,
            steps=flux_steps,
            cfg=flux_cfg,
            guidance=flux_guidance,
            seed=seed,
            sampler=k_sampler,
            scheduler=k_scheduler,
        )
    elif mode == "img2img":
        src_name = _upload_image_to_comfy(str(source_path))
        flux_denoise = denoise if denoise is not None else 0.5
        flux_denoise = float(min(max(flux_denoise, 0.10), 0.95))
        graph = _flux_img2img_graph(
            paths=paths,
            subject_prompt=subject_prompt,
            style_prompt=style_prompt,
            negative_prompt=negative_flux,
            seed=seed,
            steps=flux_steps,
            cfg=flux_cfg,
            guidance=flux_guidance,
            sampler=k_sampler,
            scheduler=k_scheduler,
            denoise=flux_denoise,
            source_filename=src_name,
        )
    else:
        raise ValueError(f"Flux backend does not support mode '{mode}'")

    pid = _post_graph(graph)
    images = _wait_for_images(pid)
    if not images:
        return {
            "ok": False,
            "error": "No images returned from ComfyUI (Flux backend)",
            "prompt_id": pid,
        }

    first = images[0]
    out = _download_image(
        first.get("filename", ""), first.get("subfolder", ""), first.get("type", "output")
    )
    final_out = _finalize_generated_image(out, base_path, force_output_to_base)

    base_url = public_base_url
    relative_url = f"{AVATAR_URL_PREFIX}/{final_out.name}"
    public = f"{base_url}{relative_url}" if base_url else relative_url
    return {
        "ok": True,
        "file": str(final_out),
        "url": public,
        "prompt_id": pid,
        "meta": {
            "backend": "flux",
            "variant": variant or DEFAULT_FLUX_VARIANT,
            "seed": seed,
            "mode": mode,
            "width": width,
            "height": height,
            "cfg": flux_cfg,
            "steps": flux_steps,
            "guidance": flux_guidance,
            "sampler": k_sampler,
            "scheduler": k_scheduler,
            "base_created": bool(force_output_to_base),
        },
    }


# ------------------------- Public API -------------------------
def _is_in(dirpath: Path, candidate: Path) -> bool:
    try:
        return str(candidate.resolve()).startswith(str(dirpath.resolve()))
    except Exception:
        return False


def generate_avatar_pipeline(
    prompt: Optional[str] = None,
    negative: Optional[str] = None,
    width: int = 832,  # SDXL sweet spot multiple of 64
    height: int = 1152,
    steps: int = 30,
    cfg_scale: float = 5.0,
    traits: Optional[Dict[str, str]] = None,
    mode: str = "txt2img",  # "txt2img" | "img2img" | "inpaint" (inpaint not wired here)
    source_path: Optional[str] = None,  # for img2img/inpaint
    mask_path: Optional[str] = None,  # for inpaint
    freeze_parts: bool = True,
    changes: Optional[str] = None,  # small delta description ("add skirt", "brown hair")
    seed: Optional[int] = None,
    refiner: bool = True,
    refiner_strength: float = 0.28,  # 0.2–0.35 typical
    upscale_factor: float = 1.0,  # 1.0 = off; 1.25–1.5 modest sharpen
    task: str = "general",  # <<< NEW: registry task key
    fresh_base: bool = False,  # force a new base via txt2img (ignore existing base)
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Unified pipeline entry. Accepts either a raw prompt or a traits dict.
    Returns: { ok, file, url, prompt_id, meta? } on success; { ok: False, error } on failure.
    """
    try:
        # Optional: fail fast if Comfy is unreachable
        try:
            r = requests.get(f"{COMFY_URL}/object_info", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Comfy unreachable at {COMFY_URL}: {e}")
        import uuid  # local import so this patch is self-contained

        # 0) Fixed identity base policy:
        #    - If avatars dir has no base yet, force a txt2img render to "lexi_base.png".
        #    - Otherwise always img2img using "lexi_base.png" as the source.
        base_name = "lexi_base.png"
        base_path = IMAGE_DIR / base_name
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        backend = (
            str(kwargs.get("backend") or kwargs.get("model") or SD_BACKEND or "sdxl")
            .strip()
            .lower()
        )
        if backend not in ("sdxl", "flux"):
            backend = "sdxl"

        if kwargs.get("flux_cfg") is not None:
            try:
                cfg_scale = float(kwargs["flux_cfg"])
            except Exception:
                pass
        if kwargs.get("flux_steps") is not None:
            try:
                steps = int(kwargs["flux_steps"])
            except Exception:
                pass

        # 1) Detect intent FIRST (use prompt if provided, otherwise traits)
        intent_text = (
            prompt
            or " ".join(
                [
                    str((traits or {}).get(k, ""))
                    for k in ("outfit", "style", "hair", "pose", "vibe")
                ]
            )
        ).lower()
        wants_outfit = any(k in intent_text for k in OUTFIT_KEYWORDS) or bool(
            traits and traits.get("outfit")
        )
        wants_hair = any(k in intent_text for k in HAIR_KEYWORDS)
        WANTS_FACE = any(t in intent_text for t in ["face", "makeup", "eyes", "nose", "lips"])
        WANTS_HAIR = wants_hair
        WANTS_BODY = any(t in intent_text for t in ["body", "curvy", "hourglass", "waist", "hips"])
        freeze_parts = bool(kwargs.get("freeze_parts", freeze_parts))
        # 1a) Build prompt (now we can safely choose composition using wants_outfit)
        if traits and not prompt:
            composition = "three_quarter" if wants_outfit else "portrait"
            prompt = _prompt_from_traits(traits, composition=composition)
        if not prompt:
            raise ValueError("Missing prompt and traits")
        style_pos, style_neg = _style_realistically_unreal()
        base_neg = _negative_prompt()
        neg_full = ", ".join(x for x in [negative, style_neg, base_neg] if x)

        # Loosen negatives for outfit edits so clothing isn't suppressed
        if wants_outfit:
            unblock = {"jeans", "shorts", "denim", "t-shirt", "tee", "crop top", "skirt", "dress"}
            toks = [t.strip() for t in neg_full.split(",") if t.strip().lower() not in unblock]
            neg_full = ", ".join(toks)

        # Lingerie/leather often cause melted clothing artifacts – counter them
        p_low = (prompt or "").lower()
        if kwargs.get("nsfw") or any(
            x in p_low for x in ("lingerie", "thigh-high", "stockings", "nsfw")
        ):
            unblock = {
                "latex",
                "shiny clothing",
                "stockings",
                "thigh-highs",
                "sheer",
                "see-through",
            }
            toks = [t.strip() for t in neg_full.split(",") if t.strip().lower() not in unblock]
            neg_full = ", ".join(toks)

        if any(x in p_low for x in ("leather", "latex", "corset")):
            neg_full = ", ".join(
                [
                    neg_full,
                    "melted clothing, warped leather, lumpy clothing, asymmetric chest, garment merging, broken seams",
                ]
            )

        # Lightly enrich LLM-provided prompt with traits and add style profile
        prompt = _augment_prompt_with_traits(prompt, traits)
        prompt = f"{prompt}, {style_pos}" if style_pos not in prompt else prompt

        # Apply tiny edit suffix for img2img/inpaint
        if changes and mode in ("img2img", "inpaint"):
            prompt = f"{prompt}, {changes}"

        # 1.5) Registry selection (base/refiner/loras & default knobs)
        if backend != "flux":
            if REG is not None:
                sel = REG.select(task, overrides=kwargs.get("model_overrides"))
            else:
                # Fallback: use env/default checkpoint names directly
                class _FallbackSel:
                    base_file = BASE_CKPT
                    refiner_file = REFINER_CKPT
                    loras: list = []
                    cfg = None
                    steps = None
                    refiner_strength = None
                    denoise = None
                    variation = None

                sel = _FallbackSel()
            # Allow registry to set better defaults if caller left them at their defaults:
            # Apply defaults from registry if the caller left the standard defaults in place
            if steps == 30 and sel.steps is not None:
                steps = int(sel.steps)
            if cfg_scale == 5.0 and sel.cfg is not None:
                cfg_scale = float(sel.cfg)
            # New: apply default denoise if caller didn’t set it (img2img/inpaint only)
            if "denoise" not in kwargs and sel.denoise is not None:
                kwargs["denoise"] = float(sel.denoise)

            # New: optional variation profile
            if sel.variation:
                import random

                seed_mode = (sel.variation.get("seed_mode") or "").lower()
                jitter = sel.variation.get("jitter") or {}

                def _jit(val, delta):
                    return (
                        val
                        if delta in (None, 0)
                        else (val + random.uniform(-float(delta), float(delta)))
                    )

                if seed_mode == "random":
                    seed = random.randint(0, 2**32 - 1)
                elif seed_mode == "semi-fixed":
                    # nudge identity seed slightly
                    seed = (int(seed) ^ random.getrandbits(12)) & 0xFFFFFFFF

                cfg_scale = _jit(cfg_scale, jitter.get("cfg"))
                steps = int(max(1, round(_jit(steps, jitter.get("steps")))))
                if "denoise" in kwargs:
                    kwargs["denoise"] = max(
                        0.0, min(1.0, _jit(float(kwargs["denoise"]), jitter.get("denoise")))
                    )
        else:
            sel = None

        # 3) Seed (continuity) — always coerce to a 32-bit int
        # For img2img edits keep composition by default (no outfit salt);
        # allow salt only for fresh txt2img or explicit strong variation.
        add_salt = not (mode == "img2img")
        seed = _coerce_seed(seed, traits, add_outfit_salt=add_salt)

        # 2.5) Determine base-or-iteration policy with explicit control
        #   - If fresh_base=True OR base doesn't exist -> txt2img and write lexi_base.png
        #   - If base exists and caller explicitly asks for txt2img -> treat as fresh base (overwrite)
        #   - If base exists and caller asks for img2img -> use base as source
        #   - Otherwise keep the requested mode (defaults to txt2img)
        force_output_to_base = False
        sp: Optional[Path] = Path(source_path).resolve() if source_path else None

        if fresh_base or not base_path.exists():
            mode = "txt2img"
            sp = None
            force_output_to_base = True
        else:
            if mode == "txt2img":
                # Explicit fresh render; overwrite the fixed base
                sp = None
                force_output_to_base = True
            elif mode == "img2img":
                sp = base_path

        if backend == "flux":
            flux_variant = kwargs.get("flux_variant") or kwargs.get("variant")
            flux_preset = kwargs.get("flux_preset") or kwargs.get("preset")
            flux_size = kwargs.get("flux_size") or kwargs.get("size")
            flux_guidance = kwargs.get("flux_guidance") or kwargs.get("guidance")
            flux_sampler = kwargs.get("flux_sampler") or kwargs.get("sampler")
            flux_scheduler = kwargs.get("flux_scheduler") or kwargs.get("scheduler")
            flux_denoise = (
                kwargs.get("flux_denoise")
                if kwargs.get("flux_denoise") is not None
                else kwargs.get("denoise")
            )
            log.info(
                "[Lexi SD][flux] mode=%s seed=%s variant=%s steps=%d cfg=%.2f guidance=%.2f width=%d height=%d",
                mode,
                seed,
                (flux_variant or DEFAULT_FLUX_VARIANT),
                steps,
                cfg_scale,
                (flux_guidance or FLUX_DEFAULT_GUIDANCE),
                width,
                height,
            )
            result = _run_flux_backend(
                prompt=prompt,
                negative=neg_full,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                mode=mode,
                base_path=base_path,
                force_output_to_base=force_output_to_base,
                source_path=sp,
                variant=flux_variant,
                preset=flux_preset,
                size=flux_size,
                guidance=flux_guidance,
                sampler=flux_sampler,
                scheduler=flux_scheduler,
                denoise=flux_denoise,
                allow_feedback_loop=bool(kwargs.get("allow_feedback_loop", False)),
                public_base_url=PUBLIC_BASE_URL,
            )
            return result

        # ---- Log selection (one-liner for quick debugging)
        try:
            base_file = sel.base_file if sel else "auto"
            ref_file = sel.refiner_file if (sel and sel.refiner_file) else "-"
            lora_names = [x[0] for x in (sel.loras if sel else [])]
        except Exception:
            base_file, ref_file, lora_names = "auto", "-", []
        log.info(
            "[Lexi SD] task=%s mode=%s seed=%s base=%s refiner=%s denoise=%s cfg=%.2f steps=%d loras=%s",
            task,
            mode,
            seed,
            base_file,
            ref_file,
            (kwargs.get("denoise") if mode == "img2img" else "-"),
            cfg_scale,
            steps,
            lora_names,
        )

        # 3) Build graph
        style = kwargs.get("style", "realistic")
        nsfw = bool(kwargs.get("nsfw", False))

        if mode == "txt2img":
            # First-time base render → use FACE workflow (portrait with refiner chain).
            flow = _use_face_workflow(
                prompt=prompt,
                negative=neg_full,
                w=width,
                h=height,
                steps=steps,
                cfg=cfg_scale,
                seed=seed,
                refiner_strength=refiner_strength,
                upscale_factor=upscale_factor,
                base_file=sel.base_file,
                refiner_file=sel.refiner_file,
            )
            # Ensure image sink exists so /history reports an image
            try:
                if not _has_image_sink(flow):
                    _attach_save_if_missing(flow)
            except Exception:
                pass

        elif mode == "img2img":
            if not sp or not sp.is_file():
                raise ValueError(
                    "img2img requested but no valid source image was provided (expected lexi_base.png)."
                )

            # Safety: allow reprocessing the fixed base, but block non-base outputs unless explicitly allowed.
            if (
                _is_in(IMAGE_DIR, sp)
                and sp.name != base_name
                and not bool(kwargs.get("allow_feedback_loop", False))
            ):
                raise ValueError(
                    "Refusing to reprocess a freshly generated avatar (non-base). Use lexi_base.png or pass allow_feedback_loop=True."
                )

            # Unique temp copy to defeat Comfy's upload cache
            tmp_copy = IMAGE_DIR / f"lexi_base_{int(time.time()*1000)}.png"
            shutil.copy(sp, tmp_copy)
            src_name = _upload_image_to_comfy(str(tmp_copy))  # <-- use tmp_copy (not sp)

            # Choose denoise ONLY for img2img
            intent = (kwargs.get("intent") or "medium").lower()  # light|medium|strong
            # More conservative defaults to preserve form; raise manually for big restyles
            denoise_default = {"light": 0.30, "medium": 0.40, "strong": 0.50}.get(intent, 0.40)
            denoise = float(kwargs.get("denoise", denoise_default))
            denoise = min(max(denoise, 0.10), 0.55)

            # Outfit/pose edits → use BODY workflow (OpenPose + IP-Adapter FaceID).
            flow = _use_body_workflow(
                prompt=prompt,
                negative=neg_full,
                w=width,
                h=height,
                steps=max(steps, 24),
                cfg=cfg_scale,
                seed=seed,
                denoise=denoise,
                src_uploaded_name=src_name,
                base_file=sel.base_file,
                refiner_file=sel.refiner_file,
                upscale_factor=upscale_factor,
            )
            # Ensure image sink exists so /history reports an image
            try:
                if not _has_image_sink(flow):
                    _attach_save_if_missing(flow)
            except Exception:
                pass

        elif mode == "inpaint":
            if not source_path or not mask_path:
                raise ValueError("inpaint requires source_path and mask_path")

            # Upload both to Comfy
            src_name = _upload_image_to_comfy(source_path)
            mask_name = _upload_image_to_comfy(mask_path)

            # Denoise for inpaint
            denoise = float(kwargs.get("denoise", 0.45))
            denoise = min(max(denoise, 0.10), 0.85)

            graph = _graph_sdxl_inpaint(
                prompt=prompt,
                negative=neg_full,
                steps=max(steps, 24),
                cfg=cfg_scale,
                seed=seed,
                denoise=denoise,
                source_filename=src_name,
                mask_filename=mask_name,
                invert_mask=bool(kwargs.get("invert_mask", False)),
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 4) Execute (graph or workflow)
        pid = _post_workflow(flow) if mode in ("txt2img", "img2img") else _post_graph(graph)
        images = _wait_for_images(pid)
        if not images:
            return {"ok": False, "error": "No images returned from ComfyUI", "prompt_id": pid}

        first = images[0]
        out = _download_image(
            first.get("filename", ""), first.get("subfolder", ""), first.get("type", "output")
        )

        final_out = _finalize_generated_image(out, base_path, force_output_to_base)

        base_url = PUBLIC_BASE_URL
        relative_url = f"{AVATAR_URL_PREFIX}/{final_out.name}"
        public = f"{base_url}{relative_url}" if base_url else relative_url
        return {
            "ok": True,
            "file": str(final_out),
            "url": public,
            "prompt_id": pid,
            "meta": {
                "seed": seed,
                "mode": mode,
                "refined": bool(refiner),
                "width": width,
                "height": height,
                "cfg": cfg_scale,
                "steps": steps,
                "base_created": bool(force_output_to_base),
                "backend": backend,
                "source_used": str(sp) if sp else None,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------- Upload helper -------------------------


def _upload_image_to_comfy(path: str) -> str:
    """
    Upload a local file to Comfy's input via /upload/image.
    Returns the filename registered in Comfy (use in LoadImage).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"source image not found: {p}")
    with p.open("rb") as f:
        files = {"image": (p.name, f, "image/png")}
        r = requests.post(f"{COMFY_URL}/upload/image", files=files, timeout=60)
        r.raise_for_status()
    # Comfy registers the original filename in its input; using p.name in LoadImage works.
    return p.name
