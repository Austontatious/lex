# ── Lexi/lexi/sd/sd_pipeline.py ────────────────────────────────────────
"""
ComfyUI-backed FLUX helpers for avatar generation.

Supported modes:
  - txt2img (default): fresh render seeded from persona traits for continuity
  - img2img: low-denoise edits to preserve identity/look using an existing base render

Continuity:
  - If no seed is provided, we derive a stable "identity seed" from traits so that the face/look
    stays consistent between sessions. You can still override the seed per call.

Env:
  COMFY_URL         (default: http://host.docker.internal:8188)
  LEX_IMAGE_DIR     (default: <repo>/frontend/public/avatars)
  FLUX_* overrides for checkpoint/text encoder/VAE paths (see configuration section below)
"""

from __future__ import annotations
import random
import base64
import hashlib
import os
import json
import time
import uuid
import logging
import threading
import re

log = logging.getLogger("lexi.sd")

BASE_IMAGE_LOCK = threading.Lock()
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_VALIDATED = False
_WARMUP_DONE = False
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from PIL import Image, ImageOps
from ..config.config import AVATAR_DIR, AVATAR_URL_PREFIX
from ..config.runtime_env import (
    BASE_MODELS_DIR,
    COMFY_ROOT,
    COMFY_URL,
    COMFY_WORKSPACE_DIR,
    resolve as resolve_model_path,
)
from .sd_prompt_styles import MODE_PRESETS
from .comfy_client import comfy_flux_generate
from .flux_defaults import FLUX_DEFAULTS

# ------------------------- Config/Paths -------------------------

PUBLIC_BASE_URL = os.getenv("LEX_PUBLIC_BASE_URL", "").rstrip("/")

LEX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = AVATAR_DIR
IMAGE_DIR = Path(os.getenv("LEX_IMAGE_DIR", str(DEFAULT_IMAGE_DIR)))
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_AVATAR_DIR = Path(os.getenv("AVATARS_PUBLIC_DIR", str(IMAGE_DIR)))
PUBLIC_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
AV_PUBLIC_DIR = PUBLIC_AVATAR_DIR
LEGACY_BASENAME = "lexi_base.png"
DEFAULT_BASE_KEY = LEGACY_BASENAME.rsplit(".", 1)[0]
PORTRAIT_EXPORT_SIZE = (1080, 1350)


def _sanitize_filename_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_-]+", "_", (value or "").strip())
    return token or "lexi"

FLUX_LORA_NAME = (os.getenv("LEX_FLUX_LORA_NAME") or "lexiverse_hybrid_v1.safetensors").strip()
_DEFAULT_LORA_PATH = (
    Path(os.getenv("LEX_FLUX_LORA_PATH"))
    if os.getenv("LEX_FLUX_LORA_PATH")
    else (Path("/mnt/data/comfy/models/loras") / FLUX_LORA_NAME)
)
FLUX_LORA_MODEL_STRENGTH = float(os.getenv("LEX_FLUX_LORA_UNET", "0.65"))
FLUX_LORA_CLIP_STRENGTH = float(os.getenv("LEX_FLUX_LORA_CLIP", "0.35"))
_FLUX_LORA_ENABLED = bool(FLUX_LORA_NAME and _DEFAULT_LORA_PATH and _DEFAULT_LORA_PATH.exists())


def _base_image_path(base_name: str) -> Path:
    """Resolve a base avatar filename under the public avatars directory."""
    base = (base_name or DEFAULT_BASE_KEY).strip() or DEFAULT_BASE_KEY
    filename = base if base.lower().endswith(".png") else f"{base}.png"
    return AV_PUBLIC_DIR / filename

# Ensure the default avatar exists inside the writable avatar directory. When the
# docker volume mounts an empty host folder, seed it with the repo asset.
_DEFAULT_AVATAR_SRC = (
    LEX_ROOT.parent.parent / "assets" / "default.png"
)
_DEFAULT_AVATAR_DST = IMAGE_DIR / "default.png"
if _DEFAULT_AVATAR_SRC.exists() and not _DEFAULT_AVATAR_DST.exists():
    try:
        _DEFAULT_AVATAR_DST.write_bytes(_DEFAULT_AVATAR_SRC.read_bytes())
    except Exception as exc:  # pragma: no cover - best effort log
        log.warning("[Lexi SD] failed to seed default avatar: %s", exc)


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve optional path overrides relative to BASE_MODELS_DIR."""
    value = os.getenv(name)
    if value:
        return resolve_model_path(value)
    return default


# ------------------------- Flux configuration -------------------------
DEFAULT_FLUX_VARIANT = os.getenv("FLUX_MODEL_VARIANT", "kontext-dev").strip().lower()

# Flux model paths
FLUX_MODELS_DIR = _path_from_env("FLUX_MODELS_DIR", COMFY_ROOT / "models")
FLUX_DIFFUSION_DIR = _path_from_env("FLUX_DIFFUSION_DIR", FLUX_MODELS_DIR / "diffusion_models")
FLUX_TEXT_ENCODER_DIR = _path_from_env("FLUX_TEXT_ENCODER_DIR", FLUX_MODELS_DIR / "clip")
FLUX_VAE_PATH = _path_from_env("FLUX_VAE_PATH", FLUX_MODELS_DIR / "vae" / "ae.safetensors")
FLUX_CLIP_L = _path_from_env("FLUX_CLIP_L", FLUX_TEXT_ENCODER_DIR / "clip_l.safetensors")
FLUX_T5XXL = _path_from_env(
    "FLUX_T5XXL", FLUX_MODELS_DIR / "text_encoders" / "t5xxl_fp8_e4m3fn.safetensors"
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

_SCHEMA_EXPECTATIONS = {
    "DualCLIPLoader": {"type"},
    "CheckpointLoaderSimple": {"ckpt_name"},
    "VAELoader": {"vae_name"},
}


def _validate_comfy_schema() -> bool:
    """Fetch Comfy object_info and ensure expected nodes/inputs exist."""
    global _SCHEMA_VALIDATED
    with _SCHEMA_LOCK:
        if _SCHEMA_VALIDATED:
            return True
        try:
            resp = requests.get(f"{COMFY_URL}/object_info", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            missing_entries = []
            for node_name, required_keys in _SCHEMA_EXPECTATIONS.items():
                required_inputs = (
                    data.get(node_name, {})
                    .get("input", {})
                    .get("required", {})
                )
                missing = [key for key in required_keys if key not in required_inputs]
                if missing:
                    missing_entries.append(f"{node_name}({', '.join(missing)})")
            if missing_entries:
                log.warning(
                    "[Lexi SD] Comfy schema missing expected inputs: %s",
                    ", ".join(missing_entries),
                )
            _SCHEMA_VALIDATED = True
        except Exception as exc:  # pragma: no cover - network safety
            log.warning("[Lexi SD] Comfy schema validation failed: %s", exc)
        return _SCHEMA_VALIDATED


def ensure_comfy_schema() -> bool:
    return _validate_comfy_schema()


def _warmup_once() -> None:
    """Lightweight warmup to exercise the Comfy HTTP path once per process."""
    global _WARMUP_DONE
    if _WARMUP_DONE:
        return
    if str(os.getenv("LEX_SKIP_COMFY_WARMUP", "0")).lower() in ("1", "true", "yes"):
        _WARMUP_DONE = True
        return
    if not _validate_comfy_schema():
        return
    try:
        requests.get(f"{COMFY_URL}/system_stats", timeout=3)
    except Exception:
        pass
    _WARMUP_DONE = True
    log.debug("[Lexi SD] Comfy warmup completed")


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


# ---- Flux model selectors ----------------------------------------------------
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


def _relname_under(root: Path, filename: Optional[str]) -> str:
    """
    Return the relative path Comfy expects for a given filename.
    If the file exists directly under root, returns 'filename'.
    If it's in a subfolder, returns that relative path.
    """
    if not filename:
        raise ValueError("Missing filename while resolving Comfy asset path")

    target = root / filename
    if target.exists():
        return filename
    candidates = list(root.rglob(filename))
    if candidates:
        rel = min((c.relative_to(root) for c in candidates), key=lambda p: len(str(p)))
        return str(rel).replace("\\", "/")
    return filename


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


def _apply_flux_lora(
    graph: Dict[str, Any],
    model_ref: list,
    clip_ref: list,
    *,
    prefix: str,
) -> tuple[list, list]:
    """
    Optionally attach a LoraLoader node for the Flux graph.
    Returns updated (model, clip) references to feed downstream nodes.
    """
    if not _FLUX_LORA_ENABLED:
        return model_ref, clip_ref
    node_name = f"{prefix}_flux_lora"
    graph[node_name] = {
        "class_type": "LoraLoader",
        "inputs": {
            "model": model_ref,
            "clip": clip_ref,
            "lora_name": FLUX_LORA_NAME,
            "strength_model": FLUX_LORA_MODEL_STRENGTH,
            "strength_clip": FLUX_LORA_CLIP_STRENGTH,
        },
    }
    return [node_name, 0], [node_name, 1]


def _flux_relname(root: Path, path: Path) -> str:
    return _relname_under(root, path.name if path.is_absolute() else str(path))


def _flux_ckpt_name(path: Path) -> str:
    return _flux_relname(FLUX_DIFFUSION_DIR, path)


def _flux_clip_name(path: Path) -> str:
    return _flux_relname(FLUX_TEXT_ENCODER_DIR, path)


def _flux_vae_name(path: Path) -> str:
    return _flux_relname(FLUX_MODELS_DIR / "vae", path)


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





# ------------------------- Prompt helpers -------------------------


def _prompt_from_traits(traits: Dict[str, str], composition: str = "portrait") -> str:
    comp = {
        "portrait": "portrait of Lexi, shoulders-up, engaging eye contact",
        "three_quarter": "3/4 body shot of Lexi, standing, outfit fully visible, natural stance",
        "full_body": "full-body shot of Lexi, outfit fully visible, natural stance",
    }.get(composition, "portrait of Lexi, shoulders-up, engaging eye contact")
    style_stub = (
        "cinematic portrait lighting, neutral color grading, editorial clarity, natural skin texture"
    )
    pieces = [comp, style_stub]

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
    payload = {"prompt": graph, "client_id": "lexi"}
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            r = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=60)
            r.raise_for_status()
            j = (
                r.json()
                if r.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            return j.get("prompt_id") or j.get("id") or ""
        except requests.RequestException as exc:  # pragma: no cover - network safety
            last_error = exc
            log.warning(
                "[Lexi SD] Comfy prompt submission failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
            if attempt == 0:
                time.sleep(0.5 + random.random())
            else:
                raise
    if last_error:
        raise last_error
    raise RuntimeError("Comfy prompt submission failed unexpectedly")


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
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for Comfy prompt {prompt_id}")


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


def normalize_portrait_image(
    image_path: Path, size: tuple[int, int] = PORTRAIT_EXPORT_SIZE
) -> Path:
    """
    Downscale + crop final renders to a consistent 4:5 portrait (1080x1350) for downstream use.
    """
    try:
        with Image.open(image_path) as img:
            fitted = ImageOps.fit(
                img.convert("RGB"),
                size,
                Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            fitted.save(image_path, format="PNG")
    except Exception as exc:  # pragma: no cover - defensive log
        log.warning("[Lexi SD] portrait normalization failed for %s: %s", image_path, exc)
    return image_path


def _finalize_generated_image(out: Path, base_path: Path, force_output_to_base: bool) -> Path:
    """
    Apply Lexi-specific naming policy:
      - Base renders overwrite lexi_base.png
      - Iterations always produce unique lexi_XXXX.png files.
    """
    base_name = base_path.name
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with BASE_IMAGE_LOCK:
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


# ------------------------- Flux workflow builders -------------------------


def _flux_txt2img_graph(
    paths: FluxPaths,
    subject_prompt: str,
    style_prompt: str,
    negative_clip: str,
    negative_t5: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    guidance_pos: float,
    guidance_neg: float,
    seed: int,
    sampler: str,
    scheduler: str,
    denoise: float,
    filename_prefix: str,
    upscale_width: int,
    upscale_height: int,
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
            "type": "flux",
        },
        "widgets_values": [
            _flux_clip_name(paths.clip_l),
            _flux_clip_name(paths.t5xxl),
            "flux",
        ],
    }
    model_ref = ["ckpt", 0]
    clip_ref = ["dual_clip", 0]
    model_ref, clip_ref = _apply_flux_lora(graph, model_ref, clip_ref, prefix="txt2img")

    graph["encode_positive"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": clip_ref,
            "clip_l": subject_prompt,
            "t5xxl": style_prompt,
            "guidance": float(guidance_pos),
        },
    }
    graph["encode_negative"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": clip_ref,
            "clip_l": negative_clip or "",
            "t5xxl": negative_t5 or "",
            "guidance": float(guidance_neg),
        },
    }
    graph["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["encode_positive", 0],
            "negative": ["encode_negative", 0],
            "latent_image": ["latent", 0],
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
    graph["scale"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["decode", 0],
            "upscale_method": "bilinear",
            "width": int(upscale_width),
            "height": int(upscale_height),
            "crop": "center",
        },
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["scale", 0], "filename_prefix": filename_prefix or "lexi"},
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
    filename_prefix: str,
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
            "type": "flux",
        },
        "widgets_values": [
            _flux_clip_name(paths.clip_l),
            _flux_clip_name(paths.t5xxl),
            "flux",
        ],
    }
    model_ref = ["ckpt", 0]
    clip_ref = ["dual_clip", 0]
    model_ref, clip_ref = _apply_flux_lora(graph, model_ref, clip_ref, prefix="img2img")
    graph["encode_positive"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": clip_ref,
            "clip_l": subject_prompt,
            "t5xxl": style_prompt,
            "guidance": float(guidance),
        },
    }
    neg_string = negative_prompt or ""
    graph["encode_negative"] = {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {
            "clip": clip_ref,
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
            "model": model_ref,
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
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix or "lexi"},
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
    positive_clip: Optional[str] = None,
    positive_t5: Optional[str] = None,
    negative_clip: Optional[str] = None,
    negative_t5: Optional[str] = None,
) -> Dict[str, Any]:
    # Canvas selection (override if preset provided)
    size_key = (size or "").strip().lower()
    canvas = FLUX_CANVAS_PRESETS.get(size_key)
    if canvas:
        width, height = canvas

    width = int(width or FLUX_DEFAULTS["width"])
    height = int(height or FLUX_DEFAULTS["height"])

    try:
        flux_steps = int(float(steps or FLUX_DEFAULTS["steps"]))
    except Exception:
        flux_steps = FLUX_DEFAULTS["steps"]
    try:
        flux_cfg = float(cfg_scale or FLUX_DEFAULTS["cfg"])
    except Exception:
        flux_cfg = FLUX_DEFAULTS["cfg"]
    flux_guidance_pos = (
        float(guidance)
        if guidance is not None
        else float(FLUX_DEFAULTS["guidance_pos"])
    )
    flux_guidance_neg = float(FLUX_DEFAULTS["guidance_neg"])
    k_sampler = (sampler or FLUX_DEFAULTS["sampler"]) or FLUX_DEFAULTS["sampler"]
    k_scheduler = (scheduler or FLUX_DEFAULTS["scheduler"]) or FLUX_DEFAULTS["scheduler"]
    flux_denoise = (
        float(denoise)
        if denoise is not None
        else float(FLUX_DEFAULTS["denoise"])
    )
    upscale_width = int(FLUX_DEFAULTS["upscale_w"])
    upscale_height = int(FLUX_DEFAULTS["upscale_h"])

    preset_cfg = FLUX_PRESETS.get((preset or "").strip().lower())
    preset_style_hint = preset_cfg.get("style") if preset_cfg else None
    preset_subject_prefix = preset_cfg.get("subject_prefix", "") if preset_cfg else ""

    subject_prompt = positive_clip or prompt
    style_prompt = positive_t5 or ""
    if not subject_prompt:
        subject_prompt = prompt
    if preset_subject_prefix:
        subject_prompt = f"{preset_subject_prefix}{subject_prompt}"
    if not style_prompt:
        style_hint = preset_style_hint
        subject_prompt, style_prompt = FluxPromptAdapter.split(subject_prompt, style_hint)
    elif preset_style_hint:
        style_prompt = ", ".join([p for p in (style_prompt, preset_style_hint) if p])

    if negative_clip and negative_t5:
        negative_flux_clip = negative_clip
        negative_flux_t5 = negative_t5
    else:
        negative_flux = FluxPromptAdapter.negatives(negative)
        negative_flux_clip = negative_flux
        negative_flux_t5 = negative_flux

    paths = _flux_variant_paths(variant)
    if mode == "img2img":
        if not source_path:
            raise ValueError("img2img requested without source image")
        source_filename = _upload_image_to_comfy(str(source_path))
        # Default to gentler denoise for identity-preserving edits
        if denoise is None:
            flux_denoise = 0.35
        else:
            flux_denoise = max(0.10, min(float(flux_denoise), 0.75))
        graph = _flux_img2img_graph(
            paths=paths,
            subject_prompt=subject_prompt,
            style_prompt=style_prompt,
            negative_prompt=negative_flux_clip,
            seed=seed,
            steps=flux_steps,
            cfg=flux_cfg,
            guidance=flux_guidance_pos,
            sampler=k_sampler,
            scheduler=k_scheduler,
            denoise=flux_denoise,
            source_filename=source_filename,
            filename_prefix=_sanitize_filename_token(base_path.stem),
        )
    else:
        graph = _flux_txt2img_graph(
            paths=paths,
            subject_prompt=subject_prompt,
            style_prompt=style_prompt,
            negative_clip=negative_flux_clip,
            negative_t5=negative_flux_t5,
            width=width,
            height=height,
            steps=flux_steps,
            cfg=flux_cfg,
            guidance_pos=flux_guidance_pos,
            guidance_neg=flux_guidance_neg,
            seed=seed,
            sampler=k_sampler,
            scheduler=k_scheduler,
            denoise=flux_denoise,
            filename_prefix=_sanitize_filename_token(base_path.stem),
            upscale_width=upscale_width,
            upscale_height=upscale_height,
        )

    start_time = time.time()
    try:
        pid = _post_graph(graph)
        if not pid:
            raise RuntimeError("Comfy response missing prompt_id")
    except Exception as exc:  # pragma: no cover - network safety
        log.error("[Lexi SD][flux] prompt submission failed: %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_PROMPT_ERROR"}

    try:
        images = _wait_for_images(pid)
    except TimeoutError as exc:
        log.error("[Lexi SD][flux] %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_TIMEOUT", "prompt_id": pid}
    except Exception as exc:  # pragma: no cover - network safety
        log.error("[Lexi SD][flux] history polling failed: %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_HISTORY_ERROR", "prompt_id": pid}

    first = images[0]
    out = _download_image(
        first.get("filename", ""), first.get("subfolder", ""), first.get("type", "output")
    )
    normalize_portrait_image(out)
    final_out = _finalize_generated_image(out, base_path, force_output_to_base)

    base_url = public_base_url
    relative_url = f"{AVATAR_URL_PREFIX}/{final_out.name}"
    public = f"{base_url}{relative_url}" if base_url else relative_url
    try:
        mtime = int(final_out.stat().st_mtime)
    except Exception:
        mtime = int(time.time())
    sep = "&" if "?" in public else "?"
    public_busted = f"{public}{sep}v={mtime}"
    elapsed = time.time() - start_time
    log.info(
        "[Lexi SD][flux] completed prompt_id=%s in %.2fs (mode=%s, seed=%s)",
        pid,
        elapsed,
        mode,
        seed,
    )
    return {
        "ok": True,
        "file": str(final_out),
        "url": public_busted,
        "avatar_url": public_busted,
        "mtime": mtime,
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
            "guidance": flux_guidance_pos,
            "sampler": k_sampler,
            "scheduler": k_scheduler,
            "base_created": bool(force_output_to_base),
            "public_path": relative_url,
            "mtime": mtime,
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
    width: int = FLUX_DEFAULTS["width"],  # Flux default canvas (multiple of 64)
    height: int = FLUX_DEFAULTS["height"],
    steps: int = FLUX_DEFAULTS["steps"],
    cfg_scale: float = FLUX_DEFAULTS["cfg"],
    traits: Optional[Dict[str, str]] = None,
    mode: str = "txt2img",  # "txt2img" | "img2img"
    source_path: Optional[str] = None,  # for img2img
    mask_path: Optional[str] = None,  # legacy: ignored in flux-only pipeline
    freeze_parts: bool = True,
    changes: Optional[str] = None,  # small delta description ("add skirt", "brown hair")
    seed: Optional[int] = None,
    refiner: bool = True,
    refiner_strength: float = 0.28,  # 0.2–0.35 typical
    upscale_factor: float = 1.0,  # 1.0 = off; 1.25–1.5 modest sharpen
    task: str = "general",  # <<< NEW: registry task key
    fresh_base: bool = False,  # force a new base via txt2img (ignore existing base)
    base_name: Optional[str] = None,
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
        _validate_comfy_schema()
        _warmup_once()
        # 0) Fixed identity base policy:
        #    - If avatars dir has no base yet, force a txt2img render to "lexi_base.png".
        #    - Otherwise always img2img using the per-IP base asset.
        base_label = (base_name or DEFAULT_BASE_KEY).strip() or DEFAULT_BASE_KEY
        base_path = _base_image_path(base_label)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_missing = not base_path.exists()

        # Seed the fixed base with the default avatar if the volume mounts empty.
        if base_missing and _DEFAULT_AVATAR_DST.exists():
            try:
                base_path.write_bytes(_DEFAULT_AVATAR_DST.read_bytes())
            except Exception:
                pass
        base_exists = base_path.exists()

        requested_backend = (
            str(kwargs.get("backend") or kwargs.get("model") or "flux")
            .strip()
            .lower()
        )
        if requested_backend and requested_backend not in ("", "flux"):
            raise ValueError(
                f"Flux pipeline only supports backend='flux' (got '{requested_backend}')"
            )

        mode = (mode or "txt2img").strip().lower()

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
        if traits and not prompt:
            composition = "three_quarter" if wants_outfit else "portrait"
            prompt = _prompt_from_traits(traits, composition=composition)
        if not prompt:
            raise ValueError("Missing prompt and traits")

        prompt = _augment_prompt_with_traits(prompt, traits)
        if changes and mode == "img2img":
            prompt = f"{prompt}, {changes}"

        style_delta = str((traits or {}).get("style", "") or "")
        outfit_desc = str((traits or {}).get("outfit", "") or "")
        mood_desc = str((traits or {}).get("vibe", "") or "")
        env_desc = str((traits or {}).get("background", "") or "")

        # Build compact Flux prompts following BFL guidance: subject first, then style/context.
        style_parts = [style_delta, outfit_desc, mood_desc, env_desc]
        style_hint = ", ".join([p.strip() for p in style_parts if p.strip()])
        subject_prompt, style_prompt = FluxPromptAdapter.split(prompt, style_hint or None)

        # Single consolidated negative prompt for both encoders
        neg_full = FluxPromptAdapter.negatives(negative)

        # Loosen negatives for outfit edits so clothing isn't suppressed
        if wants_outfit:
            unblock = {"jeans", "shorts", "denim", "t-shirt", "tee", "crop top", "skirt", "dress"}
            toks = [
                t.strip()
                for t in neg_full.replace(";", ",").split(",")
                if t.strip().lower() not in unblock
            ]
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
            toks = [
                t.strip()
                for t in neg_full.replace(";", ",").split(",")
                if t.strip().lower() not in unblock
            ]
            neg_full = ", ".join(toks)

        if any(x in p_low for x in ("leather", "latex", "corset")):
            neg_full = ", ".join(
                [
                    neg_full,
                    "melted clothing, warped leather, lumpy clothing, asymmetric chest, garment merging, broken seams",
                ]
            )

        positive_clip = subject_prompt
        positive_t5 = style_prompt
        negative_clip = neg_full
        negative_t5 = neg_full

        # 3) Seed (continuity) — always coerce to a 32-bit int
        # For img2img edits keep composition by default (no outfit salt);
        # allow salt only for fresh txt2img or explicit strong variation.
        add_salt = mode != "img2img"
        seed = _coerce_seed(seed, traits, add_outfit_salt=add_salt)

        # 2.5) Determine base-or-iteration policy:
        #   - If fresh_base=True OR the per-IP base is missing -> txt2img, write the base image.
        #   - Otherwise default to img2img using either the caller's source or the stored base.
        force_output_to_base = False
        mode = (mode or "txt2img").strip().lower()
        sp: Optional[Path] = Path(source_path).expanduser().resolve() if source_path else None

        if fresh_base or base_missing:
            mode = "txt2img"
            force_output_to_base = True
        else:
            candidate = sp or base_path
            if candidate.exists():
                mode = "img2img"
                sp = candidate
            else:
                # Fall back to txt2img (refresh base) if source is missing
                mode = "txt2img"
                force_output_to_base = True

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
            positive_clip=positive_clip,
            positive_t5=positive_t5,
            negative_clip=negative_clip,
            negative_t5=negative_t5,
        )
        if isinstance(result, dict) and result.get("url") and not result.get("avatar_url"):
            result["avatar_url"] = result["url"]
        return result

    except Exception as e:  # pragma: no cover - defensive
        log.exception("[Lexi SD] pipeline error: %s", e)
        http_status = getattr(getattr(e, "response", None), "status_code", None)
        err_code = http_status or e.__class__.__name__
        return {"ok": False, "error": str(e), "code": err_code}


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
