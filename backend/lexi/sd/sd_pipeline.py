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
# DEPRECATED FOR AVATAR RUNTIME (Dec 2025):
# The live avatar route currently uses Fusion workflow templates via comfy_client.py
# rather than these programmatic graph builders. Keep this module for reference/future
# reactivation; changes here will not affect the active avatar pipeline until routing
# is switched back to use these graphs.

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
log.setLevel(logging.INFO)

BASE_IMAGE_LOCK = threading.Lock()
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_VALIDATED = False
_WARMUP_DONE = False
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from PIL import Image, ImageOps, ImageDraw
from ..config.config import AVATAR_DIR, AVATAR_URL_PREFIX
from ..config.runtime_env import (
    BASE_MODELS_DIR,
    COMFY_ROOT,
    COMFY_URL,
    COMFY_WORKSPACE_DIR,
    resolve as resolve_model_path,
)
from .sd_prompt_styles import MODE_PRESETS
from .comfy_client import (
    comfy_flux_generate,
    comfy_flux_generate_v2,
    comfy_flux_generate_img2img_v10,
)
from .pose_selector import PoseChoice, choose_pose
from ..utils.pose_camera_intent import classify_pose_camera
from .flux_defaults import FLUX_DEFAULTS
from .flux_prompt_builder import (
    DEFAULT_NEGATIVE_PROMPT,
    BASE_LEXIVERSE_STYLE,
    AvatarTraits,
    StyleFlags,
    PoseMeta,
    build_flux_avatar_prompt_bundle,
)

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

# Small helper for logging prompt snippets without newlines.
def _truncate(text: Optional[str], limit: int = 300) -> str:
    return (text or "").replace("\n", " ").strip()[:limit]

# Disable LoRA by default while debugging styling drift.
FLUX_LORA_NAME = os.getenv("FLUX_LORA_NAME", "lexiverse_hybrid_v1.safetensors").strip()
_DEFAULT_LORA_PATH = None
FLUX_LORA_MODEL_STRENGTH = float(os.getenv("FLUX_LORA_MODEL_STRENGTH", "1.0"))
FLUX_LORA_CLIP_STRENGTH = float(os.getenv("FLUX_LORA_CLIP_STRENGTH", "1.0"))
_FLUX_LORA_ENABLED = bool(FLUX_LORA_NAME)


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
_FALLBACK_PIXEL = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ofuIAAAAASUVORK5CYII="
)


def _ensure_default_avatar_files() -> None:
    """Make sure default + base avatars exist for static serving."""
    AV_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        _DEFAULT_AVATAR_DST,
        AV_PUBLIC_DIR / "default.png",
        AV_PUBLIC_DIR / LEGACY_BASENAME,
    ]
    src_bytes: Optional[bytes] = None
    if _DEFAULT_AVATAR_SRC.exists():
        try:
            src_bytes = _DEFAULT_AVATAR_SRC.read_bytes()
        except Exception as exc:  # pragma: no cover
            log.warning("[Lexi SD] failed reading default avatar: %s", exc)
    if src_bytes is None:
        src_bytes = _FALLBACK_PIXEL

    for target in targets:
        try:
            if not target.exists() or target.stat().st_size == 0:
                target.write_bytes(src_bytes)
                log.info("[Lexi SD] seeded default avatar at %s", target)
        except Exception as exc:  # pragma: no cover
            log.warning("[Lexi SD] failed to seed default avatar %s: %s", target, exc)


_ensure_default_avatar_files()


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
# Use the whole models tree as a search root so both clip/ and text_encoders/ are discovered.
FLUX_TEXT_ENCODER_DIR = _path_from_env("FLUX_TEXT_ENCODER_DIR", FLUX_MODELS_DIR)
FLUX_VAE_PATH = _path_from_env("FLUX_VAE_PATH", FLUX_MODELS_DIR / "vae" / "ae.safetensors")
FLUX_CLIP_L = _path_from_env("FLUX_CLIP_L", FLUX_MODELS_DIR / "clip" / "clip_l.safetensors")
FLUX_T5XXL = _path_from_env(
    "FLUX_T5XXL", FLUX_MODELS_DIR / "text_encoders" / "t5xxl_fp8_e4m3fn.safetensors"
)

# Pose-control assets (optional)
_DEFAULT_POSE_ROOT = Path(
    "/data/flux_pose_assets/1000BoudoirGlamShot_boudoirGlamShotPoses"
)
if not _DEFAULT_POSE_ROOT.exists():
    _DEFAULT_POSE_ROOT = (
        LEX_ROOT.parent.parent
        / "data"
        / "flux_pose_assets"
        / "1000BoudoirGlamShot_boudoirGlamShotPoses"
    )
POSE_BUCKETS_CSV = Path(
    os.getenv("LEX_POSE_BUCKETS_CSV", str(_DEFAULT_POSE_ROOT / "boudoir_pose_buckets.csv"))
)
POSE_RENDER_DIR = Path(
    os.getenv("LEX_POSE_RENDER_DIR", str(_DEFAULT_POSE_ROOT / "BoudoirOutputWorking"))
)
POSE_CONTROLNET_NAME = os.getenv(
    "LEX_POSE_CONTROLNET_NAME", "FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8.safetensors"
).strip()
POSE_CONTROLNET_PATH = _path_from_env(
    "LEX_POSE_CONTROLNET_PATH", FLUX_MODELS_DIR / "controlnet" / POSE_CONTROLNET_NAME
)
# Fallback: if the primary path is missing, try the commonly mounted /mnt/data/comfy location.
if not POSE_CONTROLNET_PATH.exists():
    alt_controlnet = Path("/mnt/data/comfy/models/controlnet") / POSE_CONTROLNET_NAME
    if alt_controlnet.exists():
        POSE_CONTROLNET_PATH = alt_controlnet
        log.info("[Lexi SD] Using fallback pose ControlNet at %s", POSE_CONTROLNET_PATH)
POSE_CONTROLNET_AVAILABLE = POSE_CONTROLNET_PATH.exists()
POSE_CONTROL_STRENGTH = float(os.getenv("LEX_POSE_CONTROL_STRENGTH", "0.1"))
POSE_CONTROL_START = float(os.getenv("LEX_POSE_CONTROL_START", "0.0"))
POSE_CONTROL_END = float(os.getenv("LEX_POSE_CONTROL_END", "0.95"))
DISABLE_POSE_CONTROL = str(os.getenv("LEX_DISABLE_POSE_CONTROL", "0")).lower() in (
    "1",
    "true",
    "yes",
)
_cn_enabled_env = os.getenv("LEXI_CONTROLNET_ENABLED")
if _cn_enabled_env is None:
    _cn_enabled_env = os.getenv("FLUX_CONTROLNET_ENABLED", "1")
FLUX_CONTROLNET_ENABLED = str(_cn_enabled_env).lower() in ("1", "true", "yes")
_cn_strength_env = os.getenv("LEXI_CONTROLNET_STRENGTH")
if _cn_strength_env is None:
    _cn_strength_env = os.getenv("FLUX_CONTROLNET_STRENGTH", "0.55")
FLUX_CONTROLNET_STRENGTH = float(_cn_strength_env)
FLUX_CONTROLNET_START_PERCENT = float(os.getenv("FLUX_CONTROLNET_START_PERCENT", "0.0"))
FLUX_CONTROLNET_END_PERCENT = float(os.getenv("FLUX_CONTROLNET_END_PERCENT", "0.8"))

FLUX_DEFAULT_GUIDANCE = float(os.getenv("FLUX_GUIDANCE_DEFAULT", "3.5"))
FLUX_DEFAULT_CFG = float(os.getenv("FLUX_CFG_DEFAULT", "1.8"))
FLUX_DEFAULT_STEPS = int(os.getenv("FLUX_STEPS_DEFAULT", "16"))
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
    "UNETLoader": {"unet_name"},
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


def _flux_controlnet_name(path: Path) -> str:
    return _relname_under(FLUX_MODELS_DIR / "controlnet", path.name if path.is_absolute() else str(path))


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



# ------------------------- Pose map generation -------------------------

# Body-25 style skeleton pairs (OpenPose ordering: 0–24)
_POSE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
]


def _pose_map_from_keypoints(json_path: Path, canvas: tuple[int, int] = (1024, 1344)) -> Optional[Path]:
    """
    Convert OpenPose keypoints JSON into a simple stick-figure PNG for ControlNet.
    Returns the cached PNG path, or None on failure.
    """
    try:
        if not json_path.exists():
            return None
        out = json_path.with_name(json_path.stem.replace("_keypoints", "") + "_pose.png")
        if out.exists() and out.stat().st_mtime >= json_path.stat().st_mtime:
            return out

        data = json.loads(json_path.read_text())
        people = data.get("people") or []
        if not people or not isinstance(people[0], dict):
            return None
        raw = people[0].get("pose_keypoints_2d") or []
        if not raw:
            return None

        pts = []
        for i in range(0, len(raw), 3):
            x, y, c = raw[i : i + 3]
            pts.append((float(x), float(y), float(c)))

        # Accept points even if confidence is low; only drop points that are literally empty
        valid = [(x, y) for x, y, c in pts if c > 0.05 or (x != 0 or y != 0)]
        if not valid:
            return None

        xs, ys = zip(*valid)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        width, height = canvas
        margin = 80
        scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
        if not (scale > 0):
            scale = 1.0

        def project(pt):
            x, y, c = pt
            if c <= 0.0 and (x == 0 and y == 0):
                return None
            px = (x - min_x) * scale + (width - scale * span_x) / 2
            py = (y - min_y) * scale + (height - scale * span_y) / 2
            return (px, py)

        projected: list[Optional[tuple[float, float]]] = [project(p) for p in pts]
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        bone_color = (0, 255, 200)
        joint_color = (255, 120, 80)

        for a, b in _POSE_BONES:
            pa = projected[a] if a < len(projected) else None
            pb = projected[b] if b < len(projected) else None
            if pa and pb:
                draw.line([pa, pb], fill=bone_color, width=10)
        for pt in projected:
            if pt:
                r = 8
                draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=joint_color, outline=joint_color)

        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="PNG")
        return out
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[Lexi SD] failed to build pose map from %s: %s", json_path, exc)
        return None


def _pose_aspect(json_path: Path) -> Optional[float]:
    """Return aspect ratio (height/width) of pose keypoints, or None on failure."""
    try:
        data = json.loads(Path(json_path).read_text())
        people = data.get("people") or []
        if not people or not isinstance(people[0], dict):
            return None
        raw = people[0].get("pose_keypoints_2d") or []
        pts = []
        for i in range(0, len(raw), 3):
            x, y, c = raw[i : i + 3]
            pts.append((float(x), float(y), float(c)))
        valid = [(x, y) for x, y, c in pts if c > 0.05 or (x != 0 or y != 0)]
        if not valid:
            return None
        xs, ys = zip(*valid)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x <= 0:
            return None
        return span_y / span_x
    except Exception:
        return None



# ------------------------- Pose helpers -------------------------

POSE_FEEL_KEYWORDS = {
    "playful": ("playful", "fun", "flirty", "sassy"),
    "seductive": ("seductive", "intimate", "sensual", "kneeling", "lounge"),
    "cozy": ("cozy", "warm", "relaxed", "soft", "cuddle", "snuggle"),
    "confident": ("confident", "model", "powerful", "dominant", "strong"),
}


def _infer_pose_feel(traits: Optional[Dict[str, str]], intent_text: str) -> Optional[str]:
    text = (intent_text or "").lower()
    for feel, keywords in POSE_FEEL_KEYWORDS.items():
        if any(k in text for k in keywords):
            return feel
    for feel, keywords in POSE_FEEL_KEYWORDS.items():
        if traits:
            for v in traits.values():
                if isinstance(v, str) and any(k in v.lower() for k in keywords):
                    return feel
    return None


def _maybe_pick_pose_map(
    *,
    traits: Optional[Dict[str, str]],
    kwargs: Dict[str, Any],
    intent_text: str,
    seed: int,
) -> Optional[PoseChoice]:
    """
    Select a pose render to drive ControlNet. Returns None if assets are missing.
    """
    if not (POSE_BUCKETS_CSV.exists() and POSE_RENDER_DIR.exists()):
        return None

    classification = classify_pose_camera(intent_text)
    intent_lower = (intent_text or "").lower()
    pose_id = (
        kwargs.get("pose_id")
        or (traits or {}).get("pose_id")
        or kwargs.get("pose_map")
        or (traits or {}).get("pose_map")
    )
    shape_bucket = kwargs.get("pose_bucket") or (traits or {}).get("pose_bucket") or classification.pose_bucket
    camera_bucket = kwargs.get("camera_bucket") or (traits or {}).get("camera_bucket") or classification.camera_bucket
    feel = kwargs.get("pose_feel") or (traits or {}).get("pose_feel") or classification.pose_feel
    # Keyword overrides: "sitting"/"couch" → seated bucket; "seductive"/"sultry" → low-ish camera
    sitting_intent = any(k in intent_lower for k in ("sit", "sitting", "couch", "sofa", "loung"))
    aspect_cutoff = 1.2 if sitting_intent else None
    if sitting_intent:
        shape_bucket = "POSE_3"
    if not camera_bucket and any(k in intent_lower for k in ("seductive", "seduct", "sultry", "sensual")):
        camera_bucket = "CAM_SLIGHT_LOW"
    inferred_feel = _infer_pose_feel(traits, intent_text)
    if not pose_id:
        if inferred_feel and inferred_feel != feel:
            feel = inferred_feel
        if not feel and classification.camera_feel:
            feel = classification.camera_feel

    try:
        return choose_pose(
            csv_path=POSE_BUCKETS_CSV,
            render_dir=POSE_RENDER_DIR,
            feel=feel,
            shape_bucket=shape_bucket,
            camera_bucket=camera_bucket,
            pose_id=pose_id,
            rng_seed=seed,
            predicate=(lambda r: (_pose_aspect(r.keypoints_path) or 999) <= aspect_cutoff) if aspect_cutoff else None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[Lexi SD] pose selection failed: %s", exc)
        return None


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
    missing_class_nodes = [name for name, node in graph.items() if not isinstance(node, dict) or "class_type" not in node]
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
            body = None
            if exc.response is not None:
                try:
                    body = exc.response.text[:1000]
                except Exception:
                    body = None
            if missing_class_nodes:
                log.warning(
                    "[Lexi SD] Graph nodes missing class_type: %s",
                    ", ".join(missing_class_nodes),
                )
            log.warning(
                "[Lexi SD] Comfy prompt submission failed (attempt %d): %s | body=%r",
                attempt + 1,
                exc,
                body,
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


COMFY_TIMEOUT_S = int(os.getenv("LEXI_COMFY_TIMEOUT_S", "95"))


def _wait_for_images(prompt_id: str, timeout_s: int = COMFY_TIMEOUT_S) -> List[Dict[str, Any]]:
    start = time.time()
    # Poll /history/<id> and only accept images for this exact prompt_id
    while time.time() - start < timeout_s:
        hr = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=12)
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


def probe_prompt_images(prompt_id: str) -> List[Dict[str, Any]]:
    """
    Lightweight, single-shot probe of Comfy history for a prompt_id.
    Returns an images list if ready, otherwise [].
    """
    try:
        hr = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=8)
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
    except Exception as exc:  # pragma: no cover
        log.warning("[Lexi SD] probe_prompt_images failed for %s: %s", prompt_id, exc)
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
    pose_image: Optional[str] = None,
    controlnet_name: Optional[str] = None,
    control_strength: float = POSE_CONTROL_STRENGTH,
    control_start: float = POSE_CONTROL_START,
    control_end: float = POSE_CONTROL_END,
) -> Dict[str, Any]:
    """Build a Flux txt2img workflow graph for Comfy."""
    graph: Dict[str, Any] = {}

    graph["unet"] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": _flux_ckpt_name(paths.ckpt),
            "weight_dtype": "default",
        },
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
    model_ref = ["unet", 0]
    clip_ref = ["dual_clip", 0]
    model_ref, clip_ref = _apply_flux_lora(graph, model_ref, clip_ref, prefix="txt2img")

    positive_ref: list = ["encode_positive", 0]
    negative_ref: list = ["encode_negative", 0]

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

    if pose_image and controlnet_name:
        graph["pose_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image},
            "widgets_values": [pose_image, "image"],
        }
        graph["pose_controlnet"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_name},
        }
        graph["pose_controlnet_apply"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["encode_positive", 0],
                "negative": ["encode_negative", 0],
                "control_net": ["pose_controlnet", 0],
                "image": ["pose_image", 0],
                "vae": ["vae", 0],
                "strength": float(control_strength),
                "start_percent": float(control_start),
                "end_percent": float(control_end),
            },
            "widgets_values": [
                float(control_strength),
                float(control_start),
                float(control_end),
            ],
        }
        positive_ref = ["pose_controlnet_apply", 0]
        negative_ref = ["pose_controlnet_apply", 1]

    graph["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": positive_ref,
            "negative": negative_ref,
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

    graph["unet"] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": _flux_ckpt_name(paths.ckpt),
            "weight_dtype": "default",
        },
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
    model_ref = ["unet", 0]
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
    pose_map_path: Optional[Path] = None,
    pose_control_strength: Optional[float] = None,
    pose_control_start: Optional[float] = None,
    pose_control_end: Optional[float] = None,
    return_on_submit: bool = False,
    traits: Optional[Dict[str, str]] = None,
    controlnet_enabled: bool = False,
    user_text_combined: Optional[str] = None,
    pose_info: Optional[Dict[str, Any]] = None,
    controlnet_strength: Optional[float] = None,
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

    if negative_clip is not None or negative_t5 is not None:
        negative_flux_clip = negative_clip or ""
        negative_flux_t5 = negative_t5 if negative_t5 is not None else negative_flux_clip
    else:
        negative_flux = negative or ""
        negative_flux_clip = negative_flux
        negative_flux_t5 = negative_flux

    pose_filename = None
    controlnet_name = None
    pose_strength = (
        (FLUX_CONTROLNET_STRENGTH if controlnet_strength is None else float(controlnet_strength))
        if controlnet_enabled
        else 0.0
    )
    pose_strength = max(0.0, min(1.0, pose_strength))
    pose_start = (
        FLUX_CONTROLNET_START_PERCENT
        if pose_control_start is None
        else float(pose_control_start)
    )
    pose_end = (
        FLUX_CONTROLNET_END_PERCENT
        if pose_control_end is None
        else float(pose_control_end)
    )
    if DISABLE_POSE_CONTROL or not FLUX_CONTROLNET_ENABLED or not controlnet_enabled:
        pose_map_path = None

    log.info(
        "[lexi][flux prompt bundle] pos=%s neg=%s traits=%s controlnet_enabled=%s controlnet_strength=%.2f pose_id=%s pose_shape=%s pose_camera=%s steps=%s sampler=%s cfg=%.2f extra=%s lora=%s m=%.2f c=%.2f aux_required=%s",
        _truncate(positive_clip),
        _truncate(negative_clip),
        traits,
        controlnet_enabled,
        pose_strength if controlnet_enabled else 0.0,
        (pose_info or {}).get("pose_id"),
        (pose_info or {}).get("shape_bucket"),
        (pose_info or {}).get("camera_bucket"),
        steps,
        sampler or (k_sampler if "k_sampler" in locals() else None),
        cfg_scale,
        (user_text_combined or "").strip(),
        FLUX_LORA_NAME if _FLUX_LORA_ENABLED else "none",
        FLUX_LORA_MODEL_STRENGTH,
        FLUX_LORA_CLIP_STRENGTH,
        bool(controlnet_enabled and pose_map_path),
    )

    if pose_map_path:
        pose_strength = max(0.0, min(float(pose_strength), 1.0))
        if pose_strength <= 0:
            log.info("[Lexi SD][flux] pose control disabled (strength<=0); skipping ControlNet")
            pose_map_path = None
        else:
            log.info(
                "[Lexi SD][flux] pose map enabled: %s strength=%.3f start=%.2f end=%.2f",
                pose_map_path.name if hasattr(pose_map_path, "name") else pose_map_path,
                pose_strength,
                pose_start,
                pose_end,
            )
    else:
        log.info("[Lexi SD][flux] pose control disabled or no map available")
    if pose_map_path:
        try:
            pose_filename = _upload_image_to_comfy(str(pose_map_path))
            controlnet_name = _flux_controlnet_name(POSE_CONTROLNET_PATH)
        except Exception as exc:  # pragma: no cover - network safety
            pose_filename = None
            controlnet_name = None
            log.warning("[Lexi SD][flux] failed to upload pose map %s: %s", pose_map_path, exc)

    paths = _flux_variant_paths(variant)
    if mode == "img2img":
        if not source_path:
            raise ValueError("img2img requested without source image")
        source_filename = _upload_image_to_comfy(str(source_path))
        # Default to gentler denoise for identity-preserving edits
        if denoise is None:
            flux_denoise = 0.55
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
            pose_image=pose_filename,
            controlnet_name=controlnet_name,
            control_strength=pose_strength,
            control_start=pose_start,
            control_end=pose_end,
        )

    start_time = time.time()
    try:
        pid = _post_graph(graph)
        if not pid:
            raise RuntimeError("Comfy response missing prompt_id")
        log.info(
            "[Lexi SD][flux_v2] submitted prompt_id=%s seed=%s size=%dx%d steps=%d cfg=%.2f sampler=%s scheduler=%s",
            pid,
            seed,
            width,
            height,
            steps,
            flux_cfg,
            "dpmpp_2m",
            "sgm_uniform",
        )
        if return_on_submit:
            return {"ok": True, "status": "running", "prompt_id": pid}
    except Exception as exc:  # pragma: no cover - network safety
        import traceback
        log.error("[Lexi SD][flux] prompt submission failed: %s", exc)
        log.error("[Lexi SD][flux] submission traceback: %s", traceback.format_exc())
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
            "pose_control": bool(pose_filename and controlnet_name),
            "pose_image": pose_map_path.name if pose_map_path else None,
            "pose_strength": pose_strength if pose_filename else None,
        },
    }


# ------------------------- Public API -------------------------
def _lexiverse_strengths(preset: str) -> tuple[float, float]:
    preset = (preset or "soft").strip().lower()
    if preset == "off":
        return 0.0, 0.0
    # Lexiverse hybrid only; skirt LoRA disabled
    return 1.0, 0.0


def _run_flux_backend_v2(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    base_path: Path,
    force_output_to_base: bool,
    public_base_url: str,
    lexiverse_style: str,
) -> Dict[str, Any]:
    """
    Minimal txt2img wrapper around the v2 Fusion-style workflow (ModelSamplingFlux).
    """
    lora_main, lora_skirt = _lexiverse_strengths(lexiverse_style)
    # Single prompt string (style baked in) – FluxGuidance sits inside the graph.
    positive = prompt
    filename_prefix = _sanitize_filename_token(base_path.stem)

    log.info(
        "[Lexi SD][flux_v2] seed=%s width=%d height=%d lora_main=%.2f lora_skirt=%.2f",
        seed,
        width,
        height,
        lora_main,
        lora_skirt,
    )
    try:
        resp = comfy_flux_generate_v2(
            prompt=positive,
            negative_prompt=negative,
            seed=int(seed),
            width=int(width),
            height=int(height),
            steps=24,
            sampler_name="dpmpp_2m",
            scheduler="sgm_uniform",
            denoise=1.0,
            max_shift=1.15,
            base_shift=0.5,
            guidance=3.0,
            lora_main=lora_main,
            lora_main_clip=lora_main * 0.5,
            lora_skirt=lora_skirt,
            lora_skirt_clip=lora_skirt * 0.5,
            filename_prefix=filename_prefix,
        )
    except Exception as exc:  # pragma: no cover - network path
        log.error("[Lexi SD][flux_v2] prompt submission failed: %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_PROMPT_ERROR"}

    pid = resp.get("prompt_id") or resp.get("id")
    if not pid:
        return {"ok": False, "error": "Comfy response missing prompt_id", "code": "COMFY_PROMPT_ERROR"}

    start_time = time.time()
    try:
        images = _wait_for_images(pid)
    except TimeoutError as exc:
        log.error("[Lexi SD][flux_v2] %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_TIMEOUT", "prompt_id": pid}
    except Exception as exc:  # pragma: no cover - network safety
        log.error("[Lexi SD][flux_v2] history polling failed: %s", exc)
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
        "[Lexi SD][flux_v2] completed prompt_id=%s in %.2fs seed=%s",
        pid,
        elapsed,
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
            "backend": "flux_v2",
            "seed": seed,
            "mode": "txt2img",
            "width": width,
            "height": height,
            "steps": 35,
            "guidance": 3.2,
            "sampler": "dpmpp_2m",
            "scheduler": "sgm_uniform",
            "base_created": bool(force_output_to_base),
            "public_path": relative_url,
            "mtime": mtime,
            "lora_main": lora_main,
            "lora_skirt": lora_skirt,
        },
    }


def _run_flux_backend_v10_img2img(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    base_path: Path,
    force_output_to_base: bool,
    public_base_url: str,
    source_filename: str,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    guidance: float = 2.5,
) -> Dict[str, Any]:
    """
    Img2img wrapper around the official Flux Kontext v10 workflow (simplified graph).
    """
    filename_prefix = _sanitize_filename_token(base_path.stem)
    log.info(
        "[Lexi SD][flux_v10_img2img] seed=%s width=%d height=%d steps=%d cfg=%.2f sampler=%s",
        seed,
        width,
        height,
        steps,
        cfg,
        sampler,
    )
    try:
        resp = comfy_flux_generate_img2img_v10(
            prompt=prompt,
            source_image=source_filename,
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler,
            scheduler=scheduler,
            denoise=float(denoise),
            guidance=float(guidance),
            filename_prefix=filename_prefix,
        )
    except Exception as exc:  # pragma: no cover - network path
        log.error("[Lexi SD][flux_v10_img2img] prompt submission failed: %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_PROMPT_ERROR"}

    pid = resp.get("prompt_id") or resp.get("id")
    if not pid:
        return {"ok": False, "error": "Comfy response missing prompt_id", "code": "COMFY_PROMPT_ERROR"}

    start_time = time.time()
    log.info(
        "[Lexi SD][flux_v10_img2img] submitted prompt_id=%s source=%s seed=%s size=%dx%d steps=%d cfg=%.2f sampler=%s scheduler=%s denoise=%.2f",
        pid,
        source_filename,
        seed,
        width,
        height,
        steps,
        cfg,
        sampler,
        scheduler,
        denoise,
    )
    try:
        images = _wait_for_images(pid)
    except TimeoutError as exc:
        log.error("[Lexi SD][flux_v10_img2img] %s", exc)
        return {"ok": False, "error": str(exc), "code": "COMFY_TIMEOUT", "prompt_id": pid}
    except Exception as exc:  # pragma: no cover - network safety
        log.error("[Lexi SD][flux_v10_img2img] history polling failed: %s", exc)
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
        "[Lexi SD][flux_v10_img2img] completed prompt_id=%s in %.2fs seed=%s",
        pid,
        elapsed,
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
            "backend": "flux_v10_img2img",
            "seed": seed,
            "mode": "img2img",
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "sampler": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
            "base_created": bool(force_output_to_base),
            "public_path": relative_url,
            "mtime": mtime,
        },
    }


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
    return_on_submit: bool = False,  # if True, return prompt_id immediately and let status poll
    disable_controlnet: bool = False,
    controlnet_enabled: Optional[bool] = None,
    controlnet_strength: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Unified pipeline entry. Accepts either a raw prompt or a traits dict.
    Returns: { ok, file, url, prompt_id, meta? } on success; { ok: False, error } on failure.
    """
    try:
        # Optional: soft health check — log but do not fail if it errors.
        try:
            r = requests.get(f"{COMFY_URL}/object_info", timeout=3)
            r.raise_for_status()
        except Exception as e:
            log.warning("[Lexi SD] Comfy warmup/health check failed at %s: %s", COMFY_URL, e)

        _ensure_default_avatar_files()
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

        # Hard-disable img2img for now: force txt2img workflow for all avatar renders.
        mode = "txt2img"

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

        intent_text = (
            (prompt or "")
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

        # 1) Build canonical prompts from traits/user text (Lexiverse always on)
        user_text_combined = ", ".join([p for p in (prompt, changes) if p])
        bundle = build_flux_avatar_prompt_bundle(
            traits=traits or {},
            style_flags={"lexiverse_style": "promo", "lexiverse_enabled": True},
            pose_meta=None,
            user_text=user_text_combined,
        )
        built_prompt = bundle["positive"]
        neg_full = (bundle.get("negative") or "").strip()
        # Ensure both CLIP and T5 receive the full semantic prompt for adherence.
        positive_clip = built_prompt
        positive_t5 = built_prompt
        negative_clip = neg_full
        negative_t5 = neg_full

        def _truncate(txt: Optional[str], limit: int = 300) -> str:
            return (txt or "").replace("\n", " ").strip()[:limit]

        # 3) Seed (continuity) — always coerce to a 32-bit int
        # For img2img edits keep composition by default (no outfit salt);
        # allow salt only for fresh txt2img or explicit strong variation.
        add_salt = mode != "img2img"
        seed = _coerce_seed(seed, traits, add_outfit_salt=add_salt)

        # ControlNet enablement (global + request override)
        controlnet_enabled = (
            FLUX_CONTROLNET_ENABLED
            if controlnet_enabled is None
            else bool(controlnet_enabled)
        )
        controlnet_enabled = (
            controlnet_enabled
            and POSE_CONTROLNET_AVAILABLE
            and not DISABLE_POSE_CONTROL
            and not disable_controlnet
        )
        if not POSE_CONTROLNET_AVAILABLE and controlnet_enabled:
            log.warning("[Lexi SD][flux] controlnet asset missing; disabling CN for this run.")
            controlnet_enabled = False

        cn_strength = (
            FLUX_CONTROLNET_STRENGTH if controlnet_strength is None else float(controlnet_strength)
        )
        cn_strength = max(0.0, min(1.0, cn_strength))
        if not controlnet_enabled:
            cn_strength = 0.0

        pose_choice: Optional[PoseChoice] = None
        pose_map_path: Optional[Path] = None
        try:
            pose_choice = _maybe_pick_pose_map(
                traits=traits, kwargs=kwargs, intent_text=intent_text, seed=seed
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[Lexi SD] pose selection failed: %s", exc)
        if pose_choice and controlnet_enabled:
            target_canvas = (int(width or FLUX_DEFAULTS["width"]), int(height or FLUX_DEFAULTS["height"]))
            pose_map_path = _pose_map_from_keypoints(pose_choice.keypoints_path, target_canvas)
            if pose_map_path:
                log.info(
                    "[Lexi SD] pose map=%s shape=%s camera=%s",
                    pose_choice.pose_id,
                    pose_choice.shape_bucket,
                    pose_choice.camera_bucket,
                )
            else:
                log.info(
                    "[Lexi SD] pose keypoints missing for %s; skipping ControlNet",
                    pose_choice.pose_id,
                )
        else:
            log.info("[Lexi SD] no pose choice selected; skipping ControlNet")
        # Fallback: if no pose map was built, try a best-effort random pose so ControlNet is still attached.
        if (
            controlnet_enabled
            and pose_map_path is None
            and POSE_BUCKETS_CSV.exists()
            and POSE_RENDER_DIR.exists()
        ):
            try:
                fallback_choice = choose_pose(
                    csv_path=POSE_BUCKETS_CSV,
                    render_dir=POSE_RENDER_DIR,
                    rng_seed=seed,
                )
                if fallback_choice:
                    target_canvas = (int(width or FLUX_DEFAULTS["width"]), int(height or FLUX_DEFAULTS["height"]))
                    pose_map_path = _pose_map_from_keypoints(fallback_choice.keypoints_path, target_canvas)
                    if pose_map_path:
                        log.info(
                            "[Lexi SD] fallback pose map=%s shape=%s camera=%s",
                            fallback_choice.pose_id,
                            fallback_choice.shape_bucket,
                            fallback_choice.camera_bucket,
                        )
            except Exception as exc:  # pragma: no cover
                log.warning("[Lexi SD] fallback pose selection failed: %s", exc)

        # 2.5) Disable img2img: always run txt2img and refresh the base output.
        force_output_to_base = True
        sp: Optional[Path] = None

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
        # txt2img should always run a full denoise pass; keep strength only for img2img edits
        if mode == "txt2img":
            flux_denoise = float(FLUX_DEFAULTS["denoise"])
        if kwargs.get("pose_strength") is not None:
            pose_strength = float(kwargs.get("pose_strength"))
        elif kwargs.get("pose_control_strength") is not None:
            pose_strength = float(kwargs.get("pose_control_strength"))
        else:
            pose_strength = cn_strength if controlnet_enabled else None
        pose_start = kwargs.get("pose_control_start")
        pose_end = kwargs.get("pose_control_end")
        flux_pipeline = (kwargs.get("flux_pipeline") or "flux_v1").strip().lower()
        lexiverse_style = (kwargs.get("lexiverse_style") or "soft").strip().lower()
        if flux_pipeline != "flux_v1":
            log.info("[Lexi SD] forcing flux pipeline to flux_v1 (got %s)", flux_pipeline)
            flux_pipeline = "flux_v1"

        try:
            flux_steps = int(float(kwargs.get("flux_steps") or steps or FLUX_DEFAULTS["steps"]))
        except Exception:
            flux_steps = FLUX_DEFAULTS["steps"]
        try:
            flux_cfg = float(kwargs.get("flux_cfg") or cfg_scale or FLUX_DEFAULTS["cfg"])
        except Exception:
            flux_cfg = FLUX_DEFAULTS["cfg"]
        guidance_val = float(flux_guidance) if flux_guidance is not None else 2.5
        k_sampler = (flux_sampler or FLUX_DEFAULTS["sampler"]) or FLUX_DEFAULTS["sampler"]
        k_scheduler = (flux_scheduler or FLUX_DEFAULTS["scheduler"]) or FLUX_DEFAULTS["scheduler"]
        if flux_denoise is None:
            flux_denoise = float(FLUX_DEFAULTS["denoise"])
        # CN can over-constrain; keep denoise gentler when CN is active.
        # Full denoise for txt2img even when ControlNet is on; do not clamp.

        log.info(
            "[Lexi SD] pipeline=%s mode=%s width=%d height=%d seed=%s source=%s base_missing=%s",
            flux_pipeline,
            mode,
            width,
            height,
            seed,
            (sp if sp else None),
            base_missing,
        )
        log.info("[lexi][flux] POSITIVE_CLIP=%s", positive_clip or prompt or "")
        log.info("[lexi][flux] POSITIVE_T5=%s", positive_t5 or prompt or "")
        log.info("[lexi][flux] NEGATIVE_CLIP=%s", negative_clip or negative or "")
        log.info("[lexi][flux] NEGATIVE_T5=%s", negative_t5 or negative_clip or negative or "")
        log.info(
            "[lexi][flux cn] enabled=%s strength=%.2f seed=%s pos=%s neg=%s",
            controlnet_enabled,
            cn_strength,
            seed,
            _truncate(positive_clip)[:120],
            _truncate(negative_clip)[:120],
        )
        log.info(
            "[Lexi SD] final clip_prompt=%r t5_prompt=%r negative=%r",
            (positive_clip or prompt or "")[:200],
            (positive_t5 or "")[:200],
            (negative_clip or negative or "")[:200],
        )
        log.info(
            "[Lexi SD PROMPT] clip=%r | t5=%r | neg=%r | pose=%r s=%r start=%r end=%r traits=%r",
            positive_clip or prompt,
            positive_t5,
            negative_clip or negative,
            pose_map_path.name if pose_map_path else None,
            pose_strength,
            pose_start,
            pose_end,
            traits,
        )
        log.info(
            "[Lexi SD] prompt_clip=%s ... prompt_t5=%s ... negative=%s ...",
            (positive_clip or prompt)[:180],
            (positive_t5 or "")[:180],
            (negative_clip or negative)[:180],
        )
        log.info(
            "[Lexi SD][flux full prompt] positive_clip=%s | positive_t5=%s | negative_clip=%s | negative_t5=%s",
            positive_clip or built_prompt,
            positive_t5 or built_prompt,
            negative_clip or negative or neg_full,
            negative_t5 or negative_clip or negative or neg_full,
        )

        # Prefer the Fusion-style v2 pipeline for txt2img
        if flux_pipeline == "flux_v2" and mode == "txt2img":
            width = int(width or 1080)
            height = int(height or 1352)
            return _run_flux_backend_v2(
                prompt=built_prompt,
                negative=neg_full,
                width=width,
                height=height,
                seed=seed,
                base_path=base_path,
                force_output_to_base=force_output_to_base,
                public_base_url=PUBLIC_BASE_URL,
                lexiverse_style=lexiverse_style,
            )
        # Flux v2 img2img: skip the brittle v10 workflow; use classic flux graph then degrade to txt2img.
        if flux_pipeline == "flux_v2" and mode == "img2img":
            # Ensure we have a source image; if missing, generate a fresh base via txt2img first.
            if not sp or not sp.exists():
                log.info("[Lexi SD][flux_v2] img2img requested without source; rendering base first.")
                base_result = _run_flux_backend_v2(
                    prompt=built_prompt,
                    negative=neg_full,
                    width=int(width or 1080),
                    height=int(height or 1352),
                    seed=seed,
                    base_path=base_path,
                    force_output_to_base=True,
                    public_base_url=PUBLIC_BASE_URL,
                    lexiverse_style=lexiverse_style,
                )
                if not base_result.get("ok"):
                    return base_result
                sp = base_path
            # Gentler defaults for img2img
            if flux_denoise is None:
                flux_denoise = 0.35
            img2img_steps = max(12, min(int(flux_steps or FLUX_DEFAULTS["steps"]), 28))
            try:
                classic = _run_flux_backend(
                    prompt=built_prompt,
                    negative=neg_full,
                    width=width,
                    height=height,
                    steps=img2img_steps,
                    cfg_scale=flux_cfg,
                    seed=seed,
                    mode="img2img",
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
                    pose_map_path=pose_map_path,
                    pose_control_strength=pose_strength,
                    pose_control_start=pose_start,
                    pose_control_end=pose_end,
                )
                if isinstance(classic, dict) and classic.get("ok"):
                    return classic
                if isinstance(classic, dict) and classic.get("code") == "COMFY_TIMEOUT":
                    return classic
                log.warning(
                    "[Lexi SD][flux_v2 fallback] img2img returned error: %s; retrying as txt2img",
                    (classic or {}).get("error") if isinstance(classic, dict) else classic,
                )
            except Exception as exc:
                log.warning(
                    "[Lexi SD][flux_v2 fallback] img2img failed (%s); retrying as txt2img",
                    exc,
                )
            # Final attempt: degrade to txt2img so the user still gets an avatar.
            return _run_flux_backend_v2(
                prompt=built_prompt,
                negative=neg_full,
                width=int(width or 1080),
                height=int(height or 1352),
                seed=seed,
                base_path=base_path,
                force_output_to_base=True,
                public_base_url=PUBLIC_BASE_URL,
                lexiverse_style=lexiverse_style,
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
            prompt=built_prompt,
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
            denoise=min(float(flux_denoise or FLUX_DEFAULTS["denoise"]), 0.9),
            allow_feedback_loop=bool(kwargs.get("allow_feedback_loop", False)),
            public_base_url=PUBLIC_BASE_URL,
            positive_clip=positive_clip,
            positive_t5=positive_t5,
            negative_clip=negative_clip,
            negative_t5=negative_t5,
            pose_map_path=pose_map_path,
            pose_control_strength=pose_strength,
            pose_control_start=pose_start,
            pose_control_end=pose_end,
            return_on_submit=return_on_submit,
            traits=traits,
            controlnet_enabled=controlnet_enabled,
            user_text_combined=user_text_combined,
            pose_info=(
                {
                    "pose_id": getattr(pose_choice, "pose_id", None),
                    "shape_bucket": getattr(pose_choice, "shape_bucket", None),
                    "camera_bucket": getattr(pose_choice, "camera_bucket", None),
                }
                if pose_choice
                else None
            ),
            controlnet_strength=cn_strength,
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
