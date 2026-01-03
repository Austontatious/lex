# generate.py  (replace with this)
from __future__ import annotations
from typing import Any, Dict, Optional
from .sd_pipeline import generate_avatar_pipeline  # our Comfy-backed impl

_PIPELINE_KEYS = {
    "prompt",
    "negative",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "traits",
    "mode",
    "source_path",
    "mask_path",
    "changes",
    "seed",
    "refiner",
    "refiner_strength",
    "upscale_factor",
    # backend-specific knobs
    "backend",
    "model",
    "variant",
    "flux_variant",
    "preset",
    "flux_preset",
    "size",
    "flux_size",
    "guidance",
    "flux_guidance",
    "flux_cfg",
    "flux_steps",
    "flux_denoise",
    "flux_sampler",
    "flux_scheduler",
    "sampler",
    "scheduler",
    "denoise",
    "allow_feedback_loop",
    "base_name",
}


def generate_avatar(
    prompt: str,
    negative: str = "lowres, blurry, deformed, bad anatomy",
    width: int = 832,
    height: int = 1152,
    steps: int = 30,
    cfg_scale: float = 5.0,
    *,
    # NEW ↓ — allow real edits + continuity
    mode: str = "txt2img",  # "txt2img" | "img2img"
    source_path: Optional[str] = None,  # required for img2img
    denoise: float = 0.7,  # stronger img2img push to vary outfit/pose
    seed: Optional[int] = None,
    changes: Optional[str] = None,  # small delta: "brown hair", "add skirt"
    traits: Optional[Dict[str, Any]] = None,  # persona traits to keep continuity
    refiner: bool = True,
    refiner_strength: float = 0.28,
    upscale_factor: float = 1.0,
    base_name: Optional[str] = None,
    **extra: Any,
) -> str:
    res: Dict[str, Any] = generate_avatar_pipeline(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        mode=mode,
        source_path=source_path,
        denoise=denoise,
        seed=seed,
        changes=changes,
        traits=traits,
        refiner=refiner if mode == "txt2img" else False,  # edits: base only
        refiner_strength=refiner_strength,
        upscale_factor=upscale_factor,
        base_name=base_name,
        **extra,
    )
    if not res.get("ok"):
        raise RuntimeError(f"generate_avatar failed: {res.get('error', 'unknown error')}")
    return str(res["file"])


def generate_avatar_meta(**kwargs) -> Dict[str, Any]:
    """
    Normalize/validate inputs, forward to the Comfy-backed pipeline, and
    attach the effective config as `meta`.
    """
    # 1) Whitelist only supported keys
    cfg: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in _PIPELINE_KEYS}

    # 2) Defaults / normalization
    mode = str(cfg.get("mode") or "txt2img")
    cfg["mode"] = mode

    # Refiner only on txt2img
    if mode != "txt2img":
        cfg["refiner"] = False

    # Clamp denoise if caller passed it via kwargs; pipeline already clamps, but be explicit
    if "denoise" in kwargs:
        dn = float(kwargs["denoise"])
        cfg["denoise"] = max(0.10, min(dn, 0.75))

    # 3) Call the pipeline
    result = generate_avatar_pipeline(**cfg)

    # 4) Attach effective meta (don’t overwrite pipeline’s own meta if present)
    meta = dict(cfg)
    if isinstance(result, dict):
        result.setdefault("meta", {})
        # keep pipeline-calculated fields (seed, width, height, etc.), then add our cfg
        result["meta"] = {**meta, **result["meta"]}

    return result
