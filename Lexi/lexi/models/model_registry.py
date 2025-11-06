from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
import os
import yaml

COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/mnt/data/comfy"))
CKPT_ROOT = COMFY_ROOT / "models" / "checkpoints"
LORA_ROOT = COMFY_ROOT / "models" / "loras"


@dataclass
class BaseModel:
    key: str
    file: str
    family: str
    strengths: List[str]
    tags: List[str]
    default_cfg: float | None = None
    default_steps: int | None = None


@dataclass
class Refiner:
    key: str
    file: str
    family: str
    strengths: List[str]
    tags: List[str]
    default_refiner_strength: float | None = None


@dataclass
class Lora:
    key: str
    file: str
    kind: str  # adapter/style/outfit/body
    strengths: List[str]
    default_weights: Dict[str, float]


@dataclass
class Selection:
    base_file: str
    refiner_file: Optional[str]
    loras: List[Tuple[str, float, float]]  # (file, unet_w, clip_w)
    cfg: Optional[float]
    steps: Optional[int]
    refiner_strength: Optional[float]
    # --- new optional fields (added at the end to keep positional args stable) ---
    denoise: Optional[float] = None  # default denoise from policy.defaults
    variation: Optional[Dict[str, Any]] = None  # policy.defaults.variation blob


class ModelRegistry:
    def __init__(self, path: str | Path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.bases: Dict[str, BaseModel] = {}
        self.refiners: Dict[str, Refiner] = {}
        self.loras: Dict[str, Lora] = {}
        self.policies: Dict[str, dict] = data.get("policies", {})

        for k, v in data.get("bases", {}).items():
            self.bases[k] = BaseModel(k, **v)
        for k, v in data.get("refiners", {}).items():
            self.refiners[k] = Refiner(k, **v)
        for k, v in data.get("loras", {}).items():
            self.loras[k] = Lora(k, **v)

    def _exists_in_comfy(self, root: Path, filename: str) -> bool:
        try:
            if (root / filename).exists():
                return True
            # search subfolders by basename match
            for p in root.rglob("*"):
                if p.is_file() and p.name == filename:
                    return True
        except Exception:
            pass
        return False

    def select(self, task: str, overrides: Optional[dict] = None) -> Selection:
        """Pick base, optional refiner, and recommended LoRAs for a given task name."""
        overrides = overrides or {}
        policy = self.policies.get(task) or self.policies.get("general", {}) or {}
        defaults = policy.get("defaults", {}) or {}

        # pick base (first available from policy; fallback to any)
        base_file = None
        cfg = steps = None

        for key in policy.get("prefer_bases", []):
            b = self.bases.get(key)
            if not b:
                continue
            if self._exists_in_comfy(CKPT_ROOT, b.file):
                base_file = b.file
                # base-level defaults, used only if policy.defaults doesn't override
                cfg = b.default_cfg if defaults.get("cfg") is None else defaults.get("cfg")
                steps = b.default_steps if defaults.get("steps") is None else defaults.get("steps")
                break

        if not base_file:
            # fallback: any available base; still respect policy.defaults override if present
            for b in self.bases.values():
                if self._exists_in_comfy(CKPT_ROOT, b.file):
                    base_file = b.file
                    cfg = b.default_cfg if defaults.get("cfg") is None else defaults.get("cfg")
                    steps = (
                        b.default_steps if defaults.get("steps") is None else defaults.get("steps")
                    )
                    break

        if not base_file:
            want = (os.getenv("LEX_SDXL_CHECKPOINT") or "").strip()
            if want:
                if self._exists_in_comfy(CKPT_ROOT, want):
                    base_file = want
                else:
                    want_path = Path(want)
                    if want_path.exists():
                        base_file = want_path.name

        if not base_file:
            skip_preflight = os.getenv("LEX_SKIP_MODEL_PREFLIGHT", "0") == "1"
            comfy_only = os.getenv("LEX_USE_COMFY_ONLY", "0") == "1"
            if skip_preflight or comfy_only:
                logging.getLogger(__name__).warning(
                    "No base checkpoint found in Comfy model directory; continuing because "
                    "preflight is disabled%s.",
                    " (comfy-only)" if comfy_only else "",
                )
            else:
                raise RuntimeError("No base checkpoint found in Comfy model directory.")

        # optional refiner
        refiner_file = None
        refiner_strength = None
        if policy.get("allow_refiner", False):
            for r in self.refiners.values():
                if self._exists_in_comfy(CKPT_ROOT, r.file):
                    refiner_file = r.file
                    refiner_strength = r.default_refiner_strength
                    break

        # loras (with weights)
        loras_out: List[Tuple[str, float, float]] = []
        for key in policy.get("loras", []):
            l = self.loras.get(key)
            if not l:
                continue
            if self._exists_in_comfy(LORA_ROOT, l.file):
                unet_w = float(l.default_weights.get("unet", 0.6))
                clip_w = float(l.default_weights.get("clip", 0.3))
                loras_out.append((l.file, unet_w, clip_w))

        # policy-level denoise/variation (safe to be None if not provided)
        denoise = defaults.get("denoise")
        variation = defaults.get("variation")

        # apply overrides (highest precedence)
        if "base" in overrides:
            base_file = overrides["base"]
        if "refiner" in overrides:
            refiner_file = overrides["refiner"]
        if "cfg" in overrides:
            cfg = overrides["cfg"]
        if "steps" in overrides:
            steps = overrides["steps"]
        if "refiner_strength" in overrides:
            refiner_strength = overrides["refiner_strength"]
        if "loras" in overrides:
            loras_out = overrides["loras"]
        if "denoise" in overrides:
            denoise = overrides["denoise"]
        if "variation" in overrides:
            variation = overrides["variation"]

        return Selection(
            base_file=base_file,
            refiner_file=refiner_file,
            loras=loras_out,
            cfg=cfg,
            steps=steps,
            refiner_strength=refiner_strength,
            denoise=denoise,
            variation=variation,
        )
