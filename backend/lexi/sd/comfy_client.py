from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..config.runtime_env import COMFY_URL
from ..utils.ip_seed import normalize_ip
from .flux_defaults import FLUX_DEFAULTS

_DEFAULT_WORKFLOW = Path(
    os.getenv("FLUX_WORKFLOW_PATH") or "/app/docker/comfy/workflows/flux_workflow_api.json"
)
_REPO_WORKFLOW = Path(__file__).resolve().parents[3] / "docker/comfy/workflows/flux_workflow_api.json"
if not _DEFAULT_WORKFLOW.exists() and _REPO_WORKFLOW.exists():
    _DEFAULT_WORKFLOW = _REPO_WORKFLOW

_WORKFLOW_CACHE: Dict[str, tuple[float, str]] = {}
log = logging.getLogger("lexi.sd.comfy")


def _workflow_path(override: Optional[Path] = None) -> Path:
    candidate = override or _DEFAULT_WORKFLOW
    if not candidate.exists():
        raise FileNotFoundError(
            f"Flux workflow template not found at {candidate}. "
            "Set FLUX_WORKFLOW_PATH to a valid JSON file."
        )
    return candidate


def _load_workflow_template(path: Optional[Path] = None) -> Dict[str, Any]:
    target = _workflow_path(path)
    cache_key = str(target.resolve())
    mtime = target.stat().st_mtime
    cached = _WORKFLOW_CACHE.get(cache_key)
    if not cached or cached[0] != mtime:
        text = target.read_text(encoding="utf-8")
        _WORKFLOW_CACHE[cache_key] = (mtime, text)
    else:
        text = cached[1]
    return json.loads(text)


def _normalize_ip_seed(ip: str) -> int:
    digits = "".join(ch for ch in (ip or "") if ch.isdigit())
    if digits:
        return int(digits) % (2**63 - 1)
    data = hashlib.sha1(ip.encode("utf-8")).digest()[:8]
    return int.from_bytes(data, "big")


def _sanitize_prefix(token: Optional[str]) -> str:
    safe = normalize_ip(token or "")
    return safe or "lexi"


def comfy_flux_generate(
    positive_1: str,
    positive_2: str,
    negative_1: str,
    negative_2: str,
    *,
    seed: int,
    width: int = FLUX_DEFAULTS["width"],
    height: int = FLUX_DEFAULTS["height"],
    steps: int = FLUX_DEFAULTS["steps"],
    cfg: float = FLUX_DEFAULTS["cfg"],
    sampler_name: str = FLUX_DEFAULTS["sampler"],
    scheduler: str = FLUX_DEFAULTS["scheduler"],
    denoise: float = FLUX_DEFAULTS["denoise"],
    guidance: float = FLUX_DEFAULTS["guidance_pos"],
    client_id: str = "lexi-flux",
    workflow_path: Optional[Path] = None,
    timeout_s: int = 60,
    retries: int = 1,
    filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inject prompts/seed/size into the canonical Flux workflow and POST to Comfy.
    Returns the raw JSON response from /prompt.
    """
    workflow = _load_workflow_template(workflow_path)

    for node_id in ("22", "23"):
        node = workflow.get(node_id, {})
        widgets = node.get("widgets_values")
        if isinstance(widgets, list):
            for idx in range(len(widgets)):
                widgets[idx] = ""

    encode_pos = workflow.get("22", {}).get("inputs", {})
    encode_neg = workflow.get("23", {}).get("inputs", {})
    sampler_inputs = workflow.get("16", {}).get("inputs", {})
    latent_inputs = workflow.get("75", {}).get("inputs", {})
    upscale_inputs = workflow.get("74", {}).get("inputs", {})

    encode_pos["clip_l"] = positive_1
    encode_pos["t5xxl"] = positive_2
    encode_pos["guidance"] = float(guidance or FLUX_DEFAULTS["guidance_pos"])

    encode_neg["clip_l"] = negative_1
    encode_neg["t5xxl"] = negative_2
    encode_neg["guidance"] = float(guidance or FLUX_DEFAULTS["guidance_neg"])

    sampler_inputs["seed"] = int(seed)
    sampler_inputs["steps"] = int(steps or FLUX_DEFAULTS["steps"])
    sampler_inputs["cfg"] = float(cfg or FLUX_DEFAULTS["cfg"])
    sampler_inputs["sampler_name"] = sampler_name or FLUX_DEFAULTS["sampler"]
    sampler_inputs["scheduler"] = scheduler or FLUX_DEFAULTS["scheduler"]
    sampler_inputs["denoise"] = float(denoise or FLUX_DEFAULTS["denoise"])

    latent_inputs["width"] = int(width)
    latent_inputs["height"] = int(height)
    latent_inputs["batch_size"] = max(1, int(latent_inputs.get("batch_size", 1)))

    upscale_inputs["width"] = FLUX_DEFAULTS["upscale_w"]
    upscale_inputs["height"] = FLUX_DEFAULTS["upscale_h"]

    if filename_prefix:
        save_node = workflow.get("77", {})
        save_inputs = save_node.get("inputs")
        if isinstance(save_inputs, dict):
            save_inputs["filename_prefix"] = _sanitize_prefix(filename_prefix)

    payload = {"prompt": workflow, "client_id": client_id}
    log.info(
        "[lexi][flux] seed=%s pos1=%r pos2=%r",
        seed,
        positive_1[:64],
        positive_2[:64],
    )
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=timeout_s)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            return {"status_code": response.status_code}
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
            log.warning("[lexi][flux] comfy prompt attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(0.5)
    raise RuntimeError(f"Comfy Flux generation failed: {last_err}")


__all__ = ["comfy_flux_generate", "_normalize_ip_seed"]
