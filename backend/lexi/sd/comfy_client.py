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

_DEFAULT_WORKFLOW_V2 = Path(
    os.getenv("FLUX_WORKFLOW_V2_PATH") or "/app/docker/comfy/workflows/flux_workflow_api_v2.json"
)
_REPO_WORKFLOW_V2 = Path(__file__).resolve().parents[3] / "docker/comfy/workflows/flux_workflow_api_v2.json"
if not _DEFAULT_WORKFLOW_V2.exists() and _REPO_WORKFLOW_V2.exists():
    _DEFAULT_WORKFLOW_V2 = _REPO_WORKFLOW_V2

_DEFAULT_KONTEXT_IMG2IMG = Path(
    os.getenv("FLUX_WORKFLOW_IMG2IMG_PATH")
    or "/app/docker/comfy/workflows/fluxKontextOfficial_v10.json"
)
_REPO_KONTEXT_IMG2IMG = Path(__file__).resolve().parents[3] / "docker/comfy/workflows/fluxKontextOfficial_v10.json"
if not _DEFAULT_KONTEXT_IMG2IMG.exists() and _REPO_KONTEXT_IMG2IMG.exists():
    _DEFAULT_KONTEXT_IMG2IMG = _REPO_KONTEXT_IMG2IMG

_WORKFLOW_CACHE: Dict[str, tuple[float, str]] = {}
log = logging.getLogger("lexi.sd.comfy")
log.setLevel(logging.INFO)


def _log_prompt_payload(tag: str, payload: Dict[str, Any]) -> None:
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[lexi][%s] failed to serialize prompt payload: %s", tag, exc)
        return
    log.warning("[lexi][%s] comfy prompt payload: %s", tag, serialized)


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
    wf = json.loads(text)

    # Normalize nodes that use "type" instead of "class_type" (UI export vs API export).
    # Comfy /prompt requires class_type, otherwise returns invalid_prompt.
    nodes = wf.get("nodes")
    if isinstance(nodes, list):
        mutated = False
        for node in nodes:
            if isinstance(node, dict) and "class_type" not in node and "type" in node:
                node["class_type"] = node["type"]
                mutated = True
        if mutated:
            wf["nodes"] = nodes
    return wf


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
    _log_prompt_payload("flux_txt2img", payload)
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


def comfy_flux_generate_v2(
    positive: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,  # reserved for future use
    *,
    seed: int,
    width: int = 1080,
    height: int = 1352,
    steps: int = 35,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "sgm_uniform",
    denoise: float = 1.0,
    max_shift: float = 1.15,
    base_shift: float = 0.5,
    guidance: float = 3.2,
    lora_main: float = 0.0,
    lora_main_clip: float = 0.0,
    lora_skirt: float = 0.0,
    lora_skirt_clip: float = 0.0,
    client_id: str = "lexi-flux-v2",
    workflow_path: Optional[Path] = None,
    timeout_s: int = 60,
    retries: int = 1,
    filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Slim Flux txt2img graph using ModelSamplingFlux and optional Lexiverse LoRAs.
    LoRAs default to strength 0 so they can be enabled per-request.
    """
    wf_path = workflow_path or _DEFAULT_WORKFLOW_V2
    workflow = _load_workflow_template(wf_path)

    # Prompt + guidance
    prompt_node = workflow.get("5", {}).get("inputs", {})
    # NOTE: single CLIPTextEncode node for positive conditioning; compose base aesthetic + traits into one string upstream.
    prompt_node["text"] = (positive or prompt or "")  # text encoder prompt
    if "widgets_values" in workflow.get("6", {}):
        # In case widgets are used elsewhere, keep both inputs and widgets aligned.
        workflow["6"]["widgets_values"] = [float(guidance)]
    workflow.get("6", {}).get("inputs", {})["guidance"] = float(guidance)

    # Seed / sampler / scheduler
    workflow.get("10", {}).get("inputs", {})["noise_seed"] = int(seed)
    sched_inputs = workflow.get("9", {}).get("inputs", {})
    sched_inputs["steps"] = int(steps)
    sched_inputs["scheduler"] = scheduler
    sched_inputs["denoise"] = float(denoise)
    workflow.get("11", {}).get("inputs", {})["sampler_name"] = sampler_name

    # Size + sampling shifts
    for node_id in ("8", "12"):
        node_inputs = workflow.get(node_id, {}).get("inputs", {})
        node_inputs["width"] = int(width)
        node_inputs["height"] = int(height)
    ms_inputs = workflow.get("8", {}).get("inputs", {})
    ms_inputs["max_shift"] = float(max_shift)
    ms_inputs["base_shift"] = float(base_shift)
    if "widgets_values" in workflow.get("8", {}):
        workflow["8"]["widgets_values"] = [
            float(max_shift),
            float(base_shift),
            int(width),
            int(height),
        ]

    # Optional Lexiverse LoRAs (default 0.0)
    lora_main_inputs = workflow.get("2", {}).get("inputs", {})
    lora_main_inputs["strength_model"] = float(lora_main)
    lora_main_inputs["strength_clip"] = float(lora_main_clip)
    lora_skirt_inputs = workflow.get("3", {}).get("inputs", {})
    lora_skirt_inputs["strength_model"] = float(lora_skirt)
    lora_skirt_inputs["strength_clip"] = float(lora_skirt_clip)

    # Save filename prefix
    if filename_prefix:
        save_inputs = workflow.get("16", {}).get("inputs")
        if isinstance(save_inputs, dict):
            save_inputs["filename_prefix"] = _sanitize_prefix(filename_prefix)

    payload = {"prompt": workflow, "client_id": client_id}
    _log_prompt_payload("flux_v2", payload)
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
            log.warning("[lexi][flux_v2] comfy prompt attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(0.5)
    raise RuntimeError(f"Comfy Flux v2 generation failed: {last_err}")


def comfy_flux_generate_img2img_v10(
    prompt: str,
    source_image: str,
    *,
    seed: int,
    steps: int = 20,
    cfg: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    denoise: float = 1.0,
    guidance: float = 2.5,
    width: Optional[int] = None,
    height: Optional[int] = None,
    filename_prefix: Optional[str] = None,
    workflow_path: Optional[Path] = None,
    timeout_s: int = 60,
    retries: int = 1,
    client_id: str = "lexi-flux-kontext-img2img",
) -> Dict[str, Any]:
    """
    Img2img using the Flux Kontext official v10 workflow (JSON-based).
    """
    path = workflow_path or _DEFAULT_KONTEXT_IMG2IMG
    wf = _load_workflow_template(path)
    nodes = wf.get("nodes", [])
    idmap = {str(n.get("id")): n for n in nodes}

    def set_widgets(node_id: str, index: int, value: Any) -> None:
        n = idmap.get(str(node_id))
        if not n:
            return
        w = n.get("widgets_values")
        if isinstance(w, list) and index < len(w):
            w[index] = value

    # Prompt
    set_widgets("6", 0, prompt or "")
    # Guidance
    set_widgets("35", 0, float(guidance))
    # LoadImage
    set_widgets("200", 0, source_image)
    # Resize target (if provided)
    if width and height:
        set_widgets("197", 0, int(width))
        set_widgets("197", 1, int(height))
    # KSampler
    set_widgets("31", 0, int(seed))
    set_widgets("31", 1, "fixed")
    set_widgets("31", 2, int(steps))
    set_widgets("31", 3, float(cfg))
    set_widgets("31", 4, sampler_name)
    set_widgets("31", 5, scheduler)
    set_widgets("31", 6, float(denoise))
    # Save prefix
    if filename_prefix:
        set_widgets("136", 0, _sanitize_prefix(filename_prefix))

    payload = {"prompt": wf, "client_id": client_id}
    _log_prompt_payload("flux_v10_img2img", payload)
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
            log.warning("[lexi][flux_img2img_v10] comfy prompt attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(0.5)
    raise RuntimeError(f"Comfy Flux img2img v10 generation failed: {last_err}")


__all__ = [
    "comfy_flux_generate",
    "comfy_flux_generate_v2",
    "comfy_flux_generate_img2img_v10",
    "_normalize_ip_seed",
]
