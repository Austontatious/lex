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

# Dedicated edit workflow (semantic mask + Flux Fill). Kept separate from txt2img defaults.
_DEFAULT_WORKFLOW_EDIT = Path(
    os.getenv("LEXI_COMFY_WORKFLOW_EDIT")
    or "/app/docker/comfy/workflows/fusion_fill_edit_api.json"
)
_REPO_WORKFLOW_EDIT = (
    Path(__file__).resolve().parents[3] / "docker/comfy/workflows/fusion_fill_edit_api.json"
)
if not _DEFAULT_WORKFLOW_EDIT.exists() and _REPO_WORKFLOW_EDIT.exists():
    _DEFAULT_WORKFLOW_EDIT = _REPO_WORKFLOW_EDIT

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


def _edit_workflow_path(override: Optional[Path] = None) -> Path:
    candidate = override or _DEFAULT_WORKFLOW_EDIT
    if not candidate.exists():
        raise FileNotFoundError(
            f"Edit workflow template not found at {candidate}. "
            "Set LEXI_COMFY_WORKFLOW_EDIT to a valid JSON file."
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


def _nodes_map(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize workflow nodes into a mutable id->node mapping.
    Supports both API-style dicts and UI export lists under "nodes".
    """
    if isinstance(workflow.get("nodes"), list):
        nodes_list = workflow["nodes"]
        nodes = {
            str(n.get("id")): n for n in nodes_list if isinstance(n, dict) and n.get("id") is not None
        }
        return nodes
    return {str(k): v for k, v in workflow.items() if isinstance(v, dict)}


def _set_input_value(
    workflow: Dict[str, Any],
    *,
    class_types: tuple[str, ...],
    key: str,
    value: Any,
    occurrence: int = 0,
) -> Optional[str]:
    """
    Set an input value on the Nth node matching class_types.
    Returns the node id on success, None if not found.
    """
    def _sort_key(item: tuple[str, Dict[str, Any]]) -> Any:
        try:
            return int(item[0])
        except ValueError:
            return item[0]

    nodes = [
        (nid, node)
        for nid, node in sorted(_nodes_map(workflow).items(), key=_sort_key)
        if node.get("class_type") in class_types
    ]
    if occurrence >= len(nodes):
        return None
    nid, node = nodes[occurrence]
    inputs = node.setdefault("inputs", {})
    inputs[key] = value
    return nid


def _set_text_payload(node: Dict[str, Any], text: str) -> None:
    """Best-effort update for text-like fields on a node."""
    inputs = node.setdefault("inputs", {})
    for key in ("text", "clip_l", "t5xxl", "prompt"):
        if key in inputs:
            inputs[key] = text
    widgets = node.get("widgets_values")
    if isinstance(widgets, list) and widgets:
        widgets[0] = text


def _normalize_ip_seed(ip: str) -> int:
    digits = "".join(ch for ch in (ip or "") if ch.isdigit())
    if digits:
        return int(digits) % (2**63 - 1)
    data = hashlib.sha1(ip.encode("utf-8")).digest()[:8]
    return int.from_bytes(data, "big")


def _sanitize_prefix(token: Optional[str]) -> str:
    safe = normalize_ip(token or "")
    return safe or "lexi"


def _set_if_present(node: Dict[str, Any], key: str, value: Any) -> bool:
    """Set an input only if the key already exists."""
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return False
    if key not in inputs:
        return False
    inputs[key] = value
    return True


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
    sampler_inputs["steps"] = max(20, int(steps or FLUX_DEFAULTS["steps"]))
    sampler_inputs["cfg"] = float(cfg or FLUX_DEFAULTS["cfg"])
    sampler_inputs["sampler_name"] = sampler_name or FLUX_DEFAULTS["sampler"]
    sampler_inputs["scheduler"] = scheduler or FLUX_DEFAULTS["scheduler"]
    sampler_inputs["denoise"] = 1.0
    if str(sampler_inputs.get("sampler_name", "")).endswith("_sde_gpu"):
        sampler_inputs["sampler_name"] = "dpmpp_2m"

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


def comfy_flux_fill_edit(
    image_path: str,
    target_phrase: str,
    prompt: str,
    *,
    face_phrase: str = "face",
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    denoise: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    workflow_path: Optional[Path] = None,
    timeout_s: int = 90,
    retries: int = 1,
    client_id: str = "lexi-flux-fill-edit",
    filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare and submit the Flux Fill edit workflow (semantic mask + inpaint).
    Expects the workflow JSON to already include the Florence2 → SAM2 → mask math path.
    """
    path = _edit_workflow_path(workflow_path)
    workflow = _load_workflow_template(path)
    nodes = _nodes_map(workflow)
    patched_fields: list[str] = []

    # Image input
    img_node = _set_input_value(
        workflow,
        class_types=("LoadImage", "LoadImageMask", "ImageInput"),
        key="image",
        value=image_path,
        occurrence=0,
    )
    if img_node:
        patched_fields.append(f"image@{img_node}")

    # Target + face phrases via Florence2
    florence_classes = ("Florence2Run", "Florence2", "Florence2RunSE")
    target_node = _set_input_value(
        workflow,
        class_types=florence_classes,
        key="text",
        value=target_phrase,
        occurrence=0,
    )
    if target_node:
        patched_fields.append(f"target_phrase@{target_node}")
    face_node = _set_input_value(
        workflow,
        class_types=florence_classes,
        key="text",
        value=face_phrase,
        occurrence=1,
    )
    if face_node:
        patched_fields.append(f"face_phrase@{face_node}")

    # Inpaint prompt (CLIP encode or equivalent)
    encode_nodes = [
        (nid, node)
        for nid, node in nodes.items()
        if node.get("class_type")
        in (
            "CLIPTextEncodeFlux",
            "CLIPTextEncode",
            "CLIPTextEncodeSDXL",
            "FluxGuidance",
        )
    ]
    if encode_nodes:
        _set_text_payload(encode_nodes[0][1], prompt or "")
        patched_fields.append(f"prompt@{encode_nodes[0][0]}")

    # Sampler / scheduler knobs
    sampler_classes = (
        "KSampler",
        "KSamplerAdvanced",
        "SamplerCustomAdvanced",
        "ModelSamplingFlux",
        "KSamplerSelect",
        "BasicScheduler",
        "RandomNoise",
    )
    for nid, node in nodes.items():
        if node.get("class_type") not in sampler_classes:
            continue
        if seed is not None:
            if _set_if_present(node, "seed", int(seed)) or _set_if_present(node, "noise_seed", int(seed)):
                patched_fields.append(f"seed@{nid}")
        if steps is not None:
            if _set_if_present(node, "steps", int(steps)):
                patched_fields.append(f"steps@{nid}")
        if cfg is not None and _set_if_present(node, "cfg", float(cfg)):
            patched_fields.append(f"cfg@{nid}")
        if scheduler and _set_if_present(node, "scheduler", scheduler):
            patched_fields.append(f"scheduler@{nid}")
        if denoise is not None and _set_if_present(node, "denoise", float(denoise)):
            patched_fields.append(f"denoise@{nid}")
        if sampler_name and _set_if_present(node, "sampler_name", sampler_name):
            patched_fields.append(f"sampler@{nid}")

    # Save image prefix
    if filename_prefix:
        save_node = _set_input_value(
            workflow,
            class_types=("SaveImage",),
            key="filename_prefix",
            value=_sanitize_prefix(filename_prefix),
        )
        if save_node:
            patched_fields.append(f"filename_prefix@{save_node}")

    missing = []
    if not img_node:
        missing.append("LoadImage.image")
    if not target_node:
        missing.append("Florence2 target text")
    if not encode_nodes:
        missing.append("Text encoder / prompt node")
    if missing:
        raise RuntimeError(
            "Edit workflow missing expected nodes: "
            + ", ".join(missing)
            + ". Update fusion_fill_edit_api.json or set LEXI_COMFY_WORKFLOW_EDIT."
        )

    payload = {"prompt": workflow, "client_id": client_id}
    _log_prompt_payload("flux_edit_fill", payload)
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=timeout_s)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                log.info(
                    "[lexi][flux_edit_fill] submitted workflow=%s patched=%s",
                    path,
                    ",".join(patched_fields) or "none",
                )
                return response.json()
            return {"status_code": response.status_code}
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
            log.warning(
                "[lexi][flux_edit_fill] comfy prompt attempt %s failed: %s", attempt + 1, exc
            )
            if attempt < retries:
                time.sleep(0.5)
    raise RuntimeError(f"Comfy Flux edit-fill generation failed: {last_err}")


__all__ = [
    "comfy_flux_generate",
    "comfy_flux_generate_v2",
    "comfy_flux_generate_img2img_v10",
    "comfy_flux_fill_edit",
    "_normalize_ip_seed",
]
