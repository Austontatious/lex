from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .comfy_client import comfy_flux_fill_edit, _edit_workflow_path, _sanitize_prefix
from .sd_pipeline import _wait_for_images, _download_image, probe_prompt_images
from ..config.paths import AVATAR_DIR, avatar_url_for

log = logging.getLogger("lexi.sd.edit")
log.setLevel(logging.INFO)

_DEFAULT_SEED = int(os.getenv("LEXI_EDIT_DEFAULT_SEED", "0"))


def _target_phrase(target: str) -> str:
    """
    Map user-facing target labels to Florence-friendly phrases.
    """
    normalized = (target or "").strip().lower()
    hair = {"hair", "head hair", "hairstyle", "style", "bangs"}
    jacket = {"outer jacket", "jacket", "coat", "blazer"}
    dress = {"dress", "gown"}
    shirt = {"shirt", "top", "t-shirt", "tee"}
    pants = {"pants", "jeans", "trousers", "shorts"}
    shoes = {"shoes", "boots", "heels"}
    if normalized in hair:
        return "head hair"
    if normalized in jacket:
        return "outer jacket"
    if normalized in dress:
        return "dress"
    if normalized in shirt:
        return "shirt"
    if normalized in pants:
        return "pants"
    if normalized in shoes:
        return "shoes"
    return normalized or "head hair"


def _edit_output_path(prompt_id: str, src_image: Optional[str]) -> Path:
    prefix = _sanitize_prefix(Path(src_image).stem if src_image else "")
    name = f"{prefix}_{prompt_id[:8] if prompt_id else 'edit'}.png"
    return AVATAR_DIR / name


def submit_avatar_edit(
    *,
    image_path: str,
    target: str,
    prompt: str,
    preserve_identity: bool = True,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    denoise: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    workflow_path: Optional[Path] = None,
    timeout_s: int = 90,
    return_on_submit: bool = True,
) -> Dict[str, Any]:
    """
    Submit the edit workflow. If return_on_submit=False, wait for the image and download it.
    """
    path = _edit_workflow_path(workflow_path)
    target_phrase = _target_phrase(target)
    face_phrase = "face" if preserve_identity else ""
    seed_val = _DEFAULT_SEED if seed is None else int(seed)
    resp = comfy_flux_fill_edit(
        image_path=image_path,
        target_phrase=target_phrase,
        prompt=prompt,
        face_phrase=face_phrase,
        seed=seed_val,
        steps=steps,
        cfg=cfg,
        denoise=denoise,
        sampler_name=sampler_name,
        scheduler=scheduler,
        workflow_path=path,
        timeout_s=timeout_s,
        filename_prefix=_sanitize_prefix(Path(image_path).stem),
    )
    prompt_id = resp.get("prompt_id") or resp.get("id")
    if not prompt_id:
        return {"ok": False, "error": "Comfy response missing prompt_id", "code": "COMFY_PROMPT_ERROR"}
    if return_on_submit:
        return {"ok": True, "status": "running", "prompt_id": prompt_id}

    # Blocking path: wait for completion and fetch the first image.
    try:
        images = _wait_for_images(prompt_id, timeout_s=timeout_s)
    except TimeoutError as exc:
        return {"ok": False, "error": str(exc), "code": "COMFY_TIMEOUT", "prompt_id": prompt_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "code": "COMFY_HISTORY_ERROR", "prompt_id": prompt_id}

    if not images:
        return {"ok": False, "error": "No images returned", "code": "COMFY_EMPTY", "prompt_id": prompt_id}
    first = images[0]
    dst = _edit_output_path(prompt_id, image_path)
    out = _download_image(
        first.get("filename", ""),
        first.get("subfolder", ""),
        first.get("type", "output"),
        dst=dst,
    )
    url = avatar_url_for(out.name)
    return {"ok": True, "file": str(out), "url": url, "avatar_url": url, "prompt_id": prompt_id}


def check_avatar_edit_status(prompt_id: str, *, image_path_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Poll Comfy history for an edit prompt. If ready, download and return the image URL.
    """
    dst = _edit_output_path(prompt_id, image_path_hint or "")
    if dst.exists():
        url = avatar_url_for(dst.name)
        return {"status": "done", "prompt_id": prompt_id, "url": url, "avatar_url": url, "file": str(dst)}

    images = probe_prompt_images(prompt_id)
    if not images:
        return {"status": "running", "prompt_id": prompt_id}
    first = images[0]
    try:
        out = _download_image(
            first.get("filename", ""),
            first.get("subfolder", ""),
            first.get("type", "output"),
            dst=dst,
        )
    except Exception as exc:
        return {
            "status": "error",
            "prompt_id": prompt_id,
            "error": f"download failed: {exc}",
            "code": "COMFY_DOWNLOAD_ERROR",
        }
    url = avatar_url_for(out.name)
    return {"status": "done", "prompt_id": prompt_id, "url": url, "avatar_url": url, "file": str(out)}
