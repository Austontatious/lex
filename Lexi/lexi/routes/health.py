from __future__ import annotations

import logging
import time
from typing import Dict

import requests
from fastapi import APIRouter, HTTPException

from ..sd.sd_pipeline import COMFY_URL, IMAGE_DIR, ensure_comfy_schema

router = APIRouter(tags=["Health"])
log = logging.getLogger("lexi.health")


def _object_info() -> Dict:
    resp = requests.get(f"{COMFY_URL}/object_info", timeout=5)
    resp.raise_for_status()
    return resp.json()


def _touch_avatar_dir() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    probe = IMAGE_DIR / ".healthcheck"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _prompt_noop() -> None:
    payload = {
        "prompt": {
            "latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 8, "height": 8, "batch_size": 1},
            }
        },
        "client_id": "lexi-health",
    }
    resp = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
    if not resp.ok:
        raise RuntimeError(f"Comfy prompt health check failed: HTTP {resp.status_code}")


@router.get("/lexi/healthz")
def healthz() -> Dict[str, object]:
    try:
        info = _object_info()
        ensure_comfy_schema()
        _touch_avatar_dir()
        return {
            "status": "ok",
            "comfy_nodes": len(info),
            "avatar_dir": str(IMAGE_DIR),
        }
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/lexi/readyz")
def readyz() -> Dict[str, object]:
    try:
        started = time.time()
        info = _object_info()
        schema_ok = ensure_comfy_schema()
        _prompt_noop()
        elapsed = round(time.time() - started, 3)
        return {
            "status": "ok",
            "comfy_nodes": len(info),
            "schema_valid": schema_ok,
            "elapsed": elapsed,
        }
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
