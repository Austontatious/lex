from __future__ import annotations

import logging
import time
from typing import Dict

import requests
from fastapi import APIRouter, HTTPException

from ..sd.sd_pipeline import COMFY_URL, IMAGE_DIR, ensure_comfy_schema

router = APIRouter(tags=["Health"])
log = logging.getLogger("lexi.backend.health")


def _object_info() -> Dict:
    resp = requests.get(f"{COMFY_URL}/object_info", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return data


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
    """Lightweight liveness check: object_info + filesystem write."""
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
    """
    Readiness check: ensure Comfy schema expectations still hold and that the prompt endpoint
    accepts a no-op workflow.
    """
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


@router.get("/health/flux")
def flux_health() -> Dict[str, object]:
    """Check that the Flux Comfy worker is reachable and running."""
    try:
        resp = requests.get(f"{COMFY_URL}/system_stats", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        models = list((data.get("models") or {}).keys())
        return {"ok": True, "models": models}
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Flux health failed: %s", exc)
        return {"ok": False, "error": str(exc)}
