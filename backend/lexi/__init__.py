"""Lexi backend package."""

from __future__ import annotations

import os
import logging

import requests


def _flush_comfy_cache() -> None:
    url = os.getenv("COMFY_URL", "http://comfy:8188")
    log = logging.getLogger("lexi.boot")
    try:
        requests.post(f"{url}/queue", json={"clear": True}, timeout=3)
        requests.post(f"{url}/history/clear", timeout=3)
        log.info("[lexi] Flux cache cleared on boot")
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("[lexi] Flux cache flush failed: %s", exc)


_flush_comfy_cache()
