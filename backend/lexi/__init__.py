"""Lexi backend package."""

from __future__ import annotations

import importlib
import os
import logging
from typing import Any

import requests

__all__ = ["config", "model_loader", "memory", "persona", "prompt_templates"]

_EXPORT_MAP: dict[str, str] = {
    "config": "lexi.config",
    "model_loader": "lexi.core.model_loader_core",
    "memory": "lexi.memory",
    "persona": "lexi.persona",
    "prompt_templates": "lexi.persona.prompt_templates",
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target:
        module = importlib.import_module(target)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


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
