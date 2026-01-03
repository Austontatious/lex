from __future__ import annotations

FLUX_DEFAULTS = {
    "width": 768,
    "height": 1344,
    "steps": 24,
    "guidance_pos": 2.8,
    "guidance_neg": 2.8,
    "cfg": 3.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "upscale_w": 1664,
    "upscale_h": 2048,
}

__all__ = ["FLUX_DEFAULTS"]
