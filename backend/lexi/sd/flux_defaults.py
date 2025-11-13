from __future__ import annotations

FLUX_DEFAULTS = {
    "width": 832,
    "height": 1024,
    "steps": 13,
    "guidance_pos": 3.5,
    "guidance_neg": 3.5,
    "cfg": 3.4,
    "sampler": "dpmpp_3m_sde_gpu",
    "scheduler": "simple",
    "denoise": 0.95,
    "upscale_w": 1664,
    "upscale_h": 2048,
}

__all__ = ["FLUX_DEFAULTS"]
