#!/usr/bin/env python3
"""Smoke checks for avatar endpoints.

- Verifies Avatar Tools modal endpoint responds (alpha strict mode).
- Verifies legacy auto-appearance extraction returns deprecated no-op.
"""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient


def main() -> int:
    os.environ.setdefault("ALPHA_STRICT", "1")

    from lexi.core.backend_core import app

    client = TestClient(app)

    modal_payload = {
        "sd_mode": "txt2img",
        "lexiverse_style": "promo",
        "traits": {
            "hair": "brunette",
            "hair_style": "wavy",
            "skin_tone": "light_medium",
            "eyes": "hazel",
            "outfit": "lbd",
            "vibe": "soft",
        },
    }
    modal_res = client.post("/lexi/gen/avatar", json=modal_payload)
    if modal_res.status_code != 200:
        print("Modal endpoint failed:", modal_res.status_code, modal_res.text)
        return 1
    modal_body = modal_res.json()
    if not any(modal_body.get(key) for key in ("image", "image_url", "avatar_url", "url")):
        print("Modal endpoint missing avatar URL fields:", modal_body)
        return 1

    legacy_res = client.post(
        "/lexi/process",
        json={"prompt": "She has blonde hair and green eyes with red lipstick."},
    )
    if legacy_res.status_code != 200:
        print("Legacy endpoint failed:", legacy_res.status_code, legacy_res.text)
        return 1
    legacy_body = legacy_res.json()
    if not legacy_body.get("deprecated") or legacy_body.get("status") != "ignored":
        print("Legacy endpoint did not return deprecated no-op:", legacy_body)
        return 1
    if "Avatar Tools" not in legacy_body.get("cleaned", ""):
        print("Legacy endpoint missing deprecation message:", legacy_body)
        return 1

    print("Smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
