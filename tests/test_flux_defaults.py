from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = ROOT / ".pytest_state"
os.environ.setdefault("LEX_MEMORY_PATH", str(TMP_ROOT / "lex_memory.jsonl"))
os.environ.setdefault("LEX_LOG_DIR", str(TMP_ROOT / "logs"))
os.environ.setdefault("LEX_DEFAULT_AVATAR_DIR", str(TMP_ROOT / "defaults"))
os.environ.setdefault("AVATARS_PUBLIC_DIR", str(TMP_ROOT / "avatars"))
os.environ.setdefault("AVATARS_PUBLIC_URL", "/test/avatars")
os.environ.setdefault("LEX_IMAGE_DIR", str(TMP_ROOT / "avatars"))

if "orjson" not in sys.modules:  # pragma: no cover - test shim
    sys.modules["orjson"] = types.SimpleNamespace(
        dumps=lambda obj, *args, **kwargs: str(obj).encode("utf-8"),
        loads=lambda data: {},
    )

from backend.lexi.sd import comfy_client
from backend.lexi.sd import generate as sd_generate
from backend.lexi.routes.lexi import ChatRequest, process as lexi_process


def test_normalize_ip_seed_digits_preferred():
    assert comfy_client._normalize_ip_seed("100.23.10.0.1") == 100231001

    ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    # IPv6 should hash but remain stable.
    assert comfy_client._normalize_ip_seed(ipv6) == comfy_client._normalize_ip_seed(ipv6)


def test_generate_default_avatar_creates_and_reuses(monkeypatch, tmp_path: Path):
    calls = {"generate": 0, "download": 0}

    def fake_generate(*args, **kwargs):
        calls["generate"] += 1
        return {"prompt_id": "pid-123"}

    monkeypatch.setattr(sd_generate, "comfy_flux_generate", fake_generate)
    monkeypatch.setattr(
        sd_generate,
        "_wait_for_images",
        lambda pid: [{"filename": "foo.png", "subfolder": "", "type": "output"}],
    )

    def fake_download(filename, subfolder, ftype, dst=None):
        calls["download"] += 1
        target = Path(dst) if dst else tmp_path / "downloaded.png"
        target.write_bytes(b"fake")
        return target

    monkeypatch.setattr(sd_generate, "_download_image", fake_download)
    monkeypatch.setattr(sd_generate, "normalize_portrait_image", lambda path: path)

    info1 = sd_generate.generate_default_avatar_for_ip("10.0.0.42", str(tmp_path))
    assert info1["created"] is True
    expected_name = f"default_{comfy_client._normalize_ip_seed('10.0.0.42')}.png"
    assert Path(info1["path"]).name == expected_name

    info2 = sd_generate.generate_default_avatar_for_ip("10.0.0.42", str(tmp_path))
    assert info2["created"] is False

    assert calls["generate"] == 1  # second call reused cached file
    assert calls["download"] == 1


def test_avatar_edit_intent_disabled():
    import anyio

    async def _receive():
        return {"type": "http.request"}

    async def _run():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/process",
            "headers": [],
            "client": ("127.0.0.1", 5000),
            "app": types.SimpleNamespace(state=types.SimpleNamespace()),
        }
        request = Request(scope, _receive)
        response = await lexi_process(ChatRequest(prompt="let's change your look"), request)
        assert response.status_code == 200
        payload = json.loads(response.body.decode("utf-8"))
        assert payload["intent"] == "avatar_edit_disabled"
        assert "disabled" in payload["message"].lower()

    anyio.run(_run, backend="asyncio")


def test_comfy_flux_generate_injects_prompts(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"prompt_id": "pid-test"}

    def fake_post(url, json, timeout):
        captured["payload"] = json["prompt"]
        return DummyResponse()

    monkeypatch.setattr(comfy_client.requests, "post", fake_post)

    resp = comfy_client.comfy_flux_generate(
        "subject prompt",
        "style prompt",
        "neg-subject",
        "neg-style",
        seed=123,
        width=640,
        height=800,
        guidance=4.2,
    )
    assert resp["prompt_id"] == "pid-test"

    payload = captured["payload"]
    assert payload["22"]["inputs"]["clip_l"] == "subject prompt"
    assert payload["22"]["inputs"]["guidance"] == pytest.approx(4.2)
    assert payload["23"]["inputs"]["clip_l"] == "neg-subject"
    assert payload["16"]["inputs"]["seed"] == 123
    assert payload["75"]["inputs"]["width"] == 640
    assert payload["75"]["inputs"]["height"] == 800
    assert payload["74"]["inputs"]["width"] == 1664
    assert payload["74"]["inputs"]["height"] == 2048
