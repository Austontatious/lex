from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_real_version = importlib_metadata.version


def _fake_version(name: str) -> str:
    if name == "email-validator":
        return "2.0.0"
    return _real_version(name)


importlib_metadata.version = _fake_version
if "email_validator" not in sys.modules:
    sys.modules["email_validator"] = types.SimpleNamespace(
        validate_email=lambda email, *args, **kwargs: {"email": email},
        EmailNotValidError=ValueError,
    )
tmp_mem = ROOT / "tmp" / "test_memory.jsonl"
tmp_mem.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LEX_MEMORY_PATH", str(tmp_mem))
os.environ.setdefault("LEXI_SKIP_FLUX_WARMUP", "1")
os.environ.setdefault("LEXI_ENABLE_NOW", "0")

from backend.lexi.routes import gen as gen_routes
from backend.lexi.routes import lexi as lexi_routes

app = FastAPI()
app.include_router(lexi_routes.router, prefix="/lexi")
app.include_router(gen_routes.router, prefix="/lexi")


def test_modal_avatar_endpoint_ok(monkeypatch) -> None:
    def fake_pipeline(*args, **kwargs):
        return {"ok": True, "url": "/static/avatars/test.png"}

    monkeypatch.setattr(gen_routes, "generate_avatar_pipeline", fake_pipeline)

    client = TestClient(app)
    resp = client.post("/lexi/gen/avatar", json={"prompt": "test prompt"})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True


def test_legacy_auto_appearance_extraction_noop(monkeypatch) -> None:
    def fake_chat(prompt: str) -> str:
        return "chat reply"

    def explode(*args, **kwargs):
        raise AssertionError("legacy auto appearance extraction should be no-op")

    monkeypatch.setattr(lexi_routes, "extract_traits_from_text", lambda text: {"hair": "blonde"})
    monkeypatch.setattr(lexi_routes, "generate_avatar_pipeline", explode)
    monkeypatch.setattr(lexi_routes.lexi_persona, "chat", fake_chat)

    client = TestClient(app)
    resp = client.post("/lexi/process", json={"prompt": "Please change her hair to blonde."})
    assert resp.status_code == 200
    body = resp.json()
    expected = (
        "Legacy auto appearance extraction has been removed. "
        "Use the Avatar Tools modal to update Lexi's look."
    )
    assert body.get("cleaned") == expected
    assert body.get("status") == "ignored"
    assert "avatar_url" not in body
