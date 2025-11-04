import pytest
from fastapi.testclient import TestClient

from lexi.core.backend_core import app


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch, tmp_path):
    fake_avatar = tmp_path / "avatar.png"
    fake_avatar.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "lexi.routes.diagnostic.generate_avatar_pipeline",
        lambda **_: {"ok": True, "url": "/static/test.png"},
    )
    monkeypatch.setattr(
        "lexi.sd.sd_pipeline.generate_avatar_pipeline",
        lambda **_: {"ok": True, "url": "/static/test.png"},
    )
    monkeypatch.setattr(
        "lexi.sd.generate.generate_avatar",
        lambda **_: "/static/test.png",
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("lexi.persona.persona_core.lexi_persona.chat", lambda _msg: "ping")
    monkeypatch.setattr("lexi.persona.persona_core.lexi_persona._load_traits_state", lambda: True)
    monkeypatch.setattr(
        "lexi.persona.persona_core.lexi_persona.get_avatar_path",
        lambda: str(fake_avatar),
    )


def test_routes_smoke():
    client = TestClient(app)

    resp = client.get("/lexi/health")
    assert resp.status_code == 200

    resp = client.post("/lexi/gen/avatar", json={"prompt": "hi"})
    assert resp.status_code == 200

    resp = client.get("/lexi/diagnostic")
    assert resp.status_code == 200
