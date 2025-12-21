from pathlib import Path

from fastapi.testclient import TestClient

import backend.lexi.session as session_module
from backend.lexi.core.backend_core import app
from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.routes import account as account_routes
from backend.lexi.user_identity import reset_identity_store
from backend.lexi.utils.user_profile import AccountStore


def _stub_default_avatar(_ip: str, outdir: str) -> dict[str, object]:
    return {"seed": 0, "path": str(Path(outdir) / "default_0.png"), "created": False}


def test_account_bootstrap_non_500(tmp_path, monkeypatch) -> None:
    identity = IdentityStore(db_path=tmp_path / "identity.db")
    reset_identity_store(identity)
    try:
        monkeypatch.setattr(account_routes, "store", AccountStore(db_path=tmp_path / "accounts.sqlite3"))
        monkeypatch.setattr(session_module, "generate_default_avatar_for_ip", _stub_default_avatar)
        monkeypatch.setattr(session_module, "DEFAULT_AVATAR_MEDIA_DIR", tmp_path / "avatars")
        monkeypatch.setattr(session_module, "LOG_DIR", tmp_path / "sessions")
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

        client = TestClient(app)
        resp = client.post(
            "/lexi/account/bootstrap",
            json={"identifier": "testuser", "entry_mode": "new"},
            headers={"Origin": "https://lexicompanion.com"},
        )
        assert resp.status_code != 500
        body = resp.json()
        assert body["status"] in {"CREATED_NEW", "EXISTS_CONFLICT"}
        assert resp.headers.get("access-control-allow-origin") == "https://lexicompanion.com"
    finally:
        reset_identity_store()
