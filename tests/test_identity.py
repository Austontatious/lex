import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest

from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.identity.normalize import normalize_handle
from backend.lexi.user_identity import resolve_identity


class DummyRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = types.SimpleNamespace()


def _load_migrate_identity():
    path = Path(__file__).resolve().parents[1] / "tools" / "migrate_identity.py"
    spec = importlib.util.spec_from_file_location("migrate_identity", path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(module)
    return module


def test_device_binding_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "identity.db"
    store = IdentityStore(db_path=db_path)
    user_id = store.create_user()
    device_id = f"device-{uuid.uuid4().hex}"
    store.bind_device(device_id, user_id)

    store2 = IdentityStore(db_path=db_path)
    assert store2.get_device_binding(device_id) == user_id


def test_handle_collision_needs_disambiguation(tmp_path: Path) -> None:
    store = IdentityStore(db_path=tmp_path / "identity.db")
    handle_norm = normalize_handle("Auston")
    user_a = store.create_user()
    user_b = store.create_user()
    store.upsert_handle(handle_norm, user_a, "Auston")
    store.upsert_handle(handle_norm, user_b, "Auston")

    request = DummyRequest(
        headers={
            "x-lexi-device": f"device-{uuid.uuid4().hex}",
            "x-lexi-handle": "Auston",
        }
    )

    user_id, handle_out, source, needs_disambiguation, candidates = resolve_identity(
        request, store=store
    )

    assert user_id is None
    assert handle_out == handle_norm
    assert source == "handle_collision"
    assert needs_disambiguation is True
    assert len(candidates) == 2


def test_identity_select_binds_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.lexi.routes import identity as identity_routes

    store = IdentityStore(db_path=tmp_path / "identity.db")
    handle_norm = normalize_handle("Auston")
    user_a = store.create_user()
    user_b = store.create_user()
    store.upsert_handle(handle_norm, user_a, "Auston")
    store.upsert_handle(handle_norm, user_b, "Auston")

    device_id = f"device-{uuid.uuid4().hex}"
    request = DummyRequest(headers={"x-lexi-device": device_id})
    request.state.device_id = device_id
    request.state.needs_disambiguation = True

    monkeypatch.setattr(identity_routes, "_def_store", store)

    payload = identity_routes.IdentitySelectPayload(
        handle="Auston",
        selected_user_id=user_b,
        merge_others=False,
    )
    resp = identity_routes.select_identity(payload, request)

    assert resp["user_id"] == user_b
    assert store.get_device_binding(device_id) == user_b


def test_migrate_identity_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory_root = tmp_path / "memory"
    user_root = tmp_path / "user_data"
    old_id = "Auston-12345"

    (memory_root / "users" / old_id).mkdir(parents=True, exist_ok=True)
    (user_root / "users" / old_id).mkdir(parents=True, exist_ok=True)

    (memory_root / "users" / old_id / "ltm.jsonl").write_text("{\"msg\": \"hi\"}\n")
    (user_root / "users" / old_id / "profile.json").write_text("{\"id\": \"old\"}\n")

    monkeypatch.setenv("LEXI_MEMORY_ROOT", str(memory_root))
    monkeypatch.setenv("LEX_USER_DATA_ROOT", str(user_root))
    monkeypatch.setenv("LEXI_IDENTITY_DB_PATH", str(tmp_path / "identity.db"))

    migrate_identity = _load_migrate_identity()

    argv = [
        "migrate_identity.py",
        "--handle",
        "Auston",
        "--include-ids",
        old_id,
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = migrate_identity.main()
    assert rc == 0
    assert not (memory_root / "users" / old_id / "MOVED_TO.txt").exists()
    assert not (user_root / "users" / old_id / "MOVED_TO.txt").exists()
