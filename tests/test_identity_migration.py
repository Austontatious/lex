import importlib.util
import types
from pathlib import Path

import pytest

from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.user_identity import resolve_identity


class DummyRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = types.SimpleNamespace()


def _load_tool(module_name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def test_plan_groups_sessionish_suffixes(tmp_path: Path) -> None:
    inventory_mod = _load_tool("inventory_users")
    plan_mod = _load_tool("plan_identity_merge")

    mem_root = tmp_path / "memory"
    user_root = tmp_path / "user_data"
    legacy_ids = [
        "Auston-12345",
        "Auston-23456",
        "tester@example.com",
        "tester@example.com-9999",
    ]
    for legacy_id in legacy_ids:
        (mem_root / "users" / legacy_id).mkdir(parents=True, exist_ok=True)
        (user_root / "users" / legacy_id).mkdir(parents=True, exist_ok=True)

    inventory = inventory_mod.build_inventory(
        memory_root=mem_root, user_data_root=user_root, identity_db=tmp_path / "identity.db"
    )
    plan = plan_mod.build_plan(inventory, identity_db=tmp_path / "identity.db")

    grouped = [p for p in plan["plan"] if set(p["merge_from"]) == {"Auston-12345", "Auston-23456"}]
    assert grouped, "Expected Auston-* to be grouped"

    email_groups = [p for p in plan["plan"] if "tester@example.com" in p["merge_from"]]
    assert len(email_groups) == 1
    mixed = [p for p in plan["plan"] if "tester@example.com-9999" in p["merge_from"]]
    assert len(mixed) == 1
    assert email_groups[0]["canonical_user_id"] != mixed[0]["canonical_user_id"]


def test_execute_plan_and_alias_resolution(tmp_path: Path) -> None:
    inventory_mod = _load_tool("inventory_users")
    plan_mod = _load_tool("plan_identity_merge")
    merge_mod = _load_tool("merge_legacy_identities")

    mem_root = tmp_path / "memory"
    user_root = tmp_path / "user_data"

    legacy_a = "Auston-12345"
    legacy_b = "Auston-23456"

    _write_jsonl(mem_root / "users" / legacy_a / "ltm.jsonl", ["{\"msg\": \"hello\"}"])
    _write_jsonl(mem_root / "users" / legacy_b / "ltm.jsonl", ["{\"msg\": \"hi\"}"])

    inventory = inventory_mod.build_inventory(
        memory_root=mem_root, user_data_root=user_root, identity_db=tmp_path / "identity.db"
    )
    plan = plan_mod.build_plan(inventory, identity_db=tmp_path / "identity.db")

    report = merge_mod.execute_plan(
        plan,
        memory_root=mem_root,
        user_data_root=user_root,
        identity_db=tmp_path / "identity.db",
        dry_run=False,
        backup_dir=tmp_path / "backup",
    )
    assert report["created_users"] > 0

    auston_entry = None
    for entry in plan["plan"]:
        if set(entry.get("merge_from", [])) == {legacy_a, legacy_b}:
            auston_entry = entry
            break
    assert auston_entry
    canonical = auston_entry["canonical_user_id"]

    ltm_path = mem_root / "users" / canonical / "ltm.jsonl"
    assert ltm_path.exists()
    text = ltm_path.read_text(encoding="utf-8")
    assert "hello" in text and "hi" in text

    store = IdentityStore(db_path=tmp_path / "identity.db")
    request = DummyRequest(headers={"x-lexi-user": legacy_a, "x-lexi-device": "device-1"})
    user_id, _, source, _, _ = resolve_identity(request, store=store)
    assert user_id == canonical
    assert source in {"alias_header", "device_binding", "session_binding", "header_user"}
