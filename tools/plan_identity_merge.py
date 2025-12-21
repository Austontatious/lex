#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.identity.normalize import normalize_handle

SESSION_SUFFIX_RE = re.compile(
    r"^(?P<base>[A-Za-z0-9._-]{2,})[-_](?P<suffix>(?:\d{4,}|[a-f0-9]{6,}|sess[a-z0-9]+|session[a-z0-9]+))$",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_inventory(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("inventory must be a JSON object")
    return data


def _resolve_identity_db(explicit: Optional[str], inventory: Dict[str, Any]) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser().resolve()
    from_env = os.getenv("LEXI_IDENTITY_DB_PATH")
    if from_env:
        return Path(from_env).expanduser().resolve()
    inv_db = inventory.get("identity_db")
    if inv_db:
        return Path(str(inv_db)).expanduser().resolve()
    return None


def _classify_legacy(legacy_id: str) -> Tuple[str, str, bool]:
    """Return (handle_raw, handle_norm, sessionish)."""
    if EMAIL_RE.match(legacy_id):
        handle_raw = legacy_id
        return handle_raw, normalize_handle(handle_raw), False
    match = SESSION_SUFFIX_RE.match(legacy_id)
    if match:
        handle_raw = match.group("base")
        return handle_raw, normalize_handle(handle_raw), True
    handle_raw = legacy_id
    return handle_raw, normalize_handle(handle_raw), False


def build_plan(
    inventory: Dict[str, Any],
    *,
    identity_db: Optional[Path] = None,
) -> Dict[str, Any]:
    legacy_entries = [u for u in inventory.get("users", []) if u.get("kind") != "canonical"]

    alias_map: Dict[str, str] = {}
    db_path = identity_db
    if db_path and db_path.exists():
        store = IdentityStore(db_path=db_path)
        try:
            rows = store.conn.execute("SELECT old_id, user_id FROM aliases").fetchall()
            alias_map = {str(row[0]): str(row[1]) for row in rows}
        except Exception:
            alias_map = {}
    elif identity_db is None:
        inv_db = inventory.get("identity_db")
        if inv_db:
            path = Path(str(inv_db)).expanduser().resolve()
            if path.exists():
                store = IdentityStore(db_path=path)
                try:
                    rows = store.conn.execute("SELECT old_id, user_id FROM aliases").fetchall()
                    alias_map = {str(row[0]): str(row[1]) for row in rows}
                except Exception:
                    alias_map = {}

    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for entry in legacy_entries:
        legacy_id = str(entry.get("legacy_id", ""))
        if not legacy_id:
            continue

        handle_raw, handle_norm, sessionish = _classify_legacy(legacy_id)
        is_email = bool(EMAIL_RE.match(legacy_id))

        alias_user = alias_map.get(legacy_id)
        if alias_user:
            key = ("alias", alias_user)
        elif sessionish and handle_norm:
            key = ("sessionish", handle_norm)
        elif is_email:
            key = ("email", legacy_id)
        else:
            key = ("single", legacy_id)

        group = groups.setdefault(
            key,
            {
                "handle_norm": handle_norm,
                "handle_links": set(),
                "merge_from": [],
                "canonical_user_id": alias_user,
                "needs_disambiguation": False,
            },
        )
        if handle_norm and not group.get("handle_norm"):
            group["handle_norm"] = handle_norm
        if handle_raw:
            group["handle_links"].add(handle_raw)
        group["merge_from"].append(legacy_id)

    plan_entries: List[Dict[str, Any]] = []
    for group in groups.values():
        canonical_user_id = group.get("canonical_user_id") or f"user_{uuid.uuid4()}"
        handle_links = sorted(group.get("handle_links", set()))
        plan_entries.append(
            {
                "handle_norm": group.get("handle_norm") or "",
                "canonical_user_id": canonical_user_id,
                "merge_from": sorted(group.get("merge_from", [])),
                "handle_links": handle_links,
                "needs_disambiguation": False,
            }
        )

    handle_counts: Dict[str, int] = {}
    for entry in plan_entries:
        handle_norm = entry.get("handle_norm") or ""
        if handle_norm:
            handle_counts[handle_norm] = handle_counts.get(handle_norm, 0) + 1

    ambiguity_count = 0
    for entry in plan_entries:
        handle_norm = entry.get("handle_norm") or ""
        if handle_norm and handle_counts.get(handle_norm, 0) > 1:
            entry["needs_disambiguation"] = True
            ambiguity_count += 1

    return {
        "generated_at": "auto",
        "memory_root": inventory.get("memory_root"),
        "user_data_root": inventory.get("user_data_root"),
        "identity_db": inventory.get("identity_db"),
        "plan": plan_entries,
        "summary": {
            "legacy_entries": len(legacy_entries),
            "canonical_plans": len(plan_entries),
            "ambiguous": ambiguity_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a conservative identity merge plan")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output")
    parser.add_argument("--identity-db")

    args = parser.parse_args()
    inventory = _load_inventory(Path(args.inventory))
    plan = build_plan(
        inventory,
        identity_db=Path(args.identity_db).expanduser().resolve() if args.identity_db else None,
    )

    payload = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
