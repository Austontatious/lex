#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from backend.lexi.identity.normalize import is_canonical_user_id


def resolve_memory_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.getenv("LEXI_MEMORY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [
        Path("/mnt/data/Lex/data/memory"),
        Path("/data/memory"),
        REPO_ROOT / "data" / "memory",
    ]
    for cand in candidates:
        try:
            if cand.exists() or cand.parent.exists():
                return cand.resolve()
        except Exception:
            continue
    return (REPO_ROOT / "data" / "memory").resolve()


def resolve_user_data_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.getenv("LEX_USER_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    default_path = REPO_ROOT / "memory"
    legacy_path = REPO_ROOT / "backend" / "memory"
    if legacy_path.exists() and not default_path.exists():
        return legacy_path.resolve()
    return default_path.resolve()


def resolve_identity_db(explicit: Optional[str], user_data_root: Path) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.getenv("LEXI_IDENTITY_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (user_data_root / "identity" / "identity.db").resolve()


def _dir_stats(path: Path) -> Dict[str, int]:
    count = 0
    total_bytes = 0
    if not path.exists():
        return {"file_count": 0, "bytes": 0}
    for fp in path.rglob("*"):
        if fp.is_file():
            count += 1
            try:
                total_bytes += fp.stat().st_size
            except Exception:
                pass
    return {"file_count": count, "bytes": total_bytes}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _collect_hints(user_dir: Path) -> Dict[str, Any]:
    hint_sets: Dict[str, Set[str]] = {
        "email": set(),
        "display_name": set(),
        "username": set(),
        "user_id": set(),
    }
    json_files = [
        "profile.json",
        "user_profile.json",
        "avatars_manifest.json",
        "session_summaries.json",
        "session_index.json",
        "persona_state.json",
    ]
    for name in json_files:
        path = user_dir / name
        if not path.exists():
            continue
        data = _load_json(path) or {}
        for key, hint_key in (
            ("email", "email"),
            ("display_name", "display_name"),
            ("username", "username"),
            ("user_id", "user_id"),
            ("id", "user_id"),
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                hint_sets[hint_key].add(value.strip())
    hints: Dict[str, Any] = {}
    for key, values in hint_sets.items():
        if not values:
            continue
        ordered = sorted(values)
        hints[key] = ordered[0] if len(ordered) == 1 else ordered
    return hints


def _scan_user_dirs(root: Path, label: str, users: Dict[str, Dict[str, Any]]) -> None:
    base = root / "users"
    if not base.exists():
        return
    for user_dir in sorted(base.iterdir()):
        if not user_dir.is_dir():
            continue
        legacy_id = user_dir.name
        entry = users.setdefault(
            legacy_id,
            {
                "legacy_id": legacy_id,
                "kind": "canonical" if is_canonical_user_id(legacy_id) else "legacy",
                "paths": {},
                "hints": {},
                "stats": {},
            },
        )
        entry["paths"][label] = str(user_dir)
        entry["stats"][label] = _dir_stats(user_dir)
        hints = _collect_hints(user_dir)
        if hints:
            merged = dict(entry.get("hints") or {})
            for key, value in hints.items():
                existing = merged.get(key)
                if existing is None:
                    merged[key] = value
                else:
                    existing_list = existing if isinstance(existing, list) else [existing]
                    add_list = value if isinstance(value, list) else [value]
                    merged[key] = sorted({*(existing_list), *(add_list)})
            entry["hints"] = merged


def _db_counts(db_path: Path) -> Dict[str, Any]:
    if not db_path.exists():
        return {"exists": False}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(1) FROM users")
        users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM handles")
        handles = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM aliases")
        aliases = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM device_bindings")
        devices = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM sessions")
        sessions = cur.fetchone()[0]
        conn.close()
        return {
            "exists": True,
            "users": int(users),
            "handles": int(handles),
            "aliases": int(aliases),
            "device_bindings": int(devices),
            "sessions": int(sessions),
        }
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def _scan_artifacts(paths: Iterable[Path]) -> List[Dict[str, str]]:
    tokens = ["session", "registry", "account", "users", "alpha", "lexi_persona_state"]
    pattern = re.compile("|".join(tokens), re.IGNORECASE)
    seen: Set[str] = set()
    results: List[Dict[str, str]] = []
    for root in paths:
        if not root.exists():
            continue
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            name = fp.name.lower()
            if not pattern.search(name):
                continue
            key = str(fp)
            if key in seen:
                continue
            seen.add(key)
            results.append({"path": str(fp), "reason": "filename_match"})
    return results


def _scan_content(repo_root: Path) -> List[Dict[str, Any]]:
    phrases = ["alpha session registry", "session_manager.py"]
    results: List[Dict[str, Any]] = []
    for fp in repo_root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".py", ".md", ".txt", ".json"}:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for phrase in phrases:
            if phrase in text:
                results.append({"path": str(fp), "match": phrase})
    return results


def build_inventory(
    *,
    memory_root: Optional[Path] = None,
    user_data_root: Optional[Path] = None,
    identity_db: Optional[Path] = None,
) -> Dict[str, Any]:
    mem_root = resolve_memory_root(str(memory_root) if memory_root else None)
    data_root = resolve_user_data_root(str(user_data_root) if user_data_root else None)
    db_path = resolve_identity_db(str(identity_db) if identity_db else None, data_root)

    users: Dict[str, Dict[str, Any]] = {}
    _scan_user_dirs(mem_root, "memory", users)
    _scan_user_dirs(data_root, "user_data", users)

    for entry in users.values():
        stats = entry.get("stats", {})
        total_files = 0
        total_bytes = 0
        for sub in stats.values():
            total_files += sub.get("file_count", 0)
            total_bytes += sub.get("bytes", 0)
        entry["stats"] = {**stats, "total_files": total_files, "total_bytes": total_bytes}

    artifact_paths = _scan_artifacts([REPO_ROOT, mem_root, data_root])
    content_hits = _scan_content(REPO_ROOT)

    return {
        "memory_root": str(mem_root),
        "user_data_root": str(data_root),
        "identity_db": str(db_path),
        "identity_db_stats": _db_counts(db_path),
        "users": sorted(users.values(), key=lambda item: item.get("legacy_id", "")),
        "artifacts": {
            "file_matches": artifact_paths,
            "content_hits": content_hits,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory legacy and canonical users on disk")
    parser.add_argument("--memory-root")
    parser.add_argument("--user-data-root")
    parser.add_argument("--identity-db")
    parser.add_argument("--output")

    args = parser.parse_args()

    inventory = build_inventory(
        memory_root=Path(args.memory_root) if args.memory_root else None,
        user_data_root=Path(args.user_data_root) if args.user_data_root else None,
        identity_db=Path(args.identity_db) if args.identity_db else None,
    )

    payload = json.dumps(inventory, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
