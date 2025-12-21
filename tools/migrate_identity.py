#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.identity.normalize import is_canonical_user_id, normalize_handle
from backend.lexi.memory.memory_core import resolve_memory_root
from backend.lexi.utils.user_profile import user_data_root


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plan_ids(memory_root: Path, user_root: Path, include_glob: Optional[str], include_ids: List[str]) -> Set[str]:
    ids: Set[str] = {i for i in include_ids if i}
    if include_glob:
        for base in (memory_root, user_root):
            globbed = base / "users" / include_glob
            for match in globbed.parent.glob(globbed.name):
                if match.is_dir():
                    ids.add(match.name)
    return ids


def _merge_manifest(old_path: Path, target_path: Path, dry_run: bool) -> str:
    try:
        old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    except Exception:
        old = {}
    try:
        target = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else {}
    except Exception:
        target = {}

    def _event_key(evt: Dict) -> str:
        for key in ("sha256", "path", "web_url", "basename"):
            val = evt.get(key) if isinstance(evt, dict) else None
            if val:
                return f"{key}:{val}"
        return json.dumps(evt, sort_keys=True)

    history: List[Dict] = []
    seen: Set[str] = set()
    for source in (target.get("history") or [], old.get("history") or []):
        if isinstance(source, list):
            for evt in source:
                if not isinstance(evt, dict):
                    continue
                key = _event_key(evt)
                if key in seen:
                    continue
                history.append(evt)
                seen.add(key)

    def _pick_latest(entries: Iterable[Dict]) -> Optional[Dict]:
        best = None
        best_ts = ""
        for evt in entries:
            if not isinstance(evt, dict):
                continue
            ts = str(evt.get("created_at") or "")
            if ts > best_ts:
                best_ts = ts
                best = evt
        return best

    merged = dict(target)
    merged.setdefault("user_id", target.get("user_id") or old.get("user_id"))
    merged.setdefault("created_at", target.get("created_at") or old.get("created_at"))
    merged["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    merged["history"] = history

    first = target.get("first") or old.get("first")
    latest = _pick_latest([target.get("latest") or {}, old.get("latest") or {}])
    if first:
        merged["first"] = first
    if latest:
        merged["latest"] = latest

    if dry_run:
        return "merge-manifest(dry-run)"

    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return "merge-manifest"


def _copy_file(src: Path, dest: Path, old_id: str, dry_run: bool) -> str:
    if dest.exists():
        if _sha256(src) == _sha256(dest):
            return "skip-identical"
        suffix = f"__from_{old_id}__{_sha256(src)[:8]}"
        new_name = dest.with_name(f"{dest.stem}{suffix}{dest.suffix}")
        if dry_run:
            return f"rename->{new_name.name}"
        new_name.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_name)
        return f"rename->{new_name.name}"
    if dry_run:
        return "copy"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "copy"


def _merge_dir(
    old_dir: Path, target_dir: Path, old_id: str, dry_run: bool, *, exclude: Optional[Set[str]] = None
) -> List[str]:
    actions: List[str] = []
    if not old_dir.exists():
        return actions
    exclude = exclude or set()
    for src in old_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(old_dir)
        if rel.as_posix() in exclude:
            continue
        dest = target_dir / rel
        actions.append(f"{rel}: {_copy_file(src, dest, old_id, dry_run)}")
    return actions


def _write_marker(old_dir: Path, target_id: str, handle_norm: str, dry_run: bool) -> None:
    if not old_dir.exists():
        return
    marker = old_dir / "MOVED_TO.txt"
    content = f"Moved to {target_id} (handle={handle_norm}) at {datetime.now(timezone.utc).isoformat()}\n"
    if dry_run:
        return
    marker.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge legacy identity directories into canonical users")
    parser.add_argument("--handle", required=True)
    parser.add_argument("--target-user-id")
    parser.add_argument("--include-glob")
    parser.add_argument("--include-ids")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir")

    args = parser.parse_args()
    dry_run = args.dry_run and not args.apply

    memory_root = resolve_memory_root()
    user_root = user_data_root()

    handle_norm = normalize_handle(args.handle)
    if not handle_norm:
        print("Invalid handle")
        return 2

    store = IdentityStore()
    target_user_id = args.target_user_id
    if target_user_id:
        if not is_canonical_user_id(target_user_id):
            print("target-user-id must be canonical user_<uuid>")
            return 2
        store.ensure_user(target_user_id)
    else:
        candidates = store.list_handle_candidates(handle_norm)
        if len(candidates) == 1:
            target_user_id = candidates[0].get("user_id")
        elif len(candidates) > 1:
            print("Multiple candidates for handle; pass --target-user-id")
            return 2
        else:
            target_user_id = store.create_user()
            store.upsert_handle(handle_norm, target_user_id, args.handle)

    if not target_user_id:
        print("Failed to resolve target user id")
        return 2

    include_ids = []
    if args.include_ids:
        include_ids = [s.strip() for s in args.include_ids.split(",") if s.strip()]

    ids = _plan_ids(memory_root, user_root, args.include_glob, include_ids)
    ids.discard(target_user_id)

    if not ids:
        print("No legacy ids matched")
        return 0

    backup_dir = Path(args.backup_dir) if args.backup_dir else memory_root / f"_backup_{_now_ts()}"

    print(f"Target user: {target_user_id}")
    print(f"Handle norm: {handle_norm}")
    print(f"Legacy ids: {sorted(ids)}")
    print(f"Memory root: {memory_root}")
    print(f"User data root: {user_root}")
    print(f"Backup dir: {backup_dir}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")

    for old_id in sorted(ids):
        mem_old = memory_root / "users" / old_id
        mem_target = memory_root / "users" / target_user_id
        user_old = user_root / "users" / old_id
        user_target = user_root / "users" / target_user_id

        if not dry_run:
            if mem_old.exists():
                dest = backup_dir / "memory" / "users" / old_id
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(mem_old, dest, dirs_exist_ok=True)
            if user_old.exists():
                dest = backup_dir / "user_data" / "users" / old_id
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(user_old, dest, dirs_exist_ok=True)

        print(f"\nMerging {old_id} -> {target_user_id}")

        mem_actions = _merge_dir(mem_old, mem_target, old_id, dry_run)
        for action in mem_actions:
            print(f"  memory: {action}")

        manifest_old = user_old / "avatars_manifest.json"
        manifest_target = user_target / "avatars_manifest.json"
        if manifest_old.exists():
            action = _merge_manifest(manifest_old, manifest_target, dry_run)
            print(f"  user_data: avatars_manifest.json: {action}")

        user_actions = _merge_dir(
            user_old, user_target, old_id, dry_run, exclude={"avatars_manifest.json"}
        )
        for action in user_actions:
            print(f"  user_data: {action}")

        _write_marker(mem_old, target_user_id, handle_norm, dry_run)
        _write_marker(user_old, target_user_id, handle_norm, dry_run)

        if not dry_run:
            store.add_alias(old_id, target_user_id, reason="migration_merge")

    print("\nDone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
