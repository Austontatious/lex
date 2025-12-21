#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))

from backend.lexi.identity.identity_store import IdentityStore
from backend.lexi.identity.normalize import normalize_handle, normalize_user_id_for_paths
from tools.inventory_users import build_inventory
from tools.plan_identity_merge import build_plan


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _merge_json_dict(target: Dict[str, Any], incoming: Dict[str, Any], legacy_id: str) -> Dict[str, Any]:
    legacy_tag = normalize_user_id_for_paths(legacy_id) or legacy_id
    for key, value in incoming.items():
        if key not in target:
            target[key] = value
            continue
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            target[key] = _merge_json_dict(existing, value, legacy_id)
            continue
        if isinstance(existing, list) and isinstance(value, list):
            seen = {json.dumps(v, sort_keys=True) for v in existing}
            for item in value:
                marker = json.dumps(item, sort_keys=True)
                if marker not in seen:
                    existing.append(item)
                    seen.add(marker)
            target[key] = existing
            continue
        if existing == value:
            continue
        conflict_key = f"{key}__from_{legacy_tag}"
        if conflict_key not in target:
            target[conflict_key] = value
    return target


def _merge_avatar_manifest(target_path: Path, incoming_path: Path, legacy_id: str, dry_run: bool) -> str:
    target = _load_json(target_path) or {}
    incoming = _load_json(incoming_path) or {}

    def _event_key(evt: Dict[str, Any]) -> str:
        for key in ("sha256", "path", "web_url", "basename"):
            val = evt.get(key) if isinstance(evt, dict) else None
            if val:
                return f"{key}:{val}"
        return json.dumps(evt, sort_keys=True)

    history: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for source in (target.get("history") or [], incoming.get("history") or []):
        if isinstance(source, list):
            for evt in source:
                if not isinstance(evt, dict):
                    continue
                key = _event_key(evt)
                if key in seen:
                    continue
                history.append(evt)
                seen.add(key)

    def _pick_by_ts(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not a:
            return b
        if not b:
            return a
        a_ts = str(a.get("created_at") or "")
        b_ts = str(b.get("created_at") or "")
        return b if b_ts > a_ts else a

    merged = dict(target)
    merged.setdefault("user_id", target.get("user_id") or incoming.get("user_id"))
    merged.setdefault("created_at", target.get("created_at") or incoming.get("created_at"))
    merged["updated_at"] = _now_iso()
    merged["history"] = history
    merged["first"] = merged.get("first") or incoming.get("first")
    merged["latest"] = _pick_by_ts(merged.get("latest"), incoming.get("latest"))

    if dry_run:
        return "merge-manifest(dry-run)"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return "merge-manifest"


def _merge_session_summaries(
    target_path: Path, incoming_path: Path, legacy_id: str, dry_run: bool
) -> str:
    target = _load_json(target_path) or {}
    incoming = _load_json(incoming_path) or {}

    merged = dict(target)
    merged.setdefault("user_id", target.get("user_id") or incoming.get("user_id"))
    merged["updated_at"] = _now_iso()

    summaries = []
    seen_sessions: Set[str] = set()
    for source in (target.get("summaries") or [], incoming.get("summaries") or []):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("session_id") or "")
            if session_id and session_id in seen_sessions:
                continue
            summaries.append(item)
            if session_id:
                seen_sessions.add(session_id)
    if summaries:
        merged["summaries"] = summaries

    facts_target = target.get("facts") if isinstance(target.get("facts"), dict) else {}
    facts_incoming = incoming.get("facts") if isinstance(incoming.get("facts"), dict) else {}
    merged["facts"] = _merge_json_dict(dict(facts_target), dict(facts_incoming), legacy_id)

    rolling_target = target.get("rolling_summary")
    rolling_incoming = incoming.get("rolling_summary")
    if rolling_incoming and not rolling_target:
        merged["rolling_summary"] = rolling_incoming
    elif rolling_incoming and rolling_target and rolling_incoming != rolling_target:
        tag = normalize_user_id_for_paths(legacy_id) or legacy_id
        merged[f"rolling_summary__from_{tag}"] = rolling_incoming

    if dry_run:
        return "merge-session-summaries(dry-run)"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return "merge-session-summaries"


def _merge_json_generic(
    target_path: Path, incoming_path: Path, legacy_id: str, dry_run: bool
) -> str:
    target = _load_json(target_path) or {}
    incoming = _load_json(incoming_path) or {}
    merged = _merge_json_dict(dict(target), dict(incoming), legacy_id)
    if dry_run:
        return "merge-json(dry-run)"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return "merge-json"


def _merge_jsonl(target_path: Path, incoming_path: Path, dry_run: bool) -> Dict[str, int]:
    appended = 0
    skipped = 0
    bytes_written = 0
    if not target_path.exists():
        if dry_run:
            return {"appended": 0, "skipped": 0, "bytes": incoming_path.stat().st_size}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(incoming_path, target_path)
        return {"appended": 0, "skipped": 0, "bytes": incoming_path.stat().st_size}

    seen: Set[str] = set()
    with target_path.open("rb") as fh:
        for line in fh:
            seen.add(hashlib.sha256(line).hexdigest())

    if dry_run:
        with incoming_path.open("rb") as fh:
            for line in fh:
                if hashlib.sha256(line).hexdigest() in seen:
                    skipped += 1
                else:
                    appended += 1
                    bytes_written += len(line)
        return {"appended": appended, "skipped": skipped, "bytes": bytes_written}

    with target_path.open("ab") as out:
        with incoming_path.open("rb") as fh:
            for line in fh:
                digest = hashlib.sha256(line).hexdigest()
                if digest in seen:
                    skipped += 1
                    continue
                out.write(line)
                appended += 1
                bytes_written += len(line)
                seen.add(digest)

    return {"appended": appended, "skipped": skipped, "bytes": bytes_written}


def _copy_file(src: Path, dest: Path, legacy_id: str, dry_run: bool) -> Tuple[str, int]:
    if dest.exists():
        if _sha256(src) == _sha256(dest):
            return "skip-identical", 0
        tag = normalize_user_id_for_paths(legacy_id) or legacy_id
        suffix = f"__from_{tag}__{_sha256(src)[:8]}"
        new_dest = dest.with_name(f"{dest.stem}{suffix}{dest.suffix}")
        if dry_run:
            return f"rename->{new_dest.name}", 0
        new_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_dest)
        return f"rename->{new_dest.name}", new_dest.stat().st_size
    if dry_run:
        return "copy", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "copy", dest.stat().st_size


def _write_marker(old_dir: Path, target_id: str, handle_norm: str, dry_run: bool) -> None:
    if not old_dir.exists():
        return
    marker = old_dir / "MOVED_TO.txt"
    content = f"Moved to {target_id} (handle={handle_norm}) at {_now_iso()}\n"
    if dry_run:
        return
    marker.write_text(content, encoding="utf-8")


def _backup_dir(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run or not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def execute_plan(
    plan: Dict[str, Any],
    *,
    memory_root: Path,
    user_data_root: Path,
    identity_db: Path,
    dry_run: bool,
    backup_dir: Optional[Path] = None,
    max_ambiguity: Optional[int] = None,
) -> Dict[str, Any]:
    plan_entries = plan.get("plan", [])
    if not isinstance(plan_entries, list):
        raise ValueError("plan.plan must be a list")

    ambiguous = [p for p in plan_entries if p.get("needs_disambiguation")]
    if max_ambiguity is not None and len(ambiguous) > max_ambiguity:
        raise ValueError("too many ambiguous merges; review plan")

    store = IdentityStore(db_path=identity_db)

    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "memory_root": str(memory_root),
        "user_data_root": str(user_data_root),
        "identity_db": str(identity_db),
        "created_users": 0,
        "aliases_added": 0,
        "handles_added": 0,
        "bytes_copied": 0,
        "conflicts": 0,
        "jsonl_appended": 0,
        "processed": [],
        "ambiguous": ambiguous,
        "backup_dir": str(backup_dir) if backup_dir else None,
    }

    for entry in plan_entries:
        canonical_user_id = entry.get("canonical_user_id")
        if not canonical_user_id:
            canonical_user_id = f"user_{uuid.uuid4()}"
            entry["canonical_user_id"] = canonical_user_id

        if not dry_run:
            store.ensure_user(canonical_user_id)
        report["created_users"] += 1

        handle_norm = entry.get("handle_norm") or ""
        handle_links = entry.get("handle_links") or []
        for handle_raw in handle_links:
            norm = handle_norm or normalize_handle(handle_raw)
            if not norm:
                continue
            if not dry_run:
                store.upsert_handle(norm, canonical_user_id, handle_raw)
                store.increment_handle_use(norm, canonical_user_id)
            report["handles_added"] += 1

        merge_from = entry.get("merge_from") or []
        for legacy_id in merge_from:
            if not dry_run:
                store.add_alias(legacy_id, canonical_user_id, reason="legacy_backfill")
            report["aliases_added"] += 1

            mem_old = memory_root / "users" / legacy_id
            mem_target = memory_root / "users" / canonical_user_id
            user_old = user_data_root / "users" / legacy_id
            user_target = user_data_root / "users" / canonical_user_id

            if backup_dir:
                _backup_dir(mem_old, backup_dir / "memory" / "users" / legacy_id, dry_run)
                _backup_dir(user_old, backup_dir / "user_data" / "users" / legacy_id, dry_run)

            actions: List[str] = []

            if mem_old.exists():
                for src in mem_old.rglob("*"):
                    if src.is_dir():
                        continue
                    rel = src.relative_to(mem_old)
                    dest = mem_target / rel
                    if rel.name == "ltm.jsonl":
                        result = _merge_jsonl(dest, src, dry_run)
                        report["jsonl_appended"] += result["appended"]
                        report["bytes_copied"] += result["bytes"]
                        actions.append(f"memory/{rel}: merge-jsonl")
                        continue
                    if rel.name == "session_summaries.json":
                        action = _merge_session_summaries(dest, src, legacy_id, dry_run)
                        actions.append(f"memory/{rel}: {action}")
                        continue
                    if rel.suffix == ".json" and dest.exists():
                        action = _merge_json_generic(dest, src, legacy_id, dry_run)
                        actions.append(f"memory/{rel}: {action}")
                        continue
                    action, bytes_written = _copy_file(src, dest, legacy_id, dry_run)
                    if action.startswith("rename"):
                        report["conflicts"] += 1
                    report["bytes_copied"] += bytes_written
                    actions.append(f"memory/{rel}: {action}")

            if user_old.exists():
                for src in user_old.rglob("*"):
                    if src.is_dir():
                        continue
                    rel = src.relative_to(user_old)
                    dest = user_target / rel
                    if rel.name == "avatars_manifest.json" and dest.exists():
                        action = _merge_avatar_manifest(dest, src, legacy_id, dry_run)
                        actions.append(f"user_data/{rel}: {action}")
                        continue
                    if rel.suffix == ".json" and dest.exists():
                        action = _merge_json_generic(dest, src, legacy_id, dry_run)
                        actions.append(f"user_data/{rel}: {action}")
                        continue
                    action, bytes_written = _copy_file(src, dest, legacy_id, dry_run)
                    if action.startswith("rename"):
                        report["conflicts"] += 1
                    report["bytes_copied"] += bytes_written
                    actions.append(f"user_data/{rel}: {action}")

            _write_marker(mem_old, canonical_user_id, handle_norm, dry_run)
            _write_marker(user_old, canonical_user_id, handle_norm, dry_run)

            report["processed"].append(
                {
                    "legacy_id": legacy_id,
                    "canonical_user_id": canonical_user_id,
                    "actions": actions,
                }
            )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge legacy identities into canonical user ids")
    parser.add_argument("--memory-root")
    parser.add_argument("--user-data-root")
    parser.add_argument("--identity-db")
    parser.add_argument("--plan")
    parser.add_argument("--auto-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir")
    parser.add_argument("--max-ambiguity", type=int)
    parser.add_argument("--report")

    args = parser.parse_args()

    dry_run = True
    if args.apply:
        dry_run = False
    if args.dry_run:
        dry_run = True

    memory_root = Path(args.memory_root).expanduser().resolve() if args.memory_root else None
    user_data_root = Path(args.user_data_root).expanduser().resolve() if args.user_data_root else None
    identity_db = Path(args.identity_db).expanduser().resolve() if args.identity_db else None

    if not args.plan and not args.auto_plan:
        raise SystemExit("--plan or --auto-plan required")

    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    else:
        inventory = build_inventory(memory_root=memory_root, user_data_root=user_data_root)
        plan = build_plan(inventory, identity_db=identity_db)

    mem_root = Path(plan.get("memory_root") or memory_root or "").expanduser().resolve()
    data_root = Path(plan.get("user_data_root") or user_data_root or "").expanduser().resolve()
    db_path = Path(plan.get("identity_db") or identity_db or "").expanduser().resolve()

    backup_dir = None
    if args.backup_dir:
        backup_dir = Path(args.backup_dir).expanduser().resolve()
    elif not dry_run:
        backup_dir = mem_root / f"_backup_{_now_ts()}"

    report = execute_plan(
        plan,
        memory_root=mem_root,
        user_data_root=data_root,
        identity_db=db_path,
        dry_run=dry_run,
        backup_dir=backup_dir,
        max_ambiguity=args.max_ambiguity,
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
