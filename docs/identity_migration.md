# Identity Migration (Legacy IDs -> Canonical)

Use `tools/migrate_identity.py` to merge legacy user directories into a
canonical `user_<uuid>` namespace.

## Requirements
- `LEXI_MEMORY_ROOT` points at the canonical memory root.
- `LEX_USER_DATA_ROOT` points at user data root.
- (Optional) `LEXI_IDENTITY_DB_PATH` for identity DB location.

## Dry run
```
python tools/migrate_identity.py \
  --handle "Auston" \
  --include-glob "Auston-*" \
  --dry-run
```

## Apply
```
python tools/migrate_identity.py \
  --handle "Auston" \
  --include-glob "Auston-*" \
  --apply
```

## Options
- `--handle` (required): handle used to resolve/create target user.
- `--target-user-id`: canonical `user_<uuid>` to merge into.
- `--include-glob`: glob for legacy ids (e.g., `Auston-*`).
- `--include-ids`: comma list of explicit legacy ids.
- `--dry-run`: default behavior; prints plan only.
- `--apply`: execute merges.
- `--backup-dir`: where to write backups (default: `<LEXI_MEMORY_ROOT>/_backup_YYYYMMDDhhmmss`).

## Merge behavior
- Memory: copy files from `<old>` to `<target>`.
  - identical sha256 -> skip
  - conflicts -> rename incoming file with `__from_<old>__<hash>` suffix
- Avatar/user data: same file-level policy.
- `avatars_manifest.json` is merged (history deduped by path/web_url/basename/sha256).
- Each old directory receives `MOVED_TO.txt` (on apply).
- `aliases` table records the redirect (`reason = migration_merge`).

## Safety
- Always run a dry-run first.
- Backups are created before applying changes.
