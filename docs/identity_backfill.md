# Identity Backfill Runbook

This runbook migrates legacy per-user namespaces into canonical `user_<uuid>` identities
without deleting any existing data.

## Environment
Set the correct paths for your deployment:
```
export LEXI_MEMORY_ROOT=/mnt/data/Lex/backend/memory
export LEX_USER_DATA_ROOT=/mnt/data/avatars
export LEXI_IDENTITY_DB_PATH=/mnt/data/Lex/data/identity/identity.db
```

## Step 1: Inventory legacy users
```
python3 tools/inventory_users.py --output /mnt/data/legacy_users_inventory.json
```

## Step 2: Build a conservative merge plan
```
python3 tools/plan_identity_merge.py \
  --inventory /mnt/data/legacy_users_inventory.json \
  --output /mnt/data/identity_merge_plan.json
```

## Step 3: Dry-run the merge
```
python3 tools/merge_legacy_identities.py \
  --plan /mnt/data/identity_merge_plan.json \
  --dry-run
```

## Step 4: Apply
```
python3 tools/merge_legacy_identities.py \
  --plan /mnt/data/identity_merge_plan.json \
  --apply
```

## Step 5: Verify
```
sqlite3 $LEXI_IDENTITY_DB_PATH 'select count(*) from users;'
ls -1 $LEXI_MEMORY_ROOT/users | head
curl -s http://localhost:9000/lexi/whoami
```

## Notes
- No legacy directories are deleted. Each legacy dir gets a `MOVED_TO.txt` marker.
- Backup copies are created under `<LEXI_MEMORY_ROOT>/_backup_YYYYMMDDhhmmss/`.
- If multiple users share the same handle, the plan keeps separate canonical users and
  flags `needs_disambiguation` so the UI can ask.
