# Repo Audit (pre-cleanup)

Timestamp: 2026-01-03 21:16:07 UTC
Branch: fix/sonar-reliability
Last commit: b58ec7c fix: address sonar reliability and security findings

## Tracked suspicious artifacts
Exact `git ls-files | rg "\\.bak$|\\.(zip|log)$|^WIP_|sonar-hotspots\\.json|legacy_users_inventory\\.json|identity_merge_plan\\.json"` output:

```
WIP_DIFF_STAGED.txt
WIP_DIFF_UNSTAGED.txt
WIP_STATUS_BEFORE.txt
identity_merge_plan.json
legacy_users_inventory.json
```

## Untracked artifacts in repo root
From `git status --ignored` and root listing, the following root-level artifacts are untracked/ignored:

- affected_logs.zip
- lexi_backend_20250929_0644.zip
- logs.zip
- onboarding_tour_bundle.zip
- sonar-hotspots.json
- startup.log

## Duplicate trees / override trees
- backend/lexi vs Lexi/lexi (snapshot tree)
- overrides/backend_routes (compose-mounted route overrides)
- _backup (preserved snapshots)
- _quarantine (duplicates/conflicts)

## Root snapshot (top of listing)
`ls -la | head`:

```
total 334756
drwxrwxr-x 32 unix unix      4096 Jan  3 11:21 .
drwxr-xr-x 74 unix unix      4096 Dec 30 17:46 ..
-rw-r--r--  1 unix unix    160618 Dec  5 06:47 affected_logs.zip
-rw-rw-r--  1 unix unix      4516 Jan  3 12:16 AGENT.md
drwxrwxr-x  2 unix unix      4096 Jan  2 19:05 assets
drwxr-xr-x  5 unix unix      4096 Jan  2 19:05 backend
drwxr-xr-x  5 unix unix      4096 Nov 25 17:40 _backup
-rw-rw-r--  1 unix unix      1212 Jan  3 12:16 CHANGELOG.md
drwxr-xr-x  2 unix unix      4096 Jan  2 19:05 cloudflared
```

## Root file list
`find . -maxdepth 1 -type f -printf "%f\\n" | sort`:

```
affected_logs.zip
AGENT.md
CHANGELOG.md
constraints.alpha.txt
docker-compose.comfy.yml
docker-compose.override.prod.yml
docker-compose.override.yml
docker-compose.yml
.dockerignore
.env
.env.development
.env.example
.env.lexi
.env.production
Friday_AI Lex_Dreamboard.md
.gitignore
identity_merge_plan.json
__init__.py
legacy_users_inventory.json
lex_emotion_upgrade_plans.txt
lexi_backend_20250929_0644.zip
lexi-vllm-README.md
logs.zip
Makefile
onboarding_tour_bundle.zip
pyproject.toml
pytest.ini
README-ALPHA-DEPLOY.md
README.md
requirements.txt
run_backend.sh
sonar-hotspots.json
sonar-project.properties
start_lexi.py
startup.log
WIP_DIFF_STAGED.txt
WIP_DIFF_UNSTAGED.txt
WIP_STATUS_BEFORE.txt
```

## Cleanup notes (this change)
- Moved root-level artifacts (zip/log/WIP/identity/sonar) into artifacts/local/ to de-clutter the repo root.
- Moved frontend/public/asset-manifest.json (tracked build artifact) into artifacts/local/.
- Moved fusion_fill_edit_WORKFLOW.json (duplicate of fusion_fill_edit_api.json, unused) into artifacts/local/duplicates/.
- Added a startup log banner when overrides are active, detected via marker in overrides/backend_routes/lexi_persona.py.
- Moved in-tree .bak files into artifacts/local/bak/ with path-preserving filenames.
