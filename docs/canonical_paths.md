# Canonical Paths

This document defines which tree is authoritative when editing or debugging.

## Backend runtime (canonical)
- backend/lexi/** is the authoritative backend package used by tests and Docker builds.

## Frontend runtime (canonical)
- frontend/** is the authoritative frontend source.

## Snapshot tree (do not edit)
- Lexi/lexi/** is a frozen snapshot retained for historical deployments. Do not edit.

## Scripts
- backend/scripts/** is the canonical location for dev_up/dev_down/dev_status and guard scripts.
- Lexi/scripts/** is a snapshot duplicate; do not edit.
- scripts/** contains offline tooling (redteam, training, model prep); edit only if you are changing those workflows.

## Overrides
- overrides/backend_routes/** contains route overrides that are only active when mounted by docker-compose.override.yml.
- If not mounted, backend/lexi/routes/** is used as-is.

## Comfy workflows
Referenced by backend code (backend/lexi/sd/comfy_client.py):
- docker/comfy/workflows/flux_workflow_api.json (default via FLUX_WORKFLOW_PATH)
- docker/comfy/workflows/flux_workflow_api_v2.json (optional via FLUX_WORKFLOW_V2_PATH)
- docker/comfy/workflows/fusion_fill_edit_api.json (edit path via LEXI_COMFY_WORKFLOW_EDIT)

Other JSONs under docker/comfy/workflows/** and docker/comfy/workflows_canonical/** are reference templates or vendor examples.

## Deprecated
- start_lexi.py (use docker compose up -d)
- docker-compose.comfy.yml (legacy placeholder)
- backend/lexi/sd/sd_pipeline.py (programmatic graph builders, not the active avatar runtime)
