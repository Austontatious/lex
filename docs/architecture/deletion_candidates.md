# Deletion Candidates (do not delete yet)

This is a planning list only; no files have been removed.

## Programmatic avatar path (currently inactive)
- `backend/lexi/sd/sd_pipeline.py` and related helpers (programmatic Flux graphs / ControlNet wiring). Active avatar routing uses Fusion workflow templates via `comfy_client.py`.
- Any future split-off helpers under `backend/lexi/sd/` that only serve the programmatic graph path.

## Workflow templates to evaluate
- Non-Fusion/Fusion-variant JSONs that are not referenced by the backend env (`FLUX_WORKFLOW_PATH` is unset and defaults to `flux_workflow_api.json`).
- Vendor/reference workflows under `docker/comfy/workflows_canonical/9x_*` and `96-98_*` (kept for debugging; safe to remove later if not needed).

## Misc assets
- Legacy pose assets under `data/flux_pose_assets/...` that are unused once the active pose set is confirmed.

Notes:
- Do not delete until confirmed unused in running environments and container mounts.
- Re-verify references with `rg` searches before any removal.
