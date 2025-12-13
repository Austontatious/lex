# Avatar Pipeline Inventory

## Active path (observed)
- Workflow submission: `backend/lexi/sd/comfy_client.py` (Fusion workflow template injection).
- Workflow file in container: `/app/docker/comfy/workflows/flux_workflow_api.json` (FLUX_WORKFLOW_PATH is unset).
- Host source for that workflow: `/mnt/data/Lex/docker/comfy/workflows/flux_workflow_api.json` (symlinked as `docker/comfy/workflows_canonical/01_flux_workflow_api.json`).
- Recent backend logs show Fusion payload logs: `[lexi][flux_txt2img] comfy prompt payload: {...}` with numbered nodes; no programmatic ControlNet logging.

## Programmatic path (present, not active for avatar)
- Graph builders and ControlNet wiring: `backend/lexi/sd/sd_pipeline.py` (functions like `_flux_txt2img_graph`, `_flux_img2img_graph`, `_run_flux_backend`).
- Referenced by code (search):
  - `rg -n "sd_pipeline|generate_avatar_pipeline|_flux_txt2img_graph|_flux_img2img_graph" backend -S`
- Current avatar route behavior indicates the Fusion template path is being used instead of these programmatic graphs.

## Workflow file references in code (search commands)
- `rg -n "flux_workflow_.*\.json|workflows/.*\.json|FLUX_WORKFLOW_PATH" backend -S`
- `rg -n "comfy_client|comfy_flux|submit.*workflow|FLUX_WORKFLOW_PATH" backend -S`

## Files and locations
- Fusion/API templates: `/mnt/data/Lex/docker/comfy/workflows/*.json` (symlinked under `docker/comfy/workflows_canonical/`).
- Vendor/reference workflows: `/mnt/data/comfy/custom_nodes/x-flux-comfyui/workflows/*.json` (symlinked under `docker/comfy/workflows_canonical/9x_*`).
- FusionDraw reference set: `/mnt/data/comfy/input/workflowFLUXTxt2imgImg2img_v10/Workflow FLUX_FusionDraw9257/*.json` (symlinked under `docker/comfy/workflows_canonical/96-98_*`).

## Next actions (defer)
- If/when we switch avatar generation to the programmatic path, re-verify logging and ControlNet wiring there and adjust docs accordingly.
