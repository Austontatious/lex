# Avatar Generation Flow (Current State)

## What runs today
- Endpoint: `POST /lexi/gen/avatar` (frontend → backend FastAPI).
- Backend submission path: `backend/lexi/sd/comfy_client.py` (Fusion workflow template injection).
- Active workflow file (inside backend container): defaults to `/app/docker/comfy/workflows/flux_workflow_api.json` because `FLUX_WORKFLOW_PATH` is unset.
- Host copy of that workflow: `/mnt/data/Lex/docker/comfy/workflows/flux_workflow_api.json` (symlinked as `docker/comfy/workflows_canonical/01_flux_workflow_api.json`).
- ControlNet: only applies if present in the active workflow JSON. The programmatic ControlNet graph in `backend/lexi/sd/sd_pipeline.py` is **not** the active avatar path right now.

## Verification commands
Run from the host:

```bash
# Inspect Comfy/Flux env vars in the running backend
docker exec lex-lexi-backend-1 printenv | rg -i 'FLUX_WORKFLOW|COMFY|LEX|SD|PIPELINE'

# Confirm workflow payload submission (Fusion/template path)
docker logs --tail=300 lex-lexi-backend-1 | rg -i 'flux_txt2img|comfy prompt payload|prompt_id|workflow'

# Optional: confirm Comfy is receiving jobs
docker logs --tail=200 comfy-sd 2>/dev/null | rg -i 'prompt|queue|execut'
```

## Notes on programmatic graphs
- Programmatic Flux graphs live in `backend/lexi/sd/sd_pipeline.py` (includes ControlNet wiring and prompt logging), but the avatar route is currently using the Fusion workflow templates instead.
- No runtime paths were changed during this documentation pass; the active workflow remains whatever `FLUX_WORKFLOW_PATH` resolves to in the backend container.
