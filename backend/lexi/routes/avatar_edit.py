from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..sd.edit_workflow import submit_avatar_edit, check_avatar_edit_status

log = logging.getLogger(__name__)
router = APIRouter()


class AvatarEditRequest(BaseModel):
    image_path: str = Field(..., description="Path or filename Comfy can load")
    target: str = Field(..., description="Edit target (hair, jacket, dress, etc.)")
    prompt: str = Field(..., description="Inpaint prompt for the target region")
    preserve_identity: bool = True
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    denoise: float | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    return_on_submit: bool = True
    timeout_s: int = 90


@router.post("/avatar/edit")
async def avatar_edit(req: AvatarEditRequest):
    """
    Submit an avatar edit job using the dedicated Fusion Fill workflow.
    """
    try:
        result = await run_in_threadpool(
            submit_avatar_edit,
            image_path=req.image_path,
            target=req.target,
            prompt=req.prompt,
            preserve_identity=req.preserve_identity,
            seed=req.seed,
            steps=req.steps,
            cfg=req.cfg,
            denoise=req.denoise,
            sampler_name=req.sampler_name,
            scheduler=req.scheduler,
            timeout_s=req.timeout_s,
            return_on_submit=req.return_on_submit,
        )
        return result
    except Exception as exc:
        log.warning("avatar_edit failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/avatar/edit/status/{prompt_id}")
async def avatar_edit_status(prompt_id: str):
    """
    Poll the edit workflow result and download it when ready.
    """
    try:
        return await run_in_threadpool(check_avatar_edit_status, prompt_id)
    except Exception as exc:
        log.warning("avatar_edit_status failed for %s: %s", prompt_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
