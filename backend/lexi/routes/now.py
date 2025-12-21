import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from ..utils.now_models import MoviesNowRequest, NowItem, WebDoc, WebSearchRequest
from ..utils.now_ingest import query_now
from ..utils.movies_tool import movies_now
from ..utils.now_summarize import summarize_for_smalltalk
from ..config.now import settings_now
from ..memory.memory_core import memory
from ..user_identity import request_user_id

router = APIRouter(prefix="/now", tags=["now"])


@router.get("/", response_model=list[NowItem])
async def get_now(
    category: str | None = None, interests: str | None = None, limit: int | None = None
):
    interest_list = [s.strip() for s in (interests or "").split(",") if s.strip()]
    n = limit or settings_now.NOW_TOP_N_DEFAULT
    items = await query_now(category, interest_list, n)

    # Lazy summaries (cache may be cold)
    out = []
    for it in items:
        if not it.summary:
            s, bullets = await summarize_for_smalltalk(f"{it.title}\nSource: {it.source}")
            it.summary, it.talking_points = s, bullets
        out.append(it)
    return out


tools = APIRouter(prefix="/tools", tags=["tools"])


@tools.post("/web_search", response_model=list[WebDoc])
async def tool_web_search(req: WebSearchRequest):
    from ..utils.web_search_tool import web_search

    return await web_search(req)


@tools.post("/movies_now")
async def tool_movies_now(req: MoviesNowRequest):
    try:
        return await asyncio.to_thread(
            movies_now, req.location, req.start_date, req.end_date, req.limit
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


@tools.post("/memory_get_profile")
def tool_memory_get_profile(request: Request):
    if getattr(request.state, "needs_disambiguation", False):
        return {
            "needs_disambiguation": True,
            "candidates": getattr(request.state, "identity_candidates", []),
            "user_id": None,
        }
    user_id = request_user_id(request)
    memory.set_user(user_id)
    return memory.get_profile()


@tools.post("/memory_search_ltm")
def tool_memory_search_ltm(payload: MemorySearchRequest, request: Request):
    if getattr(request.state, "needs_disambiguation", False):
        return {
            "needs_disambiguation": True,
            "candidates": getattr(request.state, "identity_candidates", []),
            "results": [],
            "user_id": None,
        }
    user_id = request_user_id(request)
    memory.set_user(user_id)
    results = memory.memory_search_ltm(payload.query, k=payload.k)
    return {"results": results, "user_id": user_id}
