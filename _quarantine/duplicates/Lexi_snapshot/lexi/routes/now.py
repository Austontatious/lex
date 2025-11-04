from fastapi import APIRouter
from ..utils.now_models import NowItem, WebDoc, WebSearchRequest
from ..utils.now_ingest import query_now
from ..utils.now_summarize import summarize_for_smalltalk
from ..config.now import settings_now

router = APIRouter(prefix="/now", tags=["now"])

@router.get("/", response_model=list[NowItem])
async def get_now(category: str|None = None, interests: str|None = None, limit: int|None = None):
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
