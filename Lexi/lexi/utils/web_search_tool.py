import asyncio
import httpx
from typing import List, Callable, Awaitable
from datetime import datetime
from ..config.now import settings_now
from .now_models import WebSearchRequest, WebDoc
from .now_utils import log_now


def _as_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


async def web_search(req: WebSearchRequest) -> List[WebDoc]:
    provider = (req.provider or "auto").lower()
    pipeline: List[tuple[str, Callable[[WebSearchRequest], Awaitable[List[WebDoc]]]]] = []

    def _enqueue(name: str):
        if name == "tavily" and settings_now.TAVILY_API_KEY:
            pipeline.append((name, _tavily))
        elif name == "brave" and settings_now.BRAVE_API_KEY:
            pipeline.append((name, _brave))

    if provider == "brave":
        _enqueue("brave")
    elif provider == "tavily":
        _enqueue("tavily")
        if req.allow_brave_fallback:
            _enqueue("brave")
    else:  # auto
        _enqueue("tavily")
        if req.allow_brave_fallback:
            _enqueue("brave")

    if not pipeline:
        log_now("web_search: no providers configured; returning placeholder.")
        if req.stall_on_failure:
            await asyncio.sleep(0.5)
            return [_stall_doc("No search providers configured yet.")]
        return []

    errors: List[str] = []
    for name, fn in pipeline:
        try:
            docs = await fn(req)
            if docs:
                return docs
        except Exception as exc:
            msg = f"{name}: {exc}"
            errors.append(msg)
            log_now(f"web_search {msg}")
            continue

    if req.allow_brave_fallback and settings_now.BRAVE_API_KEY and "brave" not in [n for n, _ in pipeline]:
        log_now("web_search: brave fallback requested but no key configured.")

    if req.stall_on_failure:
        await asyncio.sleep(0.5)
        return [_stall_doc("; ".join(errors) if errors else None)]
    return []


async def _brave(req: WebSearchRequest) -> List[WebDoc]:
    headers = {"Accept": "application/json", "X-Subscription-Token": settings_now.BRAVE_API_KEY}
    params = {"q": req.query, "count": req.max_results}
    if req.time_range in {"day", "week", "month", "year"}:
        params["freshness"] = req.time_range
    if req.site_filters:
        params["q"] += " " + " ".join([f"site:{s}" for s in req.site_filters])

    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search", headers=headers, params=params
        )
        r.raise_for_status()
        j = r.json()

    docs: List[WebDoc] = []
    for it in j.get("web", {}).get("results", []):
        docs.append(
            WebDoc(
                url=it.get("url"),
                title=it.get("title"),
                snippet=it.get("description"),
                published_at=(
                    _as_dt(it.get("page_age")) if isinstance(it.get("page_age"), str) else None
                ),
                source=it.get("profile", {}).get("name"),
            )
        )
    return docs[: req.max_results]


async def _tavily(req: WebSearchRequest) -> List[WebDoc]:
    payload = {
        "api_key": settings_now.TAVILY_API_KEY,
        "query": req.query,
        "max_results": req.max_results,
        "include_answer": False,
        "include_raw_content": bool(req.include_content),
        "search_depth": "basic",
        "days": 7 if req.time_range in ("7d", "week", "day", "24h") else 30,
    }
    if req.site_filters:
        payload["query"] += " " + " ".join([f"site:{s}" for s in req.site_filters])

    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
        j = r.json()

    docs: List[WebDoc] = []
    for it in j.get("results", []):
        docs.append(
            WebDoc(
                url=it.get("url"),
                title=it.get("title"),
                snippet=it.get("content")[:400] if it.get("content") else None,
                content=it.get("raw_content") if req.include_content else None,
                published_at=_as_dt(it.get("published_date")),
                source=it.get("source"),
            )
        )
    return docs


def _stall_doc(reason: str | None = None) -> WebDoc:
    note = reason or "retrying alternate feeds"
    return WebDoc(
        url="https://lexicompanion.com/now",
        title="Still fetching live sources…",
        snippet=f"Lexi is refreshing the feed ({note}); check back shortly.",
        source="Lexi",
    )
