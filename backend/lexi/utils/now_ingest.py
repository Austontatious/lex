import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlencode

import importlib

import httpx

from ..config.now import ENABLE_NOW, settings_now
from .now_models import NowItem
from .now_utils import freshness_decay, log_now, short_id, to_aware_utc

try:
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    import feedparser as _feedparser_typing

try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:
    aioredis = None

_cache_mem: Dict[str, List[NowItem]] = {"movies": [], "tv": [], "news": []}
_cache_meta: Dict[str, Optional[datetime] | bool | Optional[str]] = {
    "last_refresh_at": None,
    "stale": True,
    "last_error": None,
}
_redis = None
_FEEDPARSER = None
_disabled_logged = False


def _mark_refresh_success(ts: datetime) -> None:
    _cache_meta["last_refresh_at"] = ts
    _cache_meta["stale"] = False
    _cache_meta["last_error"] = None


def _mark_refresh_failure(error: Exception | str) -> None:
    _cache_meta["stale"] = True
    _cache_meta["last_error"] = str(error)


def _coerce_dt(raw: object) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        try:
            tup = _feedparser()._parse_date(raw)  # type: ignore[attr-defined]
            dt = datetime(*tup[:6], tzinfo=timezone.utc) if tup else None
        except Exception:
            dt = None
    return to_aware_utc(dt)


async def _get_store():
    global _redis
    if settings_now.REDIS_URL and aioredis and _redis is None:
        try:
            _redis = aioredis.from_url(settings_now.REDIS_URL, decode_responses=True)
            await _redis.ping()
        except Exception:
            _redis = None
    return _redis


async def _write(category: str, items: List[NowItem]) -> None:
    store = await _get_store()
    if store:
        import json

        key = f"now:{category}"
        await store.set(key, json.dumps([item.model_dump() for item in items]))
    _cache_mem[category] = items


async def _read(category: str) -> List[NowItem]:
    store = await _get_store()
    if store:
        key = f"now:{category}"
        raw = await store.get(key)
        if raw:
            import json

            return [NowItem(**entry) for entry in json.loads(raw)]
    return _cache_mem.get(category, [])


async def _fetch_with_retry(
    name: str, call: Callable[[], Awaitable[httpx.Response]]
) -> httpx.Response:
    delay = 0.5
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            response: httpx.Response = await call()
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            log_now(f"[Now] {name} attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)
    raise RuntimeError(f"{name} failed after 3 attempts") from last_exc


async def ingest_tmdb() -> List[NowItem]:
    bearer = settings_now.TMDB_READ_ACCESS_TOKEN
    api_key = settings_now.TMDB_API_KEY
    if not bearer and not api_key:
        return []
    base = "https://api.themoviedb.org/3/trending"
    items: List[NowItem] = []

    async with httpx.AsyncClient(timeout=20) as client:
        for kind, cat in (("movie", "movies"), ("tv", "tv")):
            params = {"language": "en-US"}
            if api_key:
                params["api_key"] = api_key
            url = f"{base}/{kind}/day?{urlencode(params)}"

            async def _request() -> httpx.Response:
                headers = {}
                token = bearer or None
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                return await client.get(url, headers=headers)

            try:
                response = await _fetch_with_retry(f"tmdb:{kind}", _request)
            except Exception as exc:
                log_now(f"[Now] tmdb fetch skipped for {kind}: {exc}")
                continue

            for row in response.json().get("results", [])[: settings_now.NOW_MAX_ITEMS_PER_SOURCE]:
                title = row.get("title") or row.get("name") or "Untitled"
                web_url = f"https://www.themoviedb.org/{kind}/{row.get('id')}"
                img = row.get("backdrop_path") or row.get("poster_path")
                img_url = f"https://image.tmdb.org/t/p/w780{img}" if img else None
                dt = _coerce_dt(row.get("release_date") or row.get("first_air_date"))
                items.append(
                    NowItem(
                        id=short_id("tmdb", f"{kind}:{row.get('id')}"),
                        source="tmdb",
                        url=web_url,
                        title=title,
                        published_at=dt,
                        image=img_url,
                        category=cat,
                        tags=[kind],
                        score=0.0,
                    )
                )
    return items


async def ingest_tvmaze() -> List[NowItem]:
    async with httpx.AsyncClient(timeout=20) as client:

        async def _request() -> httpx.Response:
            return await client.get("https://api.tvmaze.com/schedule?country=US")

        try:
            response = await _fetch_with_retry("tvmaze", _request)
        except Exception as exc:
            log_now(f"[Now] tvmaze fetch failed: {exc}")
            return []

        out: List[NowItem] = []
        for row in response.json()[: settings_now.NOW_MAX_ITEMS_PER_SOURCE]:
            show = (row.get("show") or {}).get("name")
            episode = row.get("name")
            url = (
                row.get("url") or (row.get("show") or {}).get("url")
            ) or "https://www.tvmaze.com/"
            dt = _coerce_dt(row.get("airstamp"))
            out.append(
                NowItem(
                    id=short_id("tvmaze", url),
                    source="tvmaze",
                    url=url,
                    title=(
                        f"{show}: {episode}"
                        if show and episode
                        else (show or episode or "TV episode")
                    ),
                    published_at=dt,
                    image=None,
                    category="tv",
                    tags=["episode"],
                    score=0.0,
                )
            )
        return out


def _feedparser():
    global _FEEDPARSER
    if _FEEDPARSER is None:
        _FEEDPARSER = importlib.import_module("feedparser")
    return _FEEDPARSER


async def ingest_google_news() -> List[NowItem]:
    feeds = ["https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"]
    results: List[NowItem] = []

    async def _parse_feed(url: str) -> "_feedparser_typing.FeedParserDict":
        return await asyncio.to_thread(_feedparser().parse, url)

    for url in feeds:
        delay = 0.5
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                parsed = await _parse_feed(url)
                break
            except Exception as exc:
                last_exc = exc
                log_now(f"[Now] google-news attempt {attempt}/3 failed: {exc}")
                if attempt == 3:
                    parsed = None
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 4.0)
        if parsed is None:
            log_now(f"[Now] google-news feed skipped: {last_exc}")
            continue

        for entry in parsed.entries[: settings_now.NOW_MAX_ITEMS_PER_SOURCE]:
            link = entry.link
            dt = _coerce_dt(getattr(entry, "published", None))
            title = entry.title
            results.append(
                NowItem(
                    id=short_id("gn", link),
                    source="google-news",
                    url=link,
                    title=title,
                    published_at=dt,
                    image=None,
                    category="news",
                    tags=[],
                    score=0.0,
                )
            )
    return results


async def refresh_now_feed() -> None:
    global _disabled_logged
    if not ENABLE_NOW:
        if not _disabled_logged:
            log_now("refresh_now_feed: skipped (ENABLE_NOW!=1)")
            _disabled_logged = True
        _mark_refresh_failure("disabled")
        return

    log_now("refresh_now_feed: start")
    results = await asyncio.gather(
        ingest_tmdb(), ingest_tvmaze(), ingest_google_news(), return_exceptions=True
    )

    tmdb: List[NowItem] = []
    tvm: List[NowItem] = []
    gnews: List[NowItem] = []
    errors: List[Exception] = []

    for idx, bucket in enumerate(results):
        if isinstance(bucket, Exception):
            errors.append(bucket)
            continue
        if idx == 0:
            tmdb = bucket
        elif idx == 1:
            tvm = bucket
        else:
            gnews = bucket

    if len(errors) == len(results):
        _mark_refresh_failure(errors[0])
        log_now(f"refresh_now_feed: all sources failed ({errors[0]})")
        return

    def _score(item: NowItem) -> NowItem:
        item.score = freshness_decay(item.published_at)
        return item

    movies = [_score(item) for item in tmdb if item.category == "movies"]
    movies.sort(key=lambda item: item.score, reverse=True)

    tv_items = [_score(item) for item in (tmdb + tvm) if item.category == "tv"]
    tv_items.sort(key=lambda item: item.score, reverse=True)

    news_items = [_score(item) for item in gnews if item.category == "news"]
    news_items.sort(key=lambda item: item.score, reverse=True)

    await _write("movies", movies[:40])
    await _write("tv", tv_items[:40])
    await _write("news", news_items[:60])

    now = datetime.now(timezone.utc)
    _mark_refresh_success(now)
    log_now(
        f"refresh_now_feed: done (movies={len(movies)} tv={len(tv_items)} news={len(news_items)})"
    )


async def query_now(category: Optional[str], interests: List[str], limit: int) -> List[NowItem]:
    cats = [category] if category else ["movies", "tv", "news"]
    items: List[NowItem] = []
    for cat in cats:
        items.extend(await _read(cat))

    if interests:
        interests_lower = [i.lower() for i in interests]

        def boost(it: NowItem) -> float:
            haystack = f"{it.title.lower()} {' '.join(it.tags).lower()}"
            hit = any(key in haystack for key in interests_lower)
            return it.score + (0.2 if hit else 0.0)

        items.sort(key=boost, reverse=True)
    else:
        items.sort(key=lambda it: it.score, reverse=True)

    # Attach refresh metadata as lightweight tags.
    stamp = _cache_meta.get("last_refresh_at")
    if isinstance(stamp, datetime):
        iso = stamp.isoformat()
        for item in items:
            if f"refreshed:{iso}" not in item.tags:
                item.tags.append(f"refreshed:{iso}")
    if _cache_meta.get("stale"):
        for item in items:
            if "stale" not in item.tags:
                item.tags.append("stale")

    return items[:limit]
