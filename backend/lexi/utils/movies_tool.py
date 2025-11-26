from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..config.now import settings_now

log = logging.getLogger(__name__)


class MoviesToolError(RuntimeError):
    """Raised when the movies_now tool cannot return data."""


_STATE_OR_COUNTRY = re.compile(r",\s*([A-Za-z]{2})\b")


def _coerce_date(raw: str | None, fallback: date) -> date:
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip()).date()
        except Exception:
            pass
    return fallback


def _region_from_location(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    m = _STATE_OR_COUNTRY.search(location)
    if m:
        return m.group(1).upper()
    return None


def default_movie_window(user_text: str | None = None) -> Tuple[str, str]:
    """
    Derive a reasonable theatrical window from the user's wording.
    - today/tonight -> [today, today]
    - this week -> [today, +7 days]
    - this month -> [today, +30 days]
    - fallback -> [today, +7 days]
    """
    today = date.today()
    txt = (user_text or "").lower()
    if re.search(r"\b(today|tonight)\b", txt):
        end = today
    elif re.search(r"\bthis week\b", txt):
        end = today + timedelta(days=7)
    elif re.search(r"\bthis month\b", txt):
        end = today + timedelta(days=30)
    else:
        end = today + timedelta(days=7)
    return today.isoformat(), end.isoformat()


def movies_now(
    location: Optional[str],
    start_date: str,
    end_date: str,
    limit: int = 12,
) -> Dict[str, Any]:
    """
    Fetch theatrical releases for a given window from TMDb.
    Returns: {results: [...], start_date, end_date, region}
    """
    limit = max(1, min(20, int(limit or 12)))
    start = _coerce_date(start_date, date.today())
    end = _coerce_date(end_date, start)
    if end < start:
        start, end = end, start

    token = settings_now.TMDB_READ_ACCESS_TOKEN or settings_now.TMDB_API_KEY
    if not token:
        raise MoviesToolError("TMDB credentials missing")

    region = _region_from_location(location) or None
    params: Dict[str, Any] = {
        "language": "en-US",
        "include_adult": False,
        "include_video": False,
        "sort_by": "primary_release_date.desc",
        "with_release_type": "2|3",  # theatrical limited|wide
        "primary_release_date.gte": start.isoformat(),
        "primary_release_date.lte": end.isoformat(),
        "page": 1,
    }
    headers = {"Accept": "application/json"}
    if settings_now.TMDB_API_KEY:
        params["api_key"] = settings_now.TMDB_API_KEY
    if settings_now.TMDB_READ_ACCESS_TOKEN and not settings_now.TMDB_API_KEY:
        headers["Authorization"] = f"Bearer {settings_now.TMDB_READ_ACCESS_TOKEN}"
    if region:
        params["region"] = region

    url = "https://api.themoviedb.org/3/discover/movie"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure path
        log.warning("movies_now fetch failed: %s", exc)
        raise MoviesToolError(str(exc))

    results: List[Dict[str, Any]] = []
    for row in resp.json().get("results", []):
        title = row.get("title") or row.get("name") or "Untitled"
        rel_raw = (row.get("release_date") or "").strip()
        rel_dt = None
        if rel_raw:
            try:
                rel_dt = datetime.fromisoformat(rel_raw).date()
            except Exception:
                rel_dt = None
        if rel_dt and (rel_dt < start or rel_dt > end):
            continue

        movie_id = row.get("id")
        results.append(
            {
                "title": title,
                "release_date": rel_raw or (rel_dt.isoformat() if rel_dt else None),
                "runtime": None,
                "theatrical": True,
                "overview": row.get("overview") or "",
                "showtimes_url": f"https://www.themoviedb.org/movie/{movie_id}"
                if movie_id
                else None,
                "source": "tmdb",
            }
        )
        if len(results) >= limit:
            break

    return {
        "results": results,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "region": region,
    }


__all__ = [
    "MoviesToolError",
    "default_movie_window",
    "movies_now",
]
