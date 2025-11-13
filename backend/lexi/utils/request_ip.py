from __future__ import annotations

from typing import Any, Mapping

from .ip_seed import client_ip_from_headers


def request_ip(request: Any) -> str:
    """
    Normalize the best-effort client IP from a FastAPI/Starlette request.
    """
    headers: Mapping[str, str] = {}
    raw_headers = getattr(request, "headers", None)
    if raw_headers:
        headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}

    fallback = None
    client = getattr(request, "client", None)
    if client is not None:
        fallback = getattr(client, "host", None)

    return client_ip_from_headers(headers, fallback or "127.0.0.1")


__all__ = ["request_ip"]
