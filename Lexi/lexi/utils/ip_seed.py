from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Mapping, MutableMapping


_HEADER_ORDER = (
    "cf-connecting-ip",
    "true-client-ip",
    "x-real-ip",
    "x-client-ip",
    "x-forwarded-for",
    "forwarded",
)


def _first_forwarded(forwarded_value: str) -> str:
    """
    Handle Forwarded: for=1.2.3.4 headers by extracting the first IP-ish token.
    """
    value = forwarded_value or ""
    # Pattern: for=1.2.3.4;proto=https;by=...
    for part in value.split(";"):
        part = part.strip()
        if part.lower().startswith("for="):
            candidate = part.split("=", 1)[1].strip().strip('"')
            if candidate:
                return candidate
    return value.split(",")[0].strip()


def client_ip_from_headers(headers: Mapping[str, str], fallback: str | None = None) -> str:
    """
    Resolve the best-effort client IP using common proxy headers.
    """
    if not isinstance(headers, Mapping):
        headers = {}
    lowered: MutableMapping[str, str] = {k.lower(): v for k, v in headers.items()}
    for key in _HEADER_ORDER:
        raw = lowered.get(key)
        if not raw:
            continue
        if key == "forwarded":
            candidate = _first_forwarded(raw)
        else:
            candidate = raw.split(",")[0].strip()
        if candidate:
            return candidate
    return (fallback or "127.0.0.1").strip()


def normalize_ip(ip: str) -> str:
    """
    Collapse IPv4/IPv6 strings into a deterministic filename-safe token.
    """
    cleaned = re.sub(r"[^0-9a-fA-F]", "", (ip or "").strip())
    return cleaned.lower() or "unknown"


def ip_to_seed(ip: str) -> int:
    """
    Produce a stable 32-bit seed derived from the client IP.
    """
    digest = hashlib.sha1((ip or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def filename_for_ip(ip: str) -> str:
    """
    Map an IP string to the deterministic avatar filename (with .png suffix).
    """
    normalized = normalize_ip(ip)
    return f"{normalized}.png"


def basename_for_ip(ip: str) -> str:
    """
    Map an IP string to the deterministic avatar basename (without suffix).
    """
    return normalize_ip(ip)


def avatars_public_url_base() -> str:
    """
    Public URL prefix for avatars, defaulting to /lexi/static/avatars.
    """
    base = (os.getenv("AVATARS_PUBLIC_URL") or "/lexi/static/avatars").strip() or "/lexi/static/avatars"
    return base.rstrip("/")


def avatars_static_dir() -> Path:
    """
    Filesystem directory that backs the avatars URL prefix.
    """
    raw = os.getenv("AVATARS_STATIC_DIR") or "/app/frontend/public/avatars"
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def static_avatar_path(ip: str) -> Path:
    """
    Convenience helper returning the filesystem path for an IP's avatar.
    """
    return avatars_static_dir() / filename_for_ip(ip)


__all__ = [
    "avatars_public_url_base",
    "avatars_static_dir",
    "basename_for_ip",
    "client_ip_from_headers",
    "filename_for_ip",
    "ip_to_seed",
    "normalize_ip",
    "static_avatar_path",
]
