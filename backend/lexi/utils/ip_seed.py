from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Mapping, MutableMapping

import ipaddress

from ..config.paths import AVATAR_DIR, AVATAR_URL_BASE


_HEADER_ORDER = (
    "cf-connecting-ip",
    "true-client-ip",
    "x-real-ip",
    "x-client-ip",
    "x-forwarded-for",
    "forwarded",
)
_SEED_SALT = os.getenv("LEXI_SEED_SALT", "lexi-default-salt").strip() or "lexi-default-salt"


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
    Collapse IPv4/IPv6 strings into a filename-safe token while keeping readability.
    """
    cleaned = (ip or "").strip()
    if not cleaned:
        return "unknown"
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned).strip("_")
    return safe or "unknown"


def ip_to_seed(ip: str) -> int:
    """
    Produce a stable 64-bit seed derived from the client IP plus an optional salt.
    """
    raw = (ip or "").strip()
    try:
        canonical = ipaddress.ip_address(raw).compressed
    except ValueError:
        canonical = raw or "unknown"

    key = f"{canonical}|{_SEED_SALT}".encode("utf-8")
    digest = hashlib.blake2s(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


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
    Public URL prefix for avatars, defaulting to the mounted /lexi/static/avatars.
    """
    base = (os.getenv("AVATARS_PUBLIC_URL") or AVATAR_URL_BASE).strip() or AVATAR_URL_BASE
    return base.rstrip("/")


def avatars_static_dir() -> Path:
    """
    Filesystem directory that backs the avatars URL prefix.
    """
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    return AVATAR_DIR


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
