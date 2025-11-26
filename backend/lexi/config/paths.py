from __future__ import annotations

import os
from pathlib import Path

STATIC_URL_PREFIX = "/lexi/static"


def _resolve_static_dir() -> Path:
    override = os.getenv("LEXI_STATIC_DIR") or os.getenv("LEX_STATIC_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "frontend" / "public",
        Path("/app/static"),
        repo_root / "Lexi" / "lexi" / "static",
        Path("/mnt/data/Lexi/static"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return Path("/app/static").resolve()


STATIC_DIR = _resolve_static_dir()
AVATAR_DIR = (STATIC_DIR / "avatars").resolve()
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_URL_PREFIX = f"{STATIC_URL_PREFIX}/avatars"
# Back-compat alias (old imports still expect AVATAR_URL_BASE)
AVATAR_URL_BASE = AVATAR_URL_PREFIX


def avatar_url_for(filename: str) -> str:
    """
    Build the public URL for a given avatar filename (no leading slash required).
    """
    name = (filename or "").lstrip("/")
    return f"{AVATAR_URL_PREFIX}/{name}"


__all__ = [
    "STATIC_URL_PREFIX",
    "STATIC_DIR",
    "AVATAR_DIR",
    "AVATAR_URL_PREFIX",
    "AVATAR_URL_BASE",
    "avatar_url_for",
]
