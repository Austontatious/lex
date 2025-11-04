from __future__ import annotations

import os
from pathlib import Path
from typing import List


class Settings:
    """
    Centralized application configuration.

    Env prefix: LEX_
      - LEX_STATIC_ROOT: absolute/relative path to static assets root
      - LEX_AVATAR_DIR: override for avatar output directory
      - LEX_CORS_ORIGINS: comma-separated list of CORS origins
    """

    _repo_root = Path(__file__).resolve().parents[2]
    _default_static_root = (_repo_root / "frontend" / "public").resolve()

    STATIC_ROOT: Path = Path(
        os.getenv("LEX_STATIC_ROOT", str(_default_static_root))
    ).resolve()

    AVATAR_DIR: Path = Path(
        os.getenv("LEX_AVATAR_DIR", str(STATIC_ROOT / "avatars"))
    ).resolve()

    _default_cors = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://lexicompanion.com",
        "https://api.lexicompanion.com",
        "https://app.lexicompanion.com",
    ]

    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "LEX_CORS_ORIGINS", ",".join(_default_cors)
        ).split(",")
        if origin.strip()
    ]


settings = Settings()

