import os
from typing import List, Optional

from pydantic import AnyHttpUrl, Field

try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:  # pragma: no cover - optional dep

    class BaseSettings:  # type: ignore
        """Fallback stub when pydantic-settings is not installed."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self, *args, **kwargs):
            return dict(self.__dict__)


from .settings import env

ENABLE_NOW: bool = os.getenv("ENABLE_NOW", "0") == "1"


def _env_or_default(name: str, default: Optional[str] = None) -> Optional[str]:
    if ENABLE_NOW:
        return env(name)
    value = os.getenv(name)
    if value:
        return value
    return default


class NowSettings(BaseSettings):
    """Runtime configuration for the Now subsystem."""

    TMDB_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("TMDB_API_KEY"))
    BRAVE_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("BRAVE_API_KEY"))
    TAVILY_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("TAVILY_API_KEY"))
    NEWSAPI_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))

    SUMMARIZER_ENDPOINT: AnyHttpUrl = Field(
        default_factory=lambda: _env_or_default(
            "SUMMARIZER_ENDPOINT", "http://127.0.0.1:8008/v1/chat/completions"
        )
    )
    SUMMARIZER_MODEL: str = Field(
        default_factory=lambda: _env_or_default("SUMMARIZER_MODEL", "Lexi")
    )

    NOW_REFRESH_MINUTES: int = Field(
        default_factory=lambda: int(os.getenv("NOW_REFRESH_MINUTES", "30"))
    )
    NOW_MAX_ITEMS_PER_SOURCE: int = Field(
        default_factory=lambda: int(os.getenv("NOW_MAX_ITEMS_PER_SOURCE", "20"))
    )
    NOW_TOP_N_DEFAULT: int = Field(default_factory=lambda: int(os.getenv("NOW_TOP_N_DEFAULT", "8")))
    REDIS_URL: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    )

    DEFAULT_INTERESTS: List[str] = Field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("DEFAULT_INTERESTS", "movies,tv,technology,ai").split(",")
            if s.strip()
        ]
    )

    class Config:
        env_prefix = ""  # read raw envs


settings_now = NowSettings()
