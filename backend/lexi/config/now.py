import os
from typing import List, Optional

from pydantic import AnyHttpUrl, Field
from pydantic.fields import FieldInfo

try:
    from pydantic_settings import BaseSettings
    _P_SETTINGS = True
except ModuleNotFoundError:  # pragma: no cover - optional dep

    class BaseSettings:  # type: ignore
        """Fallback stub when pydantic-settings is not installed."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self, *args, **kwargs):
            return dict(self.__dict__)

    _P_SETTINGS = False

from .settings import env


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


ENABLE_NOW: bool = _truthy_env("ENABLE_NOW", default=True)


def _env_or_default(
    name: str, default: Optional[str] = None, *, required_when_enabled: bool = False
) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    if ENABLE_NOW and required_when_enabled and default is None:
        return env(name)
    return default


class NowSettings(BaseSettings):
    """Runtime configuration for the Now subsystem."""

    TMDB_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("TMDB_API_KEY"))
    TMDB_READ_ACCESS_TOKEN: Optional[str] = Field(
        default_factory=lambda: _env_or_default("TMDB_READ_ACCESS_TOKEN")
    )
    BRAVE_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("BRAVE_API_KEY"))
    TAVILY_API_KEY: Optional[str] = Field(default_factory=lambda: _env_or_default("TAVILY_API_KEY"))
    NEWSAPI_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))

    SUMMARIZER_ENDPOINT: AnyHttpUrl = Field(
        default_factory=lambda: _env_or_default(
            "SUMMARIZER_ENDPOINT", "http://172.17.0.1:8008/v1/chat/completions"
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

if not _P_SETTINGS:
    for attr, value in NowSettings.__dict__.items():
        if isinstance(value, FieldInfo):
            if value.default_factory is not None:
                computed = value.default_factory()
            else:
                computed = value.default
            setattr(settings_now, attr, computed)
