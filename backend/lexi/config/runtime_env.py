from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str]]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return environment variable if set and non-empty, otherwise default."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _to_path(value: str, root: Optional[Path] = None) -> Path:
    """Convert value to Path; if relative, anchor it to root (or CWD)."""
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = root or Path.cwd()
    return (anchor / path).resolve()


MODELS_MOUNT = Path("/models")
_DEFAULT_BASE = Path("/mnt/data/models")


def _select_base_models_dir() -> Path:
    override = _env("BASE_MODELS_DIR")
    if MODELS_MOUNT.exists():
        return MODELS_MOUNT
    if override:
        return _to_path(override, root=_DEFAULT_BASE)
    return _DEFAULT_BASE


BASE_MODELS_DIR: Path = _select_base_models_dir()


def resolve(path_like: PathLike) -> Path:
    """Resolve a path relative to BASE_MODELS_DIR when not absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (BASE_MODELS_DIR / path).resolve()


def _env_path(name: str, default: Path) -> Path:
    value = _env(name)
    if not value:
        return default
    return resolve(value)


COMFY_WORKSPACE_DIR: Path = _env_path("COMFY_WORKSPACE_DIR", BASE_MODELS_DIR / "comfy")
COMFY_ROOT: Path = _env_path("COMFY_ROOT", COMFY_WORKSPACE_DIR)

_comfy_raw = (
    _env("COMFY_URL")
    or _env("COMFY_BASE_URL")
    or _env("IMAGE_API_BASE")
    or "http://host.docker.internal:8188"
)
COMFY_URL: str = _comfy_raw.rstrip("/")

OPENAI_API_BASE: str = (_env("OPENAI_API_BASE") or "http://host.docker.internal:8008/v1").rstrip("/")
OPENAI_API_KEY: str = _env("OPENAI_API_KEY", "dummy")


def prefer_models_mount(path: Path) -> Path:
    """
    If /models is available and the incoming path lives under BASE_MODELS_DIR,
    rewrite it to the mounted equivalent. Keeps relative structure identical.
    """
    if not MODELS_MOUNT.exists():
        return path
    try:
        rel = path.relative_to(BASE_MODELS_DIR)
    except ValueError:
        return path
    return (MODELS_MOUNT / rel).resolve()


__all__ = [
    "BASE_MODELS_DIR",
    "COMFY_WORKSPACE_DIR",
    "COMFY_ROOT",
    "COMFY_URL",
    "MODELS_MOUNT",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "prefer_models_mount",
    "resolve",
]
