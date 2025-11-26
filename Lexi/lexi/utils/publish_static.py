from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

from .ip_seed import avatars_static_dir

PathLike = Union[str, Path]


def comfy_output_dir() -> Path:
    """
    Resolve the Comfy workspace output directory.
    """
    override = os.getenv("COMFY_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.getenv("COMFY_WORKSPACE_DIR") or "/mnt/data/comfy").expanduser()
    return base / "output"


def latest_output_png(base: PathLike | None = None) -> Optional[Path]:
    """
    Return the most recent PNG inside the Comfy output directory (if any).
    """
    directory = Path(base) if base else comfy_output_dir()
    if not directory.exists():
        return None
    try:
        files = sorted(directory.glob("*.png"), key=lambda p: p.stat().st_mtime)
    except (FileNotFoundError, PermissionError):
        return None
    return files[-1] if files else None


def publish_as(src: PathLike, target_filename: str, static_dir: PathLike | None = None) -> Path:
    """
    Copy the rendered PNG to the avatars static directory under the provided filename.
    """
    if not target_filename:
        raise ValueError("target_filename is required")
    filename = target_filename
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"

    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    dest_dir = Path(static_dir) if static_dir else avatars_static_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    shutil.copy2(src_path, dest_path)
    return dest_path


__all__ = ["comfy_output_dir", "latest_output_png", "publish_as"]
