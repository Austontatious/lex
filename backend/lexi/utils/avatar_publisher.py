from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Union

from ..config.paths import AVATAR_DIR, AVATAR_URL_BASE
from .hashes import file_md5

PathLike = Union[str, Path]


def publish_avatar(src_png: PathLike, version: str | None = None) -> Dict[str, object]:
    """
    Copy a generated PNG into the served avatar directory using a deterministic filename.
    """
    src = Path(src_png)
    if not src.exists():
        return {"ok": False, "error": f"missing: {src}"}

    try:
        digest = version or file_md5(src)
    except Exception as exc:
        return {"ok": False, "error": f"hash_failed: {exc}"}

    dest = AVATAR_DIR / f"{digest}.png"
    if src.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    try:
        os.chmod(dest, 0o644)
    except PermissionError:
        pass

    mtime = int(dest.stat().st_mtime)
    url = f"{AVATAR_URL_BASE}/{dest.name}?v={mtime}"
    return {
        "ok": True,
        "file": str(dest),
        "url": url,
        "version": digest,
        "mtime": mtime,
    }


__all__ = ["publish_avatar"]
