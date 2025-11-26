from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def file_md5(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """
    Compute the MD5 hash (hex) of a file without loading it entirely into memory.
    """
    digest = hashlib.md5()
    with open(Path(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["file_md5"]
