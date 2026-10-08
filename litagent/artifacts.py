"""Streaming artifact checks with optional, process-local validation reuse."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def file_signature(path: Path) -> tuple:
    path = Path(path).resolve()
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def file_sha256(path: Path, *, cache: dict | None = None) -> str:
    path = Path(path).resolve()
    before = file_signature(path)
    cached = cache.get(str(path)) if cache is not None else None
    if cached is not None and cached[0] == before:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if file_signature(path) != before:
        raise ValueError(f"校验过程中产物发生变化：{path}")
    value = digest.hexdigest()
    if cache is not None:
        cache[str(path)] = (before, value)
    return value


def verify_sha256(path: Path, expected: str, *, cache: dict | None = None) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"产物缺少有效 SHA-256：{path}")
    if not Path(path).is_file():
        raise ValueError(f"产物不存在：{path}")
    actual = file_sha256(path, cache=cache)
    if actual != expected.lower():
        raise ValueError(f"产物 SHA-256 不匹配：{path}")
    return actual
