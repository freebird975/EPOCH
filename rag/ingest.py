"""Load documents and turn them into traceable text chunks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _windows(text: str, size: int = 700, overlap: int = 100):
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for separator in ("\n\n", "。", "！", "？", "\n", ". "):
                cut = text.rfind(separator, start + size // 2, end)
                if cut >= 0:
                    end = cut + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def _units(path: Path):
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF 导入需要先运行: python -m pip install pypdf") from exc
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                yield f"第 {page_number} 页", text
        return

    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".txt":
        yield "全文", text
        return

    heading = "开头"
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if any(part.strip() for part in body):
                yield heading, "\n".join(body)
            heading = match.group(1)
            body = [line]
        else:
            body.append(line)
    if any(part.strip() for part in body):
        yield heading, "\n".join(body)


def build_index(docs_dir: Path, output: Path) -> dict:
    if not docs_dir.is_dir():
        raise ValueError(f"文档目录不存在: {docs_dir}")
    files = sorted(
        path for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    chunks = []
    empty_files = []
    for path in files:
        before = len(chunks)
        source = path.relative_to(docs_dir).as_posix()
        for section, unit in _units(path):
            for ordinal, content in enumerate(_windows(unit), 1):
                digest = hashlib.sha256(
                    f"{source}\0{section}\0{ordinal}\0{content}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append({
                    "id": digest,
                    "source": source,
                    "section": section,
                    "text": content,
                })
        if len(chunks) == before:
            empty_files.append(source)
    index = {"version": 1, "docs_dir": str(docs_dir.resolve()), "chunks": chunks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"files": len(files), "chunks": len(chunks), "empty_files": empty_files, "output": str(output)}


def load_index(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"索引不存在: {path}。请先运行 index 命令。")
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("version") != 1 or not isinstance(index.get("chunks"), list):
        raise ValueError("索引格式不兼容，请重新运行 index 命令。")
    return index
