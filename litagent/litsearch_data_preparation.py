"""Prepare LitSearch full papers as parent documents and searchable child chunks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .fulltext_structure import parse_fulltext_structure


DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 240
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:[A-Z][A-Za-z0-9 ,:;()'’/&-]{2,120}|"
    r"[A-Z][A-Z0-9 ,:;()'’/&-]{2,120})$"
)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [" ".join(part.split()) for part in re.split(r"\n\s*\n+", normalized) if part.strip()]


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """Split oversized paragraphs at sentence boundaries, with a hard-size fallback."""
    if len(text) <= size:
        return [text]
    sentences = [piece.strip() for piece in SENTENCE_END_RE.split(text) if piece.strip()]
    if len(sentences) <= 1:
        return [text[start:start + size] for start in range(0, len(text), size - overlap)]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[start:start + size]
                for start in range(0, len(sentence), size - overlap)
            )
        elif current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    if overlap <= 0 or len(pieces) < 2:
        return pieces
    return [pieces[0], *[
        f"{pieces[i - 1][-overlap:]} {pieces[i]}".strip()
        for i in range(1, len(pieces))
    ]]


def _section_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Use recognizable heading lines when available; otherwise keep one body section."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    section = "全文"
    body: list[str] = []
    saw_heading = False
    for line in lines:
        stripped = " ".join(line.split())
        is_heading = (
            2 <= len(stripped) <= 120
            and not stripped.endswith((".", ",", ";"))
            and bool(HEADING_RE.fullmatch(stripped))
        )
        if is_heading:
            content = "\n".join(body).strip()
            if content:
                yield section, content
            section = stripped
            body = []
            saw_heading = True
        else:
            body.append(line)
    content = "\n".join(body).strip()
    if content:
        yield section, content
    if not saw_heading and not content:
        yield "全文", text


def _chunk_section(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = _paragraphs(text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for part in _split_long_text(paragraph, size, overlap):
            if current and len(current) + 2 + len(part) > size:
                chunks.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}".strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


class DataPreparationModule:
    """Load LitSearch full-text parents, create child chunks, and restore parents."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.documents: list[dict] = []
        self.documents_by_id: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.parent_child_map: dict[str, str] = {}
        self._parent_offsets: dict[str, int] | None = None

    def iter_documents(self) -> Iterable[dict]:
        """Stream parent records, accepting either `full_text` or `full_paper`."""
        seen: set[str] = set()
        with self.data_path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                paper_id = str(row.get("paper_id", row.get("corpusid", ""))).strip()
                full_text = row.get("full_text") or row.get("full_paper") or ""
                if not paper_id or not isinstance(full_text, str) or not full_text.strip():
                    raise ValueError(f"{self.data_path}:{number} 缺少论文 ID 或全文")
                if paper_id in seen:
                    raise ValueError(f"重复论文 ID: {paper_id}")
                seen.add(paper_id)
                parent = {
                    **row,
                    "paper_id": paper_id,
                    "parent_id": paper_id,
                    "full_text": full_text.strip(),
                    "source_url": row.get("source_url", "https://huggingface.co/datasets/princeton-nlp/LitSearch"),
                    "doc_type": "parent",
                }
                parent.pop("full_paper", None)
                yield parent

    def load_documents(self) -> list[dict]:
        """Load all parents into memory; use `build_chunks` for large corpora."""
        documents = list(self.iter_documents())
        if not documents:
            raise ValueError(f"没有可用的全文父文档: {self.data_path}")
        self.documents = documents
        self.documents_by_id = {item["paper_id"]: item for item in documents}
        return documents

    @staticmethod
    def _chunk_parent(
        parent: dict,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> Iterable[dict]:
        ordinal = 0
        text = parent["full_text"]
        searchable_text = text
        title = (parent.get("title") or "").strip()
        abstract = (parent.get("abstract") or "").strip()
        if title and title.casefold() not in text.casefold():
            searchable_text = f"{title}\n\n{searchable_text}"
        if abstract and abstract.casefold() not in text.casefold():
            searchable_text = f"{abstract}\n\n{searchable_text}"

        for section, section_text in _section_blocks(searchable_text):
            for content in _chunk_section(section_text, chunk_size, overlap):
                ordinal += 1
                chunk_id = _digest(f"{parent['paper_id']}\0{ordinal}\0{section}\0{content}")
                yield {
                    "paper_id": chunk_id,
                    "parent_id": parent["paper_id"],
                    "title": f"{parent.get('title', '')} — {section}".strip(" —"),
                    "abstract": content,
                    "source_url": parent["source_url"],
                    "evidence_scope": "full_text_chunk",
                    "doc_type": "child",
                    "chunk_id": chunk_id,
                    "chunk_index": ordinal - 1,
                    "section": section,
                }

    @staticmethod
    def _chunk_parent_structured(
        parent: dict,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> Iterable[dict]:
        """Emit children with conservative section, block-type and line provenance."""
        title = (parent.get("title") or "").strip()
        ordinal = 0
        active_key: tuple | None = None
        pending_text = ""
        pending_start = pending_end = 0
        pending_meta: dict | None = None

        def emit(content: str, block: dict, start_line: int, end_line: int) -> dict:
            nonlocal ordinal
            ordinal += 1
            section_path = block["section_path"]
            source_key = (f"{parent['paper_id']}\0{section_path}\0"
                          f"{block['paragraph_type']}\0{start_line}\0"
                          f"{end_line}\0{ordinal}\0{content}")
            chunk_id = _digest(source_key)
            return {
                "paper_id": chunk_id,
                "parent_id": parent["paper_id"],
                "title": f"{title} — {' / '.join(section_path)}".strip(" —"),
                "abstract": content,
                "source_url": parent["source_url"],
                "evidence_scope": "full_text_chunk",
                "doc_type": "child",
                "chunk_id": chunk_id,
                "chunk_index": ordinal - 1,
                "section": block["section"],
                "section_path": section_path,
                "paragraph_type": block["paragraph_type"],
                "start_line": start_line,
                "end_line": end_line,
                "location_type": "source_text_line",
            }

        def flush_pending() -> dict | None:
            nonlocal pending_text, pending_meta, pending_start, pending_end
            if not pending_text or pending_meta is None:
                return None
            row = emit(pending_text, pending_meta, pending_start, pending_end)
            pending_text, pending_meta, pending_start, pending_end = "", None, 0, 0
            return row

        for block in parse_fulltext_structure(parent["full_text"]):
            key = (tuple(block["section_path"]), block["paragraph_type"])
            if key != active_key:
                row = flush_pending()
                if row:
                    yield row
                active_key = key
            parts = _split_long_text(block["text"], chunk_size, overlap)
            if len(parts) > 1 or len(block["text"]) > chunk_size:
                row = flush_pending()
                if row:
                    yield row
                for content in parts:
                    yield emit(content, block, block["start_line"], block["end_line"])
                continue
            content = parts[0]
            if pending_text and len(pending_text) + 2 + len(content) > chunk_size:
                row = flush_pending()
                if row:
                    yield row
            if not pending_text:
                pending_meta = block
                pending_start = block["start_line"]
            pending_end = block["end_line"]
            pending_text = f"{pending_text}\n\n{content}".strip()
        row = flush_pending()
        if row:
            yield row

    def iter_chunk_documents(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        structured: bool = False,
    ) -> Iterable[dict]:
        if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size 至少为 200，且 overlap 必须在 [0, chunk_size) 内")
        parents = self.documents if self.documents else self.iter_documents()
        for parent in parents:
            chunker = self._chunk_parent_structured if structured else self._chunk_parent
            yield from chunker(parent, chunk_size, overlap)

    def build_chunks(
        self,
        output_path: str | Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        structured: bool = False,
    ) -> dict:
        """Stream child chunks to JSONL to avoid holding the full corpus in memory."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        parent_count = chunk_count = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for parent in self.iter_documents():
                parent_count += 1
                chunker = self._chunk_parent_structured if structured else self._chunk_parent
                for chunk in chunker(parent, chunk_size, overlap):
                    stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1
        if not parent_count or not chunk_count:
            temporary.unlink(missing_ok=True)
            raise ValueError("没有生成可用的父文档或子块")
        temporary.replace(output)
        return {"parents": parent_count, "chunks": chunk_count, "chunk_size": chunk_size,
                "overlap": overlap, "structured": structured, "output": str(output)}

    def chunk_documents(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        structured: bool = False,
    ) -> list[dict]:
        """Create section-aware child chunks with stable IDs and parent metadata."""
        if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size 至少为 200，且 overlap 必须在 [0, chunk_size) 内")
        chunks = list(self.iter_chunk_documents(chunk_size, overlap, structured=structured))
        self.chunks = chunks
        self.parent_child_map = {chunk["chunk_id"]: chunk["parent_id"] for chunk in chunks}
        return chunks

    def save_chunks(self, output_path: str | Path) -> dict:
        if not self.chunks:
            raise ValueError("请先运行 chunk_documents")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        temporary.replace(output)
        return {"chunks": len(self.chunks), "parents": len(self.documents), "output": str(output)}

    def _ensure_parent_offsets(self) -> None:
        if self._parent_offsets is not None:
            return
        offsets: dict[str, int] = {}
        with self.data_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                paper_id = str(row.get("paper_id", row.get("corpusid", ""))).strip()
                if paper_id:
                    offsets[paper_id] = offset
        self._parent_offsets = offsets

    def _parent_by_id(self, parent_id: str) -> dict | None:
        if parent_id in self.documents_by_id:
            return self.documents_by_id[parent_id]
        self._ensure_parent_offsets()
        offset = self._parent_offsets.get(parent_id) if self._parent_offsets else None
        if offset is None:
            return None
        with self.data_path.open("rb") as stream:
            stream.seek(offset)
            row = json.loads(stream.readline())
        full_text = row.get("full_text") or row.get("full_paper") or ""
        row["paper_id"] = parent_id
        row["parent_id"] = parent_id
        row["full_text"] = full_text
        row["source_url"] = row.get("source_url", "https://huggingface.co/datasets/princeton-nlp/LitSearch")
        row["doc_type"] = "parent"
        row.pop("full_paper", None)
        return row

    def get_parent_documents(self, child_chunks: list[dict]) -> list[dict]:
        """Deduplicate child hits by parent, preserving best-hit order for generation."""
        aggregated: dict[str, dict] = {}
        order: list[str] = []
        for rank, chunk in enumerate(child_chunks, 1):
            parent_id = str(chunk.get("parent_id") or self.parent_child_map.get(str(chunk.get("paper_id", "")), ""))
            if not parent_id:
                continue
            if parent_id not in aggregated:
                found = self._parent_by_id(parent_id)
                if found is None:
                    continue
                parent = dict(found)
                parent["matched_chunks"] = []
                parent["best_chunk_rank"] = rank
                parent["retrieval_score"] = chunk.get("score")
                aggregated[parent_id] = parent
                order.append(parent_id)
            aggregated[parent_id]["matched_chunks"].append({
                "chunk_id": chunk.get("chunk_id", chunk.get("paper_id")),
                "chunk_index": chunk.get("chunk_index"),
                "section": chunk.get("section", "全文"),
                "section_path": chunk.get("section_path"),
                "paragraph_type": chunk.get("paragraph_type"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "location_type": chunk.get("location_type"),
                "score": chunk.get("score"),
                "rank": rank,
                "bm25_rank": chunk.get("bm25_rank"),
                "dense_rank": chunk.get("dense_rank"),
                "rrf_rank": chunk.get("rrf_rank"),
                "rrf_score": chunk.get("rrf_score"),
                "rerank_score": chunk.get("rerank_score"),
                "matched_queries": chunk.get("matched_queries"),
                "retrieval_method": chunk.get("retrieval_method"),
                "text": chunk.get("abstract", ""),
            })
        return [aggregated[parent_id] for parent_id in order]

