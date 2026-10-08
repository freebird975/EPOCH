"""Conservative structure extraction and streaming corpus scale diagnostics.

Offsets in this module are line numbers in the supplied text. PDF page numbers are
never inferred from flattened text. Heuristics deliberately prefer missed headings
over promoting ordinary prose to a section heading.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,5})[.)]?\s+(.{2,100})$")
_KNOWN = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "experiments", "experiment",
    "results", "discussion", "conclusion", "conclusions", "limitations",
    "acknowledgments", "acknowledgements", "references", "appendix",
}
_MONTHS = {"jan", "january", "feb", "february", "mar", "march", "apr", "april",
           "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
           "september", "oct", "october", "nov", "november", "dec", "december"}
_EQ_LINE = re.compile(r"^\s*(?:\(?\d{1,3}\)?\s*)?(?:\$\$.*\$\$|\\\[.*\\\]|\\begin\{(?:equation|align|gather|math).+|[A-Za-z]\s*=\s*[^=].{0,100})\s*$")
_TABLE_LINE = re.compile(r"\||\t{1,}|\s{2,}\S+\s{2,}\S+")


def _heading(line: str) -> tuple[int, str] | None:
    s = " ".join(line.split())
    if not s or len(s) > 140:
        return None
    md = _MD_HEADING.match(s)
    if md:
        return len(md.group(1)), md.group(2).strip()
    numbered = _NUMBERED.match(s)
    if numbered and (len(numbered.group(2).split()) <= 16):
        title = numbered.group(2)
        first_word = title.split()[0].rstrip(".:").casefold()
        if (int(numbered.group(1).split(".")[0]) <= 30
                and first_word not in _MONTHS
                and "arxiv:" not in title.casefold()
                and title[:1].isupper() and not re.search(r"[{};=@<>]", title)):
            return numbered.group(1).count(".") + 1, s
    if s.casefold().rstrip(":") in _KNOWN:
        return 1, s
    return None


def _kind(lines: list[str]) -> str:
    joined = "\n".join(lines).strip()
    if not joined:
        return "body"
    nonempty = [line for line in lines if line.strip()]
    if any(_EQ_LINE.match(line) for line in nonempty) or ("$$" in joined and joined.count("$$") >= 2):
        return "equation"
    if len(nonempty) >= 2 and sum(bool(_TABLE_LINE.search(line)) for line in nonempty) / len(nonempty) >= 0.6:
        return "table"
    return "body"


def parse_fulltext_structure(text: str) -> list[dict[str, Any]]:
    """Extract section-aware paragraphs with honest 1-based line spans."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stack: list[tuple[int, str]] = []
    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    start = 1
    active_path: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal current
        raw = "\n".join(current).strip()
        if raw:
            blocks.append({
                "section": active_path[-1] if active_path else "全文",
                "section_path": list(active_path) if active_path else ["全文"],
                "paragraph_type": _kind(current),
                "text": raw,
                "start_line": start,
                "end_line": end_line,
            })
        current = []

    for line_no, line in enumerate(lines, 1):
        detected = _heading(line)
        if (detected and stack and stack[-1][1].casefold().rstrip(":") == "references"
                and _NUMBERED.match(" ".join(line.split()))):
            # Numbered bibliography entries are not numbered section headings.
            detected = None
        if detected:
            flush(line_no - 1)
            level, title = detected
            level = max(1, min(level, 6))
            stack = stack[: level - 1]
            stack.append((level, title))
            active_path = [name for _, name in stack]
            start = line_no + 1
        elif not line.strip():
            flush(line_no - 1)
            start = line_no + 1
        else:
            if not current:
                start = line_no
            current.append(line)
    flush(len(lines))
    return blocks


def audit_corpus_scale(
    parent_path: str | Path,
    child_path: str | Path | None = None,
    *,
    target_documents: int | None = None,
    manifest_path: str | Path | None = None,
    index_paths: tuple[str | Path, ...] = (),
    structured_chunks: bool = False,
) -> dict[str, Any]:
    """Stream JSONL to report hashes, bytes, counts and linear size/time estimates.

    This measures the local file scan, not FAISS/BM25 build time. If a child file is
    present its observed bytes/chunk count is used to extrapolate the requested size.
    """
    parent = Path(parent_path)
    started = time.perf_counter()
    digest = hashlib.sha256()
    count = chars = text_bytes = 0
    structure_parse_seconds = 0.0
    structured_block_count = 0
    structured_docs = 0
    block_type_counts: dict[str, int] = {}
    with parent.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            text = row.get("full_text") or row.get("full_paper") or ""
            count += 1
            chars += len(text)
            text_bytes += len(text.encode("utf-8"))
            parse_started = time.perf_counter()
            blocks = parse_fulltext_structure(text)
            structure_parse_seconds += time.perf_counter() - parse_started
            structured_block_count += len(blocks)
            structured_docs += int(any(block["section_path"] != ["全文"] for block in blocks))
            for block in blocks:
                kind = block["paragraph_type"]
                block_type_counts[kind] = block_type_counts.get(kind, 0) + 1
    elapsed = time.perf_counter() - started
    parent_sha = digest.hexdigest()
    result: dict[str, Any] = {
        "parent_path": str(parent.resolve()), "parent_sha256": parent_sha,
        "parent_bytes": parent.stat().st_size, "parent_documents": count,
        "fulltext_chars": chars, "fulltext_utf8_bytes": text_bytes,
        "scan_seconds": round(elapsed, 6), "scan_documents_per_second": round(count / elapsed, 2) if elapsed else None,
        "structure_parse_seconds": round(structure_parse_seconds, 6),
        "structure_parse_documents_per_second": round(count / structure_parse_seconds, 2) if structure_parse_seconds else None,
        "structured_block_count": structured_block_count,
        "documents_with_detected_sections": structured_docs,
        "block_type_counts": block_type_counts,
        "license_metadata": "not_verified", "license_note": "LitSearch row/source URL alone does not establish full-text redistribution rights.",
    }
    if manifest_path:
        mp = Path(manifest_path)
        if mp.exists():
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            expected = manifest.get("parent_sha256") or manifest.get("corpus_sha256")
            result["manifest_path"] = str(mp.resolve())
            result["manifest_expected_sha256"] = expected
            result["manifest_checksum_matches"] = expected == parent_sha if expected else None
    if child_path:
        cp = Path(child_path)
        if cp.exists():
            d = hashlib.sha256()
            child_count = 0
            with cp.open("rb") as stream:
                for raw in stream:
                    d.update(raw)
                    if raw.strip():
                        child_count += 1
            result.update({"child_path": str(cp.resolve()), "child_sha256": d.hexdigest(),
                           "child_bytes": cp.stat().st_size, "child_chunks": child_count})
            if manifest_path and Path(manifest_path).exists():
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                expected_child = manifest.get("chunk_sha256")
                if expected_child:
                    result["manifest_expected_chunk_sha256"] = expected_child
                    result["manifest_chunk_checksum_matches"] = expected_child == d.hexdigest()
    # Dry-run serialization gives a measured child-JSONL size/build rate without
    # writing or replacing chunks, manifests, or indexes.
    from .litsearch_data_preparation import DataPreparationModule
    chunk_started = time.perf_counter()
    chunk_count = child_json_bytes = 0
    prep = DataPreparationModule(parent)
    for chunk in prep.iter_chunk_documents(structured=structured_chunks):
        child_json_bytes += len((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8"))
        chunk_count += 1
    chunk_seconds = time.perf_counter() - chunk_started
    result.update({
        "chunk_mode": "structured" if structured_chunks else "legacy",
        "dry_run_chunk_count": chunk_count,
        "dry_run_child_jsonl_bytes": child_json_bytes,
        "dry_run_chunk_build_seconds": round(chunk_seconds, 6),
        "dry_run_chunks_per_second": round(chunk_count / chunk_seconds, 2) if chunk_seconds else None,
    })
    indexes = []
    for item in index_paths:
        p = Path(item)
        if p.exists():
            indexes.append({"path": str(p.resolve()), "bytes": p.stat().st_size})
    result["indexes"] = indexes
    if target_documents and count:
        ratio = target_documents / count
        result["projection"] = {
            "target_documents": target_documents,
            "linear_parent_bytes": round(parent.stat().st_size * ratio),
            "linear_fulltext_utf8_bytes": round(text_bytes * ratio),
            "linear_child_bytes": round(result["child_bytes"] * ratio) if "child_bytes" in result else None,
            "linear_dry_run_child_bytes": round(child_json_bytes * ratio),
            "linear_scan_seconds": round(elapsed * ratio, 3),
            "linear_chunk_generation_seconds": round(chunk_seconds * ratio, 3),
            "linear_index_bytes": round(sum(row["bytes"] for row in indexes) * ratio) if indexes else None,
            "assumption": "linear extrapolation from this local corpus; index bytes require supplied existing indexes; excludes embedding/index construction time",
        }
    return result
