"""Persistent, inspectable BM25 baseline over paper titles and abstracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import load_papers
from .artifacts import file_signature, verify_sha256

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "using", "was", "were", "what", "when", "which", "with",
}
K1 = 1.2
B = 0.75
SQLITE_BM25_MMAP_SIZE_BYTES = 2_147_418_112


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_sqlite_index(papers_path: Path, index_path: Path) -> dict:
    """Build an exact BM25 index incrementally on disk for large corpora."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    paper_count = term_count = token_count = 0
    total_length = 0
    scope: str | None = None
    corpus_digest = hashlib.sha256()
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript("""
            CREATE TABLE docs (
                doc_id INTEGER PRIMARY KEY,
                paper_id TEXT NOT NULL,
                doc_length INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE terms (
                term TEXT PRIMARY KEY,
                df INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE postings (
                term TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE title_terms (
                term TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                PRIMARY KEY (term, doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
        """)
        with papers_path.open("rb") as source:
            for line in source:
                corpus_digest.update(line)
                if not line.strip():
                    continue
                paper = json.loads(line)
                current_scope = paper.get("evidence_scope", "abstract")
                if scope is None:
                    scope = current_scope
                elif current_scope != scope:
                    raise ValueError("BM25 语料证据范围不一致")
                if current_scope not in {"abstract", "full_text_chunk"}:
                    raise ValueError("BM25 语料证据范围不一致")

                doc_id = paper_count
                title = str(paper.get("title") or "")
                title_terms = set(tokenize(title))
                frequencies = Counter(tokenize(title + " " + str(paper.get("abstract") or "")))
                doc_length = sum(frequencies.values())
                connection.execute(
                    "INSERT INTO docs(doc_id, paper_id, doc_length, payload) VALUES (?, ?, ?, ?)",
                    (doc_id, str(paper.get("paper_id", doc_id)), doc_length,
                     json.dumps(paper, ensure_ascii=False, separators=(",", ":"))),
                )
                connection.executemany(
                    "INSERT INTO postings(term, doc_id, tf) VALUES (?, ?, ?)",
                    ((term, doc_id, frequency) for term, frequency in frequencies.items()),
                )
                connection.executemany(
                    "INSERT INTO terms(term, df) VALUES (?, 1) "
                    "ON CONFLICT(term) DO UPDATE SET df = df + 1",
                    ((term,) for term in frequencies),
                )
                connection.executemany(
                    "INSERT INTO title_terms(term, doc_id) VALUES (?, ?)",
                    ((term, doc_id) for term in title_terms),
                )
                paper_count += 1
                total_length += doc_length
                token_count += len(frequencies)
                if paper_count % 1000 == 0:
                    connection.commit()
                if paper_count % 10000 == 0:
                    print(f"BM25 磁盘索引进度：{paper_count} 篇子块", flush=True)

        if not paper_count or scope is None:
            raise ValueError("没有可建立索引的文档")
        connection.execute("CREATE INDEX postings_doc_id_idx ON postings(doc_id)")
        metadata = {
            "version": 1,
            "backend": "sqlite_exact_bm25",
            "corpus_sha256": corpus_digest.hexdigest(),
            "corpus_path": papers_path.resolve().as_posix(),
            "evidence_scope": "full_text_chunk" if scope == "full_text_chunk" else "abstract_only",
            "k1": K1,
            "b": B,
            "avg_length": total_length / paper_count,
            "documents": paper_count,
            "terms": connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0],
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        temporary.replace(index_path)
        print(f"BM25 磁盘索引完成：{paper_count} 篇子块", flush=True)
        return metadata
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise


def build_index(papers_path: Path, index_path: Path) -> dict:
    if index_path.suffix.lower() in {".sqlite", ".sqlite3"}:
        return _build_sqlite_index(papers_path, index_path)
    papers = load_papers(papers_path)
    scopes = {paper["evidence_scope"] for paper in papers}
    if scopes not in ({"abstract"}, {"full_text_chunk"}):
        raise ValueError("BM25 语料证据范围不一致")
    postings: dict[str, list[list[int]]] = defaultdict(list)
    lengths: list[int] = []
    title_terms: list[list[str]] = []
    for doc_id, paper in enumerate(papers):
        title_terms.append(sorted(set(tokenize(paper["title"]))))
        terms = Counter(tokenize(paper["title"] + " " + paper["abstract"]))
        lengths.append(sum(terms.values()))
        for term, frequency in terms.items():
            postings[term].append([doc_id, frequency])
    index = {
        "version": 1,
        "corpus_sha256": hashlib.sha256(papers_path.read_bytes()).hexdigest(),
        "corpus_path": papers_path.resolve().as_posix(),
        "evidence_scope": "full_text_chunk" if scopes == {"full_text_chunk"} else "abstract_only",
        "k1": K1,
        "b": B,
        "papers": papers,
        "lengths": lengths,
        "title_terms": title_terms,
        "avg_length": sum(lengths) / len(lengths),
        "postings": dict(postings),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=index_path.name, suffix=".tmp", dir=index_path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(index, stream, ensure_ascii=False)
        os.replace(temporary, index_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"papers": len(papers), "terms": len(postings), "index_path": str(index_path)}


def _validate_index_metadata(metadata: dict) -> None:
    required = {"version", "evidence_scope", "documents", "corpus_sha256"}
    if not required <= metadata.keys() or type(metadata.get("version")) is not int or metadata["version"] != 1:
        raise ValueError("BM25 索引元数据不兼容")
    if metadata.get("backend") not in {None, "sqlite_exact_bm25"}:
        raise ValueError("BM25 索引后端不兼容")
    if metadata.get("evidence_scope") not in {"abstract_only", "full_text_chunk"}:
        raise ValueError("BM25 索引证据范围不兼容")
    if type(metadata.get("documents")) is not int or metadata["documents"] <= 0:
        raise ValueError("BM25 索引文档数量无效")
    for name, valid in (
        ("k1", lambda x: x > 0),
        ("b", lambda x: 0 <= x <= 1),
        ("avg_length", lambda x: x >= 0),
    ):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"BM25 索引参数无效：{name}")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite or not valid(value):
            raise ValueError(f"BM25 索引参数无效：{name}")


def _effective_corpus_path(stored_path, override: Path | None) -> Path:
    if override is not None:
        if not isinstance(override, (str, Path)):
            raise ValueError("显式语料路径无效")
        return Path(override)
    if not isinstance(stored_path, str) or not stored_path.strip():
        raise ValueError("索引缺少有效的语料路径")
    return Path(stored_path)


def _check_artifact_signatures(signatures: dict) -> None:
    try:
        if any(file_signature(Path(sig[0])) != sig for sig in signatures.values()):
            raise ValueError("BM25 索引或语料在校验后发生变化，请重新加载索引")
    except OSError as exc:
        raise ValueError("BM25 索引或语料文件已不可用") from exc


def load_index(path: Path, *, corpus_path: Path | None = None,
               validation_cache: dict | None = None) -> dict:
    if not path.is_file():
        raise ValueError(f"索引不存在: {path}。请先运行 python lit.py index")
    if path.suffix.lower() in {".sqlite", ".sqlite3"}:
        connection = None
        try:
            signatures = {"index": file_signature(path)}
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            connection.execute("PRAGMA query_only=ON")
            # Large full-text BM25 indexes are read-only during search. Mapping
            # a bounded window lets the OS page postings directly and avoids
            # repeated SQLite page-cache copies. SQLite may clamp the request
            # to a platform-specific maximum; record the effective value.
            mmap_size_bytes = int(connection.execute(
                f"PRAGMA mmap_size={SQLITE_BM25_MMAP_SIZE_BYTES}"
            ).fetchone()[0])
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(metadata)")}
            if not {"key", "value"} <= columns:
                raise ValueError("SQLite BM25 metadata schema is invalid")
            metadata = {
                key: json.loads(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
            if not isinstance(metadata, dict) or metadata.get("backend") != "sqlite_exact_bm25":
                raise ValueError("SQLite 索引格式不兼容")
            _validate_index_metadata(metadata)
            stored_path = metadata.get("corpus_path")
            effective_path = _effective_corpus_path(stored_path, corpus_path)
            if not effective_path.is_file():
                raise ValueError("索引的语料文件不存在")
            signatures["corpus"] = file_signature(effective_path)
            verify_sha256(effective_path, metadata["corpus_sha256"], cache=validation_cache)
            for table, needed in {
                "docs": {"doc_id", "paper_id", "doc_length", "payload"},
                "terms": {"term", "df"}, "postings": {"term", "doc_id", "tf"},
                "title_terms": {"term", "doc_id"},
            }.items():
                actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                if not needed <= actual:
                    raise ValueError(f"SQLite BM25 table schema is invalid: {table}")
            docs_info = list(connection.execute("PRAGMA table_info(docs)"))
            primary_keys = [row for row in docs_info if row[5]]
            if (len(primary_keys) != 1 or primary_keys[0][1] != "doc_id"
                    or primary_keys[0][2].upper() != "INTEGER"):
                raise ValueError("SQLite BM25 docs.doc_id must be a primary key")
            count, min_id, max_id = connection.execute(
                "SELECT COUNT(*), MIN(doc_id), MAX(doc_id) FROM docs").fetchone()
            if count != metadata["documents"] or min_id != 0 or max_id != count - 1:
                raise ValueError("SQLite BM25 文档数量或 doc_id 不连续")
            _check_artifact_signatures(signatures)
            return {**metadata, "stored_corpus_path": metadata["corpus_path"],
                    "corpus_path": str(effective_path.resolve()), "connection": connection,
                    "artifact_signatures": signatures,
                    "mmap_size_bytes": mmap_size_bytes}
        except sqlite3.DatabaseError as exc:
            if connection is not None:
                connection.close()
            raise ValueError("SQLite BM25 索引损坏或格式无效") from exc
        except Exception as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("SQLite BM25 metadata or schema is invalid") from exc
    try:
        signatures = {"index": file_signature(path)}
        index = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise ValueError("JSON BM25 索引必须是对象")
        if "documents" not in index:
            papers = index.get("papers")
            if not isinstance(papers, list):
                raise ValueError("JSON BM25 索引文档列表无效")
            index["documents"] = len(papers)
        _validate_index_metadata(index)
        stored_path = index.get("corpus_path")
        effective_path = _effective_corpus_path(stored_path, corpus_path)
        if not effective_path.is_file():
            raise ValueError("索引的语料文件不存在")
        signatures["corpus"] = file_signature(effective_path)
        verify_sha256(effective_path, index.get("corpus_sha256"), cache=validation_cache)
        _check_artifact_signatures(signatures)
        index["stored_corpus_path"] = stored_path
        index["corpus_path"] = str(effective_path.resolve())
        index["artifact_signatures"] = signatures
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON BM25 索引或语料文件无效") from exc
    return index


def search(index: dict, question: str, top_k: int = 5, *, title_weight: float = 1.25) -> list[dict]:
    if index.get("backend") == "sqlite_exact_bm25":
        if index.get("evidence_scope") == "full_text_chunk":
            # The full-text index has millions of postings. Cache document
            # lengths once and score the inverted lists in NumPy so each
            # posting does not require random docs/title_terms joins.
            from .sqlite_bm25_numpy import search_sqlite_numpy

            return search_sqlite_numpy(index, question, top_k, title_weight=title_weight)
        # Keep the public API unchanged while moving large on-disk postings
        # scoring into SQLite. The optimized path validates the same loaded
        # artifact signatures and returns the same hit payload shape.
        from .sqlite_bm25_fast import search_sqlite_fast

        return search_sqlite_fast(index, question, top_k)

    _check_artifact_signatures(index.get("artifact_signatures", {}))
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    terms = set(tokenize(question))
    if not terms:
        return []
    if index.get("backend") == "sqlite_exact_bm25":
        connection = index["connection"]
        document_count = index["documents"]
        average_length = index["avg_length"] or 1
        k1 = index["k1"]
        b = index["b"]
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        paper_ids: dict[int, str] = {}
        for term in terms:
            term_row = connection.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
            if term_row is None:
                continue
            df = term_row[0]
            idf = math.log1p((document_count - df + 0.5) / (df + 0.5))
            rows = connection.execute("""
                SELECT p.doc_id, p.tf, d.doc_length, d.paper_id,
                       CASE WHEN tt.doc_id IS NULL THEN 0 ELSE 1 END AS in_title
                FROM postings AS p
                JOIN docs AS d ON d.doc_id = p.doc_id
                LEFT JOIN title_terms AS tt ON tt.doc_id = p.doc_id AND tt.term = p.term
                WHERE p.term = ?
            """, (term,))
            for row in rows:
                doc_id, frequency, length, paper_id, in_title = row
                denominator = frequency + k1 * (1 - b + b * length / average_length)
                contribution = idf * frequency * (k1 + 1) / denominator
                if in_title:
                    contribution *= 1.25
                scores[doc_id] += contribution
                matched[doc_id].add(term)
                paper_ids[doc_id] = paper_id
        ranked_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], paper_ids[doc_id]))[:top_k]
        results = []
        for doc_id in ranked_ids:
            row = connection.execute("SELECT payload FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
            paper = json.loads(row[0])
            results.append({**paper, "score": round(scores[doc_id], 5),
                            "matched_terms": sorted(matched[doc_id])})
        return results
    papers = index["papers"]
    lengths = index["lengths"]
    title_terms = index["title_terms"]
    average_length = index["avg_length"] or 1
    n = len(papers)
    scores: dict[int, float] = defaultdict(float)
    matched: dict[int, set[str]] = defaultdict(set)
    for term in terms:
        matches = index["postings"].get(term, [])
        if not matches:
            continue
        df = len(matches)
        idf = math.log1p((n - df + 0.5) / (df + 0.5))
        for doc_id, frequency in matches:
            denominator = frequency + K1 * (1 - B + B * lengths[doc_id] / average_length)
            contribution = idf * frequency * (K1 + 1) / denominator
            if term in title_terms[doc_id]:
                contribution *= 1.25
            scores[doc_id] += contribution
            matched[doc_id].add(term)
    ranked_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], papers[doc_id]["paper_id"]))
    return [
        {**papers[doc_id], "score": round(scores[doc_id], 5), "matched_terms": sorted(matched[doc_id])}
        for doc_id in ranked_ids[:top_k]
    ]
