"""Exact-postings BM25 prototype that removes per-hit SQLite joins.

This module is opt-in and read-only with respect to the index. It lazily caches
document lengths in a NumPy array (about 11.3 MiB for 2,960,164 int32 rows).
The float64 score and argpartition index arrays use about 45 MiB together per
query, plus a small posting batch and temporary masks. Scoring follows
deterministic sorted-term order. NumPy's vector operations can still differ by
a few floating-point ulps from scalar Python arithmetic.
"""

from __future__ import annotations

import json
import math
import sqlite3
import numbers
from collections.abc import Iterator

from .artifacts import file_signature
from .retrieval import tokenize


_DOC_LENGTH_CACHE_KEY = "_sqlite_numpy_doc_lengths"
_POSTING_BATCH_SIZE = 65_536
_ID_QUERY_BATCH_SIZE = 400


def _check_signatures(signatures: dict) -> None:
    """Fail closed when a loaded index or its corpus has changed."""
    try:
        for expected in signatures.values():
            if file_signature(expected[0]) != expected:
                raise ValueError("BM25 索引或语料在校验后发生变化，请重新加载索引")
    except (OSError, IndexError, TypeError) as exc:
        raise ValueError("BM25 索引或语料文件已不可用") from exc


def _load_document_lengths(index: dict, connection: sqlite3.Connection, np):
    """Load the contiguous doc_id -> doc_length array once per loaded index."""
    cached = index.get(_DOC_LENGTH_CACHE_KEY)
    document_count = int(index["documents"])
    if (isinstance(cached, np.ndarray) and cached.shape == (document_count,)
            and cached.dtype == np.int32):
        return cached

    lengths = np.empty(document_count, dtype=np.int32)
    cursor = connection.execute(
        "SELECT doc_id, doc_length FROM docs ORDER BY doc_id"
    )
    offset = 0
    while batch := cursor.fetchmany(_POSTING_BATCH_SIZE):
        count = len(batch)
        if offset + count > document_count:
            raise ValueError("SQLite BM25 docs 数量与索引元数据不一致")
        expected_ids = np.arange(offset, offset + count, dtype=np.int64)
        doc_ids = np.fromiter((int(row[0]) for row in batch), dtype=np.int64, count=count)
        if not np.array_equal(doc_ids, expected_ids):
            raise ValueError("SQLite BM25 docs.doc_id 不连续")
        batch_lengths = np.fromiter(
            (int(row[1]) for row in batch), dtype=np.int64, count=count
        )
        if np.any(batch_lengths < 0) or np.any(batch_lengths > np.iinfo(np.int32).max):
            raise ValueError("SQLite BM25 doc_length 超出 int32 范围")
        lengths[offset:offset + count] = batch_lengths
        offset += count
    if offset != document_count:
        raise ValueError("SQLite BM25 docs 数量与索引元数据不一致")

    index[_DOC_LENGTH_CACHE_KEY] = lengths
    return lengths


def prepare_numpy_index(index: dict):
    """Preload and cache per-document lengths outside a timed query loop."""
    if not isinstance(index, dict) or index.get("backend") != "sqlite_exact_bm25":
        raise ValueError("prepare_numpy_index 仅支持 sqlite_exact_bm25 索引")
    if index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("prepare_numpy_index 仅支持 full_text_chunk 索引")
    _check_signatures(index.get("artifact_signatures", {}))
    connection = index.get("connection")
    if not isinstance(connection, sqlite3.Connection):
        raise ValueError("SQLite BM25 索引缺少已加载的只读连接")
    import numpy as np

    lengths = _load_document_lengths(index, connection, np)
    _check_signatures(index.get("artifact_signatures", {}))
    return {"documents": int(lengths.size), "dtype": str(lengths.dtype),
            "bytes": int(lengths.nbytes)}


def _fetch_paper_ids(connection: sqlite3.Connection, doc_ids) -> dict[int, str]:
    """Fetch paper IDs for a small set of candidates in SQLite-safe batches."""
    ids = [int(doc_id) for doc_id in doc_ids]
    paper_ids: dict[int, str] = {}
    for start in range(0, len(ids), _ID_QUERY_BATCH_SIZE):
        batch = ids[start:start + _ID_QUERY_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT doc_id, paper_id FROM docs WHERE doc_id IN ({placeholders})", batch
        ).fetchall()
        paper_ids.update({int(row[0]): str(row[1]) for row in rows})
    if len(paper_ids) != len(ids):
        raise ValueError("SQLite BM25 候选文档缺失")
    return paper_ids


def _paper_id_rows_for_ids(
    connection: sqlite3.Connection, doc_ids
) -> Iterator[tuple[int, str]]:
    for start in range(0, len(doc_ids), _ID_QUERY_BATCH_SIZE):
        batch = [int(doc_id) for doc_id in doc_ids[start:start + _ID_QUERY_BATCH_SIZE]]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT doc_id, paper_id FROM docs WHERE doc_id IN ({placeholders})", batch
        )
        for row in rows:
            yield int(row[0]), str(row[1])


def search_sqlite_numpy(index: dict, question: str, top_k: int = 5, *, title_weight: float = 1.25) -> list[dict]:
    """Search an already loaded full-text SQLite BM25 index using NumPy.

    Postings are streamed term by term. SQLite returns only ``doc_id`` and
    ``tf``; document lengths and title membership are applied from cached or
    per-term arrays, avoiding docs/title_terms joins for every posting hit.
    """
    if not isinstance(index, dict) or index.get("backend") != "sqlite_exact_bm25":
        raise ValueError("search_sqlite_numpy 仅支持 sqlite_exact_bm25 索引")
    if index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("search_sqlite_numpy 仅支持 full_text_chunk 索引")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是大于 0 的整数")
    if (isinstance(title_weight, bool) or not isinstance(title_weight, numbers.Real)
            or not math.isfinite(float(title_weight)) or title_weight < 0):
        raise ValueError("title_weight 必须是有限的非负数")
    title_weight = float(title_weight)

    signatures = index.get("artifact_signatures", {})
    _check_signatures(signatures)
    terms = sorted(set(tokenize(question)))
    if not terms:
        _check_signatures(signatures)
        return []

    connection = index.get("connection")
    if not isinstance(connection, sqlite3.Connection):
        raise ValueError("SQLite BM25 索引缺少已加载的只读连接")

    import numpy as np

    document_count = int(index["documents"])
    average_length = float(index.get("avg_length") or 1.0)
    k1 = float(index["k1"])
    b = float(index["b"])
    lengths = _load_document_lengths(index, connection, np)
    scores = np.zeros(document_count, dtype=np.float64)

    for term in terms:
        df_row = connection.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
        if df_row is None:
            continue
        df = int(df_row[0])
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))

        title_cursor = connection.execute(
            "SELECT doc_id FROM title_terms WHERE term = ? ORDER BY doc_id", (term,)
        )
        title_ids = np.fromiter((int(row[0]) for row in title_cursor), dtype=np.int64)

        posting_cursor = connection.execute(
            "SELECT doc_id, tf FROM postings WHERE term = ?", (term,)
        )
        while batch := posting_cursor.fetchmany(_POSTING_BATCH_SIZE):
            count = len(batch)
            doc_ids = np.fromiter((int(row[0]) for row in batch), dtype=np.int64, count=count)
            frequencies = np.fromiter((int(row[1]) for row in batch), dtype=np.float64, count=count)
            doc_lengths = lengths[doc_ids].astype(np.float64, copy=False)
            denominator = frequencies + k1 * (1.0 - b + b * doc_lengths / average_length)
            contributions = idf * frequencies * (k1 + 1.0) / denominator

            if title_ids.size:
                positions = np.searchsorted(title_ids, doc_ids)
                title_mask = positions < title_ids.size
                matching_positions = np.flatnonzero(title_mask)
                title_mask[matching_positions] = (
                    title_ids[positions[matching_positions]] == doc_ids[matching_positions]
                )
                contributions[title_mask] *= title_weight
            scores[doc_ids] += contributions

    matching_count = int(np.count_nonzero(scores))
    if not matching_count:
        _check_signatures(signatures)
        return []

    # Select an initial score cutoff, then resolve ties by paper_id. The tie
    # resolver below only fetches paper_id values, never payloads or terms.
    if matching_count <= top_k:
        ranked_ids = [int(doc_id) for doc_id in np.flatnonzero(scores > 0)]
        selected_paper_ids = _fetch_paper_ids(connection, ranked_ids)
        ranked_ids.sort(
            key=lambda doc_id: (-float(scores[doc_id]), selected_paper_ids[doc_id], doc_id)
        )
    else:
        partitioned_ids = np.argpartition(scores, document_count - top_k)
        cutoff = float(np.min(scores[partitioned_ids[-top_k:]]))
        higher_ids = np.flatnonzero(scores > cutoff)
        tied_ids = np.flatnonzero(scores == cutoff)
        higher_paper_ids = _fetch_paper_ids(connection, higher_ids)
        higher_sorted = sorted(
            (int(doc_id) for doc_id in higher_ids),
            key=lambda doc_id: (-float(scores[doc_id]), higher_paper_ids[doc_id], doc_id),
        )
        slots = top_k - len(higher_sorted)
        # Top-k ties may be numerous. Use a bounded heap-style selection via
        # nsmallest while streaming paper IDs in batches from SQLite.
        import heapq

        tied_rows = _paper_id_rows_for_ids(connection, tied_ids)
        chosen_ties = heapq.nsmallest(
            slots, tied_rows, key=lambda item: (item[1], item[0])
        )
        ranked_ids = higher_sorted + [doc_id for doc_id, _ in chosen_ties]
        selected_paper_ids = dict(higher_paper_ids)
        selected_paper_ids.update({doc_id: paper_id for doc_id, paper_id in chosen_ties})

    ranked_ids.sort(
        key=lambda doc_id: (-float(scores[doc_id]), selected_paper_ids[doc_id], doc_id)
    )

    selected_sql = ",".join("(?)" for _ in ranked_ids)
    query_terms_sql = ",".join("(?)" for _ in terms)
    matched_rows = connection.execute(
        f"""
        WITH selected_docs(doc_id) AS (VALUES {selected_sql}),
             query_terms(term) AS (VALUES {query_terms_sql})
        SELECT p.doc_id, p.term
        FROM selected_docs AS s
        JOIN postings AS p INDEXED BY postings_doc_id_idx ON p.doc_id = s.doc_id
        JOIN query_terms AS q ON q.term = p.term
        ORDER BY p.doc_id, p.term
        """,
        (*ranked_ids, *terms),
    ).fetchall()
    matched_by_id: dict[int, list[str]] = {doc_id: [] for doc_id in ranked_ids}
    for row in matched_rows:
        matched_by_id[int(row[0])].append(str(row[1]))

    placeholders = ",".join("?" for _ in ranked_ids)
    payload_rows = connection.execute(
        f"SELECT doc_id, payload FROM docs WHERE doc_id IN ({placeholders})",
        ranked_ids,
    ).fetchall()
    paper_by_id = {int(row[0]): json.loads(row[1]) for row in payload_rows}
    if len(paper_by_id) != len(ranked_ids):
        raise ValueError("SQLite BM25 候选文档缺失")

    results = []
    for doc_id in ranked_ids:
        results.append({
            **paper_by_id[doc_id],
            "score": round(float(scores[doc_id]), 5),
            "matched_terms": matched_by_id[doc_id],
        })
    _check_signatures(signatures)
    return results

