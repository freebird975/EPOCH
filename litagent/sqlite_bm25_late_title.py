"""Candidate-pool BM25 prototype with title boosts applied late.

This is approximate until evaluation shows that the body-only candidate pool
preserves the exact Top100. It does not build or modify the SQLite index.
"""

from __future__ import annotations

import json
import math
import sqlite3

from .artifacts import file_signature
from .retrieval import tokenize


def _check_signatures(signatures: dict) -> None:
    try:
        for expected in signatures.values():
            if file_signature(expected[0]) != expected:
                raise ValueError("BM25 索引或语料在校验后发生变化，请重新加载索引")
    except (OSError, IndexError, TypeError) as exc:
        raise ValueError("BM25 索引或语料文件已不可用") from exc


def _ordered_unique_terms(question: str) -> list[str]:
    return sorted(set(tokenize(question)))


def search_sqlite_late_title(
    index: dict,
    question: str,
    top_k: int = 5,
    candidate_pool: int | None = None,
) -> list[dict]:
    """Search a loaded SQLite BM25 index with a late title-boost pass.

    The first SQL query ranks by the unboosted body score and fetches an
    over-sized candidate pool (default ``max(top_k * 10, 1000)``). The second
    pass applies the usual 1.25 title multiplier to query-term matches only
    within those candidates. This is approximate until evaluation confirms
    that the pool retains the exact Top100 for the target corpus and queries.

    Returned records match ``retrieval.search``: original payload fields,
    score rounded to five decimals, and sorted matched terms. Only
    ``sqlite_exact_bm25`` indexes are supported; callers may fall back to
    ``retrieval.search`` for other backends.
    """
    if not isinstance(index, dict) or index.get("backend") != "sqlite_exact_bm25":
        raise ValueError("search_sqlite_late_title 仅支持 sqlite_exact_bm25 索引")
    if index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("search_sqlite_late_title 仅支持 full_text_chunk 索引")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是大于 0 的整数")
    if candidate_pool is None:
        candidate_pool = max(top_k * 10, 1000)
    if (isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int)
            or candidate_pool < top_k):
        raise ValueError("candidate_pool 必须是大于等于 top_k 的整数")

    signatures = index.get("artifact_signatures", {})
    _check_signatures(signatures)

    terms = _ordered_unique_terms(question)
    if not terms:
        _check_signatures(signatures)
        return []

    connection = index.get("connection")
    if not isinstance(connection, sqlite3.Connection):
        raise ValueError("SQLite BM25 索引缺少已加载的只读连接")

    document_count = int(index["documents"])
    average_length = float(index.get("avg_length") or 1.0)
    k1 = float(index["k1"])
    b = float(index["b"])
    term_placeholders = ",".join("?" for _ in terms)
    term_rows = connection.execute(
        f"SELECT term, df FROM terms WHERE term IN ({term_placeholders})", terms
    ).fetchall()
    df_by_term = {str(row["term"]): int(row["df"]) for row in term_rows}
    scored_terms = [
        (term, math.log1p((document_count - df + 0.5) / (df + 0.5)))
        for term in terms
        if (df := df_by_term.get(term)) is not None
    ]
    if not scored_terms:
        _check_signatures(signatures)
        return []

    query_terms_sql = ",".join("(?, ?)" for _ in scored_terms)
    term_values: list[object] = []
    idf_by_term: dict[str, float] = {}
    for term, idf in scored_terms:
        term_values.extend((term, idf))
        idf_by_term[term] = idf

    # This aggregation avoids joining title_terms for every posting hit.
    # Scores can differ by a few ulps from Python's accumulation order.
    body_rows = connection.execute(
        f"""
        WITH query_terms(term, idf) AS (VALUES {query_terms_sql})
        SELECT
            d.doc_id AS doc_id,
            d.paper_id AS paper_id,
            SUM(
                query_terms.idf * p.tf * (? + 1.0)
                / (p.tf + ? * (1.0 - ? + ? * d.doc_length / ?))
            ) AS body_score
        FROM query_terms
        JOIN postings AS p ON p.term = query_terms.term
        JOIN docs AS d ON d.doc_id = p.doc_id
        GROUP BY d.doc_id, d.paper_id
        ORDER BY body_score DESC, d.paper_id ASC
        LIMIT ?
        """,
        (*term_values, k1, k1, b, b, average_length, candidate_pool),
    ).fetchall()
    if not body_rows:
        _check_signatures(signatures)
        return []

    candidate_ids = [int(row["doc_id"]) for row in body_rows]
    candidates = {
        int(row["doc_id"]): {
            "paper_id": str(row["paper_id"]),
            "score": float(row["body_score"]),
            "matched_terms": [],
        }
        for row in body_rows
    }

    # Restrict the exact title check to the body-only candidate pool. The
    # postings doc-id index narrows the work before matching query terms.
    candidate_values_sql = ",".join("(?)" for _ in candidate_ids)
    query_term_names_sql = ",".join("(?)" for _ in scored_terms)
    title_rows = connection.execute(
        f"""
        WITH candidates(doc_id) AS (VALUES {candidate_values_sql}),
             query_terms(term) AS (VALUES {query_term_names_sql})
        SELECT p.doc_id, p.term, p.tf, d.doc_length
        FROM candidates AS c
        CROSS JOIN query_terms AS q
        JOIN postings AS p INDEXED BY postings_doc_id_idx
            ON p.doc_id = c.doc_id AND p.term = q.term
        JOIN title_terms AS tt ON tt.doc_id = c.doc_id AND tt.term = q.term
        JOIN docs AS d ON d.doc_id = c.doc_id
        ORDER BY p.doc_id, p.term
        """,
        (*candidate_ids, *(term for term, _ in scored_terms)),
    ).fetchall()

    for row in title_rows:
        doc_id = int(row["doc_id"])
        term = str(row["term"])
        frequency = int(row["tf"])
        length = int(row["doc_length"])
        denominator = frequency + k1 * (1 - b + b * length / average_length)
        contribution = idf_by_term[term] * frequency * (k1 + 1) / denominator
        candidates[doc_id]["score"] += contribution * 0.25

    ranked_ids = sorted(
        candidate_ids,
        key=lambda doc_id: (-candidates[doc_id]["score"], candidates[doc_id]["paper_id"]),
    )[:top_k]
    final_values_sql = ",".join("(?)" for _ in ranked_ids)
    matched_terms_sql = ",".join("(?)" for _ in scored_terms)
    matched_rows = connection.execute(
        f"""
        WITH selected_docs(doc_id) AS (VALUES {final_values_sql}),
             query_terms(term) AS (VALUES {matched_terms_sql})
        SELECT p.doc_id, p.term
        FROM selected_docs AS s
        JOIN postings AS p INDEXED BY postings_doc_id_idx ON p.doc_id = s.doc_id
        JOIN query_terms AS q ON q.term = p.term
        ORDER BY p.doc_id, p.term
        """,
        (*ranked_ids, *(term for term, _ in scored_terms)),
    ).fetchall()
    matched_by_id: dict[int, list[str]] = {doc_id: [] for doc_id in ranked_ids}
    for row in matched_rows:
        matched_by_id[int(row["doc_id"])].append(str(row["term"]))

    payload_placeholders = ",".join("?" for _ in ranked_ids)
    payload_rows = connection.execute(
        f"SELECT doc_id, payload FROM docs WHERE doc_id IN ({payload_placeholders})",
        ranked_ids,
    ).fetchall()
    payload_by_id = {int(row["doc_id"]): json.loads(row["payload"]) for row in payload_rows}

    results = []
    for doc_id in ranked_ids:
        results.append({
            **payload_by_id[doc_id],
            "score": round(candidates[doc_id]["score"], 5),
            "matched_terms": matched_by_id[doc_id],
        })

    _check_signatures(signatures)
    return results

