"""Fast SQL-side ranking for an already loaded SQLite BM25 index.

This module is an opt-in prototype. It does not build or mutate the index and
does not replace :func:`litagent.retrieval.search`.
"""

from __future__ import annotations

import json
import math
import sqlite3

from .artifacts import file_signature
from .retrieval import tokenize


def _check_signatures(signatures: dict) -> None:
    """Fail closed if a source artifact changed after the index was loaded."""
    try:
        for expected in signatures.values():
            if file_signature(expected[0]) != expected:
                raise ValueError("BM25 索引或语料在校验后发生变化，请重新加载索引")
    except (OSError, IndexError, TypeError) as exc:
        raise ValueError("BM25 索引或语料文件已不可用") from exc


def _ordered_unique_terms(question: str) -> list[str]:
    # Keep tokenization identical to retrieval.search, while ensuring the
    # bound values and SQL aggregation inputs have stable ordering.
    return sorted(set(tokenize(question)))


def search_sqlite_fast(index: dict, question: str, top_k: int = 5) -> list[dict]:
    """Search a loaded ``sqlite_exact_bm25`` index using SQL aggregation.

    Query terms are looked up once, and SQLite computes and ranks the document
    scores in one aggregation query. Full JSON payloads are returned only for
    the requested top-k rows. Non-SQLite indexes raise ``ValueError`` so a
    caller can explicitly fall back to ``retrieval.search``.
    """
    if not isinstance(index, dict) or index.get("backend") != "sqlite_exact_bm25":
        raise ValueError("search_sqlite_fast 仅支持 sqlite_exact_bm25 索引")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是大于 0 的整数")

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

    placeholders = ",".join("?" for _ in terms)
    term_rows = connection.execute(
        f"SELECT term, df FROM terms WHERE term IN ({placeholders})", terms
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

    values_sql = ",".join("(?, ?)" for _ in scored_terms)
    term_values: list[object] = []
    for term, idf in scored_terms:
        term_values.extend((term, idf))

    # The query's VALUES rows are sorted by term. SQLite SUM can use a
    # different floating-point accumulation order from the old Python loop,
    # so scores extremely close to a ranking boundary may differ by a few ulps.
    # Do not aggregate matched terms here: high-df queries can touch millions
    # of docs, while matched terms are needed only for the selected top-k.
    rows = connection.execute(
        f"""
        WITH query_terms(term, idf) AS (VALUES {values_sql}),
        ranked AS (
            SELECT
                d.doc_id AS doc_id,
                d.paper_id AS paper_id,
                SUM(
                    query_terms.idf * p.tf * (? + 1.0)
                    / (p.tf + ? * (1.0 - ? + ? * d.doc_length / ?))
                    * CASE WHEN tt.doc_id IS NULL THEN 1.0 ELSE 1.25 END
                    ORDER BY query_terms.term
                ) AS score
            FROM query_terms
            JOIN postings AS p ON p.term = query_terms.term
            JOIN docs AS d ON d.doc_id = p.doc_id
            LEFT JOIN title_terms AS tt
                ON tt.term = p.term AND tt.doc_id = p.doc_id
            GROUP BY d.doc_id, d.paper_id
            ORDER BY score DESC, d.paper_id ASC
            LIMIT ?
        )
        SELECT d.doc_id, d.payload, ranked.score
        FROM ranked
        JOIN docs AS d ON d.doc_id = ranked.doc_id
        ORDER BY ranked.score DESC, ranked.paper_id ASC
        """,
        (*term_values, k1, k1, b, b, average_length, top_k),
    ).fetchall()

    if not rows:
        _check_signatures(signatures)
        return []

    # The shipped SQLite schema creates postings_doc_id_idx. Force its use so
    # this top-k-only lookup visits each selected document's postings instead
    # of starting from high-frequency query terms across the whole corpus.
    selected_ids = [int(row["doc_id"]) for row in rows]
    selected_sql = ",".join("(?)" for _ in selected_ids)
    query_terms_sql = ",".join("(?)" for _ in scored_terms)
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
        (*selected_ids, *(term for term, _ in scored_terms)),
    ).fetchall()
    matched_by_id: dict[int, list[str]] = {doc_id: [] for doc_id in selected_ids}
    for matched_row in matched_rows:
        matched_by_id[int(matched_row["doc_id"])].append(str(matched_row["term"]))

    results = []
    for row in rows:
        doc_id = int(row["doc_id"])
        paper = json.loads(row["payload"])
        results.append({
            **paper,
            "score": round(float(row["score"]), 5),
            "matched_terms": matched_by_id[doc_id],
        })

    _check_signatures(signatures)
    return results

