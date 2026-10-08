"""Offline post-hoc Cross Encoder evaluation over full-text LitSearch qrels.

This module reuses a completed v4 retrieval checkpoint. It reads no FAISS
vectors, runs no retrieval, and makes no hosted API calls. The only model work
is local Cross Encoder inference over the top 20 chunks from a fixed RRF pool.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .fulltext_eval_metrics import summarize_fulltext_eval
from .fulltext_fusion_tuning import (
    CHECKPOINT_EVALUATOR_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    _load_checkpoint,
    _load_parent_ids,
    _read_jsonl,
    _sha256,
)
from .hybrid import fuse_rrf
from .reranker import DEFAULT_RERANK_MODEL, load_reranker, rerank_children, reranker_metadata


RRF_CANDIDATE_K = 100
RRF_CONSTANT = 60
RRF_CHILD_POOL_SIZE = 200
RERANK_LIMIT = 20
SQLITE_BATCH_SIZE = 400


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite input does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validate_docs_table(connection: sqlite3.Connection, path: Path, required: set[str], *, primary_key: str) -> None:
    columns = list(connection.execute("PRAGMA table_info(docs)"))
    if not columns:
        raise ValueError(f"SQLite input has no docs table: {path}")
    names = {str(row[1]) for row in columns}
    if not required.issubset(names):
        raise ValueError(f"SQLite docs table at {path} is missing columns: {sorted(required - names)}")
    primary_keys = {str(row[1]) for row in columns if int(row[5]) > 0}
    if primary_key not in primary_keys:
        raise ValueError(f"SQLite docs.{primary_key} is not declared as a primary key: {path}")
    if primary_key == "doc_id":
        doc_id_column = next(row for row in columns if str(row[1]) == "doc_id")
        if str(doc_id_column[2]).upper() != "INTEGER" or int(doc_id_column[5]) != 1:
            raise ValueError(f"SQLite docs.doc_id must be an INTEGER PRIMARY KEY: {path}")


def _scan_sidecar_doc_ids(
    sidecar_connection: sqlite3.Connection,
    sidecar_path: Path,
    requested_child_ids: set[str],
) -> dict[str, tuple[int, str]]:
    """Scan the sidecar once; retain mappings only for requested chunks."""
    _validate_docs_table(
        sidecar_connection,
        sidecar_path,
        {"doc_id", "paper_id", "byte_offset"},
        primary_key="doc_id",
    )
    found: dict[str, tuple[int, str]] = {}
    cursor = sidecar_connection.execute("SELECT doc_id, paper_id, byte_offset FROM docs")
    while batch := cursor.fetchmany(20_000):
        for raw_doc_id, raw_paper_id, raw_offset in batch:
            if raw_doc_id is None or raw_paper_id is None or raw_offset is None:
                raise ValueError(f"Dense sidecar contains a NULL docs row: {sidecar_path}")
            paper_id = str(raw_paper_id).strip()
            if paper_id not in requested_child_ids:
                continue
            if paper_id in found:
                raise ValueError(f"Dense sidecar has duplicate paper_id {paper_id!r}")
            try:
                doc_id, byte_offset = int(raw_doc_id), int(raw_offset)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Dense sidecar has invalid doc_id/byte_offset for {paper_id!r}") from exc
            if doc_id < 0 or byte_offset < 0:
                raise ValueError(f"Dense sidecar has negative doc_id/byte_offset for {paper_id!r}")
            found[paper_id] = (doc_id, paper_id)
    missing = sorted(requested_child_ids - found.keys())
    if missing:
        raise ValueError(f"Dense sidecar is missing {len(missing)} requested child IDs; first={missing[:5]}")
    return found


def _fetch_bm25_passages(
    bm25_connection: sqlite3.Connection,
    bm25_path: Path,
    child_to_doc: Mapping[str, tuple[int, str]],
) -> dict[str, dict[str, str]]:
    """Fetch requested payloads in bounded batches and verify both ID maps."""
    _validate_docs_table(
        bm25_connection,
        bm25_path,
        {"doc_id", "paper_id", "payload"},
        primary_key="doc_id",
    )
    child_ids = sorted(child_to_doc)
    passages: dict[str, dict[str, str]] = {}
    for start in range(0, len(child_ids), SQLITE_BATCH_SIZE):
        batch_ids = child_ids[start:start + SQLITE_BATCH_SIZE]
        doc_ids = [child_to_doc[child_id][0] for child_id in batch_ids]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Dense sidecar maps multiple requested child IDs to one doc_id")
        placeholders = ",".join("?" for _ in doc_ids)
        rows = bm25_connection.execute(
            f"SELECT doc_id, paper_id, payload FROM docs WHERE doc_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        by_doc_id = {int(row[0]): row for row in rows}
        if len(by_doc_id) != len(doc_ids):
            missing = sorted(set(doc_ids) - by_doc_id.keys())
            raise ValueError(f"BM25 docs is missing {len(missing)} sidecar doc_ids; first={missing[:5]}")
        for child_id in batch_ids:
            doc_id, sidecar_paper_id = child_to_doc[child_id]
            _, bm25_paper_id, payload_text = by_doc_id[doc_id]
            bm25_paper_id = str(bm25_paper_id).strip()
            if bm25_paper_id != sidecar_paper_id or bm25_paper_id != child_id:
                raise ValueError(
                    f"BM25/sidecar paper_id mismatch for child {child_id!r}: "
                    f"BM25={bm25_paper_id!r}, sidecar={sidecar_paper_id!r}"
                )
            try:
                payload = json.loads(payload_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid BM25 payload JSON for child {child_id!r}") from exc
            if not isinstance(payload, dict) or str(payload.get("paper_id") or "").strip() != sidecar_paper_id:
                raise ValueError(f"BM25 payload paper_id does not match sidecar for child {child_id!r}")
            title = str(payload.get("title") or "")
            abstract = str(payload.get("abstract") or "")
            if not title and not abstract:
                raise ValueError(f"BM25 payload has no rerankable text for child {child_id!r}")
            passages[child_id] = {"title": title, "abstract": abstract}
    return passages


def _parent_ids_in_rank_order(hits: Sequence[Mapping[str, Any]], corpus_parent_ids: set[str]) -> tuple[list[str], list[str], int]:
    ranked: list[str] = []
    candidate_pool: list[str] = []
    seen_ranked: set[str] = set()
    seen_candidate: set[str] = set()
    orphan_count = 0
    for hit in hits:
        parent_id = str(hit.get("parent_id") or "").strip()
        if not parent_id or parent_id not in corpus_parent_ids:
            orphan_count += 1
            continue
        if parent_id not in seen_candidate:
            candidate_pool.append(parent_id)
            seen_candidate.add(parent_id)
        if parent_id not in seen_ranked:
            ranked.append(parent_id)
            seen_ranked.add(parent_id)
    return ranked, candidate_pool, orphan_count


def _metric_delta(swept: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in ("bm25", "dense", "hybrid"):
        current = swept[mode]["metrics_macro_mean"]
        reference = baseline[mode]["metrics_macro_mean"]
        output[mode] = {
            key: (value - reference[key] if value is not None and reference.get(key) is not None else None)
            for key, value in current.items()
        }
    return output


def run_fulltext_rerank_eval(
    queries_path: str | Path,
    checkpoint_path: str | Path,
    bm25_path: str | Path,
    dense_docs_path: str | Path,
    *,
    report_path: str | Path | None = None,
    reranker_model: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate cached local Cross Encoder reranking on full-text queries.

    This is exploratory because it reuses the same official qrels that selected
    and diagnosed the retrieval run; it is not a held-out estimate. The parent
    corpus path is taken from checkpoint metadata and verified by hash. BM25
    and Dense child ranks are reconstructed from stored checkpoint hit order,
    then fused through the production :func:`fuse_rrf` implementation.
    """
    queries_path = Path(queries_path)
    checkpoint_path = Path(checkpoint_path)
    bm25_path = Path(bm25_path)
    dense_docs_path = Path(dense_docs_path)
    for label, path in (("queries", queries_path), ("checkpoint", checkpoint_path),
                        ("BM25 SQLite", bm25_path), ("Dense sidecar", dense_docs_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    queries = _read_jsonl(queries_path)
    checkpoint_retrievals, source_metadata = _load_checkpoint(checkpoint_path, queries, queries_path)
    parameters = source_metadata.get("parameters", {})
    if parameters.get("candidate_k_per_channel") != RRF_CANDIDATE_K or parameters.get("rrf_k") != RRF_CONSTANT:
        raise ValueError("Reranker evaluation requires the baseline candidate_k=100 and RRF k=60 checkpoint")
    paths = source_metadata.get("input_paths", {})
    hashes = source_metadata.get("hashes", {})
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError("Checkpoint metadata lacks input_paths or hashes")
    if _sha256(bm25_path) != hashes.get("bm25_index_sha256"):
        raise ValueError("Supplied BM25 SQLite hash does not match the checkpoint")
    if _sha256(dense_docs_path) != hashes.get("dense_docs_sha256"):
        raise ValueError("Supplied Dense sidecar hash does not match the checkpoint")
    parent_path_value = paths.get("parents")
    if not parent_path_value or not isinstance(hashes.get("parents_sha256"), str):
        raise ValueError("Checkpoint metadata lacks parent corpus path/hash")
    parent_path = Path(parent_path_value)
    corpus_parent_ids = _load_parent_ids(parent_path, hashes["parents_sha256"])
    expected_parent_count = source_metadata.get("corpus", {}).get("parent_count")
    if expected_parent_count is not None and expected_parent_count != len(corpus_parent_ids):
        raise ValueError("Parent corpus count does not match checkpoint metadata")

    candidate_child_hits: dict[str, list[dict[str, Any]]] = {}
    baseline_query_retrievals: dict[str, dict[str, Any]] = {}
    requested_child_ids: set[str] = set()
    for query in queries:
        query_id = str(query["id"])
        query_channels = checkpoint_retrievals[query_id]
        bm25_hits = query_channels["bm25"]["child_hits"]
        dense_hits = query_channels["dense"]["child_hits"]
        pool = fuse_rrf(bm25_hits, dense_hits, top_k=RRF_CHILD_POOL_SIZE, rrf_k=RRF_CONSTANT)
        if not pool:
            raise ValueError(f"Query {query_id!r} has no RRF candidates")
        candidate_child_hits[query_id] = pool
        requested_child_ids.update(str(hit["paper_id"]) for hit in pool[:RERANK_LIMIT])
        baseline_modes: dict[str, Any] = {}
        for mode, channel_hits in (("bm25", bm25_hits), ("dense", dense_hits), ("hybrid", pool)):
            parent_ids, candidate_parent_ids, orphan_hits = _parent_ids_in_rank_order(channel_hits, corpus_parent_ids)
            original = query_channels[mode]
            baseline_modes[mode] = {
                "parent_ids": parent_ids,
                "candidate_parent_ids": candidate_parent_ids,
                "child_hits": [{"paper_id": str(hit.get("paper_id") or ""),
                                 "parent_id": str(hit.get("parent_id") or "")} for hit in channel_hits],
                "orphan_child_hits": orphan_hits,
                "latency_seconds": original.get("latency_seconds"),
            }
        baseline_query_retrievals[query_id] = baseline_modes

    if len(requested_child_ids) > len(queries) * RERANK_LIMIT:
        raise ValueError("Unique reranker input count exceeded the top-20-per-query bound")

    sidecar_connection = _readonly_connection(dense_docs_path)
    bm25_connection = _readonly_connection(bm25_path)
    try:
        child_to_doc = _scan_sidecar_doc_ids(sidecar_connection, dense_docs_path, requested_child_ids)
    finally:
        sidecar_connection.close()
    try:
        passages = _fetch_bm25_passages(bm25_connection, bm25_path, child_to_doc)
    finally:
        bm25_connection.close()

    if reranker_model is None:
        if progress:
            progress("Loading cached Cross Encoder in offline mode")
        reranker_model = load_reranker(DEFAULT_RERANK_MODEL, Path(".rag/models"), offline=True)
    model_metadata = reranker_metadata(reranker_model)
    if model_metadata.get("cross_encoder_model") != DEFAULT_RERANK_MODEL:
        raise ValueError(f"Unexpected Cross Encoder model: {model_metadata.get('cross_encoder_model')!r}")

    reranked_query_retrievals: dict[str, dict[str, Any]] = {}
    rerank_seconds_by_query: dict[str, float] = {}
    for number, query in enumerate(queries, 1):
        query_id = str(query["id"])
        pool = candidate_child_hits[query_id]
        enriched_pool = []
        for hit in pool:
            child_id = str(hit["paper_id"])
            text = passages.get(child_id)
            enriched_pool.append({**hit, **(text if text is not None else {})})
        started = time.perf_counter()
        reranked_pool = rerank_children(query["question"], enriched_pool, reranker_model, limit=RERANK_LIMIT)
        elapsed = time.perf_counter() - started
        rerank_seconds_by_query[query_id] = round(elapsed, 6)

        baseline_modes = baseline_query_retrievals[query_id]
        reranked_parent_ids, _, reranked_orphans = _parent_ids_in_rank_order(reranked_pool, corpus_parent_ids)
        _, original_candidate_parents, _ = _parent_ids_in_rank_order(pool, corpus_parent_ids)
        modes = {
            "bm25": baseline_modes["bm25"],
            "dense": baseline_modes["dense"],
            "hybrid": {
                "parent_ids": reranked_parent_ids,
                # Keep the original RRF pool for candidate-recall coverage.
                "candidate_parent_ids": original_candidate_parents,
                "child_hits": [{"paper_id": str(hit.get("paper_id") or ""),
                                 "parent_id": str(hit.get("parent_id") or "")} for hit in reranked_pool],
                "orphan_child_hits": reranked_orphans,
                "latency_seconds": {"cross_encoder_rerank": elapsed, "mode_total": elapsed},
            },
        }
        reranked_query_retrievals[query_id] = modes
        if progress:
            progress(f"Reranked {number}/{len(queries)}: {query_id} ({elapsed:.3f}s)")

    checkpoint_sha256 = _sha256(checkpoint_path)
    report_metadata = {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "evaluator_version": CHECKPOINT_EVALUATOR_VERSION,
            "fingerprint": _checkpoint_fingerprint_from_metadata(checkpoint_path),
        },
        "input_hashes": {
            "queries_sha256": _sha256(queries_path),
            "bm25_index_sha256": _sha256(bm25_path),
            "dense_docs_sha256": _sha256(dense_docs_path),
            "parents_sha256": hashes["parents_sha256"],
            "checkpoint_source_hashes": hashes,
        },
        "reranker": model_metadata,
        "candidate_limits": {
            "bm25_top_k_per_channel": RRF_CANDIDATE_K,
            "dense_top_k_per_channel": RRF_CANDIDATE_K,
            "rrf_k": RRF_CONSTANT,
            "rrf_pool_top_k": RRF_CHILD_POOL_SIZE,
            "cross_encoder_top_k": RERANK_LIMIT,
        },
        "evaluation_protocol": {
            "qrels": "official LitSearch gold_paper_ids",
            "same_queries_used_for_retrieval_evaluation": True,
            "held_out": False,
            "interpretation": "exploratory post-hoc result; not an unbiased held-out estimate",
            "candidate_pool_coverage": "computed from pre-rerank RRF candidate pool",
            "unjudged_documents": "remain unjudged; only official positive labels are relevant",
            "rerank_latency": "per-query local model inference only; excludes payload lookup and model load",
        },
        "corpus": {
            "parent_count": len(corpus_parent_ids),
            "evidence_scope": "full_text_chunk",
        },
    }
    baseline_report = summarize_fulltext_eval(
        queries, baseline_query_retrievals, corpus_parent_ids,
        {**report_metadata, "evaluation_variant": "hybrid_rrf_before_cross_encoder"}, ks=(5, 10, 20)
    )
    reranked_report = summarize_fulltext_eval(
        queries, reranked_query_retrievals, corpus_parent_ids,
        {**report_metadata, "evaluation_variant": "hybrid_rrf_then_cross_encoder_top20"}, ks=(5, 10, 20)
    )
    baseline_hybrid = baseline_report["metrics"]["overall_macro_mean"]["hybrid"]["metrics_macro_mean"]
    reranked_hybrid = reranked_report["metrics"]["overall_macro_mean"]["hybrid"]["metrics_macro_mean"]
    delta_hybrid = {
        name: (reranked_hybrid[name] - baseline_hybrid[name]
               if reranked_hybrid.get(name) is not None and baseline_hybrid.get(name) is not None else None)
        for name in reranked_hybrid
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "evaluation": "fulltext_cross_encoder_posthoc",
        "source": report_metadata,
        "baseline": {
            "description": "BM25 + Dense fused with production RRF, before Cross Encoder",
            "coverage": baseline_report["coverage"],
            "metrics": baseline_report["metrics"],
        },
        "reranked": {
            "description": "Same RRF pool after Cross Encoder reorders its top 20 child chunks",
            "coverage": reranked_report["coverage"],
            "metrics": reranked_report["metrics"],
            "queries": reranked_report["queries"],
        },
        "hybrid_delta_vs_rrf_baseline": delta_hybrid,
        "rerank_seconds": {
            "per_query": rerank_seconds_by_query,
            "total": round(sum(rerank_seconds_by_query.values()), 6),
            "mean": round(sum(rerank_seconds_by_query.values()) / len(rerank_seconds_by_query), 6)
            if rerank_seconds_by_query else None,
        },
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        output["report_path"] = str(destination.resolve())
    return output


def _checkpoint_fingerprint_from_metadata(path: Path) -> str:
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
    try:
        header = json.loads(first)
    except json.JSONDecodeError as exc:
        raise ValueError("Checkpoint header is invalid JSON") from exc
    return str(header.get("fingerprint") or "")
