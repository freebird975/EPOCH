"""Exploratory title-weight sweep for full-text BM25 using a v4 checkpoint.

The default production multiplier remains 1.25. This evaluator reruns only
BM25 for weights 1.0 and 1.5; the 1.25 BM25 and Dense rankings are reused from
the completed official checkpoint. It reads existing artifacts in read-only
mode and never calls a hosted API or rebuilds an index.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from .fulltext_eval_metrics import summarize_fulltext_eval
from .fulltext_fusion_tuning import (
    CHECKPOINT_EVALUATOR_VERSION,
    _load_checkpoint,
    _load_parent_ids,
    _read_jsonl,
    _sha256,
)
from .hybrid import fuse_rrf, DEFAULT_RRF_K
from .retrieval import load_index, search
from .sqlite_bm25_numpy import prepare_numpy_index


TITLE_WEIGHTS = (1.0, 1.25, 1.5)
SAMPLE_LIMIT = 100


def _child_hits(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize saved/current hits to the fields used by parent metrics/RRF."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        child_id = str(row.get("paper_id") or "").strip()
        parent_id = str(row.get("parent_id") or "").strip()
        if not child_id or child_id in seen:
            continue
        seen.add(child_id)
        result.append({"paper_id": child_id, "parent_id": parent_id})
    return result


def _mode_record(hits: list[dict[str, str]], corpus_parent_ids: set[str], seconds: float) -> dict[str, Any]:
    parent_ids: list[str] = []
    candidate_ids: list[str] = []
    seen: set[str] = set()
    candidate_seen: set[str] = set()
    orphans = 0
    for hit in hits:
        parent_id = hit["parent_id"]
        if not parent_id or parent_id not in corpus_parent_ids:
            orphans += 1
            continue
        if parent_id not in candidate_seen:
            candidate_seen.add(parent_id)
            candidate_ids.append(parent_id)
        if parent_id not in seen:
            seen.add(parent_id)
            parent_ids.append(parent_id)
    return {
        "parent_ids": parent_ids,
        "candidate_parent_ids": candidate_ids,
        "child_hits": hits,
        "orphan_child_hits": orphans,
        "latency_seconds": {"mode_total": max(0.0, float(seconds))},
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_fulltext_title_weight_sweep(
    queries_path: str | Path,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    sample_size: int = SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Compare title multipliers on a deterministic sample of official qrels.

    Sampling selects the ``sample_size`` smallest SHA-256 digests of query IDs,
    then evaluates those queries in official qrels file order. This avoids
    relying on ordering or hand-picking favorable queries. The evaluation is
    exploratory: it reuses the same qrels used to select/tune parameters and
    does not constitute held-out validation.
    """
    queries_path, checkpoint_path = Path(queries_path), Path(checkpoint_path)
    if type(sample_size) is not int or not 1 <= sample_size <= SAMPLE_LIMIT:
        raise ValueError(f"sample_size must be an integer from 1 through {SAMPLE_LIMIT}")
    if not queries_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Official qrels or full-text evaluation checkpoint does not exist")
    queries = _read_jsonl(queries_path)
    saved, checkpoint_metadata = _load_checkpoint(checkpoint_path, queries, queries_path)
    if checkpoint_metadata.get("indexes", {}).get("bm25", {}).get("evidence_scope") != "full_text_chunk":
        raise ValueError("Checkpoint does not use the full-text BM25 child index")
    input_paths = checkpoint_metadata.get("input_paths", {})
    index_path_value = input_paths.get("bm25") if isinstance(input_paths, dict) else None
    chunk_path_value = input_paths.get("chunks") if isinstance(input_paths, dict) else None
    parent_path_value = input_paths.get("parents") if isinstance(input_paths, dict) else None
    if not all(isinstance(value, str) and value for value in (index_path_value, chunk_path_value, parent_path_value)):
        raise ValueError("Checkpoint is missing paths for full-text corpus/index artifacts")

    sampled_ids = set(sorted((str(row["id"]) for row in queries),
                             key=lambda query_id: (hashlib.sha256(query_id.encode("utf-8")).hexdigest(), query_id))[:sample_size])
    sample = [row for row in queries if str(row["id"]) in sampled_ids]
    parent_sha = checkpoint_metadata.get("hashes", {}).get("parents_sha256")
    parent_ids = _load_parent_ids(Path(parent_path_value), parent_sha)

    # Load the production index in read-only mode. load_index validates both
    # its signature and the full-text child corpus hash before any scoring.
    index = load_index(Path(index_path_value), corpus_path=Path(chunk_path_value), validation_cache={})
    if index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("BM25 index is not the full-text child index")
    expected_index_sha = checkpoint_metadata.get("hashes", {}).get("bm25_index_sha256")
    if _sha256(Path(index_path_value)) != expected_index_sha:
        raise ValueError("BM25 index SHA-256 does not match the checkpoint")
    numpy_cache = prepare_numpy_index(index)

    configs: dict[str, Any] = {}
    timings: dict[str, dict[str, Any]] = {str(weight): {} for weight in TITLE_WEIGHTS}
    try:
        for weight in TITLE_WEIGHTS:
            per_query: dict[str, dict[str, Any]] = {}
            for query in sample:
                query_id = str(query["id"])
                saved_row = saved[query_id]
                dense_hits = _child_hits(saved_row["dense"]["child_hits"])
                if weight == 1.25:
                    # Reuse exactly the saved production-default BM25 ranking.
                    bm25_hits = _child_hits(saved_row["bm25"]["child_hits"])
                    bm25_seconds = saved_row["bm25"].get("latency_seconds", {}).get("mode_total")
                    bm25_source = "completed_v4_checkpoint"
                else:
                    started = time.perf_counter()
                    bm25_hits = _child_hits(search(index, query["question"], 100, title_weight=weight))
                    bm25_seconds = time.perf_counter() - started
                    bm25_source = "fresh_read_only_bm25_search"
                rrf_started = time.perf_counter()
                hybrid_hits = _child_hits(fuse_rrf(bm25_hits, dense_hits, top_k=200, rrf_k=DEFAULT_RRF_K))
                rrf_seconds = time.perf_counter() - rrf_started
                dense_seconds = saved_row["dense"].get("latency_seconds", {}).get("mode_total")
                per_query[query_id] = {
                    "bm25_seconds": bm25_seconds,
                    "bm25_timing_source": bm25_source,
                    "dense_seconds_from_checkpoint": dense_seconds,
                    "rrf_seconds": rrf_seconds,
                }
                per_query[query_id + "\0retrieval"] = {
                    "bm25": bm25_hits,
                    "dense": dense_hits,
                    "hybrid": hybrid_hits,
                }

            metric_inputs: dict[str, dict[str, Any]] = {}
            for query in sample:
                query_id = str(query["id"])
                hitsets = per_query.pop(query_id + "\0retrieval")
                timing = per_query[query_id]
                metric_inputs[query_id] = {
                    mode: _mode_record(hitsets[mode], parent_ids,
                                       timing["bm25_seconds"] if mode == "bm25" else
                                       (timing["dense_seconds_from_checkpoint"] or 0.0) if mode == "dense" else
                                       (timing["bm25_seconds"] or 0.0) + (timing["dense_seconds_from_checkpoint"] or 0.0) + timing["rrf_seconds"])
                    for mode in ("bm25", "dense", "hybrid")
                }
            code_files = [Path(__file__), Path(__file__).with_name("retrieval.py"),
                          Path(__file__).with_name("sqlite_bm25_numpy.py"), Path(__file__).with_name("hybrid.py")]
            code_hashes = {path.name: _sha256(path) for path in code_files}
            metadata = {
                "evaluation": "fulltext_title_weight_exploratory_sweep",
                "title_weight": weight,
                "rrf_k": DEFAULT_RRF_K,
                "candidate_k_per_channel": 100,
                "sample_count": len(sample),
                "sample_selection": "lowest SHA-256(query_id) values, tie-broken by query_id; evaluated in official qrels order",
                "qrels_reused_for_tuning": True,
                "held_out_validation": False,
            }
            summary = summarize_fulltext_eval(sample, metric_inputs, parent_ids, metadata)
            configs[str(weight)] = {
                "metrics": summary["metrics"],
                "coverage": summary["coverage"],
                "query_results": summary["queries"],
                "per_query_timings": per_query,
            }
    finally:
        index["connection"].close()

    baseline = configs["1.25"]["metrics"]["overall_macro_mean"]
    for config in configs.values():
        config["delta_vs_default_1_25"] = {
            mode: {
                metric: (value - baseline[mode]["metrics_macro_mean"][metric]
                         if value is not None and baseline[mode]["metrics_macro_mean"].get(metric) is not None else None)
                for metric, value in config["metrics"]["overall_macro_mean"][mode]["metrics_macro_mean"].items()
            }
            for mode in ("bm25", "dense", "hybrid")
        }
    output = {
        "schema_version": 1,
        "status": "complete",
        "limitations": [
            "The deterministic 100-query subset is exploratory and does not establish full-set performance.",
            "The same official qrels are reused for tuning; results are not held-out validation.",
            "Dense rankings are reused from the v4 checkpoint; only BM25 is re-scored for non-default weights.",
            "Default 1.25 BM25 timings come from the checkpoint; alternate-weight timings are fresh searches and may differ with OS cache state.",
        ],
        "source": {
            "queries_path": str(queries_path.resolve()), "queries_sha256": _sha256(queries_path),
            "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_fingerprint": checkpoint_metadata.get("fingerprint"),
            "parent_corpus_sha256": parent_sha,
            "bm25_index_path": str(Path(index_path_value).resolve()),
            "bm25_index_sha256": _sha256(Path(index_path_value)),
            "child_corpus_sha256": checkpoint_metadata.get("hashes", {}).get("chunks_sha256"),
            "code_sha256": code_hashes,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy_cache": numpy_cache},
        "selection": {"method": "sort by (sha256(query_id), query_id), take first sample_size, emit in official file order",
                      "sample_size": len(sample), "query_ids": [str(row["id"]) for row in sample]},
        "baseline": {"title_weight": 1.25, "bm25_source": "saved completed v4 checkpoint ranking"},
        "configurations": configs,
    }
    if output_path is not None:
        destination = Path(output_path)
        _atomic_write(destination, output)
        output["output_path"] = str(destination.resolve())
    return output
