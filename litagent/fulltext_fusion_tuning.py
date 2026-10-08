"""Offline, deterministic post-hoc tuning of full-text BM25/Dense RRF.

The sweep consumes the official LitSearch qrels and a completed full-text
evaluation JSONL checkpoint. It does not load an index, encode queries, or call
any retrieval or hosted model. Child ``score`` fields are deliberately ignored:
RRF is recomputed from the stored, ordered child-hit lists using ranks only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fulltext_eval_metrics import summarize_fulltext_eval


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_EVALUATOR_VERSION = "fulltext-offline-v4-sqlite-numpy-mmap"
DEFAULT_CANDIDATE_KS = (20, 50, 100)
DEFAULT_RRF_KS = (20, 40, 60, 100)
DEFAULT_BASELINE = {"candidate_k": 100, "rrf_k": 60}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(item)
    if not rows:
        raise ValueError(f"JSONL contains no records: {path}")
    return rows


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path, queries: Sequence[Mapping[str, Any]], queries_path: Path) -> tuple[dict[str, dict], dict]:
    """Load and validate the runner's append-only header/query checkpoint."""
    records = _read_jsonl(path)
    header = records[0]
    if header.get("record_type") != "header":
        raise ValueError(f"Checkpoint has no header record: {path}")
    if header.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version: {header.get('schema_version')!r}")
    if header.get("evaluator_version") != CHECKPOINT_EVALUATOR_VERSION:
        raise ValueError(f"Unsupported checkpoint evaluator_version: {header.get('evaluator_version')!r}")
    fingerprint = header.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Checkpoint fingerprint is missing or is not a lowercase SHA-256")
    metadata = header.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint header is missing source_metadata")
    hashes = metadata.get("hashes")
    parameters = metadata.get("parameters")
    model = metadata.get("model")
    if not all(isinstance(item, dict) for item in (hashes, parameters, model)):
        raise ValueError("Checkpoint metadata must include hashes, parameters, and model objects")
    expected_query_hash = hashes.get("queries_sha256")
    if not isinstance(expected_query_hash, str) or _sha256(queries_path) != expected_query_hash:
        raise ValueError("Official qrels file SHA-256 does not match the checkpoint fingerprint inputs")
    if parameters.get("candidate_k_per_channel") != 100:
        raise ValueError(
            "Fusion tuning requires top-100 BM25 and Dense child hits per query; "
            f"checkpoint candidate_k_per_channel={parameters.get('candidate_k_per_channel')!r}"
        )
    if parameters.get("reranker") is not None or parameters.get("query_translation") is not False:
        raise ValueError("Checkpoint must contain unreranked, untranslated BM25/Dense rankings")
    query_source = metadata.get("query_source")
    if not isinstance(query_source, dict) or query_source.get("label_status") != "LitSearch_official":
        raise ValueError("Checkpoint query source is not marked LitSearch_official")

    expected_fingerprint = _canonical_sha256({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evaluator_version": CHECKPOINT_EVALUATOR_VERSION,
        "hashes": hashes,
        "dense_model": model,
        "parameters": parameters,
    })
    if expected_fingerprint != fingerprint:
        raise ValueError("Checkpoint fingerprint does not match its recorded hashes and evaluation parameters")

    query_ids: list[str] = []
    for row in queries:
        query_id = str(row.get("id") or "").strip()
        if not query_id:
            raise ValueError("Official qrels contain an empty query ID")
        if row.get("label_status") != "LitSearch_official":
            raise ValueError(f"Query {query_id!r} is not marked LitSearch_official")
        query_ids.append(query_id)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Official qrels contain duplicate query IDs")

    completed: dict[str, dict] = {}
    known_query_ids = set(query_ids)
    for line_number, record in enumerate(records[1:], 2):
        if record.get("record_type") != "query":
            raise ValueError(f"Unexpected checkpoint record type at {path}:{line_number}")
        query_id = str(record.get("query_id") or "").strip()
        if query_id not in known_query_ids:
            raise ValueError(f"Checkpoint contains an unknown query ID: {query_id!r}")
        if query_id in completed:
            raise ValueError(f"Checkpoint contains duplicate query ID: {query_id!r}")
        retrievals = record.get("retrievals")
        if not isinstance(retrievals, dict) or set(retrievals) != {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Checkpoint query {query_id!r} does not contain all retrieval modes")
        for channel in ("bm25", "dense"):
            channel_record = retrievals[channel]
            hits = channel_record.get("child_hits") if isinstance(channel_record, dict) else None
            if not isinstance(hits, list):
                raise ValueError(f"Checkpoint query {query_id!r} lacks {channel} child_hits")
            if len(hits) > 100:
                raise ValueError(f"Checkpoint query {query_id!r} has more than 100 {channel} child hits")
            for hit in hits:
                if not isinstance(hit, dict) or not str(hit.get("paper_id") or "").strip():
                    raise ValueError(f"Checkpoint query {query_id!r} has malformed {channel} child hits")
                if hit.get("parent_id") is not None and not isinstance(hit.get("parent_id"), (str, int)):
                    raise ValueError(f"Checkpoint query {query_id!r} has an invalid {channel} child parent_id")
        completed[query_id] = retrievals

    missing_ids = [query_id for query_id in query_ids if query_id not in completed]
    if missing_ids:
        raise ValueError(f"Checkpoint is incomplete: {len(missing_ids)} of {len(query_ids)} queries missing; first={missing_ids[:5]}")
    if len(records) != len(queries) + 1:
        raise ValueError("Checkpoint contains an unexpected number of records")
    return completed, metadata


def _load_parent_ids(path: Path, expected_sha256: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Parent full-text corpus from checkpoint is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Parent full-text corpus SHA-256 does not match checkpoint metadata")
    parent_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid parent corpus JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not str(row.get("paper_id") or "").strip():
                raise ValueError(f"Parent corpus row lacks paper_id at {path}:{line_number}")
            paper_id = str(row["paper_id"]).strip()
            if paper_id in parent_ids:
                raise ValueError(f"Duplicate parent paper_id in corpus: {paper_id!r}")
            parent_ids.add(paper_id)
    return parent_ids


def _channel_hits(hits: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, str]]:
    """Keep the first occurrence of each child ID while retaining rank order."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in hits[:limit]:
        child_id = str(hit["paper_id"]).strip()
        if child_id in seen:
            continue
        seen.add(child_id)
        result.append({"paper_id": child_id, "parent_id": str(hit["parent_id"]).strip()})
    return result


def _rrf_hits(
    bm25_hits: Sequence[Mapping[str, str]],
    dense_hits: Sequence[Mapping[str, str]],
    rrf_k: int,
) -> list[dict[str, str]]:
    """Fuse child rankings using only 1-based reciprocal-rank contributions."""
    scores: dict[str, float] = {}
    parent_by_child: dict[str, str] = {}
    for channel_hits in (bm25_hits, dense_hits):
        for rank, hit in enumerate(channel_hits, 1):
            child_id, parent_id = hit["paper_id"], hit["parent_id"]
            previous_parent = parent_by_child.setdefault(child_id, parent_id)
            if previous_parent != parent_id:
                raise ValueError(f"Chunk {child_id!r} maps to conflicting parent IDs across retrieval channels")
            scores[child_id] = scores.get(child_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda child_id: (-scores[child_id], child_id))
    return [{"paper_id": child_id, "parent_id": parent_by_child[child_id]} for child_id in ordered]


def _parent_ranking(child_hits: Sequence[Mapping[str, str]], corpus_parent_ids: set[str]) -> tuple[list[str], list[str], int]:
    ranked_parents: list[str] = []
    candidate_parents: list[str] = []
    seen_ranked: set[str] = set()
    seen_candidates: set[str] = set()
    orphan_count = 0
    for hit in child_hits:
        parent_id = hit["parent_id"]
        if not parent_id or parent_id not in corpus_parent_ids:
            orphan_count += 1
            continue
        if parent_id not in seen_candidates:
            candidate_parents.append(parent_id)
            seen_candidates.add(parent_id)
        if parent_id not in seen_ranked:
            ranked_parents.append(parent_id)
            seen_ranked.add(parent_id)
    return ranked_parents, candidate_parents, orphan_count


def _metric_delta(swept: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("bm25", "dense", "hybrid"):
        swept_mode = swept.get(mode, {})
        baseline_mode = baseline.get(mode, {})
        swept_metrics = swept_mode.get("metrics_macro_mean", {})
        baseline_metrics = baseline_mode.get("metrics_macro_mean", {})
        result[mode] = {
            key: (value - baseline_metrics[key] if value is not None and baseline_metrics.get(key) is not None else None)
            for key, value in swept_metrics.items()
        }
    return result


def _deltas_by_slice(
    swept: Mapping[str, Any], baseline: Mapping[str, Any], *, slice_key: str = "by_query_set"
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    swept_slices = swept.get(slice_key, {})
    baseline_slices = baseline.get("by_query_set", {})
    for query_set in sorted(set(swept_slices) | set(baseline_slices)):
        if query_set not in swept_slices or query_set not in baseline_slices:
            result[query_set] = None
            continue
        result[query_set] = _metric_delta(swept_slices[query_set], baseline_slices[query_set])
    return result


def run_fulltext_fusion_sweep(
    queries_path: str | Path,
    checkpoint_path: str | Path,
    *,
    parent_corpus_path: str | Path | None = None,
    candidate_ks: Sequence[int] = DEFAULT_CANDIDATE_KS,
    rrf_ks: Sequence[int] = DEFAULT_RRF_KS,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read official qrels and a complete checkpoint, then sweep RRF offline.

    Assumptions: the checkpoint was produced with 100 BM25 and 100 Dense child
    hits for every official query; each ``child_hits`` list preserves descending
    retrieval rank; each child has stable ``paper_id`` and ``parent_id``; ranks
    are 1-based after duplicate chunk IDs are removed. Candidate cutoffs only
    truncate those saved lists. RRF uses ``1 / (rrf_k + rank)`` for each channel
    rank, with ties ordered by descending fused score then ascending child ID,
    matching the production ``fuse_rrf`` ordering. Original
    retrieval scores and the checkpoint's saved Hybrid list are not used.

    The parent corpus defaults to the path recorded in the checkpoint header.
    It is read only to identify the full-text parent universe and validate the
    checkpoint's parent-corpus hash, so unavailable gold labels are not mistaken
    for corpus negatives. If ``output_path`` is supplied, JSON is written via an
    atomic temporary file replacement.
    """
    queries_path = Path(queries_path)
    checkpoint_path = Path(checkpoint_path)
    if not queries_path.is_file():
        raise FileNotFoundError(f"Official qrels file does not exist: {queries_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Full-text evaluation checkpoint does not exist: {checkpoint_path}")
    candidate_values = tuple(candidate_ks)
    rrf_values = tuple(rrf_ks)
    if not candidate_values or any(type(value) is not int or value < 1 or value > 100 for value in candidate_values):
        raise ValueError("candidate_ks must contain unique integers from 1 through 100")
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("candidate_ks contains duplicates")
    if not rrf_values or any(type(value) is not int or value < 1 for value in rrf_values):
        raise ValueError("rrf_ks must contain unique positive integers")
    if len(set(rrf_values)) != len(rrf_values):
        raise ValueError("rrf_ks contains duplicates")
    if 20 not in candidate_values or 50 not in candidate_values or 100 not in candidate_values:
        raise ValueError("The planned sweep must include candidate_k 20, 50, and 100")
    if not {20, 40, 60, 100}.issubset(set(rrf_values)):
        raise ValueError("The planned sweep must include RRF constants 20, 40, 60, and 100")
    candidate_values = tuple(sorted(candidate_values))
    rrf_values = tuple(sorted(rrf_values))

    queries = _read_jsonl(queries_path)
    retrievals, header_metadata = _load_checkpoint(checkpoint_path, queries, queries_path)
    paths = header_metadata.get("input_paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Checkpoint metadata is missing input_paths")
    parent_value = parent_corpus_path or paths.get("parents")
    if not parent_value:
        raise ValueError("Supply parent_corpus_path or use a checkpoint with input_paths.parents")
    parent_path = Path(parent_value)
    expected_parent_hash = header_metadata["hashes"].get("parents_sha256")
    if not isinstance(expected_parent_hash, str) or len(expected_parent_hash) != 64:
        raise ValueError("Checkpoint metadata has no valid parents_sha256")
    corpus_parent_ids = _load_parent_ids(parent_path, expected_parent_hash)
    expected_parent_count = header_metadata.get("corpus", {}).get("parent_count")
    if expected_parent_count is not None and expected_parent_count != len(corpus_parent_ids):
        raise ValueError("Parent corpus count does not match checkpoint metadata")

    configs: dict[str, dict[str, Any]] = {}
    for candidate_k in candidate_values:
        for rrf_k in rrf_values:
            query_retrievals: dict[str, dict[str, Any]] = {}
            for query in queries:
                query_id = str(query["id"])
                saved = retrievals[query_id]
                bm25_hits = _channel_hits(saved["bm25"]["child_hits"], candidate_k)
                dense_hits = _channel_hits(saved["dense"]["child_hits"], candidate_k)
                hybrid_hits = _rrf_hits(bm25_hits, dense_hits, rrf_k)
                modes: dict[str, dict[str, Any]] = {}
                for mode, hits in (("bm25", bm25_hits), ("dense", dense_hits), ("hybrid", hybrid_hits)):
                    parent_ids, candidate_parent_ids, orphan_hits = _parent_ranking(hits, corpus_parent_ids)
                    modes[mode] = {
                        "parent_ids": parent_ids,
                        "candidate_parent_ids": candidate_parent_ids,
                        "child_hits": list(hits),
                        "orphan_child_hits": orphan_hits,
                        # These are post-hoc rankings, so original retrieval
                        # latency is not comparable across sweep configurations.
                        "latency_seconds": None,
                    }
                query_retrievals[query_id] = modes

            config_metadata = {
                "kind": "offline_posthoc_rrf_sweep",
                "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_path),
                "checkpoint_evaluator_version": CHECKPOINT_EVALUATOR_VERSION,
                "query_source": header_metadata.get("query_source"),
                "source_hashes": header_metadata.get("hashes"),
                "candidate_k_per_channel": candidate_k,
                "rrf_k": rrf_k,
                "rrf_formula": "sum(1 / (rrf_k + 1-based rank))",
                "score_policy": "stored retrieval scores ignored; rank order only",
                "tie_policy": "RRF score desc, child paper_id asc; matches production fuse_rrf",
                "ranked_parent_deduplication": "first occurrence in ranked child list",
                "latency_policy": "not measured by post-hoc sweep",
            }
            report = summarize_fulltext_eval(
                queries, query_retrievals, corpus_parent_ids, config_metadata, ks=(5, 10, 20)
            )
            key = f"candidate_k={candidate_k};rrf_k={rrf_k}"
            configs[key] = {
                "candidate_k_per_channel": candidate_k,
                "rrf_k": rrf_k,
                "coverage": report["coverage"],
                "metrics": report["metrics"],
            }

    baseline_key = f"candidate_k={DEFAULT_BASELINE['candidate_k']};rrf_k={DEFAULT_BASELINE['rrf_k']}"
    baseline = configs.get(baseline_key)
    if baseline is None:
        raise ValueError("Sweep did not produce the required baseline candidate_k=100, rrf_k=60")
    baseline_overall = baseline["metrics"]["overall_macro_mean"]
    baseline_slices = baseline["metrics"]["by_query_set"]
    for value in configs.values():
        value["baseline"] = {"candidate_k_per_channel": 100, "rrf_k": 60}
        value["delta_vs_baseline"] = _metric_delta(value["metrics"]["overall_macro_mean"], baseline_overall)
        value["query_set_delta_vs_baseline"] = _deltas_by_slice(value["metrics"], {"by_query_set": baseline_slices})
        value["label_status_delta_vs_baseline"] = _deltas_by_slice(
            value["metrics"], {"by_query_set": baseline["metrics"]["by_label_status"]},
            slice_key="by_label_status",
        )

    output = {
        "schema_version": 1,
        "status": "complete",
        "evaluation": "fulltext_litsearch_posthoc_rrf_tuning",
        "assumptions": {
            "checkpoint_top_k": 100,
            "query_sequence": "official qrels file order",
            "child_ranks": "list order, 1-based; duplicate paper_id rows keep first rank",
            "rrf": "rank-only sum of 1 / (rrf_k + rank)",
            "candidate_k": "truncate each saved BM25 and Dense child ranking independently",
            "hybrid_candidate_pool": "RRF-ranked union, at most 2 * candidate_k distinct child IDs",
            "parent_ranking": "first occurrence while traversing ranked child hits; deduplicate by parent_id",
            "unjudged_documents": "remain unjudged; only official positive gold_paper_ids are relevant",
            "latency": "not recomputed or compared by this post-hoc sweep",
            "baseline": "candidate_k=100, rrf_k=60",
        },
        "source": {
            "queries_path": str(queries_path.resolve()),
            "queries_sha256": _sha256(queries_path),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_path),
            "parent_corpus_path": str(parent_path.resolve()),
            "parent_corpus_sha256": expected_parent_hash,
            "parent_count": len(corpus_parent_ids),
            "query_count": len(queries),
        },
        "sweep": {"candidate_ks": list(candidate_values), "rrf_ks": list(rrf_values)},
        "baseline_config": baseline_key,
        "configurations": configs,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        output["output_path"] = str(destination.resolve())
    return output


def _checkpoint_fingerprint(path: Path) -> str:
    """Read only the header to obtain the already-validated fingerprint."""
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
    try:
        header = json.loads(first)
    except json.JSONDecodeError as exc:
        raise ValueError("Checkpoint first line is not valid JSON") from exc
    return str(header.get("fingerprint") or "")
