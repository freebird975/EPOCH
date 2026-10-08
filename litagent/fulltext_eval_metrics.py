"""Pure metric aggregation for full-text LitSearch retrieval evaluations.

This module intentionally performs no I/O or retrieval.  In particular, full
text gold papers that are absent from the evaluated corpus are reported as
unavailable and are excluded from metric denominators; they are never treated
as nonrelevant papers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


_MODES = ("bm25", "dense", "hybrid")


def _id(value: Any, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be a non-empty string or number")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must be a non-empty string or number")
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _unique_ids(values: Any, field: str, *, deduplicate: bool = False) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of IDs")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _id(value, field)
        if item in seen:
            if deduplicate:
                continue
            raise ValueError(f"{field} contains duplicate ID {item!r}")
        seen.add(item)
        result.append(item)
    return result


def _normalise_latency(record: Mapping[str, Any], context: str) -> dict[str, float] | None:
    raw = record.get("latency_seconds")
    if raw is None:
        raw = record.get("timing")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {str(key): _finite_number(value, f"{context}.timing.{key}") for key, value in sorted(raw.items(), key=lambda kv: str(kv[0]))}
    return {"total": _finite_number(raw, f"{context}.latency_seconds")}


def _dcg(relevant_ranks: list[int], cutoff: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= cutoff)


def _metric_values(ranked: list[str], gold: set[str], ks: tuple[int, ...]) -> dict[str, float]:
    ranks = [position for position, paper_id in enumerate(ranked, 1) if paper_id in gold]
    values = {f"recall_at_{k}": sum(rank <= k for rank in ranks) / len(gold) for k in ks}
    values["mrr_at_10"] = 1.0 / ranks[0] if ranks and ranks[0] <= 10 else 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    values["ndcg_at_10"] = _dcg(ranks, 10) / ideal if ideal else 0.0
    return values


def _mean_metrics(records: list[dict[str, Any]], modes: tuple[str, ...], ks: tuple[int, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in modes:
        eligible = [row["modes"][mode] for row in records if row["modes"][mode]["metrics"] is not None]
        metric_names = [*(f"recall_at_{k}" for k in ks), "mrr_at_10", "ndcg_at_10"]
        means: dict[str, float | None] = {
            name: (sum(entry["metrics"][name] for entry in eligible) / len(eligible) if eligible else None)
            for name in metric_names
        }
        coverage = [entry["child_pool_gold_coverage"] for entry in eligible if entry["child_pool_gold_coverage"] is not None]
        timing_names = sorted({name for entry in eligible if entry["latency_seconds"] for name in entry["latency_seconds"]})
        mean_timing = {
            name: sum(entry["latency_seconds"][name] for entry in eligible if entry["latency_seconds"] and name in entry["latency_seconds"])
            / sum(bool(entry["latency_seconds"] and name in entry["latency_seconds"]) for entry in eligible)
            for name in timing_names
        }
        output[mode] = {
            "metric_query_count": len(eligible),
            "metrics_macro_mean": means,
            "child_pool_gold_coverage_macro_mean": sum(coverage) / len(coverage) if coverage else None,
            "latency_seconds_macro_mean": mean_timing,
            "orphan_child_hits_total": sum(entry["orphan_child_hits"] for entry in (row["modes"][mode] for row in records)),
        }
    return output


def summarize_fulltext_eval(
    queries: Sequence[Mapping[str, Any]],
    retrievals: Mapping[Any, Mapping[str, Mapping[str, Any]]],
    corpus_parent_ids: set[Any] | frozenset[Any],
    source_metadata: Mapping[str, Any],
    ks: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    """Validate and summarize full-text retrieval rankings.

    Each query must contain ``id``, ``question`` and a nonempty
    ``gold_paper_ids`` sequence. Retrievals are keyed by query ID and contain
    ``bm25``, ``dense`` and ``hybrid`` records. Parent rankings are made unique
    by first occurrence. Child-pool coverage uses ``parent_id`` on child hits
    (or ``candidate_parent_ids`` when supplied), since a chunk's own
    ``paper_id`` is not necessarily a parent paper ID.

    Scores use only relevant parent IDs available in ``corpus_parent_ids``.
    Queries with no available relevant parent remain in all coverage counts
    and per-query audit rows, but have null ranking metrics and do not enter
    macro means. ``source_metadata`` is copied into the JSON-serializable
    result so callers can include hashes, model/index versions, and parameters.
    """
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence) or not queries:
        raise ValueError("queries must be a non-empty sequence")
    if not isinstance(retrievals, Mapping):
        raise ValueError("retrievals must be a mapping keyed by query ID")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("source_metadata must be a mapping")
    try:
        serial_metadata = json.loads(json.dumps(source_metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("source_metadata must contain only JSON-serializable values") from exc

    normal_ks: list[int] = []
    for value in ks:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("ks must contain positive integers")
        if value in normal_ks:
            raise ValueError(f"ks contains duplicate cutoff {value}")
        normal_ks.append(value)
    if not normal_ks:
        raise ValueError("ks must not be empty")
    cutoffs = tuple(sorted(normal_ks))

    corpus_ids = {_id(value, "corpus_parent_ids") for value in corpus_parent_ids}
    query_rows: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for index, query in enumerate(queries):
        context = f"queries[{index}]"
        if not isinstance(query, Mapping):
            raise ValueError(f"{context} must be a mapping")
        query_id = _id(query.get("id"), f"{context}.id")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query ID {query_id!r}")
        seen_query_ids.add(query_id)
        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{context} ({query_id}) has an empty or invalid question")
        gold_ids = _unique_ids(query.get("gold_paper_ids"), f"{context}.gold_paper_ids")
        if not gold_ids:
            raise ValueError(f"{context} ({query_id}) has an empty qrel row")
        available = [paper_id for paper_id in gold_ids if paper_id in corpus_ids]
        missing = [paper_id for paper_id in gold_ids if paper_id not in corpus_ids]
        query_rows.append({
            "id": query_id,
            "_key": query_id,
            "question": question,
            "query_set": str(query.get("query_set") or "unspecified"),
            "label_status": str(query.get("label_status") or "unspecified"),
            "quality": str(query.get("quality") or "unspecified"),
            "gold_paper_ids": gold_ids,
            "available_gold_paper_ids": available,
            "missing_fulltext_gold_paper_ids": missing,
        })

    retrieval_keys: dict[str, Any] = {}
    for raw_key in retrievals:
        key = _id(raw_key, "retrieval query ID")
        if key in retrieval_keys:
            raise ValueError(f"retrievals has colliding query IDs after normalization: {key!r}")
        retrieval_keys[key] = raw_key
    unexpected_queries = sorted(set(retrieval_keys) - seen_query_ids)
    if unexpected_queries:
        raise ValueError(f"retrievals contains unknown query IDs: {unexpected_queries}")

    rows: list[dict[str, Any]] = []
    for query in query_rows:
        query_id = query["_key"]
        raw_key = retrieval_keys.get(query_id)
        if raw_key is None:
            raise ValueError(f"retrievals is missing query ID {query_id!r}")
        mode_records = retrievals[raw_key]
        if not isinstance(mode_records, Mapping):
            raise ValueError(f"retrievals[{query_id!r}] must be a mapping of modes")
        unknown_modes = sorted(set(mode_records) - set(_MODES), key=str)
        if unknown_modes:
            raise ValueError(f"retrievals[{query_id!r}] has unknown modes: {unknown_modes}")
        missing_modes = sorted(set(_MODES) - set(mode_records))
        if missing_modes:
            raise ValueError(f"retrievals[{query_id!r}] is missing modes: {missing_modes}")

        available_gold = set(query["available_gold_paper_ids"])
        modes: dict[str, Any] = {}
        for mode in _MODES:
            record = mode_records[mode]
            context = f"retrievals[{query_id!r}][{mode}]"
            if not isinstance(record, Mapping):
                raise ValueError(f"{context} must be a mapping")
            ranked_ids = _unique_ids(record.get("parent_ids"), f"{context}.parent_ids", deduplicate=True)
            candidate_ids: list[str] | None = None
            if "candidate_parent_ids" in record and record["candidate_parent_ids"] is not None:
                candidate_ids = _unique_ids(record["candidate_parent_ids"], f"{context}.candidate_parent_ids", deduplicate=True)
            child_hits_raw = record.get("child_hits")
            if child_hits_raw is not None and (isinstance(child_hits_raw, (str, bytes)) or not isinstance(child_hits_raw, Sequence)):
                raise ValueError(f"{context}.child_hits must be a sequence")
            child_hits = list(child_hits_raw or [])
            inferred_orphans = 0
            child_parent_ids: list[str] = []
            for hit_index, hit in enumerate(child_hits):
                if not isinstance(hit, Mapping):
                    raise ValueError(f"{context}.child_hits[{hit_index}] must be a mapping")
                parent_value = hit.get("parent_id")
                if parent_value is None or not str(parent_value).strip():
                    inferred_orphans += 1
                    continue
                parent_id = _id(parent_value, f"{context}.child_hits[{hit_index}].parent_id")
                child_parent_ids.append(parent_id)
                if parent_id not in corpus_ids:
                    inferred_orphans += 1
            if candidate_ids is None and child_hits_raw is not None:
                candidate_ids = list(dict.fromkeys(child_parent_ids))
            orphan_value = record.get("orphan_child_hits")
            if orphan_value is not None:
                if isinstance(orphan_value, bool) or not isinstance(orphan_value, int) or orphan_value < 0:
                    raise ValueError(f"{context}.orphan_child_hits must be a non-negative integer")
            # When child rows are present, derive orphan count from their
            # parent IDs. An optional precomputed value is only a fallback
            # when the caller does not provide those rows.
            orphan_count = inferred_orphans if child_hits_raw is not None else (orphan_value or 0)
            if available_gold and candidate_ids is not None:
                candidate_coverage: float | None = len(available_gold.intersection(candidate_ids)) / len(available_gold)
            else:
                candidate_coverage = None
            metrics = _metric_values(ranked_ids, available_gold, cutoffs) if available_gold else None
            modes[mode] = {
                "ranked_parent_ids": ranked_ids,
                "metrics": metrics,
                "child_candidate_parent_ids": candidate_ids,
                "child_pool_gold_coverage": candidate_coverage,
                "orphan_child_hits": orphan_count,
                "latency_seconds": _normalise_latency(record, context),
            }
        query_without_internal = {key: value for key, value in query.items() if key != "_key"}
        rows.append({
            **query_without_internal,
            "eligible_for_ranking_metrics": bool(available_gold),
            "partial_fulltext_gold": bool(query["missing_fulltext_gold_paper_ids"] and available_gold),
            "modes": modes,
        })

    by_set: dict[str, list[dict[str, Any]]] = {}
    by_status: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(row["query_set"], []).append(row)
        by_status.setdefault(row["label_status"], []).append(row)

    def coverage_slice(items: list[dict[str, Any]]) -> dict[str, Any]:
        original = sum(len(item["gold_paper_ids"]) for item in items)
        available = sum(len(item["available_gold_paper_ids"]) for item in items)
        return {
            "query_count": len(items),
            "eligible_query_count": sum(bool(item["available_gold_paper_ids"]) for item in items),
            "partial_fulltext_gold_query_count": sum(bool(item["missing_fulltext_gold_paper_ids"] and item["available_gold_paper_ids"]) for item in items),
            "no_fulltext_gold_query_count": sum(not item["available_gold_paper_ids"] for item in items),
            "gold_id_count": original,
            "available_gold_id_count": available,
            "missing_fulltext_gold_id_count": original - available,
            "gold_id_coverage": available / original if original else None,
        }

    return {
        "schema_version": 1,
        "source_metadata": serial_metadata,
        "modes": list(_MODES),
        "cutoffs": list(cutoffs),
        "coverage": {
            **coverage_slice(rows),
            "query_set_slices": {key: coverage_slice(by_set[key]) for key in sorted(by_set)},
            "label_status_slices": {key: coverage_slice(by_status[key]) for key in sorted(by_status)},
            "quality_slices": {
                key: coverage_slice([row for row in rows if str(row.get("quality") or "unspecified") == key])
                for key in sorted({str(row.get("quality") or "unspecified") for row in rows})
            },
        },
        "metrics": {
            "overall_macro_mean": _mean_metrics(rows, _MODES, cutoffs),
            "by_query_set": {key: _mean_metrics(by_set[key], _MODES, cutoffs) for key in sorted(by_set)},
            "by_label_status": {key: _mean_metrics(by_status[key], _MODES, cutoffs) for key in sorted(by_status)},
        },
        "queries": rows,
    }
