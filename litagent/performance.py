"""Offline summaries for literature-search run reports.

Reports may be individual full-text runs or aggregate end-to-end reports with
``systems`` and per-question ``rows``. No network access is used.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PERCENTILE_METHOD = "linear interpolation, type 7 (h=(n-1)*p)"


def percentile(values: Iterable[float], p: float) -> float | None:
    """Return a type-7 (linear interpolation) percentile, or None if empty."""
    ordered = sorted(float(v) for v in values if _number(v) is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _latency(values: Iterable[Any]) -> dict[str, Any]:
    nums = [n for v in values if (n := _number(v)) is not None]
    return {
        "sample_count": len(nums),
        "p50_seconds": percentile(nums, 0.50),
        "p95_seconds": percentile(nums, 0.95),
        "method": PERCENTILE_METHOD,
    }


def _outcome(report: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    status = report.get("status")
    answer = (row or report).get("answer")
    if not isinstance(answer, dict):
        answer = report.get("answer") if isinstance(report.get("answer"), dict) else {}
    if status in {"failed", "error", "exception"} or answer.get("failed") is True:
        return "failed"
    if status and str(status).lower().startswith("refus"):
        return "refused"
    if answer.get("refused") is True:
        return "refused"
    if status in {"verified", "success", "succeeded", "complete", "completed"}:
        return "successful"
    if status is None and answer.get("failed") is False and answer.get("refused") is False:
        return "successful"
    return "unlabeled"


def _metadata(report: dict[str, Any], system: str | None = None) -> dict[str, Any]:
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    # Keep prompt and corpus identity plus execution configuration. Missing values
    # remain explicit nulls so unknown provenance never silently joins known runs.
    keys = (
        "prompt_version", "judge_version", "parent_sha256", "child_sha256",
        "chunk_sha256", "dense_index_sha256", "embedding_model", "chat_model",
        "cross_encoder_model", "corpus_paper_count", "top_k", "retriever",
        "pipeline", "rerank", "rerank_k", "candidate_k", "context_strategy",
        "max_context_chars", "max_chunks_per_parent",
    )
    meta = {key: run.get(key) for key in keys}
    meta["child_sha256"] = run.get("child_sha256", run.get("chunk_sha256"))
    meta["chunk_sha256"] = meta["child_sha256"]
    meta["dataset_sha256"] = dataset.get("sha256")
    meta["runtime_policy"] = (report.get("runtime") or {}).get("policy")
    trace = report.get("retrieval_trace") or {}
    meta["effective_config"] = {key: trace.get(key) for key in
                                ("effective_pipeline", "effective_retriever", "effective_rerank")}
    meta["degraded"] = bool(report.get("degraded", False))
    normalized_system = {"classic_hybrid": "classic", "agentic_hybrid": "agentic",
                         "agentic_rerank": "agentic_rerank"}.get(system, system) if system else run.get("pipeline")
    if not system and run.get("pipeline") == "agentic" and run.get("rerank"):
        normalized_system = "agentic_rerank"
    meta["system"] = normalized_system
    if system:
        meta["pipeline"] = {"classic_hybrid": "classic", "agentic_hybrid": "agentic",
                             "agentic_rerank": "agentic"}.get(system, run.get("pipeline"))
        if system == "agentic_rerank":
            meta["rerank"] = True
    return meta


def _records(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten aggregate e2e rows; leave a normal per-run report as one record."""
    systems = report.get("systems")
    if isinstance(systems, dict):
        result = []
        for system, payload in systems.items():
            if not isinstance(payload, dict):
                continue
            for row in payload.get("rows", []):
                if isinstance(row, dict):
                    diagnostics = (row.get("answer") or {}).get("diagnostics") or {}
                    result.append({"report": {**report, **diagnostics}, "row": row, "system": str(system)})
        return result
    return [{"report": report, "row": None, "system": None}]


def _stage_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[str, list[float]] = defaultdict(list)
    statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        report = record["report"]
        row = record.get("row")
        timings = report.get("timings", [])
        # Row-shaped nested reports may carry their own timing list.
        if isinstance(row, dict) and isinstance(row.get("timings"), list):
            timings = row["timings"]
        if not isinstance(timings, list):
            continue
        for timing in timings:
            if not isinstance(timing, dict) or not isinstance(timing.get("stage"), str):
                continue
            stage = timing["stage"]
            seconds = _number(timing.get("seconds"))
            if seconds is not None:
                samples[stage].append(seconds)
            status = timing.get("status")
            statuses[stage][str(status) if status is not None else "unlabeled"] += 1
    return {stage: {**_latency(samples.get(stage, [])), "status_counts": dict(statuses[stage])}
            for stage in sorted(set(samples) | set(statuses))}


def _api_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        report = record["report"]
        row = record.get("row")
        calls = (row.get("api_calls") or report.get("api_calls")) if isinstance(row, dict) else report.get("api_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict):
                grouped[str(call.get("stage") or "unlabeled")].append(call)
    result = {}
    for stage, calls in sorted(grouped.items()):
        latencies = [c.get("latency_seconds", c.get("seconds")) for c in calls]
        tokens = [c.get("usage", {}).get("total_tokens") for c in calls if isinstance(c.get("usage"), dict)]
        costs = [_number(c.get("cost_usd")) for c in calls]
        cost_values = [c for c in costs if c is not None]
        result[stage] = {
            "call_count": len(calls), "latency": _latency(latencies),
            "prompt_tokens": sum(_number(c.get("usage", {}).get("prompt_tokens")) or 0 for c in calls if isinstance(c.get("usage"), dict)),
            "completion_tokens": sum(_number(c.get("usage", {}).get("completion_tokens")) or 0 for c in calls if isinstance(c.get("usage"), dict)),
            "total_tokens": sum(_number(t) or 0 for t in tokens),
            "cost_usd": sum(cost_values) if cost_values else None,
        }
        from .fulltext_answer import estimate_flash_cost_range_usd
        known_usage = [call for call in calls if isinstance(call.get("usage"), dict)
                       and any(key in call["usage"] for key in ("prompt_tokens", "completion_tokens"))]
        result[stage]["usage_observed_call_count"] = len(known_usage)
        result[stage]["estimated_cost"] = estimate_flash_cost_range_usd(known_usage) if known_usage else None
    return result


def _summarize(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    outcomes = {name: [] for name in ("successful", "refused", "failed", "unlabeled")}
    for record in records:
        row = record.get("row")
        report = record["report"]
        outcome = _outcome(report, row)
        elapsed = row.get("elapsed_seconds", report.get("elapsed_seconds")) if isinstance(row, dict) else report.get("elapsed_seconds")
        cost = row.get("cost", report.get("cost")) if isinstance(row, dict) else report.get("cost")
        outcomes[outcome].append({"elapsed": elapsed, "cost": cost})
    counts = {k: len(v) for k, v in outcomes.items()}
    latency = {}
    for outcome, values in outcomes.items():
        latency[outcome] = _latency(v["elapsed"] for v in values)
    latency["completed"] = _latency(v["elapsed"] for key in ("successful", "refused") for v in outcomes[key])
    total_cost = {}
    for outcome, values in outcomes.items():
        cost_samples = []
        for v in values:
            cost = v["cost"]
            if isinstance(cost, dict):
                cost = cost.get("estimated_usd_offpeak", cost.get("estimated_usd_peak"))
            n = _number(cost)
            if n is not None:
                cost_samples.append(n)
        total_cost[outcome] = {"sample_count": len(cost_samples), "sum_usd": sum(cost_samples) if cost_samples else None}
    quality = _retrieval_quality(records)
    return {
        "metadata": metadata, "run_count": len(records), "outcome_counts": counts,
        "latency": latency, "cost": total_cost,
        "stages": _stage_stats(records), "api_calls_by_stage": _api_stats(records),
        "retrieval_quality": quality,
    }


def _retrieval_quality(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        row = record.get("quality_row", record.get("row"))
        if not isinstance(row, dict) or not isinstance(row.get("retrieval"), dict):
            continue
        retrieval = row["retrieval"]
        category = str(row.get("category") or "unknown")
        # Dataset IDs with zh suffix are translated questions. Original-language
        # rows are retained as their own bucket instead of pooling the two.
        question_id = str(row.get("id") or "")
        language = "translated" if question_id.lower().endswith("_zh") else "original"
        bucket = f"{language}:{category}"
        if retrieval.get("status") not in {None, "scored"}:
            continue
        buckets[bucket].append(retrieval)
    if not buckets:
        return None
    result = {}
    for name, rows in sorted(buckets.items()):
        metrics = {}
        for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
            values = [n for row in rows if (n := _number(row.get(metric))) is not None]
            metrics[metric] = {"sample_count": len(values), "mean": sum(values) / len(values) if values else None}
        result[name] = metrics
    return result


def analyze_reports(inputs: Iterable[str | Path]) -> dict[str, Any]:
    """Analyze JSON report files/directories, preserving incompatible groups.

    Directories are searched recursively. Aggregate e2e files are expanded into
    rows; their separate detail files remain separate observations by design.
    """
    paths: set[Path] = set()
    for item in inputs:
        text = str(item)
        matches = [Path(p) for p in glob.glob(text, recursive=True)]
        if not matches:
            candidate = Path(text)
            matches = list(candidate.rglob("*.json")) if candidate.is_dir() else [candidate]
        for path in matches:
            if path.is_dir():
                paths.update(path.rglob("*.json"))
            elif path.suffix.lower() == ".json":
                paths.add(path)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bad_files = []
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(paths):
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(report, dict):
                raise ValueError("top-level JSON value must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            bad_files.append({"path": str(path), "error": str(exc)})
            continue
        loaded.append((path, report))

    aggregate_roots = {path.parent.resolve() for path, report in loaded if isinstance(report.get("systems"), dict)}
    def signature_for(path, system, question, run):
        resolved = path.resolve()
        roots = [root for root in aggregate_roots if resolved.is_relative_to(root)]
        scope = max(roots, key=lambda root: len(root.parts)) if roots else resolved.parent
        return (str(scope), str(system), question.strip(), str(run.get("parent_sha256") or ""),
                str(run.get("child_sha256", run.get("chunk_sha256")) or ""), run.get("top_k"))
    # Build signatures for detailed question reports. Aggregate rows that have a
    # matching detail file are omitted to prevent counting the same run twice.
    detail_signatures: set[tuple[str, str, str, str]] = set()
    aggregate_metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    aggregate_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    aggregate_datasets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _path, report in loaded:
        if isinstance(report.get("systems"), dict):
            top_run = report.get("run", {})
            for record in _records(report):
                row = record.get("row") or {}
                system_name = record.get("system", "")
                system = {"classic_hybrid": "classic", "agentic_hybrid": "agentic", "agentic_rerank": "agentic_rerank"}.get(system_name, system_name)
                signature = signature_for(_path, system, str(row.get("question") or ""), top_run)
                aggregate_metadata[signature] = top_run
                aggregate_rows[signature] = row
                if isinstance(report.get("dataset"), dict):
                    aggregate_datasets[signature] = report["dataset"]
            continue
        run = report.get("run", {})
        question = run.get("question") if isinstance(run, dict) else None
        if isinstance(question, str):
            system = "agentic_rerank" if run.get("pipeline") == "agentic" and run.get("rerank") else run.get("pipeline")
            detail_signatures.add(signature_for(_path, system, question, run))
    for path, original_report in loaded:
        report = original_report
        aggregate = isinstance(report.get("systems"), dict)
        signature = None
        if not aggregate:
            run = report.get("run", {})
            question = run.get("question") if isinstance(run, dict) else None
            if isinstance(question, str):
                pipeline = run.get("pipeline")
                system = "agentic_rerank" if pipeline == "agentic" and run.get("rerank") else pipeline
                signature = signature_for(path, system, question, run)
                inherited = aggregate_metadata.get(signature)
                if inherited:
                    # A stored e2e aggregate provides the prompt/data provenance
                    # omitted from its per-question detail reports.
                    report = dict(report)
                    report["run"] = {**inherited, **run}
                    if signature in aggregate_datasets:
                        report["dataset"] = aggregate_datasets[signature]
        for record in _records(report):
            record["source"] = str(path)
            if not aggregate and signature is not None:
                record["quality_row"] = aggregate_rows.get(signature)
            if aggregate:
                row = record.get("row") or {}
                meta_run = report.get("run", {})
                sysname = record.get("system", "")
                system = {"classic_hybrid": "classic", "agentic_hybrid": "agentic", "agentic_rerank": "agentic_rerank"}.get(sysname, sysname)
                signature = signature_for(path, system, str(row.get("question") or ""), meta_run)
                if signature in detail_signatures:
                    continue
            meta = _metadata(report, record.get("system"))
            key = json.dumps(meta, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            groups[key].append(record)
    summaries = []
    for key, records in sorted(groups.items()):
        summaries.append(_summarize(records, json.loads(key)))
    return {"schema_version": 1, "percentile_method": PERCENTILE_METHOD,
            "input_file_count": len(paths), "groups": summaries, "invalid_files": bad_files}


def _expand_inputs(values: list[str]) -> list[str]:
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON files, glob patterns, or directories")
    parser.add_argument("-o", "--output", help="write JSON to this path; stdout by default")
    args = parser.parse_args(argv)
    result = analyze_reports(_expand_inputs(args.inputs))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
