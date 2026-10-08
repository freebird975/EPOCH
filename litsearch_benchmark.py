"""Offline retrieval-only benchmark on the existing 100-paper full-text corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from litagent.runtime import RunRuntime, RuntimePolicy, use_runtime
from litagent.provenance import collect_provenance
from litsearch_fulltext import (BGE_MODEL, CHILD_DATA, CHILD_INDEX, DENSE_CHILD_INDEX,
                                PARENT_DATA, retrieve_parents)


ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "eval" / "fulltext_e2e_questions.jsonl"
PARENT_PATH = ROOT / PARENT_DATA
CHILD_PATH = ROOT / CHILD_DATA
INDEX_PATH = ROOT / CHILD_INDEX
DENSE_PATH = ROOT / DENSE_CHILD_INDEX
OUTPUT = ROOT / "eval" / "p2_retrieval_benchmark.json"
PERCENTILE_METHOD = "linear interpolation, type 7 (h=(n-1)*p)"

PROFILES = [
    {"name": "bm25_k100", "retriever": "bm25", "top_k": 5, "candidate_k": 100, "rerank": False},
    {"name": "dense_k100", "retriever": "dense", "top_k": 5, "candidate_k": 100, "rerank": False},
    {"name": "hybrid_k20", "retriever": "hybrid", "top_k": 5, "candidate_k": 20, "rerank": False},
    {"name": "hybrid_k50", "retriever": "hybrid", "top_k": 5, "candidate_k": 50, "rerank": False},
    {"name": "hybrid_k100", "retriever": "hybrid", "top_k": 5, "candidate_k": 100, "rerank": False},
    {"name": "hybrid_k100_ce20", "retriever": "hybrid", "top_k": 5, "candidate_k": 100, "rerank": True, "rerank_k": 20},
    {"name": "hybrid_k100_ce50", "retriever": "hybrid", "top_k": 5, "candidate_k": 100, "rerank": True, "rerank_k": 50},
]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _percentile(values: list[float], p: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, Any]:
    return {"sample_count": len(values), "p50_seconds": _percentile(values, .50),
            "p95_seconds": _percentile(values, .95), "method": PERCENTILE_METHOD}


def _load_questions(parent_ids: set[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = []
    for row in rows:
        source = row.get("source") or {}
        gold = row.get("gold_paper_ids") or []
        if (source.get("type") == "official_litsearch_query"
                and row.get("qrels_status") == "official_gold_complete_in_pilot"
                and not str(row.get("id", "")).endswith("_zh")
                and gold and set(map(str, gold)).issubset(parent_ids)):
            eligible.append(row)
    if len(eligible) != 8:
        raise ValueError(f"Expected 8 complete original-English official questions in the parent corpus; found {len(eligible)}")
    return eligible


def _metrics(gold_ids: set[str], ranked: list[dict], k: int = 5) -> dict[str, float]:
    ids = [str(hit["paper_id"]) for hit in ranked[:k]]
    relevant_positions = [rank for rank, paper_id in enumerate(ids, 1) if paper_id in gold_ids]
    recall = len(relevant_positions) / len(gold_ids)
    mrr = 1 / relevant_positions[0] if relevant_positions else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_positions)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), k) + 1))
    return {"recall_at_5": recall, "mrr_at_5": mrr, "ndcg_at_5": dcg / ideal if ideal else 0.0}


def _required_resources(profile: dict[str, Any]) -> list[str]:
    result = []
    if profile["retriever"] in {"bm25", "hybrid"}:
        result.append("bm25")
    if profile["retriever"] in {"dense", "hybrid"}:
        result.extend(("dense", "embedding_model"))
    if profile["rerank"]:
        result.append("reranker_model")
    return result


def run_benchmark(*, replay_agentic: bool = False) -> dict[str, Any]:
    for required in (PARENT_PATH, CHILD_PATH, INDEX_PATH, DENSE_PATH,
                     DENSE_PATH.with_suffix(".meta.json")):
        if not required.is_file():
            raise FileNotFoundError(f"Required existing corpus/index artifact is missing: {required}")
    parent_rows = [json.loads(line) for line in PARENT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    parent_ids = {str(row["paper_id"]) for row in parent_rows}
    questions = _load_questions(parent_ids)
    manifest_path = CHILD_PATH.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dense_meta = json.loads(DENSE_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    resources: dict[str, Any] = {}
    systems = []

    profiles = list(PROFILES)
    if replay_agentic:
        for rounds in (1, 2):
            profiles.append({"name": f"agentic_replay_{rounds}round_ce20", "retriever": "hybrid",
                             "top_k": 5, "candidate_k": 100, "rerank": True, "rerank_k": 20,
                             "pipeline": "agentic", "rounds": rounds})
        profiles.append({"name": "agentic_replay_2round_ce50", "retriever": "hybrid",
                         "top_k": 5, "candidate_k": 100, "rerank": True, "rerank_k": 50,
                         "pipeline": "agentic", "rounds": 2})
    replay_hashes = {}
    for profile in profiles:
        rows = []
        cache_state = []
        for item in questions:
            needed = _required_resources(profile)
            is_warm = all(name in resources for name in needed)
            runtime = RunRuntime(RuntimePolicy(deadline_seconds=180, failure_policy="strict",
                                               max_retrieval_rounds=profile.get("rounds", 2)))
            trace: dict[str, Any] = {}
            agents = None
            if profile.get("pipeline") == "agentic":
                source = ROOT / "data/litsearch/runs/p0_final/e2e_details/agentic_hybrid" / f"{item['id']}.json"
                stored = json.loads(source.read_text(encoding="utf-8"))["retrieval_trace"]
                replay_hashes[str(source.relative_to(ROOT))] = _digest(source)
                class ReplayAgents:
                    def plan(self, question):
                        return {"queries": stored["first_round_queries"], "intent": "saved p0-v1 query replay"}
                    def follow_up(self, question, queries, hits):
                        return stored["follow_up_queries"]
                agents = ReplayAgents()
            start = time.perf_counter()
            try:
                with use_runtime(runtime):
                    parents, _contexts = retrieve_parents(
                        item["question"], top_k=profile["top_k"], candidate_k=profile["candidate_k"],
                        parent_path=PARENT_PATH, child_path=CHILD_PATH, index_path=INDEX_PATH,
                        dense_index_path=DENSE_PATH, retriever=profile["retriever"], pipeline=profile.get("pipeline", "classic"),
                        rerank=profile["rerank"], rerank_k=profile.get("rerank_k", 50),
                        trace=trace, resources=resources, build_missing_indexes=False,
                        query_agents=agents,
                    )
                elapsed = time.perf_counter() - start
                rows.append({"question_id": item["id"], "status": "ok", "elapsed_seconds": elapsed,
                             "cache_state": "warm" if is_warm else "initialization_miss",
                             "metrics": _metrics(set(map(str, item["gold_paper_ids"])), parents),
                             "retrieved_paper_ids_at_5": [str(hit["paper_id"]) for hit in parents[:5]],
                             "timings": runtime.timings})
                rows[-1]["retrieval_trace"] = trace
            except Exception as exc:
                elapsed = time.perf_counter() - start
                rows.append({"question_id": item["id"], "status": "failed", "error_type": type(exc).__name__,
                             "elapsed_seconds": elapsed, "cache_state": "warm" if is_warm else "initialization_miss",
                             "timings": runtime.timings})
            cache_state.append("warm" if is_warm else "initialization_miss")

        successful = [row for row in rows if row["status"] == "ok"]
        elapsed_values = [row["elapsed_seconds"] for row in successful]
        initialization_values = [row["elapsed_seconds"] for row in successful if row["cache_state"] == "initialization_miss"]
        warm_values = [row["elapsed_seconds"] for row in successful if row["cache_state"] == "warm"]
        metric_means = {}
        for key in ("recall_at_5", "mrr_at_5", "ndcg_at_5"):
            vals = [row["metrics"][key] for row in successful]
            metric_means[key] = sum(vals) / len(vals) if vals else None
        stage_values: dict[str, list[float]] = {}
        stage_statuses: dict[str, dict[str, int]] = {}
        for row in successful:
            for timing in row["timings"]:
                stage_values.setdefault(timing["stage"], []).append(timing["seconds"])
                statuses = stage_statuses.setdefault(timing["stage"], {})
                status = timing.get("status", "unlabeled")
                statuses[status] = statuses.get(status, 0) + 1
        systems.append({
            "profile": profile, "question_count": len(rows),
            "success_count": len(successful), "failure_count": len(rows) - len(successful),
            "quality_mean_at_5": metric_means,
            "elapsed_seconds": _summary(elapsed_values),
            "first_query_elapsed_seconds": (rows[0]["elapsed_seconds"] if rows else None),
            "first_query_cache_state": (rows[0]["cache_state"] if rows else None),
            "initialization_miss_elapsed_seconds": _summary(initialization_values),
            "warm_elapsed_seconds": _summary(warm_values),
            "initialization_miss_count": cache_state.count("initialization_miss"),
            "stages": {stage: {**_summary(values), "status_counts": stage_statuses[stage]}
                       for stage, values in sorted(stage_values.items())},
            "rows": rows,
        })

    return {
        "schema_version": 1,
        "benchmark": "offline full-text parent retrieval; optional saved p0-v1 query replay; no answer generation or live agentic planning",
        "saved_query_replay_hashes": replay_hashes,
        "network_or_api_calls": 0,
        "provenance": collect_provenance(PARENT_PATH, CHILD_PATH, INDEX_PATH, DENSE_PATH, rerank=True,
                                          parameters={"profiles": profiles, "query_sha256": _digest(QUESTIONS)}),
        "selection": {"question_count": len(questions), "language": "original English",
                      "qrels_status": "official_gold_complete_in_pilot",
                      "question_ids": [row["id"] for row in questions],
                      "gold_ids_verified_inside_parent_corpus": True,
                      "scope_warning": "Eight complete in-corpus official pilot questions provide a small retrieval diagnostic, not a full-corpus quality estimate."},
        "corpus": {"paper_count": len(parent_rows), "parent_sha256": _digest(PARENT_PATH),
                   "child_sha256": _digest(CHILD_PATH), "parent_path": str(PARENT_PATH.relative_to(ROOT)),
                   "child_path": str(CHILD_PATH.relative_to(ROOT)), "chunk_manifest": manifest,
                   "bm25_index_sha256": _digest(INDEX_PATH), "dense_index_sha256": _digest(DENSE_PATH),
                   "dense_index_metadata": dense_meta, "embedding_model": BGE_MODEL},
        "measurement": {"deadline_seconds_per_query": 180, "metric_cutoff": 5,
                        "resources_cache_shared_across_all_profiles": True,
                        "warm_elapsed_definition": "successful query elapsed values whose required indexes/models were already cached before that call",
                        "deadline_semantics": "RunRuntime checks deadlines cooperatively; native local model operations cannot be preempted mid-call.",
                        "percentile_method": PERCENTILE_METHOD,
                        "timing_note": "Nested stage durations are reported independently and are never summed into elapsed time."},
        "systems": systems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay-agentic", action="store_true", help="增加保存查询的一轮/两轮检索；不调用规划模型")
    args = parser.parse_args(argv)
    report = run_benchmark(replay_agentic=args.replay_agentic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "systems": [
        {"name": entry["profile"]["name"], "success_count": entry["success_count"],
         "failure_count": entry["failure_count"], "quality_mean_at_5": entry["quality_mean_at_5"],
         "elapsed_seconds": entry["elapsed_seconds"], "warm_elapsed_seconds": entry["warm_elapsed_seconds"]}
        for entry in report["systems"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
