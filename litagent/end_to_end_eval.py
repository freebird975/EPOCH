"""Small, injectable end-to-end evaluator for full-text RAG configurations.

This module deliberately does not construct prompts or call a model. Callers
provide retrieval and answer functions, making runs reproducible and tests safe
without network/API access. See ``eval/fulltext_e2e_questions.jsonl`` for label
provenance and ``docs/REMAINING_WORK.md`` for the 100-paper pilot caveat.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


Retriever = Callable[[str, int, dict[str, Any]], Sequence[dict[str, Any]]]
Answerer = Callable[[str, Sequence[dict[str, Any]], dict[str, Any]], dict[str, Any]]
AnswerJudge = Callable[[dict[str, Any], dict[str, Any], Sequence[dict[str, Any]]], dict[str, Any]]


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate the versioned JSONL dataset."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not item.get("id") or not item.get("question"):
            raise ValueError(f"{path}:{line_no}: question id and text are required")
        if item["id"] in seen:
            raise ValueError(f"duplicate question id: {item['id']}")
        seen.add(item["id"])
        if item.get("answerability") not in {"answerable", "unanswerable", "unknown"}:
            raise ValueError(f"{item['id']}: invalid answerability")
        gold = item.get("gold_paper_ids")
        if gold is not None and (not isinstance(gold, list) or not all(isinstance(x, str) for x in gold)):
            raise ValueError(f"{item['id']}: gold_paper_ids must be a list or null")
        rows.append(item)
    if not rows:
        raise ValueError("question set is empty")
    return rows


def _paper_id(hit: dict[str, Any]) -> str | None:
    value = hit.get("parent_id", hit.get("corpusid", hit.get("paper_id")))
    return str(value) if value is not None else None


def _retrieval_scores(item: dict[str, Any], hits: Sequence[dict[str, Any]], top_k: int,
                      corpus_paper_ids: set[str] | None) -> dict[str, Any]:
    gold = item.get("gold_paper_ids")
    if not gold:
        return {"status": "no_qrels", "recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None}
    if corpus_paper_ids is not None and not set(gold).issubset(corpus_paper_ids):
        return {"status": "incomplete_gold_out_of_scope", "recall_at_k": None,
                "mrr_at_k": None, "ndcg_at_k": None,
                "gold_in_scope": sorted(set(gold) & corpus_paper_ids),
                "gold_out_of_scope": sorted(set(gold) - corpus_paper_ids)}
    ids = [_paper_id(hit) for hit in hits[:top_k]]
    relevant_ranks = [rank for rank, pid in enumerate(ids, 1) if pid in set(gold)]
    ideal_len = min(len(gold), top_k)
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return {"status": "scored", "gold_paper_ids": gold,
            "retrieved_paper_ids": ids,
            "recall_at_k": len(relevant_ranks) / len(gold),
            "mrr_at_k": (1 / relevant_ranks[0]) if relevant_ranks else 0.0,
            "ndcg_at_k": dcg / ideal if ideal else 0.0}


def _citation_ids(prediction: dict[str, Any]) -> list[str]:
    values = prediction.get("citations", []) or []
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("paper_id", value.get("parent_id", value.get("id")))
        if value is not None:
            result.append(str(value))
    return result


def _answer_metrics(item: dict[str, Any], prediction: dict[str, Any], hits: Sequence[dict[str, Any]],
                    judge: AnswerJudge | None) -> dict[str, Any]:
    citations = _citation_ids(prediction)
    retrieved = {_paper_id(hit) for hit in hits}
    retrieved.discard(None)
    citation_validity = (sum(cid in retrieved for cid in citations) / len(citations)) if citations else None
    answerable = item.get("answerability")
    refused = bool(prediction.get("refused", False))
    failure = prediction.get("status") == "failed"
    refusal_correct = ((refused if answerable == "unanswerable" else not refused)
                       if answerable != "unknown" and not failure else None)
    result: dict[str, Any] = {
        "refused": refused,
        "failed": failure,
        "refusal_correct": refusal_correct,
        "citation_ids": citations,
        "citation_validity": citation_validity,
        "citation_count": len(citations),
        "supported_claim_rate": None,
        "supported_claims": None,
        "claim_count": None,
        "answer_quality_status": "judge_not_supplied",
    }
    if judge is not None:
        judgment = judge(item, prediction, hits)
        total = int(judgment.get("claim_count", 0))
        supported = int(judgment.get("supported_claims", 0))
        if total < 0 or supported < 0 or supported > total:
            raise ValueError(f"invalid answer judgment for {item['id']}")
        result.update({"supported_claim_rate": supported / total if total else None,
                       "supported_claims": supported, "claim_count": total,
                       "answer_quality_status": judgment.get("status", "judged"),
                       "fact_correctness": judgment.get("fact_correctness"),
                       "citation_support_rate": judgment.get("citation_support_rate")})
    return result


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def evaluate_e2e(
    questions: Sequence[dict[str, Any]],
    systems: Mapping[str, tuple[Retriever, Answerer]],
    *,
    run_metadata: dict[str, Any],
    top_k: int = 5,
    corpus_paper_ids: set[str] | None = None,
    answer_judge: AnswerJudge | None = None,
) -> dict[str, Any]:
    """Run named (retriever, answerer) pairs and return row and aggregate data.

    Retriever signature: ``retriever(question, top_k, item) -> hits``.
    Answerer signature: ``answerer(question, hits, item) -> prediction`` where
    prediction contains ``answer``, optional ``citations`` and ``refused``.
    An optional independent judge returns claim counts and factual/citation
    assessments; without it, answer support and correctness remain null.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    digest_payload = json.dumps(list(questions), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {"question_count": len(questions),
                    "sha256": hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
                    "qrel_question_count": sum(bool(q.get("gold_paper_ids")) for q in questions),
                    "answerability_counts": {key: sum(q.get("answerability") == key for q in questions)
                                             for key in ("answerable", "unanswerable", "unknown")}},
        "run": {**run_metadata, "top_k": top_k,
                "corpus_paper_count": len(corpus_paper_ids) if corpus_paper_ids is not None else None,
                "answer_judge": "injected" if answer_judge else None,
                "scope_warning": "100-paper pilot is gold-enriched; incomplete official qrels are unscored. "
                                 "Results do not represent the full LitSearch corpus."},
        "systems": {},
    }
    for name, (retriever, answerer) in systems.items():
        rows = []
        for item in questions:
            hits = list(retriever(item["question"], top_k, item))
            prediction = answerer(item["question"], hits, item)
            if not isinstance(prediction, dict):
                raise TypeError(f"{name}/{item['id']}: answerer must return a dict")
            diagnostic_fields = ("verification", "contexts", "api_calls", "cost", "latency_seconds",
                                 "elapsed_seconds", "status", "retrieval_trace", "candidate_parents",
                                 "candidate_parent_ids", "token_usage", "failure", "failure_reason",
                                 "run", "timings", "runtime", "provenance", "degradations", "degraded")
            diagnostics = {key: prediction[key] for key in diagnostic_fields if key in prediction}
            rows.append({"id": item["id"], "question": item["question"],
                         "category": item.get("category"),
                         "source_query_id": (item.get("source") or {}).get("query_id"),
                         "source_type": (item.get("source") or {}).get("type"),
                         "retrieval": ({"status": "retrieval_failed"}
                                       if prediction.get("status") == "failed" and not hits
                                       else _retrieval_scores(item, hits, top_k, corpus_paper_ids)),
                         "answer": {"text": prediction.get("answer", ""),
                                    **_answer_metrics(item, prediction, hits, answer_judge),
                                    "diagnostics": diagnostics}})
        retrieval_rows = [row["retrieval"] for row in rows]
        scored = [row for row in retrieval_rows if row["status"] == "scored"]
        original_scored = [row["retrieval"] for row in rows
                           if row["retrieval"]["status"] == "scored"
                           and row["source_type"] == "official_litsearch_query"]
        translated_scored = [row["retrieval"] for row in rows
                             if row["retrieval"]["status"] == "scored"
                             and row["source_type"] == "official_litsearch_query_translation"]
        answer_rows = [row["answer"] for row in rows]
        report["systems"][name] = {
            "aggregate": {
                "question_count": len(rows), "retrieval_scored_count": len(scored),
                "retrieval_unscored_count": len(rows) - len(scored),
                "recall_at_k": _mean(scored, "recall_at_k"), "mrr_at_k": _mean(scored, "mrr_at_k"),
                "ndcg_at_k": _mean(scored, "ndcg_at_k"),
                "original_query_count": len(original_scored),
                "original_recall_at_k": _mean(original_scored, "recall_at_k"),
                "original_mrr_at_k": _mean(original_scored, "mrr_at_k"),
                "original_ndcg_at_k": _mean(original_scored, "ndcg_at_k"),
                "translated_query_count": len(translated_scored),
                "translated_recall_at_k": _mean(translated_scored, "recall_at_k"),
                "translated_mrr_at_k": _mean(translated_scored, "mrr_at_k"),
                "translated_ndcg_at_k": _mean(translated_scored, "ndcg_at_k"),
                "supported_claim_rate": _mean(answer_rows, "supported_claim_rate"),
                "fact_correctness": _mean(answer_rows, "fact_correctness"),
                "citation_validity": _mean(answer_rows, "citation_validity"),
                "citation_support_rate": _mean(answer_rows, "citation_support_rate"),
                "refusal_accuracy": _mean(answer_rows, "refusal_correct"),
                "refusal_scored_count": sum(row["refusal_correct"] is not None for row in answer_rows),
                "verified_count": sum(row["diagnostics"].get("status") == "verified" for row in answer_rows),
                "failure_count": sum(row["failed"] for row in answer_rows),
                "api_call_count": sum(len(row["diagnostics"].get("api_calls", [])) for row in answer_rows),
                "prompt_tokens": sum((row["diagnostics"].get("cost") or {}).get("prompt_tokens", 0)
                                     for row in answer_rows),
                "completion_tokens": sum((row["diagnostics"].get("cost") or {}).get("completion_tokens", 0)
                                         for row in answer_rows),
                "estimated_usd_cold_peak": sum(
                    (row["diagnostics"].get("cost") or {}).get("estimated_usd_cold_peak", 0)
                    for row in answer_rows),
                "mean_elapsed_seconds": _mean([row["diagnostics"] for row in answer_rows], "elapsed_seconds"),
            },
            "rows": rows,
        }
    return report


def evaluate_file(path: str | Path, systems: Mapping[str, tuple[Retriever, Answerer]], **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper to load the fixed JSONL set and run the evaluator."""
    questions = load_questions(path)
    return evaluate_e2e(questions, systems, **kwargs)
