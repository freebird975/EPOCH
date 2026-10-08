"""Run a bounded, full-corpus cited-QA check for DEVELOPMENT_GUIDE step 5.

This script uses the configured chat API. It never reads or prints API key
values; run reports are written under data/litsearch/runs/fulltext_step5/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litagent.end_to_end_eval import load_questions
from litagent.runtime import RuntimePolicy
from litsearch_fulltext import run_fulltext_ask


QUESTION_SET = ROOT / "eval/fulltext_e2e_questions.jsonl"
OUTPUT_DIR = ROOT / "data/litsearch/runs/fulltext_step5"
SUMMARY_PATH = ROOT / "eval/fulltext_step5_eval.json"
SELECTED_IDS = ("e2e_pilot_518", "e2e_pilot_518_zh", "e2e_noanswer_001")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _citation_summary(report: dict) -> list[dict]:
    contexts = report.get("contexts", [])
    citation_groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", report.get("answer", ""))
    cited_numbers = sorted({int(value.strip()) for group in citation_groups for value in group.split(",")})
    result = []
    for number in cited_numbers:
        if not 1 <= number <= len(contexts):
            result.append({"number": number, "valid_context_number": False})
            continue
        context = contexts[number - 1]
        result.append({
            "number": number,
            "valid_context_number": True,
            "paper_id": context.get("paper_id"),
            "chunk_id": context.get("chunk_id"),
            "section": context.get("section"),
            "source_url": context.get("source_url"),
            "text_sha256": context.get("text_sha256"),
        })
    return result


def _row_summary(question: dict, report: dict | None, error: str | None, report_path: Path) -> dict:
    if report is None:
        return {"id": question["id"], "question": question["question"], "status": "failed",
                "error": error, "report_path": str(report_path)}
    return {
        "id": question["id"],
        "question": question["question"],
        "answerability": question["answerability"],
        "gold_paper_ids": question.get("gold_paper_ids"),
        "status": report.get("status"),
        "answer": report.get("answer"),
        "failure": report.get("failure"),
        "citations": _citation_summary(report),
        "candidate_parents": report.get("candidate_parents", []),
        "verification": report.get("verification"),
        "retrieval_trace": report.get("retrieval_trace"),
        "api_calls": report.get("api_calls", []),
        "cost": report.get("cost"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "runtime": report.get("runtime"),
        "degradations": report.get("degradations", []),
        "provenance": report.get("provenance"),
        "report_path": str(report_path),
    }


def main() -> int:
    questions_by_id = {item["id"]: item for item in load_questions(QUESTION_SET)}
    missing = [item_id for item_id in SELECTED_IDS if item_id not in questions_by_id]
    if missing:
        raise ValueError(f"Step 5 question set is missing IDs: {missing}")
    questions = [questions_by_id[item_id] for item_id in SELECTED_IDS]
    resources: dict = {}
    policy = RuntimePolicy(
        deadline_seconds=900,
        api_timeout_seconds=60,
        max_api_calls=6,
        max_completion_tokens=9_000,
        max_first_queries=1,
        max_followup_queries=0,
        max_retrieval_rounds=1,
        failure_policy="strict",
    )
    summary = {
        "schema_version": 1,
        "status": "running",
        "evaluation": "full_corpus_cited_qa_smoke",
        "scope": "two official answerable query translations and one adversarial closed-corpus abstention challenge",
        "interpretation": "Small smoke check only; automated verification is not independent human fact review.",
        "prompt_version": "fulltext-answer-p2-v2-citation-source-identity",
        "prior_iteration_report": str(ROOT / "eval/fulltext_step5_eval_v1.json"),
        "parameters": {"pipeline": "classic", "retriever": "hybrid", "candidate_k": 100,
                       "top_k": 5, "rerank": True, "rerank_k": 20,
                       "context_strategy": "chunks", "runtime_policy": policy.__dict__},
        "questions": [],
    }
    _write_json(SUMMARY_PATH, summary)

    for question in questions:
        report_path = OUTPUT_DIR / f"{question['id']}.json"
        print(f"Running {question['id']} ({question['answerability']})", flush=True)
        report = None
        error = None
        try:
            report = run_fulltext_ask(
                question["question"],
                top_k=5,
                candidate_k=100,
                retriever="hybrid",
                pipeline="classic",
                rerank=True,
                rerank_k=20,
                context_strategy="chunks",
                max_context_chars=12_000,
                max_chunks_per_parent=2,
                report_path=report_path,
                parent_path=ROOT / "data/litsearch/corpus_fulltext.jsonl",
                child_path=ROOT / "data/litsearch/could_full/fulltext_chunks.jsonl",
                index_path=ROOT / "data/litsearch/could_full/fulltext_bm25_index.sqlite3",
                dense_index_path=ROOT / "data/litsearch/could_full/fulltext_dense_bge.faiss",
                dense_docs_path=ROOT / "data/litsearch/could_full/fulltext_dense_bge.docs.sqlite3",
                runtime_policy=policy,
                resources=resources,
                build_missing_indexes=False,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    report = None
            print(f"Failed {question['id']}: {error}", flush=True)
        row = _row_summary(question, report, error, report_path)
        summary["questions"].append(row)
        _write_json(SUMMARY_PATH, summary)
        if report is not None:
            print(f"Finished {question['id']}: {report.get('status')} · "
                  f"API calls {len(report.get('api_calls', []))} · "
                  f"{report.get('elapsed_seconds')}s", flush=True)
            if report.get("status") == "failed" and any(call.get("error") for call in report.get("api_calls", [])):
                summary["skipped_question_ids"] = [item["id"] for item in questions
                                                    if item["id"] not in {row["id"] for row in summary["questions"]}]
                summary["api_blocked"] = True
                break

    failed = sum(row.get("status") == "failed" for row in summary["questions"])
    summary["status"] = "complete" if not failed and not summary.get("api_blocked") else "partial"
    summary["failed_count"] = failed
    _write_json(SUMMARY_PATH, summary)
    print(f"Step 5 report: {SUMMARY_PATH}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
