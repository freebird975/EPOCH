"""Run the fixed pilot end-to-end comparison with explicit live API opt-in."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from litagent.end_to_end_eval import evaluate_e2e, load_questions
from litsearch_fulltext import (BGE_MODEL, CHILD_DATA, DENSE_CHILD_INDEX,
                                PARENT_DATA, _digest, run_fulltext_ask)


QUESTIONS = Path("eval/fulltext_e2e_questions.jsonl")
OUTPUT_ROOT = Path("data/litsearch/runs")
CONFIGS = {
    "classic_hybrid": {"pipeline": "classic", "rerank": False},
    "agentic_hybrid": {"pipeline": "agentic", "rerank": False},
    "agentic_rerank": {"pipeline": "agentic", "rerank": True},
}


def _pilot_parent_ids() -> set[str]:
    return {str(json.loads(line)["paper_id"]) for line in PARENT_DATA.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def select_pilot_questions(questions: list[dict], parent_ids: set[str]) -> list[dict]:
    """Score only complete in-scope official qrels; retain no-answer probes."""
    return [item for item in questions
            if (item.get("answerability") == "unanswerable" and not item.get("gold_paper_ids"))
            or (bool(item.get("gold_paper_ids"))
                and set(item["gold_paper_ids"]).issubset(parent_ids))]


def _answer_judge(item: dict, prediction: dict, _hits: list[dict]) -> dict:
    verification = prediction.get("verification") or {}
    claims = verification.get("claims") or []
    supported = sum(claim.get("verdict") == "support" for claim in claims)
    return {"claim_count": len(claims), "supported_claims": supported,
            "citation_support_rate": supported / len(claims) if claims else None,
            "fact_correctness": None,
            "status": "automatic_evidence_check; factual_correctness_requires_human_review"}


def run_e2e(questions_path: Path = QUESTIONS, output_path: Path | None = None,
            *, selected_configs: tuple[str, ...] = tuple(CONFIGS), max_questions: int | None = None,
            ask_fn=None, reuse_reports: bool = False) -> dict:
    questions = load_questions(questions_path)
    parent_ids = _pilot_parent_ids()
    selected = select_pilot_questions(questions, parent_ids)
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("--max-questions 必须大于 0")
        selected = selected[:max_questions]
    if not selected:
        raise ValueError("测试集中没有适用于当前全文试点的题目")
    unknown = set(selected_configs) - set(CONFIGS)
    if unknown:
        raise ValueError(f"未知配置: {sorted(unknown)}")
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = OUTPUT_ROOT / f"e2e-pilot-{stamp}.json"
    output_path = Path(output_path)
    ask_fn = ask_fn or run_fulltext_ask
    prior_run = {}
    if reuse_reports and output_path.is_file():
        prior_run = (json.loads(output_path.read_text(encoding="utf-8")).get("run") or {})
    if reuse_reports:
        missing = [output_path.parent / "e2e_details" / name / f"{item['id']}.json"
                   for name in selected_configs for item in selected
                   if not (output_path.parent / "e2e_details" / name / f"{item['id']}.json").is_file()]
        if missing:
            raise FileNotFoundError(f"缺少 {len(missing)} 份逐题运行记录；首个缺失文件：{missing[0]}")
    systems = {}
    resources = {}
    for name in selected_configs:
        settings = CONFIGS[name]
        cache: dict[str, dict] = {}

        def retrieve(question, top_k, item, *, _cache=cache, _settings=settings, _name=name):
            run_path = output_path.parent / "e2e_details" / _name / f"{item['id']}.json"
            print(f"{_name}: {item['id']}", flush=True)
            try:
                if reuse_reports:
                    if not run_path.is_file():
                        raise FileNotFoundError(f"缺少可回放运行记录: {run_path}")
                    result = json.loads(run_path.read_text(encoding="utf-8"))
                else:
                    result = ask_fn(
                        question, top_k=top_k, candidate_k=100, retriever="hybrid",
                        pipeline=_settings["pipeline"], rerank=_settings["rerank"],
                        context_strategy="chunks", max_context_chars=12000,
                        report_path=run_path,
                        resources=resources,
                    )
            except Exception as exc:
                if reuse_reports:
                    raise
                if run_path.is_file():
                    result = json.loads(run_path.read_text(encoding="utf-8"))
                else:
                    result = {"status": "failed", "answer": "", "candidate_parents": [],
                              "failure": {"type": type(exc).__name__}}
            _cache[item["id"]] = result
            return [{"paper_id": parent["paper_id"]} for parent in result.get("candidate_parents", [])]

        def answer(_question, _hits, item, *, _cache=cache):
            result = _cache[item["id"]]
            text = result.get("answer", "")
            from litagent.fulltext_answer import is_refusal_answer, refusal_message

            explicit_refusal = is_refusal_answer(text)
            if explicit_refusal and result.get("status") == "verified":
                result = {**result, "status": "refused_evidence_insufficient",
                          "validated_refusal_text": text,
                          "answer": refusal_message(item["question"], "evidence_insufficient")}
                text = result["answer"]
            contexts = result.get("contexts") or []
            from litsearch_fulltext import _cited_numbers

            citations = [contexts[number - 1]["paper_id"] for number in _cited_numbers(text)
                         if 1 <= number <= len(contexts)]
            return {**result, "citations": citations,
                    "refused": result.get("status", "").startswith("refused")}

        systems[name] = (retrieve, answer)

    dense_meta_path = DENSE_CHILD_INDEX.with_suffix(".meta.json")
    dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8")) if dense_meta_path.is_file() else {}
    run_metadata = {
        "scope": "fixed 100-paper gold-enriched pilot; manual_iclr in-scope qrels",
        "questions_path": questions_path.as_posix(),
        "questions_file_sha256": _digest(questions_path),
        "parent_sha256": _digest(PARENT_DATA), "chunk_sha256": _digest(CHILD_DATA),
        "dense_index_sha256": dense_meta.get("index_sha256"),
        "embedding_model": BGE_MODEL, "embedding_snapshot": dense_meta.get("model_snapshot"),
        "chat_model": "deepseek-flash", "cross_encoder_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "prompt_version": prior_run.get("prompt_version", "unknown_saved_version") if reuse_reports else "fulltext-answer-p2-v1",
        "judge_version": prior_run.get("judge_version", "batch-entailment-p0-v1"),
        "selected_question_ids": [item["id"] for item in selected],
        "excluded_out_of_scope_question_ids": [item["id"] for item in questions if item not in selected],
        "run_at_utc": prior_run.get("run_at_utc", datetime.now(timezone.utc).isoformat()),
        **({"replayed_at_utc": datetime.now(timezone.utc).isoformat()} if reuse_reports else {}),
    }
    report = evaluate_e2e(selected, systems, run_metadata=run_metadata, top_k=5,
                          corpus_paper_ids=parent_ids, answer_judge=_answer_judge)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    report["report_path"] = str(output_path)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="固定全文试点的端到端评测")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--live", action="store_true", help="允许实际调用已配置的 DeepSeek API")
    parser.add_argument("--replay", action="store_true", help="从已保存的逐题运行文件重算汇总，不调用 API")
    args = parser.parse_args()
    try:
        questions = select_pilot_questions(load_questions(args.questions), _pilot_parent_ids())
        if args.max_questions:
            questions = questions[:args.max_questions]
        if not args.live and not args.replay:
            print(f"适用题目 {len(questions)} 条：" + ", ".join(item["id"] for item in questions))
            print("添加 --live 后运行真实 API 端到端对照。")
            return 0
        if args.live and not args.replay:
            from rag.generate import resolve_api_key

            resolve_api_key()
        report = run_e2e(args.questions, args.out, selected_configs=tuple(args.configs),
                         max_questions=args.max_questions, reuse_reports=args.replay)
        for name, system in report["systems"].items():
            summary = system["aggregate"]
            print(f"{name}: scored={summary['retrieval_scored_count']} "
                  f"Recall@5={summary['recall_at_k']} MRR@5={summary['mrr_at_k']} "
                  f"verified={summary['verified_count']} failures={summary['failure_count']} "
                  f"calls={summary['api_call_count']}")
        print(f"报告：{report['report_path']}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
