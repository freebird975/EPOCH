"""Export and score human reviews for saved full-text RAG answers.

The exporter keeps model verification separate from human labels. It accepts
the injectable end-to-end report format and the full-corpus Step 5 summary plus
its per-question detail files. It never calls a model or mutates source reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ANSWER_LABELS = {"correct", "partially_correct", "incorrect", "uncertain", "unverifiable", "not_applicable"}
CITATION_LABELS = {"all_supported", "partially_supported", "unsupported", "unclear", "not_applicable"}
HANDLING_LABELS = {"appropriate", "inappropriate", "uncertain", "not_applicable"}
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_question_labels(path: str | Path | None,
                         expected_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file() and not source.is_absolute():
        source = Path(__file__).resolve().parent.parent / source
    if not source.is_file():
        raise FileNotFoundError(f"Question set required by the report does not exist: {source}")
    raw = source.read_bytes()
    actual_sha256 = _digest(raw)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"Question set SHA-256 mismatch for {source}: expected {expected_sha256}, got {actual_sha256}")
    result = {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError(f"{source}:{line_no}: question id is required")
        question_id = str(item["id"])
        if question_id in result:
            raise ValueError(f"{source}:{line_no}: duplicate question id {question_id}")
        result[question_id] = item
    return result


def _minimal_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {key: context[key] for key in (
        "number", "paper_id", "title", "source_url", "chunk_id", "section",
        "text", "text_sha256", "granularity",
    ) if context.get(key) is not None}


def _candidate_summary(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    result = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        item = {key: candidate[key] for key in ("paper_id", "title", "best_chunk_rank")
                if candidate.get(key) is not None}
        if item:
            result.append(item)
    return result


def _claim_summary(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for claim in value:
        if not isinstance(claim, Mapping):
            continue
        item = {key: claim[key] for key in (
            "claim", "citations", "supported", "verdict", "status", "reason", "evidence",
        ) if key in claim}
        result.append(item)
    return result


def _evidence(contexts: Any, answer: str, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(contexts, list):
        contexts = []
    by_number = {str(item.get("number")): item for item in contexts
                 if isinstance(item, Mapping) and item.get("number") is not None}
    used_numbers = set(_CITATION_RE.findall(answer or ""))
    for claim in claims:
        for number in claim.get("citations", []) if isinstance(claim.get("citations"), list) else []:
            if isinstance(number, (str, int)) and str(number).isdigit():
                used_numbers.add(str(number))
    selected = [by_number[number] for number in sorted(used_numbers, key=int) if number in by_number]
    # Include a small evidence sample for refusals, failed citations, and answers
    # without inline citation markers so those decisions remain reviewable.
    if not selected:
        selected = [item for item in contexts[:5] if isinstance(item, Mapping)]
    return [_minimal_context(item) for item in selected]


def _make_row(*, source_kind: str, source_name: str, source_sha256: str,
              system: str, question_id: str, question: str, answerability: str | None,
              answer: str, status: str | None, refused: bool | None, failed: bool | None,
              citations: Any, contexts: Any, claims: Any, claim_source: str, candidates: Any,
              provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    stable = "\0".join((source_sha256, source_kind, system, question_id))
    review_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]
    claim_rows = _claim_summary(claims)
    return {
        "schema_version": 1,
        "review_id": review_id,
        "source": {"kind": source_kind, "report": source_name, "sha256": source_sha256,
                   "system": system, "question_id": question_id, "provenance": dict(provenance or {})},
        "question": {"text": question, "answerability": answerability},
        "result": {
            "status": status,
            "answer": answer,
            "refused": refused,
            "failed": failed,
            "citations": citations if isinstance(citations, list) else [],
            "inline_citation_numbers": sorted({int(value) for value in _CITATION_RE.findall(answer or "")}),
            "candidate_papers": _candidate_summary(candidates),
            "verification_claims": claim_rows,
            "verification_claims_source": claim_source,
            "evidence": _evidence(contexts, answer, claim_rows),
        },
        "annotation": {
            "answer_correctness": "",
            "citation_support": "",
            "answerability_handling": "",
            "notes": "",
            "reviewer": "",
            "reviewed_at": "",
        },
    }


def export_e2e_report(report: Mapping[str, Any], source_name: str = "e2e.json",
                      source_sha256: str | None = None,
                      question_set: str | Path | None = None) -> list[dict[str, Any]]:
    """Create one human-review row per question and system in an E2E report."""
    if not isinstance(report.get("systems"), Mapping):
        raise ValueError("E2E report must contain a systems object")
    digest = source_sha256 or _digest(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                                  default=str).encode("utf-8"))
    run = report.get("run") if isinstance(report.get("run"), Mapping) else {}
    question_path = question_set or run.get("questions_path")
    question_metadata = _load_question_labels(question_path, run.get("questions_file_sha256"))
    rows: list[dict[str, Any]] = []
    for system, system_report in report["systems"].items():
        if not isinstance(system_report, Mapping) or not isinstance(system_report.get("rows"), list):
            continue
        for result_row in system_report["rows"]:
            if not isinstance(result_row, Mapping):
                continue
            answer_row = result_row.get("answer") if isinstance(result_row.get("answer"), Mapping) else {}
            diagnostics = answer_row.get("diagnostics") if isinstance(answer_row.get("diagnostics"), Mapping) else {}
            verification = diagnostics.get("verification") if isinstance(diagnostics.get("verification"), Mapping) else {}
            question_id = str(result_row.get("id", ""))
            question_info = question_metadata.get(question_id, {})
            if question_metadata and not question_info:
                raise ValueError(f"Question {question_id} is not present in the recorded question set")
            if question_info and question_info.get("question") != result_row.get("question"):
                raise ValueError(f"Question text mismatch for {question_id} between report and question set")
            claims = verification.get("claims")
            rows.append(_make_row(
                source_kind="e2e", source_name=source_name, source_sha256=digest,
                system=str(system), question_id=question_id,
                question=str(result_row.get("question", "")),
                answerability=question_info.get("answerability"),
                answer=str(answer_row.get("text", "")), status=diagnostics.get("status"),
                refused=bool(answer_row.get("refused", False)), failed=bool(answer_row.get("failed", False)),
                citations=answer_row.get("citation_ids", []), contexts=diagnostics.get("contexts", []),
                claims=claims, claim_source="verification.claims" if isinstance(claims, list) else "unavailable",
                candidates=diagnostics.get("candidate_parents", []),
                provenance={key: run[key] for key in ("prompt_version", "chat_model", "embedding_model",
                                                       "cross_encoder_model") if run.get(key) is not None},
            ))
    return rows


def export_step5_report(summary: Mapping[str, Any], details_dir: str | Path,
                        source_name: str = "fulltext_step5_eval.json",
                        source_sha256: str | None = None) -> list[dict[str, Any]]:
    """Create review rows from Step 5 summary and its per-question details."""
    questions = summary.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Step 5 summary must contain a questions list")
    details_root = Path(details_dir)
    digest = source_sha256 or _digest(json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                                  default=str).encode("utf-8"))
    rows: list[dict[str, Any]] = []
    parameters = summary.get("parameters") if isinstance(summary.get("parameters"), Mapping) else {}
    rerank_name = "rerank" if parameters.get("rerank") else "no_rerank"
    system = "full_corpus_" + "_".join(str(parameters.get(key, "unknown")) for key in
                                         ("pipeline", "retriever")) + "_" + rerank_name
    for item in questions:
        if not isinstance(item, Mapping) or not item.get("id"):
            continue
        question_id = str(item["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", question_id):
            raise ValueError(f"unsafe Step 5 question id for detail lookup: {question_id}")
        detail_path = details_root / f"{question_id}.json"
        if not detail_path.is_file():
            raise FileNotFoundError(f"Step 5 detail report missing for {question_id}: {detail_path}")
        loaded = json.loads(detail_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Step 5 detail report must be a JSON object: {detail_path}")
        detail: dict[str, Any] = loaded
        run = detail.get("run") if isinstance(detail.get("run"), Mapping) else {}
        detail_question = run.get("question")
        if not isinstance(detail_question, str) or detail_question != item.get("question"):
            raise ValueError(f"Step 5 question text mismatch for {question_id}: {detail_path}")
        if item.get("status") is not None and detail.get("status") != item.get("status"):
            raise ValueError(f"Step 5 status mismatch for {question_id}: {detail_path}")
        verification = detail.get("verification") if isinstance(detail.get("verification"), Mapping) else {}
        claims = verification.get("claims")
        status = detail.get("status", item.get("status"))
        answer = detail.get("answer", item.get("answer", ""))
        failed = status == "failed" or bool(item.get("error"))
        refused = bool(status and str(status).startswith("refused"))
        row_digest = digest
        if detail_path.is_file():
            row_digest = _digest(detail_path.read_bytes() + digest.encode("ascii"))
        rows.append(_make_row(
            source_kind="step5", source_name=source_name, source_sha256=row_digest,
            system=system, question_id=question_id,
            question=str(run.get("question", item.get("question", ""))),
            answerability=item.get("answerability"), answer=str(answer or ""), status=status,
            refused=refused, failed=failed,
            citations=detail.get("citations", item.get("citations", [])),
            contexts=detail.get("contexts", []), claims=claims,
            claim_source="verification.claims" if isinstance(claims, list) else "unavailable",
            candidates=detail.get("candidate_parents", []),
            provenance={"prompt_version": run.get("prompt_version", summary.get("prompt_version")),
                        "detail_report": detail_path.name if detail else None},
        ))
    return rows


def _validate_annotations(rows: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        review_id = row.get("review_id")
        if review_id:
            if review_id in seen:
                raise ValueError(f"duplicate review_id: {review_id}")
            seen.add(str(review_id))
        annotation = row.get("annotation") if isinstance(row.get("annotation"), Mapping) else {}
        for key, allowed in (("answer_correctness", ANSWER_LABELS),
                             ("citation_support", CITATION_LABELS),
                             ("answerability_handling", HANDLING_LABELS)):
            value = annotation.get(key, "")
            if value not in (None, "", *allowed):
                raise ValueError(f"invalid {key} for {review_id}: {value}")


def _category_score(rows: list[Mapping[str, Any]], key: str, labels: set[str],
                    decided: set[str]) -> dict[str, Any]:
    annotations = [r.get("annotation") if isinstance(r.get("annotation"), Mapping) else {} for r in rows]
    values = [str(a.get(key, "")) for a in annotations]
    counts = dict(sorted(Counter(value for value in values if value in labels).items()))
    valid = [value for value in values if value in labels]
    resolved = [value for value in valid if value in decided]
    return {"labeled_rows": len(valid), "coverage": len(valid) / len(rows) if rows else 0.0,
            "labels": counts, "decided_rows": len(resolved)}


def _score_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    answer = _category_score(rows, "answer_correctness", ANSWER_LABELS,
                             {"correct", "partially_correct", "incorrect"})
    answer_decided = answer["decided_rows"]
    answer_counts = answer["labels"]
    citation = _category_score(rows, "citation_support", CITATION_LABELS,
                               {"all_supported", "partially_supported", "unsupported"})
    citation_decided = citation["decided_rows"]
    citation_counts = citation["labels"]
    refusal = _category_score(rows, "answerability_handling", HANDLING_LABELS,
                              {"appropriate", "inappropriate"})
    refusal_decided = refusal["decided_rows"]
    refusal_correct = 0
    refusal_decision_scored = 0
    refusal_decision_skipped_failed = 0
    for row in rows:
        question = row.get("question") if isinstance(row.get("question"), Mapping) else {}
        expected = question.get("answerability")
        refused = bool((row.get("result") or {}).get("refused"))
        if expected in {"answerable", "unanswerable"}:
            if (row.get("result") or {}).get("failed") is True:
                refusal_decision_skipped_failed += 1
            else:
                refusal_decision_scored += 1
                refusal_correct += refused == (expected == "unanswerable")
    refusal_counts = refusal["labels"]
    return {
        "row_count": len(rows),
        "unique_question_count": len({(r.get("source") or {}).get("question_id") for r in rows}),
        "answer_correctness": {**answer,
                               "strict_correct_rate_among_decided": answer_counts.get("correct", 0) / answer_decided if answer_decided else None,
                               "non_incorrect_rate_among_decided": (answer_counts.get("correct", 0) + answer_counts.get("partially_correct", 0)) / answer_decided if answer_decided else None},
        "citation_support": {**citation,
                             "fully_supported_rate_among_decided": citation_counts.get("all_supported", 0) / citation_decided if citation_decided else None},
        "answerability_handling": {**refusal,
                                   "appropriate_rate_among_decided": refusal_counts.get("appropriate", 0) / refusal_decided if refusal_decided else None},
        "answerability_refusal_decision": {
            "accuracy_against_question_labels": refusal_correct / refusal_decision_scored if refusal_decision_scored else None,
            "scored_rows": refusal_decision_scored,
            "skipped_failed_rows": refusal_decision_skipped_failed,
        },
    }


def score_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize only explicit human labels; blank/uncertain rows aren't correct."""
    _validate_annotations(rows)
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
        by_system[str(source.get("system", "unknown"))].append(row)
    return {"schema_version": 1, "total_rows": len(rows), "overall": _score_group(rows),
            "by_system": {system: _score_group(group) for system, group in sorted(by_system.items())}}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        rows.append(value)
    _validate_annotations(rows)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; refusing to overwrite annotations (use --force only for a fresh export)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    e2e = commands.add_parser("export-e2e", help="export an end-to-end report for manual review")
    e2e.add_argument("report", type=Path)
    e2e.add_argument("output", type=Path)
    e2e.add_argument("--questions", type=Path, help="override the question JSONL used to join answerability labels")
    e2e.add_argument("--force", action="store_true")
    step5 = commands.add_parser("export-step5", help="export full-corpus Step 5 answers for manual review")
    step5.add_argument("summary", type=Path)
    step5.add_argument("details_dir", type=Path)
    step5.add_argument("output", type=Path)
    step5.add_argument("--force", action="store_true")
    score = commands.add_parser("score", help="score a filled JSONL review file")
    score.add_argument("reviewed", type=Path)
    score.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export-e2e":
        raw = args.report.read_bytes()
        rows = export_e2e_report(json.loads(raw.decode("utf-8")), args.report.name, _digest(raw), args.questions)
        _write_jsonl(args.output, rows, force=args.force)
        print(f"Exported {len(rows)} answer rows to {args.output}")
        return 0
    if args.command == "export-step5":
        raw = args.summary.read_bytes()
        rows = export_step5_report(json.loads(raw.decode("utf-8")), args.details_dir,
                                   args.summary.name, _digest(raw))
        _write_jsonl(args.output, rows, force=args.force)
        print(f"Exported {len(rows)} answer rows to {args.output}")
        return 0
    rows = _read_jsonl(args.reviewed)
    result = score_rows(rows)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"{args.output} already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
