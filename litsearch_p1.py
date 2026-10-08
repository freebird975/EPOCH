"""Paper-understanding and related-work workflows over LitSearch full text."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from litagent.fulltext_answer import estimate_flash_cost_range_usd
from litagent.paper_understanding import FIELDS, understand_paper
from litagent.related_work import synthesize_related_work
from litagent.runtime import BudgetExceeded, RunRuntime, RuntimePolicy, timed, use_runtime
from litagent.source_links import semantic_scholar_record_api_url
from litsearch_fulltext import (CHILD_DATA, CHILD_INDEX, DENSE_CHILD_INDEX,
                                PARENT_DATA, _digest, retrieve_parents)
from rag.generate import capture_api_calls


RUN_ROOT = Path("data/litsearch/runs/p1")
PAPER_PROMPT_VERSION = "paper-understanding-v1"
RELATED_PROMPT_VERSION = "related-work-v1"


class _BudgetGuard:
    def __init__(self):
        self.error: BudgetExceeded | None = None

    def call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BudgetExceeded as exc:
            self.error = exc
            raise

    def check(self):
        if self.error is not None:
            raise self.error


def _provenance(parent_path: Path, child_path: Path, *, index_path: Path | None = None,
                dense_index_path: Path | None = None, parameters: dict | None = None) -> dict:
    from litagent.provenance import collect_provenance, sha256_file

    result = collect_provenance(parent_path, child_path, index_path, dense_index_path,
                                parameters=parameters)
    result.setdefault("code_sha256", {})["litsearch_p1.py"] = sha256_file(Path(__file__))
    result["prompt_versions"] = {"paper_understanding": PAPER_PROMPT_VERSION,
                                  "related_work": RELATED_PROMPT_VERSION}
    return result


def _runtime_report(runtime: RunRuntime) -> dict:
    return {"runtime": runtime.summary(), "timings": runtime.timings,
            "degradations": runtime.degradations, "degraded": bool(runtime.degradations)}


def _policy(deadline_seconds: float | None, api_timeout_seconds: float | None,
            max_api_calls: int | None, max_completion_tokens: int | None,
            failure_policy: str | None, max_first_queries: int | None,
            max_followup_queries: int | None, max_retrieval_rounds: int | None) -> RuntimePolicy:
    defaults = RuntimePolicy()
    return RuntimePolicy(
        deadline_seconds=deadline_seconds if deadline_seconds is not None else defaults.deadline_seconds,
        api_timeout_seconds=api_timeout_seconds if api_timeout_seconds is not None else defaults.api_timeout_seconds,
        max_api_calls=max_api_calls if max_api_calls is not None else defaults.max_api_calls,
        max_completion_tokens=max_completion_tokens if max_completion_tokens is not None else defaults.max_completion_tokens,
        failure_policy=failure_policy if failure_policy is not None else defaults.failure_policy,
        max_first_queries=max_first_queries if max_first_queries is not None else defaults.max_first_queries,
        max_followup_queries=max_followup_queries if max_followup_queries is not None else defaults.max_followup_queries,
        max_retrieval_rounds=max_retrieval_rounds if max_retrieval_rounds is not None else defaults.max_retrieval_rounds,
    )


def _save(report: dict, output_path: Path | None, kind: str) -> dict:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RUN_ROOT / f"{kind}-{stamp}-{uuid4().hex[:8]}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    report["report_path"] = str(output_path)
    return report


def load_paper_and_chunks(paper_id: str, parent_path: Path = PARENT_DATA,
                          child_path: Path = CHILD_DATA) -> tuple[dict, list[dict]]:
    manifest_path = Path(child_path).with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("缺少子块 manifest；请先切块并验证父子语料版本")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (_digest(Path(parent_path)) != manifest.get("parent_sha256")
            or _digest(Path(child_path)) != manifest.get("chunk_sha256")):
        raise ValueError("论文理解的父/子语料与 manifest 不一致，请重新切块")
    paper_id = str(paper_id).strip()
    if not paper_id:
        raise ValueError("论文 ID 不能为空")
    parent = None
    with Path(parent_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if str(row.get("paper_id", row.get("corpusid", ""))) == paper_id:
                    parent = row
                    break
    if parent is None:
        raise ValueError(f"全文父文档中没有论文 {paper_id}")
    chunks = []
    with Path(child_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if str(row.get("parent_id", "")) == paper_id:
                    chunks.append(row)
    chunks.sort(key=lambda row: (int(row.get("chunk_index", 0)), str(row.get("chunk_id", ""))))
    if not chunks:
        raise ValueError(f"论文 {paper_id} 没有可用子块")
    return parent, chunks


def partition_paper_chunks(chunks: list[dict], *, window_chars: int = 28000,
                           max_windows: int | None = None) -> tuple[list[list[dict]], int]:
    """Partition all exact chunks in source order; explicitly report omissions."""
    if window_chars < 500 or window_chars > 80000:
        raise ValueError("window_chars 必须在 500 到 80000 之间")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows 必须大于 0")
    windows: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for chunk in chunks:
        body = str(chunk.get("text", chunk.get("abstract", "")))
        if not body or len(body) > window_chars:
            raise ValueError(f"子块 {chunk.get('chunk_id')} 超出窗口长度或为空")
        if current and used + len(body) > window_chars:
            windows.append(current)
            current, used = [], 0
        current.append(chunk)
        used += len(body)
    if current:
        windows.append(current)
    selected = windows[:max_windows] if max_windows is not None else windows
    omitted = sum(len(window) for window in windows[len(selected):])
    return selected, omitted


def run_paper_understanding(paper_id: str, *, parent_path: Path = PARENT_DATA,
                            child_path: Path = CHILD_DATA, window_chars: int = 28000,
                            max_windows: int | None = None, output_path: Path | None = None,
                            completion_fn=None, verifier_fn=None, dry_run: bool = False,
                            runtime_policy: RuntimePolicy | None = None) -> dict:
    parent, chunks = load_paper_and_chunks(paper_id, parent_path, child_path)
    windows, omitted = partition_paper_chunks(chunks, window_chars=window_chars,
                                               max_windows=max_windows)
    report = {
        "schema_version": 1, "task": "paper_understanding",
        "paper_id": str(parent["paper_id"]), "title": parent.get("title", ""),
        "source_url": parent.get("source_url"), "source_record": parent.get("source_record"),
        "paper_record_api_url": semantic_scholar_record_api_url(parent),
        "parent_path": str(parent_path), "child_path": str(child_path),
        "parent_sha256": _digest(Path(parent_path)), "child_sha256": _digest(Path(child_path)),
        "total_chunks": len(chunks), "selected_chunks": sum(map(len, windows)),
        "omitted_chunks": omitted, "window_count": len(windows),
        "window_chars": window_chars,
        "selected_chunk_ids": [[str(chunk["chunk_id"]) for chunk in window] for window in windows],
        "fields": {field: [] for field in FIELDS}, "window_statuses": [],
        "status": "dry_run" if dry_run else "started", "api_calls": [],
        "prompt_version": PAPER_PROMPT_VERSION,
    }
    if dry_run:
        runtime = RunRuntime(runtime_policy)
        report.update(_runtime_report(runtime))
        report["provenance"] = _provenance(parent_path, child_path,
                                            parameters={"window_chars": window_chars,
                                                        "max_windows": max_windows,
                                                        "runtime_policy": runtime.summary()["policy"]})
        return _save(report, output_path, "understand")
    seen_claims = {field: set() for field in FIELDS}
    runtime = RunRuntime(runtime_policy)
    budget_guard = _BudgetGuard()
    report["failed_windows"] = []
    attempted_window_numbers: list[int] = []
    try:
        with use_runtime(runtime), capture_api_calls() as calls:
            if completion_fn is None:
                from litagent.paper_understanding import _default_completion
                effective_completion = lambda messages, **kwargs: budget_guard.call(
                    _default_completion, messages, **kwargs)
            else:
                effective_completion = lambda messages, **kwargs: budget_guard.call(
                    completion_fn, messages, **kwargs)
            effective_verifier = (lambda *a, **kw: budget_guard.call(verifier_fn, *a, **kw)) if verifier_fn else None
            for window_number, window in enumerate(windows, 1):
                attempted_window_numbers.append(window_number)
                try:
                    runtime.check()
                    with timed("paper_understanding_window", window=window_number,
                               chunks=len(window)):
                        result = understand_paper(parent, window, completion_fn=effective_completion,
                                                  verifier_fn=effective_verifier,
                                                  max_context_chars=window_chars)
                        # understand_paper intentionally converts model errors to failed cards;
                        # recheck here so budget exceptions swallowed by that layer still stop work.
                        runtime.check()
                        budget_guard.check()
                    report["window_statuses"].append(result["status"])
                    if result["status"] == "failed":
                        report["failed_windows"].append(window_number)
                    for field in FIELDS:
                        for item in result["fields"][field]:
                            key = item["claim"].casefold().strip()
                            if key not in seen_claims[field]:
                                report["fields"][field].append(item)
                                seen_claims[field].add(key)
                    report["dropped_claim_count"] = report.get("dropped_claim_count", 0) + result.get("dropped_claim_count", 0)
                except BudgetExceeded as exc:
                    report["failed_windows"].append(window_number)
                    report["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
                    break
            report["api_calls"] = list(calls)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:300]}
        report["api_calls"] = list(locals().get("calls", []))
    report["cost"] = estimate_flash_cost_range_usd(report["api_calls"])
    attempted = len(attempted_window_numbers)
    unattempted_omissions = sum(len(window) for window in windows[attempted:])
    failed_chunk_count = sum(len(windows[number - 1]) for number in report["failed_windows"])
    report["omitted_chunks"] = omitted + unattempted_omissions + failed_chunk_count
    report["processed_chunks"] = sum(len(windows[number - 1])
                                     for number, status in enumerate(report["window_statuses"], 1)
                                     if status != "failed")
    report["attempted_windows"] = attempted
    report["completed_windows"] = sum(status != "failed" for status in report["window_statuses"])
    report.update(_runtime_report(runtime))
    report["provenance"] = _provenance(parent_path, child_path,
                                        parameters={"window_chars": window_chars,
                                                    "max_windows": max_windows,
                                                    "runtime_policy": runtime.summary()["policy"]})
    failed_windows = len(report["failed_windows"])
    if failed_windows == len(windows):
        report["status"] = "failed"
    elif report.get("failure") and not report["completed_windows"]:
        report["status"] = "failed"
    elif omitted or unattempted_omissions or failed_windows or report.get("failure"):
        report["status"] = "partial"
    else:
        report["status"] = "verified" if any(report["fields"].values()) else "insufficient_evidence"
    return _save(report, output_path, "understand")


def run_related_work(topic: str, *, top_k: int = 6, candidate_k: int = 100,
                     retriever: str = "hybrid", parent_path: Path = PARENT_DATA,
                     child_path: Path = CHILD_DATA, index_path: Path = CHILD_INDEX,
                     dense_index_path: Path = DENSE_CHILD_INDEX,
                     output_path: Path | None = None, complete=None, verifier_fn=None,
                     dry_run: bool = False, retrieve_fn=None,
                     runtime_policy: RuntimePolicy | None = None) -> dict:
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("top_k 必须大于 0，candidate_k 不得小于 top_k")
    retrieve_fn = retrieve_fn or retrieve_parents
    trace: dict = {}
    runtime = RunRuntime(runtime_policy)
    parents, child_hits, synthesis = [], [], None
    failure_type = None
    try:
        with use_runtime(runtime), capture_api_calls() as calls:
            with timed("related_work_retrieval"):
                parents, child_hits = retrieve_fn(
                    topic, top_k, candidate_k, parent_path=parent_path, child_path=child_path,
                    index_path=index_path, dense_index_path=dense_index_path,
                    retriever=retriever, pipeline="classic", rerank=False, trace=trace,
                )
            if not dry_run:
                with timed("related_work_generation_and_verification"):
                    synthesis = synthesize_related_work(topic, parents, complete=complete,
                                                        verifier_fn=verifier_fn)
            api_calls = list(calls)
    except Exception as exc:
        failure_type = type(exc).__name__
        api_calls = list(locals().get("calls", []))
    report = {
        "schema_version": 1, "task": "related_work", "topic": topic,
        "retriever": retriever, "pipeline": "classic", "top_k": top_k,
        "candidate_k": candidate_k, "retrieval_trace": trace,
        "candidate_parents": [{"paper_id": parent["paper_id"],
                               "title": parent.get("title", ""),
                               "source_url": parent.get("source_url", ""),
                               "matched_chunk_ids": [chunk.get("chunk_id") for chunk in parent.get("matched_chunks", [])]}
                              for parent in parents],
        "retrieved_child_count": len(child_hits),
        "parent_sha256": _digest(Path(parent_path)), "child_sha256": _digest(Path(child_path)),
        "status": "failed" if failure_type else "dry_run" if dry_run else "completed",
        "failure_type": failure_type,
        "result": synthesis, "api_calls": api_calls,
        "cost": estimate_flash_cost_range_usd(api_calls),
        "prompt_version": RELATED_PROMPT_VERSION,
    }
    report.update(_runtime_report(runtime))
    report["provenance"] = _provenance(parent_path, child_path, index_path=index_path,
                                        dense_index_path=dense_index_path if retriever in {"dense", "hybrid"} else None,
                                        parameters={"top_k": top_k, "candidate_k": candidate_k,
                                                    "retriever": retriever,
                                                    "runtime_policy": runtime.summary()["policy"]})
    return _save(report, output_path, "related")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="LitSearch P1：论文理解与 Related Work")
    commands = parser.add_subparsers(dest="command", required=True)
    paper = commands.add_parser("understand", help="为单篇论文生成有原文证据的研究卡片")
    paper.add_argument("paper_id")
    paper.add_argument("--parents", type=Path, default=PARENT_DATA)
    paper.add_argument("--chunks", type=Path, default=CHILD_DATA)
    paper.add_argument("--window-chars", type=int, default=28000)
    paper.add_argument("--max-windows", type=int)
    paper.add_argument("--out", type=Path)
    paper.add_argument("--dry-run", action="store_true")
    related = commands.add_parser("related", help="检索并生成带证据的相关工作比较")
    related.add_argument("topic")
    related.add_argument("--top-k", type=int, default=6)
    related.add_argument("--candidate-k", type=int, default=100)
    related.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    related.add_argument("--parents", type=Path, default=PARENT_DATA)
    related.add_argument("--chunks", type=Path, default=CHILD_DATA)
    related.add_argument("--index", type=Path, default=CHILD_INDEX)
    related.add_argument("--dense-index", type=Path, default=DENSE_CHILD_INDEX)
    related.add_argument("--out", type=Path)
    related.add_argument("--dry-run", action="store_true")
    for command in (paper, related):
        command.add_argument("--deadline-seconds", type=float)
        command.add_argument("--api-timeout", type=float)
        command.add_argument("--max-api-calls", type=int)
        command.add_argument("--max-completion-tokens", type=int)
        command.add_argument("--failure-policy", choices=("strict", "degrade"))
        command.add_argument("--max-first-queries", type=int)
        command.add_argument("--max-followup-queries", type=int)
        command.add_argument("--max-retrieval-rounds", type=int, choices=(1, 2))
    args = parser.parse_args()
    try:
        policy = _policy(args.deadline_seconds, args.api_timeout, args.max_api_calls,
                         args.max_completion_tokens, args.failure_policy,
                         args.max_first_queries, args.max_followup_queries,
                         args.max_retrieval_rounds)
        if args.command == "understand":
            result = run_paper_understanding(
                args.paper_id, parent_path=args.parents, child_path=args.chunks,
                window_chars=args.window_chars, max_windows=args.max_windows,
                output_path=args.out, dry_run=args.dry_run, runtime_policy=policy)
            print(f"{result['title']} · {result['status']} · 来源块 {result['selected_chunks']}/{result['total_chunks']}")
            if not args.dry_run:
                for field in FIELDS:
                    print(f"{field}: {len(result['fields'][field])} 项")
        else:
            result = run_related_work(
                args.topic, top_k=args.top_k, candidate_k=args.candidate_k,
                retriever=args.retriever, parent_path=args.parents, child_path=args.chunks,
                index_path=args.index, dense_index_path=args.dense_index,
                output_path=args.out, dry_run=args.dry_run, runtime_policy=policy)
            print(f"相关工作候选 {len(result['candidate_parents'])} 篇 · {result['status']}")
            if not args.dry_run and result["result"]:
                print(f"有证据论文 {len(result['result']['papers'])} 篇，方法组 {len(result['result']['groups'])} 组")
        print(f"报告：{result['report_path']}")
        return 1 if result["status"] == "failed" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
