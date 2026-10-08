"""Offline full-corpus retrieval checks with independent source and rank verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import file_sha256, file_signature
from .hybrid import DEFAULT_RRF_K
from .runtime import RunRuntime, RuntimePolicy, use_runtime
from litsearch_fulltext import close_retrieval_resources, retrieve_parents

MODES = ("bm25", "dense", "hybrid")
CHILD_FIELDS = ("paper_id", "parent_id", "chunk_id", "chunk_index", "title", "section",
                "abstract", "source_url", "evidence_scope", "doc_type", "section_path",
                "paragraph_type", "start_line", "end_line", "location_type")


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_questions(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not row["id"] or row["id"] in seen
                    or not isinstance(row.get("question"), str) or not row["question"].strip()
                    or re.search(r"[\u3400-\u9fff]", row["question"])):
                raise ValueError("冒烟题集要求唯一 ID 和非空英文问题；中文查询需要单独的翻译链路")
            gold = row.get("gold_paper_ids", [])
            if not isinstance(gold, list) or any(not isinstance(item, str) or not item for item in gold):
                raise ValueError("冒烟题集的 gold_paper_ids 必须是论文 ID 字符串列表")
            seen.add(row["id"])
            rows.append(row)
    if not 1 <= len(rows) <= 10:
        raise ValueError("冒烟检查需要 1–10 条问题；全量题集使用独立评测入口")
    return rows


def _summarize(parents: list[dict], hits: list[dict], top_k: int) -> tuple[list[dict], list[dict]]:
    if len(parents) > top_k or len({item["paper_id"] for item in parents}) != len(parents):
        raise ValueError("父文档数量或去重检查失败")
    if len({item["paper_id"] for item in hits}) != len(hits):
        raise ValueError("子块检索结果有重复 ID")
    parent_records = [{"paper_id": item["paper_id"], "title": item.get("title", ""),
                       "source_url": item.get("source_url"),
                       "full_text_sha256": hashlib.sha256(item["full_text"].encode("utf-8")).hexdigest(),
                       "best_chunk_rank": item.get("best_chunk_rank"),
                       "matched_chunks": item["matched_chunks"]} for item in parents]
    child_records = []
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit.get("score"), (int, float)) or not math.isfinite(hit["score"]):
            raise ValueError("检索结果缺少有效分数")
        child_records.append({**{key: hit.get(key) for key in CHILD_FIELDS}, "rank": rank,
                              **{key: hit.get(key) for key in
                                 ("score", "retrieval_method", "bm25_rank", "dense_rank", "rrf_rank", "rrf_score")}})
    return parent_records, child_records


def _check_rrf(runs: list[dict], questions: list[dict], candidate_k: int) -> dict:
    verified = dual_channel = 0
    for question in questions:
        by_mode = {run["retriever"]: run for run in runs if run["query_id"] == question["id"]}
        sparse = {hit["paper_id"]: rank for rank, hit in enumerate(by_mode["bm25"]["child_hits"], 1)}
        dense = {hit["paper_id"]: rank for rank, hit in enumerate(by_mode["dense"]["child_hits"], 1)}
        scores = {key: ((1 / (DEFAULT_RRF_K + sparse[key]) if key in sparse else 0)
                        + (1 / (DEFAULT_RRF_K + dense[key]) if key in dense else 0))
                  for key in sparse.keys() | dense.keys()}
        expected = sorted(scores, key=lambda key: (-scores[key], key))[:2 * candidate_k]
        actual = by_mode["hybrid"]["child_hits"]
        if [hit["paper_id"] for hit in actual] != expected:
            raise ValueError("Hybrid 排序与独立两路召回的 RRF 结果不一致")
        for rank, hit in enumerate(actual, 1):
            key = hit["paper_id"]
            if (hit.get("bm25_rank") != sparse.get(key) or hit.get("dense_rank") != dense.get(key)
                    or hit.get("rrf_rank") != rank
                    or abs(hit["score"] - round(scores[key], 8)) > 1e-8
                    or hit.get("rrf_score") != hit["score"]):
                raise ValueError(f"Hybrid 通道排名或 RRF 分数错误：{key}")
            dual_channel += int(key in sparse and key in dense)
            verified += 1
    return {"status": "passed", "verified_hybrid_hits": verified,
            "dual_channel_hits": dual_channel, "rrf_k": DEFAULT_RRF_K}


def _check_sources(runs: list[dict], questions: list[dict], parent_path: Path,
                   child_path: Path, dense_index: dict, top_k: int) -> tuple[dict, dict]:
    """Compare all returned records to original JSONL, not to another result list."""
    child_ids = {hit["paper_id"] for run in runs for hit in run["child_hits"]}
    originals = {}
    ordered_ids = sorted(child_ids)
    with child_path.open("rb") as source:
        for start in range(0, len(ordered_ids), 900):
            batch = ordered_ids[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = dense_index["docs_connection"].execute(
                f"SELECT paper_id, byte_offset FROM docs WHERE paper_id IN ({placeholders})", batch)
            for child_id, offset in rows:
                source.seek(offset)
                record = json.loads(source.readline())
                if record.get("paper_id") != child_id or child_id in originals:
                    raise ValueError("来源核验发现错误或重复子块映射")
                originals[child_id] = record
    if originals.keys() != child_ids:
        raise ValueError("检索子块在原始语料中缺失")
    wanted_parents = {str(record.get("parent_id", "")) for record in originals.values()}
    wanted_parents.update(gold for question in questions for gold in question.get("gold_paper_ids", []))
    source_parents = {}
    parent_rows = full_parent_rows = 0
    with parent_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            parent_rows += 1
            text = row.get("full_text") or row.get("full_paper") or ""
            full_parent_rows += int(bool(text.strip()))
            key = str(row.get("paper_id", row.get("corpusid", "")))
            if key in wanted_parents:
                if key in source_parents:
                    raise ValueError(f"原始父语料有重复论文 ID：{key}")
                source_parents[key] = {**row, "full_text": text}
    orphan = sum(str(item.get("parent_id", "")) not in source_parents for item in originals.values())
    if orphan:
        raise ValueError(f"来源核验发现 {orphan} 个孤儿子块")
    verified_hits = verified_parents = 0
    for run in runs:
        expected_order = list(dict.fromkeys(hit["parent_id"] for hit in run["child_hits"]))[:top_k]
        if [parent["paper_id"] for parent in run["parents"]] != expected_order:
            raise ValueError("父文档回溯排序或数量与子块命中不一致")
        for hit in run["child_hits"]:
            original = originals[hit["paper_id"]]
            if any(hit.get(key) != original.get(key) for key in CHILD_FIELDS):
                raise ValueError(f"子块文本、标题、章节或 ID 与原始语料不一致：{hit['paper_id']}")
            verified_hits += 1
        for parent in run["parents"]:
            original = source_parents[parent["paper_id"]]
            digest = hashlib.sha256(original["full_text"].encode("utf-8")).hexdigest()
            if (parent["title"] != original.get("title", "") or parent["full_text_sha256"] != digest
                    or parent["source_url"] != original.get("source_url", "https://huggingface.co/datasets/princeton-nlp/LitSearch")):
                raise ValueError(f"回溯父文档与原始记录不一致：{parent['paper_id']}")
            expected_matches = [hit for hit in run["child_hits"] if hit["parent_id"] == parent["paper_id"]]
            if (len(parent["matched_chunks"]) != len(expected_matches)
                    or parent["best_chunk_rank"] != expected_matches[0]["rank"]):
                raise ValueError("父文档的命中子块数量或最早排名不一致")
            for match, hit in zip(parent["matched_chunks"], expected_matches):
                if (match["chunk_id"] != hit["chunk_id"] or match["text"] != hit["abstract"]
                        or match["section"] != hit["section"] or match["rank"] != hit["rank"]
                        or any(match.get(key) != hit.get(key) for key in ("bm25_rank", "dense_rank", "rrf_rank"))):
                    raise ValueError("父文档的命中子块或排名信息丢失")
            verified_parents += 1
    gold_coverage = {row["id"]: {"gold_paper_ids": row.get("gold_paper_ids", []),
                               "gold_with_full_text": [key for key in row.get("gold_paper_ids", [])
                                                       if source_parents.get(key, {}).get("full_text", "").strip()]}
                     for row in questions}
    return ({"status": "passed", "verified_child_hits": verified_hits,
             "unique_verified_chunks": len(originals), "verified_parent_results": verified_parents,
             "unique_hit_parents": len({item["parent_id"] for item in originals.values()}),
             "orphan_child_hits": orphan, "source_parent_rows": parent_rows,
             "source_full_text_parents": full_parent_rows}, gold_coverage)


def run_fulltext_smoke(parent_path: Path, child_path: Path, bm25_path: Path, dense_path: Path,
                       *, dense_docs_path: Path, queries_path: Path, report_path: Path,
                       log_path: Path | None = None, top_k: int = 5, candidate_k: int = 20,
                       deadline_seconds: float = 900, progress=None) -> dict:
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("冒烟检查要求 candidate_k >= top_k > 0")
    paths = {"parents": Path(parent_path).resolve(), "chunks": Path(child_path).resolve(),
             "bm25": Path(bm25_path).resolve(), "dense": Path(dense_path).resolve(),
             "dense_docs": Path(dense_docs_path).resolve(), "questions": Path(queries_path).resolve()}
    paths.update({"chunk_manifest": paths["chunks"].with_suffix(".manifest.json"),
                  "dense_metadata": paths["dense"].with_suffix(".meta.json")})
    report_path = Path(report_path).resolve()
    log_path = Path(log_path).resolve() if log_path else report_path.with_suffix(".jsonl")
    if report_path == log_path or report_path in paths.values() or log_path in paths.values():
        raise ValueError("报告和日志不能覆盖输入文件，且必须使用不同路径")
    questions = _load_questions(paths["questions"])
    policy = RuntimePolicy(deadline_seconds=deadline_seconds, failure_policy="strict")
    signatures = {name: file_signature(path) for name, path in paths.items() if path.is_file()}
    resources = {}
    started = time.perf_counter()
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "full-corpus source/rank smoke checks; not a retrieval quality benchmark",
              "parameters": {"top_k": top_k, "candidate_k_per_channel": candidate_k,
                             "rrf_k": DEFAULT_RRF_K, "pipeline": "classic", "rerank": False,
                             "build_missing_indexes": False, "deadline_seconds_per_call": deadline_seconds},
              "questions": questions, "runs": [], "negative_checks": [], "checks": {}, "api_attempts": 0}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, report)

    def announce(message):
        if progress is not None:
            progress(message)

    def retrieve(question, mode, **overrides):
        runtime = RunRuntime(policy)
        start = time.perf_counter()
        try:
            with use_runtime(runtime):
                parents, hits = retrieve_parents(
                    question, top_k, candidate_k, parent_path=paths["parents"], child_path=paths["chunks"],
                    index_path=overrides.get("index_path", paths["bm25"]), dense_index_path=paths["dense"],
                    dense_docs_path=paths["dense_docs"], retriever=mode, pipeline="classic", rerank=False,
                    resources=resources, build_missing_indexes=False)
            if runtime.api_attempts or runtime.degradations:
                raise ValueError("离线冒烟检查不允许 API 调用或检索降级")
            return parents, hits, round(time.perf_counter() - start, 6), runtime.timings
        finally:
            report["api_attempts"] += runtime.api_attempts

    try:
        with log_path.open("w", encoding="utf-8") as log:
            for question in questions:
                for mode in MODES:
                    announce(f"检索 {question['id']} / {mode}…")
                    parents, hits, seconds, timings = retrieve(question["question"], mode)
                    parent_records, child_records = _summarize(parents, hits, top_k)
                    gold = set(question.get("gold_paper_ids", []))
                    row = {"query_id": question["id"], "question": question["question"], "retriever": mode,
                           "status": "ok" if hits else "no_evidence", "elapsed_seconds": seconds,
                           "recall_seconds": round(sum(item["seconds"] for item in timings
                                                        if item["stage"] in {"bm25_recall", "dense_recall"}), 6),
                           "timings": timings, "parents": parent_records, "child_hits": child_records,
                           "gold_hit_parent_ids": sorted(gold & {item["paper_id"] for item in parents}),
                           "gold_ids_in_child_pool": sorted(gold & {item["parent_id"] for item in hits})}
                    report["runs"].append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    _write_report(report_path, report)
                    announce(f"完成 {question['id']} / {mode}：{len(parents)} 篇父论文，{len(hits)} 个子块，{seconds:.3f}s")
            announce("检查空查询、无词项命中及缺失索引…")
            for mode in MODES:
                parents, hits, _, _ = retrieve("", mode)
                if parents or hits:
                    raise ValueError("空查询必须明确返回无证据")
                report["negative_checks"].append({"case": "empty_query", "retriever": mode,
                                                  "status": "passed", "result": "no_evidence"})
            parents, hits, _, _ = retrieve("zzzxqvsmokenomatchqvxzz", "bm25")
            if parents or hits:
                raise ValueError("无词项命中检查出现意外结果")
            report["negative_checks"].append({"case": "unmatched_lexical_query", "retriever": "bm25",
                                              "status": "passed", "result": "no_evidence"})
            missing = paths["bm25"].with_name(f".missing-smoke-{uuid4().hex}.sqlite3")
            try:
                retrieve("graph retrieval", "bm25", index_path=missing)
            except FileNotFoundError:
                report["negative_checks"].append({"case": "missing_index", "status": "passed",
                                                  "result": "FileNotFoundError", "file_created": missing.exists()})
                if missing.exists():
                    raise ValueError("缺失索引检查意外创建了文件")
            else:
                raise ValueError("缺失索引必须报错停止")
            announce("独立核对子块原文、父论文与 RRF 通道排名…")
            report["checks"]["rrf_provenance"] = _check_rrf(report["runs"], questions, candidate_k)
            integrity, coverage = _check_sources(report["runs"], questions, paths["parents"], paths["chunks"],
                                                resources["dense"][1], top_k)
            report["checks"]["source_integrity"] = integrity
            report["query_gold"] = coverage
            report["indexes"] = {"bm25_documents": resources["bm25"][1]["documents"],
                                  "dense_documents": int(resources["dense"][1]["faiss_index"].ntotal),
                                  "dimension": resources["dense"][1]["dimension"],
                                  "model_name": resources["dense"][1]["model_name"],
                                  "model_snapshot": resources["dense"][1]["model_snapshot"]}
            announce("记录哈希和执行结果…")
            cache = resources["validation_cache"]
            report["artifacts"] = {name: {"path": str(path), "bytes": signatures[name][1],
                                         "sha256": file_sha256(path, cache=cache)}
                                   for name, path in paths.items()}
            if any(file_signature(path) != signatures[name] for name, path in paths.items()):
                raise ValueError("冒烟检查期间输入产物发生变化")
            report["checks"]["artifacts_unchanged"] = True
            root = Path(__file__).resolve().parents[1]
            report["code_sha256"] = {name: file_sha256(root / name) for name in
                                     ("litagent/fulltext_smoke.py", "litsearch_fulltext.py", "litagent/retrieval.py",
                                      "litagent/dense.py", "litagent/hybrid.py", "litagent/litsearch_data_preparation.py")}
            for check in report["negative_checks"]:
                log.write(json.dumps({"type": "negative_check", **check}, ensure_ascii=False) + "\n")
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__,
                             "reason": str(exc)[:500] if isinstance(exc, (ValueError, OSError, RuntimeError)) else "unexpected_error"}
        announce(f"冒烟检查停止：{report['failure']['type']}，{report['failure']['reason']}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"type": "failure", **report["failure"]}, ensure_ascii=False) + "\n")
    finally:
        close_retrieval_resources(resources)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["report_path"], report["log_path"] = str(report_path), str(log_path)
        _write_report(report_path, report)
    return report
