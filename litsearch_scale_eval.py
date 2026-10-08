"""Offline parent-level retrieval evaluation for a local full-text corpus slice."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from litagent.evaluation import evaluate
from litagent.litsearch_data_preparation import DataPreparationModule
from litagent.retrieval import load_index
from litsearch_fulltext import BGE_MODEL, MODEL_CACHE, _digest, _rank_child_hits


def run_local_eval(*, parent_path: Path, child_path: Path, query_path: Path,
                   bm25_path: Path | None = None, dense_path: Path | None = None,
                   retriever: str = "bm25", top_k: int = 5, candidate_k: int = 100,
                   output_path: Path | None = None) -> dict:
    if retriever not in {"bm25", "dense", "hybrid"}:
        raise ValueError("retriever 必须是 bm25、dense 或 hybrid")
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("top_k 必须大于 0，candidate_k 不得小于 top_k")
    parent_path, child_path, query_path = map(Path, (parent_path, child_path, query_path))
    parent_manifest = parent_path.with_name("corpus_fulltext_manifest.json")
    child_manifest = child_path.with_suffix(".manifest.json")
    if not parent_manifest.is_file() or not child_manifest.is_file():
        raise FileNotFoundError("缺少父文档或子块 manifest")
    parent_meta = json.loads(parent_manifest.read_text(encoding="utf-8"))
    child_meta = json.loads(child_manifest.read_text(encoding="utf-8"))
    parent_sha, child_sha = _digest(parent_path), _digest(child_path)
    if (parent_sha != parent_meta.get("parent_sha256")
            or parent_sha != child_meta.get("parent_sha256")
            or child_sha != child_meta.get("chunk_sha256")):
        raise ValueError("父文档或子块与 manifest 哈希不一致")

    ids = set()
    with parent_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.add(str(json.loads(line)["paper_id"]))
    queries = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    eligible = [row for row in queries if row.get("gold_paper_ids")
                and set(map(str, row["gold_paper_ids"])).issubset(ids)]
    partial = sum(bool(set(map(str, row.get("gold_paper_ids", []))) & ids)
                  and row not in eligible for row in queries)
    if not eligible:
        raise ValueError("当前本地语料没有完整金标的官方查询，不能评分")

    bm25 = dense = model = None
    if retriever in {"bm25", "hybrid"}:
        if bm25_path is None:
            raise ValueError("BM25 评测需要 --bm25")
        bm25 = load_index(Path(bm25_path))
        if bm25["corpus_sha256"] != child_sha:
            raise ValueError("BM25 索引与子块文件不一致")
    if retriever in {"dense", "hybrid"}:
        if dense_path is None:
            raise ValueError("Dense 评测需要 --dense")
        from litagent.dense import load_dense_index, load_model

        dense = load_dense_index(Path(dense_path))
        if dense["corpus_sha256"] != child_sha or dense["model_name"] != BGE_MODEL:
            raise ValueError("Dense 索引模型或子块文件不匹配")
        model = load_model(BGE_MODEL, MODEL_CACHE, offline=True)

    preparer = DataPreparationModule(parent_path)
    ranked: dict[str, list[str]] = {}
    rows = []
    started = time.perf_counter()
    for row in eligible:
        hits = _rank_child_hits(row["question"], candidate_k, retriever, bm25, dense, model)
        parents = preparer.get_parent_documents(hits)
        parent_ids = [paper["paper_id"] for paper in parents[:top_k]]
        ranked[row["id"]] = parent_ids
        rows.append({"id": row["id"], "query_set": row.get("query_set"),
                     "gold_paper_ids": list(map(str, row["gold_paper_ids"])),
                     "retrieved_paper_ids": parent_ids,
                     "retrieved_child_count": len(hits)})
    elapsed = time.perf_counter() - started
    by_question = {row["question"]: ranked[row["id"]] for row in eligible}
    scores = evaluate({}, eligible, top_k, lambda question, limit: [
        {"paper_id": paper_id} for paper_id in by_question[question][:limit]])
    report = {
        "run": {"scope": "local sequential full-text slice; complete official qrels only",
                "parent_path": str(parent_path), "child_path": str(child_path),
                "parent_sha256": parent_sha, "chunk_sha256": child_sha,
                "queries_sha256": _digest(query_path),
                "parent_count": len(ids), "chunk_count": child_meta.get("chunks"),
                "retriever": retriever, "top_k": top_k, "candidate_k": candidate_k,
                "complete_qrel_queries": len(eligible), "partial_qrel_queries_excluded": partial,
                "query_sets": sorted({str(row.get("query_set")) for row in eligible}),
                "search_seconds": round(elapsed, 3),
                "warning": "Sequential local slice and its qrels do not represent the full LitSearch corpus."},
        "scores": scores, "rows": rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        report["report_path"] = str(output_path)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="本地全文切片的离线父论文检索评测")
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("data/litsearch/queries.jsonl"))
    parser.add_argument("--bm25", type=Path)
    parser.add_argument("--dense", type=Path)
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = run_local_eval(parent_path=args.parents, child_path=args.chunks,
                                query_path=args.queries, bm25_path=args.bm25,
                                dense_path=args.dense, retriever=args.retriever,
                                top_k=args.top_k, candidate_k=args.candidate_k,
                                output_path=args.out)
        score = report["scores"]
        print(f"完整金标 {report['run']['complete_qrel_queries']} 题："
              f"Recall@{args.top_k}={score['recall_at_k']:.3f} "
              f"MRR={score['mrr_at_k']:.3f} nDCG={score['ndcg_at_k']:.3f}")
        print(f"检索耗时 {report['run']['search_seconds']} 秒")
        if args.out:
            print(f"报告：{args.out}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
