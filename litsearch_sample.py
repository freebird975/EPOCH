"""Small LitSearch hard-negative candidate-subset comparison.

The resulting metrics describe retrieval inside a constructed candidate set,
not retrieval over the complete LitSearch corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from litagent.evaluation import evaluate
from litagent.hybrid import fuse_rrf
from litagent.retrieval import build_index, load_index, search


ROOT = Path("data/litsearch")
SAMPLE_ROOT = ROOT / "sample"
SOURCE_CORPUS = ROOT / "corpus.jsonl"
SOURCE_QUERIES = ROOT / "queries.jsonl"
FULL_BM25_REPORT = Path("eval/litsearch_bm25.json")
SEED = 20260929
QUERY_SET_SAMPLE_SIZE = 3
BM25_NEGATIVES_PER_QUERY = 8
MODEL_NAMES = {
    "harrier": "microsoft/harrier-oss-v1-0.6b",
    "bge": "BAAI/bge-small-en-v1.5",
}
MODEL_CACHE = Path(".rag/models")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _paper_text(paper: dict) -> str:
    return " ".join((paper.get("title", "") + " " + paper.get("abstract", "")).split())


def prepare(
    root: Path = ROOT,
    sample_root: Path = SAMPLE_ROOT,
    bm25_report: Path = FULL_BM25_REPORT,
) -> dict:
    """Create the deterministic candidate set from official data and full BM25 ranks."""
    for path in (root / "manifest.json", bm25_report):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件: {path}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    actual_corpus_hash = _digest(root / "corpus.jsonl")
    actual_queries_hash = _digest(root / "queries.jsonl")
    if actual_corpus_hash != manifest.get("corpus_sha256") or actual_queries_hash != manifest.get("queries_sha256"):
        raise ValueError("LitSearch 原始语料/查询与其 manifest 哈希不符")

    full_report = json.loads(bm25_report.read_text(encoding="utf-8"))
    full_run = full_report.get("run", {})
    if (full_run.get("corpus_sha256") != actual_corpus_hash
            or full_run.get("queries_sha256") != actual_queries_hash
            or full_run.get("top_k") != 20):
        raise ValueError("全量 BM25 报告与当前 LitSearch 数据不匹配，或不是 Top-20 报告")
    full_rows = full_report.get("results", {}).get("20", {}).get("rows", [])
    if len(full_rows) != len(_read_jsonl(root / "queries.jsonl")):
        raise ValueError("全量 BM25 Top-20 逐题排名数量不匹配")
    ranks_by_id = {row.get("id"): row.get("retrieved_paper_ids", []) for row in full_rows}
    if len(ranks_by_id) != len(full_rows):
        raise ValueError("全量 BM25 排名中存在重复或缺失查询 ID")

    papers = _read_jsonl(root / "corpus.jsonl")
    papers_by_id = {paper["paper_id"]: paper for paper in papers}
    if len(papers_by_id) != len(papers):
        raise ValueError("原始语料中 paper_id 重复")
    queries = _read_jsonl(root / "queries.jsonl")
    eligible: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        gold = query.get("gold_paper_ids", [])
        if (len(gold) == 1 and gold[0] in papers_by_id and _paper_text(papers_by_id[gold[0]])
                and query.get("query_set")):
            eligible[str(query["query_set"])].append(query)

    rng = random.Random(SEED)
    selected: list[dict] = []
    shortages = {query_set: len(rows) for query_set, rows in eligible.items() if len(rows) < QUERY_SET_SAMPLE_SIZE}
    if shortages:
        raise ValueError(f"以下 query_set 可用单金标查询不足 {QUERY_SET_SAMPLE_SIZE} 条: {shortages}")
    for query_set in sorted(eligible):
        selected.extend(rng.sample(sorted(eligible[query_set], key=lambda row: row["id"]), QUERY_SET_SAMPLE_SIZE))
    if len(selected) != 12:
        raise ValueError(f"预期每个 4 个 query_set 选 3 条，共 12 条；实际为 {len(selected)} 条")

    candidate_ids: set[str] = set()
    candidate_ids_by_query: dict[str, list[str]] = {}
    for query in selected:
        query_id = query["id"]
        gold_ids = set(query["gold_paper_ids"])
        if query_id not in ranks_by_id:
            raise ValueError(f"全量 BM25 报告缺少查询 {query_id}")
        negatives = [paper_id for paper_id in ranks_by_id[query_id] if paper_id not in gold_ids][:BM25_NEGATIVES_PER_QUERY]
        if len(negatives) < BM25_NEGATIVES_PER_QUERY:
            raise ValueError(f"查询 {query_id} 的 BM25 非金标负例不足 {BM25_NEGATIVES_PER_QUERY} 篇")
        chosen = list(query["gold_paper_ids"]) + negatives
        unknown = set(chosen) - papers_by_id.keys()
        if unknown:
            raise ValueError(f"查询 {query_id} 的候选文档不在原始语料: {sorted(unknown)}")
        candidate_ids.update(chosen)
        candidate_ids_by_query[query_id] = chosen

    # Keep source corpus order so the sample bytes and document-ID tie breaks are reproducible.
    sample_papers = [paper for paper in papers if paper["paper_id"] in candidate_ids]
    selected_queries = selected
    sample_corpus_path = sample_root / "corpus.jsonl"
    sample_queries_path = sample_root / "queries.jsonl"
    _write_jsonl(sample_corpus_path, sample_papers)
    _write_jsonl(sample_queries_path, selected_queries)
    sample_manifest = {
        "dataset": "princeton-nlp/LitSearch",
        "revision": manifest.get("revision"),
        "description": "hard-negative candidate subset; metrics are not full-corpus LitSearch retrieval results",
        "construction_method": {
            "name": "seeded stratified query sample with full-corpus BM25 hard negatives",
            "seed": SEED,
            "query_selection": f"randomly select {QUERY_SET_SAMPLE_SIZE} eligible single-gold queries per query_set; gold title/abstract must be nonempty",
            "eligible_query_count": sum(map(len, eligible.values())),
            "selected_per_query_set": QUERY_SET_SAMPLE_SIZE,
            "query_sets": sorted(eligible),
            "candidate_rule": f"union of each selected query's gold document and first {BM25_NEGATIVES_PER_QUERY} non-gold IDs from its full-corpus BM25 Top-20 ranking",
            "candidate_document_count": len(sample_papers),
            "full_bm25_report_sha256": _digest(bm25_report),
        },
        "selected_query_ids": [query["id"] for query in selected_queries],
        "candidate_ids_by_query": candidate_ids_by_query,
        "original_corpus_sha256": actual_corpus_hash,
        "original_queries_sha256": actual_queries_hash,
        "sample_corpus_sha256": _digest(sample_corpus_path),
        "sample_queries_sha256": _digest(sample_queries_path),
        "sample_corpus_count": len(sample_papers),
        "sample_query_count": len(selected_queries),
    }
    (sample_root / "manifest.json").write_text(json.dumps(sample_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return sample_manifest


def _assert_sample(sample_root: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = sample_root / "manifest.json"
    corpus_path = sample_root / "corpus.jsonl"
    queries_path = sample_root / "queries.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _digest(corpus_path) != manifest.get("sample_corpus_sha256") or _digest(queries_path) != manifest.get("sample_queries_sha256"):
        raise ValueError("样本文件与 manifest 哈希不符；请重新运行 prepare")
    corpus = _read_jsonl(corpus_path)
    queries = _read_jsonl(queries_path)
    if len(corpus) != manifest.get("sample_corpus_count") or len(queries) != manifest.get("sample_query_count"):
        raise ValueError("样本文件条数与 manifest 不符")
    ids = {paper["paper_id"] for paper in corpus}
    if any(not set(query["gold_paper_ids"]).issubset(ids) for query in queries):
        raise ValueError("样本语料缺少官方金标文档")
    return manifest, corpus, queries


def run_eval(model_key: str, sample_root: Path = SAMPLE_ROOT, report_path: Path | None = None) -> dict:
    """Compare BM25, dense, and RRF hybrid on the same candidate documents."""
    if model_key not in MODEL_NAMES:
        raise ValueError(f"未知模型: {model_key}")
    from litagent.dense import build_dense_index, dense_search, load_dense_index, load_model

    manifest, _, rows = _assert_sample(sample_root)
    model_name = MODEL_NAMES[model_key]
    bm25_index_path = sample_root / "bm25_index.json"
    dense_index_path = sample_root / f"dense_index_{model_key}.faiss"
    if not bm25_index_path.is_file():
        print("建立样本候选集 BM25 索引…", flush=True)
        build_index(sample_root / "corpus.jsonl", bm25_index_path)
    bm25_index = load_index(bm25_index_path)
    if not dense_index_path.is_file():
        print(f"建立样本候选集 Dense 索引（{model_name}）…", flush=True)
        build_dense_index(sample_root / "corpus.jsonl", dense_index_path, model_name, MODEL_CACHE)
    dense_index = load_dense_index(dense_index_path)
    if dense_index.get("model_name") != model_name:
        raise ValueError("已缓存的 Dense 索引模型与所选模型不符，请删除该样本索引后重建")
    model = load_model(model_name, MODEL_CACHE, offline=True)

    started = time.perf_counter()
    ranked: dict[str, dict[str, list[str]]] = {method: {} for method in ("bm25", "dense", "hybrid")}
    for n, row in enumerate(rows, 1):
        question = row["question"]
        bm25_hits = search(bm25_index, question, 20)
        dense_hits = dense_search(dense_index, question, model, 20)
        hybrid_hits = fuse_rrf(bm25_hits, dense_hits, 20)
        for method, hits in (("bm25", bm25_hits), ("dense", dense_hits), ("hybrid", hybrid_hits)):
            ranked[method][question] = [hit["paper_id"] for hit in hits]
        print(f"已检索样本查询 {n}/{len(rows)}", flush=True)
    search_seconds = time.perf_counter() - started

    results: dict[str, dict[str, dict]] = {}
    for method in ranked:
        search_fn: Callable[[str, int], list[dict]] = lambda question, limit, cache=ranked[method]: [
            {"paper_id": paper_id} for paper_id in cache[question][:limit]
        ]
        results[method] = {str(k): evaluate({}, rows, k, search_fn) for k in (5, 20)}

    from litagent.dense import _snapshot
    report = {
        "run": {
            "dataset": "princeton-nlp/LitSearch",
            "revision": manifest.get("revision"),
            "scope": "hard-negative candidate subset",
            "scope_warning": "These results measure ranking within a constructed candidate subset. They are not complete LitSearch retrieval results and must not be presented as such.",
            "retriever_methods": ["BM25", "Dense", "Hybrid (RRF, k=60)"],
            "model_name": model_name,
            "model_snapshot": _snapshot(model),
            "embedding_backend": dense_index.get("embedding_backend"),
            "embedding_backend_version": dense_index.get("backend_version"),
            "embedding_mode": dense_index.get("embedding_mode"),
            "index_backend": dense_index.get("index_backend"),
            "index_type": dense_index.get("index_type"),
            "rrf_k": 60,
            "top_k": 20,
            "corpus_count": manifest["sample_corpus_count"],
            "query_count": len(rows),
            "original_corpus_sha256": manifest["original_corpus_sha256"],
            "original_queries_sha256": manifest["original_queries_sha256"],
            "sample_corpus_sha256": manifest["sample_corpus_sha256"],
            "sample_queries_sha256": manifest["sample_queries_sha256"],
            "construction_method": manifest["construction_method"],
            "selected_query_ids": manifest["selected_query_ids"],
            "search_seconds": round(search_seconds, 3),
        },
        "results": results,
    }
    if report_path is None:
        report_path = Path("eval") / f"litsearch_sample_{model_key}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LitSearch hard-negative candidate subset comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="构造固定种子的 hard-negative candidate subset")
    prepare_parser.add_argument("--root", type=Path, default=ROOT)
    prepare_parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    prepare_parser.add_argument("--bm25-report", type=Path, default=FULL_BM25_REPORT)
    eval_parser = subparsers.add_parser("eval", help="评测 BM25/Dense/Hybrid 候选子集排名")
    eval_parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    eval_parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    eval_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.sample_root, args.bm25_report)
        print(f"已生成 hard-negative candidate subset：{result['sample_corpus_count']} 篇，{result['sample_query_count']} 题")
    else:
        report = run_eval(args.model, args.sample_root, args.report)
        for method, by_k in report["results"].items():
            for k, metrics in by_k.items():
                print(f"{method} candidate-subset Top-{k}: Recall={metrics['recall_at_k']:.3f}, MRR={metrics['mrr_at_k']:.3f}, nDCG={metrics['ndcg_at_k']:.3f}")
        print(f"结果（hard-negative candidate subset；非完整 LitSearch 检索）：{args.report or Path('eval') / f'litsearch_sample_{args.model}.json'}")


if __name__ == "__main__":
    main()
