"""Reproducible LitSearch title/abstract retrieval benchmark for this project."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from litagent.evaluation import evaluate
from litagent.retrieval import build_index, load_index, search


REPO = "princeton-nlp/LitSearch"
REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
SOURCE_URL = "https://huggingface.co/datasets/princeton-nlp/LitSearch"
ROOT = Path("data/litsearch")
CORPUS = ROOT / "corpus.jsonl"
QUERIES = ROOT / "queries.jsonl"
MANIFEST = ROOT / "manifest.json"
BM25_INDEX = ROOT / "bm25_index.json"
REPORT = Path("eval/litsearch_bm25.json")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def prepare(root: Path = ROOT) -> dict:
    try:
        import fsspec
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError as exc:
        raise RuntimeError("请先安装 requirements.txt") from exc

    root.mkdir(parents=True, exist_ok=True)
    query_parquet = Path(hf_hub_download(
        REPO, "query/full-00000-of-00001.parquet", repo_type="dataset",
        revision=REVISION, local_dir=root / "raw",
    ))
    raw_queries = pq.read_table(query_parquet).to_pylist()
    corpus_path = root / "corpus.jsonl"
    query_path = root / "queries.jsonl"
    corpus_tmp = corpus_path.with_suffix(".jsonl.tmp")
    seen: set[str] = set()
    textless_ids: set[str] = set()
    counts = Counter()
    with corpus_tmp.open("w", encoding="utf-8", newline="\n") as output:
        for shard in range(6):
            filename = f"corpus_clean/full-{shard:05d}-of-00006.parquet"
            url = hf_hub_url(REPO, filename, repo_type="dataset", revision=REVISION)
            for attempt in range(3):
                try:
                    with fsspec.open(url, "rb", block_size=4 * 1024 * 1024,
                                     client_kwargs={"trust_env": True}) as remote:
                        table = pq.ParquetFile(remote).read(columns=["corpusid", "title", "abstract"])
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            for row in table.to_pylist():
                paper_id = str(row["corpusid"])
                if paper_id in seen:
                    raise ValueError(f"LitSearch 语料有重复 corpusid: {paper_id}")
                seen.add(paper_id)
                title = " ".join((row["title"] or "").split())
                abstract = " ".join((row["abstract"] or "").split())
                counts["empty_title"] += not bool(title)
                counts["empty_abstract"] += not bool(abstract)
                counts["empty_both"] += not bool(title or abstract)
                if not title and not abstract:
                    textless_ids.add(paper_id)
                paper = {
                    "paper_id": paper_id, "title": title, "abstract": abstract,
                    "source_url": SOURCE_URL, "evidence_scope": "abstract",
                    "dataset": "LitSearch", "corpusid": row["corpusid"],
                }
                output.write(json.dumps(paper, ensure_ascii=False) + "\n")
                counts["papers"] += 1
            print(f"投影读取 LitSearch 语料分片 {shard + 1}/6：累计 {counts['papers']} 篇", flush=True)
    corpus_tmp.replace(corpus_path)

    query_tmp = query_path.with_suffix(".jsonl.tmp")
    missing_gold: set[str] = set()
    sets = Counter()
    affected_queries = 0
    with query_tmp.open("w", encoding="utf-8", newline="\n") as output:
        for number, row in enumerate(raw_queries, 1):
            gold = [str(value) for value in row["corpusids"]]
            missing_gold.update(set(gold) - seen)
            textless_gold = set(gold) & textless_ids
            counts["textless_gold_labels"] += len(textless_gold)
            affected_queries += bool(textless_gold)
            query = {
                "id": f"litsearch_{number:03d}", "question": row["query"],
                "gold_paper_ids": gold, "query_set": row["query_set"],
                "specificity": row["specificity"], "quality": row["quality"],
                "label_status": "LitSearch_official",
            }
            if not query["question"] or not gold:
                raise ValueError(f"LitSearch 第 {number} 条查询缺少问题或相关论文")
            output.write(json.dumps(query, ensure_ascii=False) + "\n")
            sets[row["query_set"]] += 1
    if missing_gold:
        query_tmp.unlink(missing_ok=True)
        raise ValueError(f"LitSearch 有 {len(missing_gold)} 个相关论文 ID 未出现在语料中")
    query_tmp.replace(query_path)
    manifest = {
        "source": SOURCE_URL, "revision": REVISION,
        "corpus_config": "corpus_clean", "query_config": "query",
        "text_fields": ["title", "abstract"], "corpus_count": counts["papers"],
        "query_count": len(raw_queries), "query_sets": dict(sets),
        "empty_title": counts["empty_title"],
        "empty_abstract": counts["empty_abstract"],
        "empty_both": counts["empty_both"],
        "textless_gold_labels": counts["textless_gold_labels"],
        "queries_with_textless_gold": affected_queries,
        "corpus_sha256": _digest(corpus_path), "queries_sha256": _digest(query_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_bm25(root: Path = ROOT, report_path: Path = REPORT, top_k: int = 20) -> dict:
    if top_k < 5:
        raise ValueError("LitSearch 评测至少取 Top-5")
    corpus_path = root / "corpus.jsonl"
    query_path = root / "queries.jsonl"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if _digest(corpus_path) != manifest["corpus_sha256"] or _digest(query_path) != manifest["queries_sha256"]:
        raise ValueError("LitSearch 数据文件与 manifest 哈希不符")
    index_path = root / "bm25_index.json"
    if not index_path.is_file():
        print("建立 LitSearch 全量 BM25 索引…", flush=True)
        build_index(corpus_path, index_path)
    index = load_index(index_path)
    papers = index["papers"]
    if len(papers) != manifest["corpus_count"]:
        raise ValueError("LitSearch 语料篇数与 manifest 不符")
    rows = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["query_count"]:
        raise ValueError("LitSearch 查询条数与 manifest 不符")
    textless_ids = {paper["paper_id"] for paper in papers if not paper["title"] and not paper["abstract"]}
    textless_gold = sum(len(set(row["gold_paper_ids"]) & textless_ids) for row in rows)
    textless_queries = sum(bool(set(row["gold_paper_ids"]) & textless_ids) for row in rows)
    recall_ceiling = sum(
        1 - len(set(row["gold_paper_ids"]) & textless_ids) / len(set(row["gold_paper_ids"]))
        for row in rows
    ) / len(rows)
    cache = {}
    started = time.perf_counter()
    for number, row in enumerate(rows, 1):
        cache[row["question"]] = [hit["paper_id"] for hit in search(index, row["question"], top_k)]
        if number % 100 == 0:
            print(f"已检索 {number}/{len(rows)} 条查询", flush=True)
    elapsed = time.perf_counter() - started
    search_fn = lambda question, limit: [{"paper_id": paper_id} for paper_id in cache[question][:limit]]
    results = {str(k): evaluate({}, rows, k, search_fn) for k in (5, top_k)}
    metric_names = ("recall_at_k", "mrr_at_k", "ndcg_at_k", "all_gold_at_k")
    slices = {}
    for field in ("query_set", "specificity"):
        slices[field] = {}
        for value in sorted({row[field] for row in rows}):
            subset = [row for row in rows if row[field] == value]
            slices[field][str(value)] = {}
            for k in (5, top_k):
                summary = evaluate({}, subset, k, search_fn)
                slices[field][str(value)][str(k)] = {
                    "question_count": len(subset),
                    **{metric: summary[metric] for metric in metric_names},
                }
    report = {
        "run": {
            "dataset": REPO, "revision": manifest["revision"],
            "corpus_config": "corpus_clean", "text_fields": ["title", "abstract"],
            "corpus_count": len(papers), "query_count": len(rows),
            "corpus_sha256": manifest["corpus_sha256"],
            "queries_sha256": manifest["queries_sha256"],
            "retriever": "LitAgent BM25 with title boost", "top_k": top_k,
            "search_seconds": round(elapsed, 3),
            "textless_corpus_count": len(textless_ids),
            "textless_gold_labels": textless_gold,
            "queries_with_textless_gold": textless_queries,
            "text_only_recall_ceiling": recall_ceiling,
        },
        "results": results,
        "slices": slices,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report



def main() -> None:
    parser = argparse.ArgumentParser(description="LitSearch 官方查询和完整语料检索评测")
    parser.add_argument("command", choices=("prepare", "bm25"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
        print(f"LitSearch 已准备：{result['corpus_count']} 篇，{result['query_count']} 题")
    else:
        report = run_bm25(args.root, args.report, args.top_k)
        for k, result in report["results"].items():
            print(f"BM25 Top-{k}: Recall={result['recall_at_k']:.3f}, MRR={result['mrr_at_k']:.3f}, nDCG={result['ndcg_at_k']:.3f}")
        print(f"结果：{args.report}")


if __name__ == "__main__":
    main()
