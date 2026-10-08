"""Command line interface for the computer-science literature RAG track."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from litagent.audit import audit
from litagent.corpus import DEFAULT_QUERY, fetch_arxiv, import_atom, load_papers
from litagent.evaluation import evaluate, load_questions
from litagent.hybrid import DEFAULT_RRF_K, fuse_rrf
from litagent.retrieval import build_index, load_index, search as sparse_search
from rag.generate import answer, resolve_api_key, translate_query_to_english

RAW_PATH = Path("data/raw/arxiv.xml")
PAPERS_PATH = Path("data/papers.jsonl")
MANIFEST_PATH = Path("data/manifest.json")
INDEX_PATH = Path("data/bm25_index.json")
DENSE_INDEX_PATH = Path("data/dense_index.faiss")
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = Path(".rag/models")
QUESTIONS_PATH = Path("eval/seed_questions.jsonl")
HAN_RE = re.compile(r"[\u3400-\u9fff]")


def _retrieval_query(question: str, *, no_translate: bool = False) -> tuple[str, str | None]:
    if no_translate or not HAN_RE.search(question):
        return question, None
    api_key = resolve_api_key()
    return translate_query_to_english(question, api_key=api_key), api_key


def _make_searchers(methods: set[str], bm25_path: Path, dense_path: Path, model_cache: Path, candidate_k: int):
    if candidate_k < 1:
        raise ValueError("candidate_k 必须大于 0")
    need_bm25 = bool(methods & {"bm25", "hybrid"})
    need_dense = bool(methods & {"dense", "hybrid"})
    bm25 = load_index(bm25_path) if need_bm25 else None
    dense = None
    model = None
    if need_dense:
        from litagent.dense import load_dense_index, load_model, dense_search

        dense = load_dense_index(dense_path)
        model = load_model(dense["model_name"], model_cache, offline=True)
        if bm25 and dense["corpus_sha256"] != bm25["corpus_sha256"]:
            raise ValueError("BM25 与向量索引对应的语料不同，请重建索引")

    searchers = {}
    if "bm25" in methods:
        searchers["bm25"] = lambda question, limit: sparse_search(bm25, question, limit)
    if "dense" in methods:
        searchers["dense"] = lambda question, limit: dense_search(dense, question, model, limit)
    if "hybrid" in methods:
        def hybrid(question, limit):
            pool = max(limit, candidate_k)
            lexical = sparse_search(bm25, question, pool)
            semantic = dense_search(dense, question, model, pool)
            return fuse_rrf(lexical, semantic, limit)
        searchers["hybrid"] = hybrid
    papers = bm25["papers"] if bm25 else dense["papers"]
    metadata = {
        "corpus_sha256": (bm25 or dense)["corpus_sha256"],
        "dense_model": dense["model_name"] if dense else None,
        "dense_model_snapshot": dense["model_snapshot"] if dense else None,
        "embedding_backend": dense.get("embedding_backend", "fastembed") if dense else None,
        "backend_version": dense.get("backend_version", dense.get("fastembed_version")) if dense else None,
        "index_backend": dense.get("index_backend") if dense else None,
        "index_type": dense.get("index_type") if dense else None,
    }
    return searchers, papers, metadata


def _cited_numbers(answer_text: str) -> list[int]:
    """Return unique numeric citations in the order they appear in the answer."""
    numbers = []
    for group in re.findall(r"\[(\d+(?:\s*[,，;；]\s*\d+)*)\]", answer_text):
        for part in re.split(r"\s*[,，;；]\s*", group):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers


def main() -> int:
    parser = argparse.ArgumentParser(description="计算机论文 RAG：arXiv 摘要语料与检索基线")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_cmd = commands.add_parser("fetch", help="从 arXiv 获取论文元数据与摘要")
    fetch_cmd.add_argument("--query", default=DEFAULT_QUERY)
    fetch_cmd.add_argument("--limit", type=int, default=100)
    fetch_cmd.add_argument("--raw", type=Path, default=RAW_PATH)
    fetch_cmd.add_argument("--papers", type=Path, default=PAPERS_PATH)
    fetch_cmd.add_argument("--manifest", type=Path, default=MANIFEST_PATH)

    import_cmd = commands.add_parser("import", help="离线导入已保存的 arXiv Atom 快照")
    import_cmd.add_argument("--raw", type=Path, default=RAW_PATH)
    import_cmd.add_argument("--papers", type=Path, default=PAPERS_PATH)
    import_cmd.add_argument("--manifest", type=Path, default=MANIFEST_PATH)

    index_cmd = commands.add_parser("index", help="构建论文级 BM25 倒排索引")
    index_cmd.add_argument("--papers", type=Path, default=PAPERS_PATH)
    index_cmd.add_argument("--index", type=Path, default=INDEX_PATH)

    dense_cmd = commands.add_parser("index-dense", help="下载本地 embedding 模型并构建 FAISS 向量索引")
    dense_cmd.add_argument("--papers", type=Path, default=PAPERS_PATH)
    dense_cmd.add_argument("--index", type=Path, default=DENSE_INDEX_PATH)
    dense_cmd.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    dense_cmd.add_argument("--model", default=DEFAULT_DENSE_MODEL)

    migrate_cmd = commands.add_parser("migrate-dense", help="将旧版 JSON 向量索引无重编码迁移到 FAISS")
    migrate_cmd.add_argument("--legacy", type=Path, required=True,
                             help="要迁移的旧版 JSON Dense 索引；仓库不再附带旧格式索引副本")
    migrate_cmd.add_argument("--index", type=Path, default=DENSE_INDEX_PATH)

    audit_cmd = commands.add_parser("audit", help="检查论文语料元数据与摘要质量")
    audit_cmd.add_argument("--papers", type=Path, default=PAPERS_PATH)

    for name in ("search", "ask"):
        command = commands.add_parser(name, help="检索论文摘要" if name == "search" else "摘要证据问答")
        command.add_argument("question")
        command.add_argument("--index", type=Path, default=INDEX_PATH)
        command.add_argument("--dense-index", type=Path, default=DENSE_INDEX_PATH)
        command.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
        command.add_argument("--retriever", choices=("auto", "bm25", "dense", "hybrid"), default="auto")
        command.add_argument("--candidate-k", type=int, default=20)
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--no-translate", action="store_true", help="保留原问题检索，不调用查询翻译")

    eval_cmd = commands.add_parser("eval", help="运行论文级检索评测")
    eval_cmd.add_argument("--index", type=Path, default=INDEX_PATH)
    eval_cmd.add_argument("--dense-index", type=Path, default=DENSE_INDEX_PATH)
    eval_cmd.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    eval_cmd.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    eval_cmd.add_argument("--candidate-k", type=int, default=20)
    eval_cmd.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    eval_cmd.add_argument("--top-k", type=int, default=5)
    eval_cmd.add_argument("--json", type=Path, help="可选：保存完整结果到 JSON")

    compare_cmd = commands.add_parser("compare", help="同一题集对比 BM25、Dense、Hybrid")
    compare_cmd.add_argument("--index", type=Path, default=INDEX_PATH)
    compare_cmd.add_argument("--dense-index", type=Path, default=DENSE_INDEX_PATH)
    compare_cmd.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    compare_cmd.add_argument("--candidate-k", type=int, default=20)
    compare_cmd.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    compare_cmd.add_argument("--top-k", type=int, default=5)
    compare_cmd.add_argument("--json", type=Path, default=Path("eval/retrieval_comparison.json"))

    args = parser.parse_args()
    try:
        if args.command == "fetch":
            url = fetch_arxiv(args.raw, args.query, args.limit)
            result = import_atom(args.raw, args.papers, args.manifest)
            print(f"arXiv 获取 {result['normalized_count']} 篇摘要；快照: {args.raw}")
            print(f"查询: {url}")
            return 0
        if args.command == "import":
            result = import_atom(args.raw, args.papers, args.manifest)
            print(f"已导入 {result['normalized_count']} 篇摘要；清单: {args.manifest}")
            return 0
        if args.command == "index":
            result = build_index(args.papers, args.index)
            print(f"已索引 {result['papers']} 篇论文、{result['terms']} 个词项 → {result['index_path']}")
            return 0
        if args.command == "index-dense":
            from litagent.dense import build_dense_index

            result = build_dense_index(args.papers, args.index, args.model, args.model_cache)
            print(f"已生成 {result['papers']} 篇 × {result['dimension']} 维向量；模型快照 {result['model_snapshot']} → {result['index_path']}")
            return 0
        if args.command == "migrate-dense":
            from litagent.dense import migrate_legacy_index

            result = migrate_legacy_index(args.legacy, args.index)
            print(f"已迁移 {result['papers']} 篇 × {result['dimension']} 维向量 → {result['index_path']}；元数据: {result['metadata_path']}")
            return 0
        if args.command == "audit":
            print(json.dumps(audit(load_papers(args.papers)), ensure_ascii=False, indent=2))
            return 0
        if args.command in ("search", "ask"):
            contains_chinese = bool(HAN_RE.search(args.question))
            retriever = ("dense" if contains_chinese else "hybrid") if args.retriever == "auto" else args.retriever
            searchers, _, metadata = _make_searchers({retriever}, args.index, args.dense_index, args.model_cache, args.candidate_k)
            retrieval_query, api_key = _retrieval_query(args.question, no_translate=args.no_translate)
            if retrieval_query != args.question:
                print(f"英文检索查询：{retrieval_query}")
            if args.retriever == "auto":
                print(f"自动选择检索方式：{retriever}")
            hits = searchers[retriever](retrieval_query, args.top_k)
            if not hits:
                print("没有检索命中；请检查问题、语料或检索方式。")
                return 0
            if args.command == "search":
                for rank, hit in enumerate(hits, 1):
                    print(f"\n[{rank}] {hit['title']} ({hit['year']})")
                    print(f"arXiv:{hit['paper_id']} · {retriever} score={hit['score']} · 摘要证据")
                    if retriever == "hybrid":
                        print(f"BM25 排名: {hit.get('bm25_rank', '—')} · Dense 排名: {hit.get('dense_rank', '—')}")
                    print(hit["source_url"])
                    print(hit["abstract"][:450] + ("…" if len(hit["abstract"]) > 450 else ""))
                return 0
            context_hits = [
                {
                    "source": f"arXiv:{hit['paper_id']} {hit['title']} ({hit['source_url']})",
                    "section": "Abstract / 摘要",
                    "text": hit["abstract"],
                }
                for hit in hits
            ]
            print("以下回答仅基于检索到的论文摘要，不代表已核对全文。\n")
            answer_text = answer(args.question, context_hits, api_key=api_key) if api_key else answer(args.question, context_hits)
            print(answer_text)
            cited = _cited_numbers(answer_text)
            valid = [number for number in cited if 1 <= number <= len(hits)]
            invalid = [number for number in cited if number not in valid]
            if valid:
                print("\n引用来源：")
                for number in sorted(valid):
                    hit = hits[number - 1]
                    print(f"[{number}] arXiv:{hit['paper_id']} · {hit['title']} · {hit['source_url']} · Abstract")
            else:
                print("\n回答中没有有效的来源引用，请核对后再使用。")
            if invalid:
                print(f"注意：回答引用了未提供的编号 {invalid}，对应内容未经来源核验。")
            return 0
        methods = {"bm25", "dense", "hybrid"} if args.command == "compare" else {args.retriever}
        searchers, papers, metadata = _make_searchers(methods, args.index, args.dense_index, args.model_cache, args.candidate_k)
        questions = load_questions(args.questions, {paper["paper_id"]: paper for paper in papers})
        if args.command == "compare":
            results = {}
            for method in ("bm25", "dense", "hybrid"):
                result = evaluate({}, questions, args.top_k, searchers[method])
                results[method] = result
                print(f"{method:6} Recall@{args.top_k}={result['recall_at_k']:.3f} MRR={result['mrr_at_k']:.3f} nDCG={result['ndcg_at_k']:.3f} AllGold={result['all_gold_at_k']:.3f}")
            report = {
                "run": {
                    **metadata,
                    "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                    "top_k": args.top_k,
                    "candidate_k_per_channel": max(args.top_k, args.candidate_k),
                    "rrf_k": DEFAULT_RRF_K,
                },
                "results": results,
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"同源题目，指标可能偏乐观；详细结果: {args.json}")
            return 0
        result = evaluate({}, questions, args.top_k, searchers[args.retriever])
        result["retriever"] = args.retriever
        print(f"检索评测：{result['question_count']} 题，Top-{args.top_k}")
        print(f"标注状态：{result['label_status']}（同源题目，指标可能偏乐观）")
        for field in ("recall_at_k", "mrr_at_k", "ndcg_at_k", "all_gold_at_k"):
            print(f"{field}: {result[field]:.3f}")
        misses = [row for row in result["rows"] if not row["all_gold_found"]]
        for row in misses:
            print(f"未找全 {row['id']}: gold={row['gold_paper_ids']} hits={row['retrieved_paper_ids']}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"详细结果: {args.json}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
