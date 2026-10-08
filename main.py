"""CLI entry point for the first RAG vertical slice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.generate import answer
from rag.ingest import build_index, load_index
from rag.retrieve import search


def main() -> int:
    parser = argparse.ArgumentParser(description="本地文档 RAG：导入、检索、回答")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="读取 docs 并建立索引")
    index_parser.add_argument("--docs", type=Path, default=Path("docs"))
    index_parser.add_argument("--index", type=Path, default=Path(".rag/index.json"))

    for name in ("search", "ask"):
        command_parser = subparsers.add_parser(name, help="检索片段" if name == "search" else "检索并生成回答")
        command_parser.add_argument("question")
        command_parser.add_argument("--index", type=Path, default=Path(".rag/index.json"))
        command_parser.add_argument("--top-k", type=int, default=4)

    args = parser.parse_args()
    try:
        if args.command == "index":
            result = build_index(args.docs, args.index)
            print(f"已导入 {result['files']} 个文件、{result['chunks']} 个片段 → {result['output']}")
            if result["empty_files"]:
                print("未提取到文字的文件：" + "、".join(result["empty_files"]))
            return 0

        if args.top_k < 1:
            raise ValueError("--top-k 必须大于 0")
        hits = search(load_index(args.index)["chunks"], args.question, args.top_k)
        if args.command == "search":
            if not hits:
                print("没有检索到相关片段。")
            for number, hit in enumerate(hits, 1):
                print(f"\n[{number}] {hit['source']} · {hit['section']} · score={hit['score']}")
                print(hit["text"])
            return 0

        response = answer(args.question, hits)
        print(response)
        if hits:
            print("\n来源：")
            for number, hit in enumerate(hits, 1):
                print(f"[{number}] {hit['source']} · {hit['section']}")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
