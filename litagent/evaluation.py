"""Retrieval metrics for draft paper-level relevance labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from .retrieval import search


def load_questions(path: Path, papers_by_id: dict[str, dict]) -> list[dict]:
    questions = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("id") or not item.get("question") or not item.get("gold_paper_ids"):
            raise ValueError(f"{path}:{number} 缺少问题、ID 或相关论文标注")
        if item["id"] in seen:
            raise ValueError(f"重复问题 ID: {item['id']}")
        seen.add(item["id"])
        gold_ids = set(item["gold_paper_ids"])
        unknown = gold_ids - papers_by_id.keys()
        if unknown:
            raise ValueError(f"{item['id']} 的相关论文不在语料中: {sorted(unknown)}")
        evidence_ids = set()
        for evidence in item.get("evidence", []):
            paper_id = evidence.get("paper_id", "")
            quote = " ".join(evidence.get("quote", "").lower().split())
            if paper_id not in gold_ids or not quote:
                raise ValueError(f"{item['id']} 的证据项缺少有效论文 ID 或原句")
            abstract = " ".join(papers_by_id[paper_id]["abstract"].lower().split())
            if quote not in abstract:
                raise ValueError(f"{item['id']} 的证据原句不在 arXiv:{paper_id} 摘要中")
            evidence_ids.add(paper_id)
        if evidence_ids != gold_ids:
            raise ValueError(f"{item['id']} 未给每篇相关论文标注摘要原句")
        questions.append(item)
    if not questions:
        raise ValueError("评测集为空")
    return questions


def evaluate(
    index: dict,
    questions: list[dict],
    top_k: int = 5,
    search_fn: Callable[[str, int], list[dict]] | None = None,
) -> dict:
    search_fn = search_fn or (lambda question, limit: search(index, question, limit))
    rows = []
    for item in questions:
        gold = set(item["gold_paper_ids"])
        retrieved = [hit["paper_id"] for hit in search_fn(item["question"], top_k)]
        relevant = [position for position, paper_id in enumerate(retrieved, 1) if paper_id in gold]
        recall = len(relevant) / len(gold)
        mrr = 1 / relevant[0] if relevant else 0.0
        dcg = sum(1 / math.log2(position + 1) for position in relevant)
        ideal = sum(1 / math.log2(position + 1) for position in range(1, min(len(gold), top_k) + 1))
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "gold_paper_ids": sorted(gold),
            "retrieved_paper_ids": retrieved,
            "recall": recall,
            "mrr": mrr,
            "ndcg": dcg / ideal if ideal else 0.0,
            "all_gold_found": gold.issubset(retrieved),
            "label_status": item.get("label_status", "unspecified"),
        })
    count = len(rows)
    statuses = sorted({row["label_status"] for row in rows})
    return {
        "top_k": top_k,
        "question_count": count,
        "label_status": statuses[0] if len(statuses) == 1 else "mixed: " + ", ".join(statuses),
        "evaluation_warning": (
            "LitSearch official labels; see run metadata for the evaluated corpus scope."
            if statuses == ["LitSearch_official"] else
            "Questions were created from the evaluated abstracts and reviewed by the same assistant; metrics may be optimistic."
        ),
        "recall_at_k": sum(row["recall"] for row in rows) / count,
        "mrr_at_k": sum(row["mrr"] for row in rows) / count,
        "ndcg_at_k": sum(row["ndcg"] for row in rows) / count,
        "all_gold_at_k": sum(row["all_gold_found"] for row in rows) / count,
        "rows": rows,
    }
