"""Derive a small Chinese retrieval probe from the reviewed English abstract questions."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRANSLATIONS = {
    "q01": "原始 Corrective RAG 如何评估检索文档并决定是否采取其他检索动作？",
    "q02": "哪个框架无需人工标注的标准答案就能评估 RAG 流程？",
    "q03": "哪篇论文提出了用于测试 RAG 是否拒答不可回答问题的分类法和指标？",
    "q04": "在长篇结构化学位论文的测试条件下，基于聚类的语义分块是否优于固定长度和递归分块？",
    "q05": "哪种无监督查询路由方法通过评估上界回答来选择搜索引擎？",
    "q06": "哪个图检索模型使用超过 1400 万条三元组的 60 个知识图谱训练？",
    "q07": "哪项研究使用中心核对齐和检索结果重叠度比较嵌入模型？",
    "q08": "哪份技术报告计划将向量索引、知识图谱、全文搜索和结构化数据库结合为统一检索层？",
    "q09": "哪两篇论文分别研究无需人工标准答案的 RAG 评估和不可回答请求的评估？",
    "q10": "哪两篇原始论文分别提出低质量检索证据的纠错动作，以及六类不可回答 RAG 问题的分类法？",
}


def main() -> None:
    source = ROOT / "seed_questions.jsonl"
    target = ROOT / "zh_questions.jsonl"
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        item["question"] = TRANSLATIONS[item["id"]]
        item["id"] = "zh_" + item["id"]
        item["label_status"] = "assistant_translated_from_reviewed_abstract"
        item["review_basis"] = "Chinese paraphrase of the corresponding English seed question; same abstract evidence and gold IDs"
        rows.append(item)
    if len(rows) != len(TRANSLATIONS):
        raise ValueError("Chinese question mapping does not match the English seed set")
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} Chinese questions to {target}")


if __name__ == "__main__":
    main()
