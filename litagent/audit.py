"""Small corpus audit before retrieval experiments."""

from __future__ import annotations

from collections import Counter


def audit(papers: list[dict]) -> dict:
    titles = Counter(" ".join(paper["title"].lower().split()) for paper in papers)
    primary_categories = Counter(paper.get("primary_category", "unknown") for paper in papers)
    abstract_words = [len(paper["abstract"].split()) for paper in papers]
    return {
        "paper_count": len(papers),
        "year_range": [min(paper["year"] for paper in papers), max(paper["year"] for paper in papers)],
        "primary_categories": dict(primary_categories.most_common()),
        "missing_doi": sum(not paper.get("doi") for paper in papers),
        "missing_pdf_link": sum(not paper.get("pdf_url") for paper in papers),
        "abstract_words_min": min(abstract_words),
        "abstract_words_median": sorted(abstract_words)[len(abstract_words) // 2],
        "abstract_words_max": max(abstract_words),
        "duplicate_titles": [title for title, count in titles.items() if count > 1],
        "short_abstract_ids": [
            paper["paper_id"] for paper, length in zip(papers, abstract_words) if length < 50
        ],
    }
