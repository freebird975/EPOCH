"""Rank fusion for lexical and semantic retrieval results."""

from __future__ import annotations

DEFAULT_RRF_K = 60


def fuse_rrf(sparse_hits: list[dict], dense_hits: list[dict], top_k: int = 5, rrf_k: int = DEFAULT_RRF_K) -> list[dict]:
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k 和 rrf_k 必须大于 0")
    merged: dict[str, dict] = {}
    for channel, hits in (("bm25", sparse_hits), ("dense", dense_hits)):
        for rank, hit in enumerate(hits, 1):
            paper_id = hit["paper_id"]
            if paper_id not in merged:
                merged[paper_id] = {**hit, "score": 0.0, "retrieval_method": "hybrid"}
            merged[paper_id]["score"] += 1 / (rrf_k + rank)
            merged[paper_id][f"{channel}_rank"] = rank
    ranked = sorted(merged.values(), key=lambda hit: (-hit["score"], hit["paper_id"]))
    for rank, hit in enumerate(ranked[:top_k], 1):
        hit["score"] = round(hit["score"], 8)
        hit["rrf_rank"] = rank
        hit["rrf_score"] = hit["score"]
    return ranked[:top_k]
