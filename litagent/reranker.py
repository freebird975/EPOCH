"""Local ONNX Cross Encoder reranking over a bounded RRF child pool."""

from __future__ import annotations

import math
import os
from pathlib import Path
from importlib.metadata import version


DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def reranker_metadata(model) -> dict:
    model_dir = getattr(getattr(model, "model", None), "_model_dir", None)
    return {"cross_encoder_model": getattr(model, "model_name", DEFAULT_RERANK_MODEL),
            "cross_encoder_snapshot": Path(model_dir).name if model_dir else "unknown",
            "cross_encoder_backend": "fastembed",
            "cross_encoder_backend_version": version("fastembed")}


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL, cache_dir: Path = Path(".rag/models"), *, offline: bool = True):
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError("缺少 Cross Encoder 依赖；请运行 python -m pip install -r requirements.txt") from exc
    threads = int(os.getenv("LIT_RERANK_THREADS", str(min(os.cpu_count() or 2, 8))))
    if threads < 1:
        raise ValueError("LIT_RERANK_THREADS 必须大于 0")
    try:
        return TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir), threads=threads,
                                local_files_only=offline)
    except Exception as exc:
        raise RuntimeError(f"无法加载 Cross Encoder {model_name}；请检查网络或本地模型缓存") from exc


def rerank_children(question: str, hits: list[dict], model, limit: int = 50) -> list[dict]:
    """Score only top RRF hits and preserve the remaining pool after them."""
    if limit < 1:
        raise ValueError("rerank limit 必须大于 0")
    if not hits:
        return []
    selected = hits[:limit]
    passages = [f"{hit.get('title', '')}\n{hit.get('abstract', '')}" for hit in selected]
    scores = list(model.rerank(question, passages, batch_size=16))
    if len(scores) != len(selected) or any(not math.isfinite(float(score)) for score in scores):
        raise RuntimeError("Cross Encoder 返回的分数数量或数值无效")
    reranked = []
    for rank, (hit, score) in enumerate(zip(selected, scores), 1):
        reranked.append({**hit, "rrf_score": hit["score"], "rrf_rank": rank,
                         "rerank_score": float(score), "score": float(score),
                         "retrieval_method": "cross_encoder"})
    reranked.sort(key=lambda hit: (-hit["rerank_score"], hit["rrf_rank"], hit["paper_id"]))
    return reranked + hits[limit:]
