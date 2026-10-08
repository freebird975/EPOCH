"""Run a resumable, offline full-corpus LitSearch retrieval evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .artifacts import file_sha256
from .dense import dense_search, load_dense_index, load_model
from .fulltext_eval_metrics import summarize_fulltext_eval
from .hybrid import DEFAULT_RRF_K, fuse_rrf
from .retrieval import SQLITE_BM25_MMAP_SIZE_BYTES, load_index, search
from .sqlite_bm25_numpy import prepare_numpy_index


SCHEMA_VERSION = 1
EVALUATOR_VERSION = "fulltext-offline-v4-sqlite-numpy-mmap"
RETRIEVAL_MODES = ("bm25", "dense", "hybrid")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 无效：{path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 每行都必须是对象：{path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL 没有记录：{path}")
    return rows


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _child_parent_ids(hits: list[dict], corpus_parent_ids: set[str]) -> tuple[list[str], int]:
    """Return deduplicated parent ranks and the number of orphan child hits."""
    parent_ids: list[str] = []
    seen: set[str] = set()
    orphans = 0
    for hit in hits:
        parent_id = str(hit.get("parent_id") or "").strip()
        if not parent_id or parent_id not in corpus_parent_ids:
            orphans += 1
            continue
        if parent_id not in seen:
            seen.add(parent_id)
            parent_ids.append(parent_id)
    return parent_ids, orphans


def _load_checkpoint(path: Path, fingerprint: str, query_ids: set[str]) -> dict[str, dict]:
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"检查点为空，拒绝续跑：{path}")
    lines = raw.splitlines(keepends=True)
    records: list[dict] = []
    valid_bytes = 0
    has_partial_tail = False
    for number, line in enumerate(lines, 1):
        # A crash during the last append may leave one incomplete record. It is
        # safe to discard only that unterminated tail; all earlier lines must
        # remain valid so that no completed ranking is silently lost.
        if number == len(lines) and not line.endswith((b"\n", b"\r")):
            has_partial_tail = True
            break
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"检查点损坏：{path}:{number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"检查点记录必须是 JSON 对象：{path}:{number}")
        records.append(record)
        valid_bytes += len(line)
    if has_partial_tail:
        # The checkpoint is an agent-owned append-only artifact. Remove only
        # bytes after the last complete JSONL record before future appends.
        with path.open("r+b") as checkpoint:
            checkpoint.truncate(valid_bytes)
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
    if not records or records[0].get("record_type") != "header":
        raise ValueError(f"检查点缺少 header：{path}")
    header = records[0]
    if header.get("schema_version") != SCHEMA_VERSION or header.get("fingerprint") != fingerprint:
        raise ValueError("检查点来源哈希或评测参数不匹配；请指定新的 --checkpoint 路径")
    completed: dict[str, dict] = {}
    for record in records[1:]:
        if record.get("record_type") != "query":
            raise ValueError(f"检查点包含未知记录类型：{path}")
        query_id = str(record.get("query_id") or "")
        if query_id not in query_ids:
            raise ValueError(f"检查点包含不在当前 qrels 中的查询：{query_id}")
        if query_id in completed:
            raise ValueError(f"检查点重复记录查询：{query_id}")
        payload = record.get("retrievals")
        if not isinstance(payload, dict) or set(payload) != set(RETRIEVAL_MODES):
            raise ValueError(f"检查点缺少 BM25/Dense/Hybrid 排名：{query_id}")
        completed[query_id] = payload
    return completed


def _metadata_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_fulltext_offline_eval(
    parents_path: Path,
    chunks_path: Path,
    bm25_path: Path,
    dense_path: Path,
    dense_docs_path: Path,
    queries_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    *,
    candidate_k: int = 100,
    rrf_k: int = DEFAULT_RRF_K,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Score all official qrels using one BM25 and one Dense call per query.

    This runner loads existing indexes in read-only mode. It never builds an
    index or calls a hosted model. Every completed query is fsynced to a JSONL
    checkpoint so a long full-corpus run can continue after interruption.
    """
    paths = {
        "parents": Path(parents_path),
        "chunks": Path(chunks_path),
        "bm25": Path(bm25_path),
        "dense": Path(dense_path),
        "dense_docs": Path(dense_docs_path),
        "queries": Path(queries_path),
    }
    report_path, checkpoint_path = Path(report_path), Path(checkpoint_path)
    if candidate_k < 20:
        raise ValueError("candidate_k 至少为 20，才能计算 Recall@20")
    if rrf_k < 1:
        raise ValueError("rrf_k 必须为正整数")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"评测输入文件不存在：{name}={path}")
    if report_path.resolve() == checkpoint_path.resolve():
        raise ValueError("报告与检查点必须使用不同路径")

    started = time.perf_counter()
    validation_cache: dict = {}
    queries = _read_jsonl(paths["queries"])
    for row in queries:
        if row.get("label_status") != "LitSearch_official":
            raise ValueError("全量离线评测只接受官方 LitSearch qrels；发现非官方查询行")

    child_manifest_path = paths["chunks"].with_suffix(".manifest.json")
    if not child_manifest_path.is_file():
        raise FileNotFoundError(f"全文子块 manifest 不存在：{child_manifest_path}")
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(child_manifest, dict):
        raise ValueError("全文子块 manifest 格式无效")
    parent_sha256 = file_sha256(paths["parents"], cache=validation_cache)
    chunks_sha256 = file_sha256(paths["chunks"], cache=validation_cache)
    if parent_sha256 != child_manifest.get("parent_sha256"):
        raise ValueError("父语料哈希与子块 manifest 不符")
    if chunks_sha256 != child_manifest.get("chunk_sha256"):
        raise ValueError("子块哈希与子块 manifest 不符")

    if progress:
        progress("只读加载 BM25 索引…")
    bm25_started = time.perf_counter()
    bm25_index = load_index(paths["bm25"], corpus_path=paths["chunks"],
                            validation_cache=validation_cache)
    bm25_load_seconds = time.perf_counter() - bm25_started
    if bm25_index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("BM25 索引不是全文子块索引")
    if bm25_index.get("corpus_sha256") != chunks_sha256:
        raise ValueError("BM25 索引与全文子块语料不匹配")
    if progress:
        progress("顺序预载 BM25 文档长度缓存…")
    numpy_cache_started = time.perf_counter()
    numpy_cache_metadata = prepare_numpy_index(bm25_index)
    bm25_numpy_cache_seconds = time.perf_counter() - numpy_cache_started

    if progress:
        progress("只读加载 FAISS 索引与映射库…")
    dense_started = time.perf_counter()
    dense_index = load_dense_index(paths["dense"], corpus_path=paths["chunks"],
                                   docs_path=paths["dense_docs"],
                                   validation_cache=validation_cache)
    dense_load_seconds = time.perf_counter() - dense_started
    if dense_index.get("evidence_scope") != "full_text_chunk":
        raise ValueError("Dense 索引不是全文子块索引")
    if dense_index.get("corpus_sha256") != chunks_sha256:
        raise ValueError("Dense 索引与全文子块语料不匹配")
    if dense_index.get("model_name") != "BAAI/bge-small-en-v1.5":
        raise ValueError("全量离线基线固定使用 BAAI/bge-small-en-v1.5")
    if dense_index.get("documents") != bm25_index.get("documents"):
        raise ValueError("BM25 与 Dense 的子块数量不一致")

    if progress:
        progress("离线加载 BGE 查询编码器…")
    model_started = time.perf_counter()
    model = load_model(dense_index["model_name"], Path(".rag/models"), offline=True)
    model_load_seconds = time.perf_counter() - model_started

    # Build a reusable paper-ID/byte-offset map with one sequential parent scan.
    from .litsearch_data_preparation import DataPreparationModule

    preparer = DataPreparationModule(paths["parents"])
    parent_index_started = time.perf_counter()
    preparer._ensure_parent_offsets()
    parent_index_seconds = time.perf_counter() - parent_index_started
    corpus_parent_ids = set(preparer._parent_offsets or {})
    expected_parent_count = child_manifest.get("parents")
    if type(expected_parent_count) is int and len(corpus_parent_ids) != expected_parent_count:
        raise ValueError("父语料论文数与子块 manifest 不一致")
    expected_chunk_count = child_manifest.get("chunks")
    if type(expected_chunk_count) is int and expected_chunk_count != bm25_index.get("documents"):
        raise ValueError("全文子块数与 BM25 索引不一致")

    hashes = {
        "parents_sha256": parent_sha256,
        "chunks_sha256": chunks_sha256,
        "queries_sha256": file_sha256(paths["queries"], cache=validation_cache),
        "bm25_index_sha256": file_sha256(paths["bm25"], cache=validation_cache),
        "dense_index_sha256": dense_index.get("index_sha256"),
        "dense_docs_sha256": file_sha256(paths["dense_docs"], cache=validation_cache),
        "chunk_manifest_sha256": file_sha256(child_manifest_path, cache=validation_cache),
        "dense_metadata_sha256": file_sha256(paths["dense"].with_suffix(".meta.json"), cache=validation_cache),
    }
    code_paths = {
        "evaluator": Path(__file__).resolve(),
        "metrics": Path(__file__).with_name("fulltext_eval_metrics.py").resolve(),
        "bm25_retrieval": Path(__file__).with_name("retrieval.py").resolve(),
        "dense_retrieval": Path(__file__).with_name("dense.py").resolve(),
        "rrf": Path(__file__).with_name("hybrid.py").resolve(),
        "parent_recovery": Path(__file__).with_name("litsearch_data_preparation.py").resolve(),
        "bm25_sql_aggregation": Path(__file__).with_name("sqlite_bm25_fast.py").resolve(),
        "bm25_numpy_postings": Path(__file__).with_name("sqlite_bm25_numpy.py").resolve(),
    }
    code_hashes = {name: file_sha256(path, cache=validation_cache)
                   for name, path in code_paths.items()}
    hashes["code_sha256"] = hashlib.sha256(json.dumps(
        code_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if not all(isinstance(value, str) and len(value) == 64 for value in hashes.values()):
        raise ValueError("评测输入缺少有效的 SHA-256")
    source_metadata = {
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "query_source": {
            "repository": "princeton-nlp/LitSearch",
            "revision": "9573fb284a1026c998df47024b888a163f0f0e25",
            "label_status": "LitSearch_official",
            "qrels": "gold_paper_ids",
            "unjudged_documents": "unjudged; not treated as nonrelevant",
        },
        "corpus": {
            "parent_count": len(corpus_parent_ids),
            "chunk_count": bm25_index.get("documents"),
            "evidence_scope": "full_text_chunk",
            "parser_version": child_manifest.get("parser_version"),
            "chunk_size_chars": child_manifest.get("chunk_size_chars"),
            "overlap_chars": child_manifest.get("overlap_chars"),
        },
        "hashes": hashes,
        "code_hashes": code_hashes,
        "indexes": {
            "bm25": {key: bm25_index.get(key) for key in
                      ("version", "backend", "evidence_scope", "documents", "k1", "b", "avg_length",
                       "mmap_size_bytes")},
            "dense": {key: dense_index.get(key) for key in
                       ("version", "index_backend", "index_type", "dimension", "model_name",
                        "model_snapshot", "embedding_mode", "documents")},
        },
        "model": {
            "name": dense_index["model_name"],
            "snapshot": dense_index["model_snapshot"],
            "backend_version": _metadata_version("fastembed") or _metadata_version("fastembed-gpu"),
            "offline": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "faiss": _metadata_version("faiss-cpu") or _metadata_version("faiss-gpu"),
            "numpy": _metadata_version("numpy"),
        },
        "parameters": {
            "candidate_k_per_channel": candidate_k,
            "hybrid_candidate_k": 2 * candidate_k,
            "bm25_engine": "sqlite_numpy_postings_v1",
            "bm25_mmap_size_request_bytes": SQLITE_BM25_MMAP_SIZE_BYTES,
            "bm25_doc_length_cache": numpy_cache_metadata,
            "rrf_k": rrf_k,
            "rrf_method": "reciprocal_rank_fusion",
            "reranker": None,
            "query_translation": False,
            "retrieval_calls_per_query": {"bm25": 1, "dense": 1, "hybrid_extra": 0},
            "cutoffs": [5, 10, 20],
        },
    }
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "hashes": hashes,
        "dense_model": source_metadata["model"],
        "parameters": source_metadata["parameters"],
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True,
                                            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    query_ids = {str(row.get("id") or "") for row in queries}
    if len(query_ids) != len(queries) or "" in query_ids:
        raise ValueError("官方 qrels 中查询 ID 重复或为空")
    completed = _load_checkpoint(checkpoint_path, fingerprint, query_ids) if resume and checkpoint_path.is_file() else {}
    if checkpoint_path.is_file() and not resume:
        raise ValueError("检查点已存在；请启用 resume 或显式更换 --checkpoint 路径")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not checkpoint_path.is_file()
    with checkpoint_path.open("a", encoding="utf-8", newline="\n") as checkpoint:
        if needs_header:
            _append_jsonl(checkpoint, {
                "record_type": "header", "schema_version": SCHEMA_VERSION,
                "evaluator_version": EVALUATOR_VERSION, "fingerprint": fingerprint,
                "source_metadata": source_metadata,
            })
        total = len(queries)
        for number, row in enumerate(queries, 1):
            query_id = str(row["id"])
            if query_id in completed:
                if progress and number % 10 == 0:
                    progress(f"已恢复 {len(completed)}/{total} 条查询检查点")
                continue
            question = row["question"]
            if progress:
                progress(f"评测 {number}/{total}：{query_id}（BM25 + Dense + RRF）")
            query_started = time.perf_counter()
            bm25_started = time.perf_counter()
            bm25_hits = search(bm25_index, question, candidate_k)
            bm25_seconds = time.perf_counter() - bm25_started
            dense_started = time.perf_counter()
            dense_hits = dense_search(dense_index, question, model, candidate_k)
            dense_seconds = time.perf_counter() - dense_started
            rrf_started = time.perf_counter()
            hybrid_hits = fuse_rrf(bm25_hits, dense_hits, top_k=2 * candidate_k, rrf_k=rrf_k)
            rrf_seconds = time.perf_counter() - rrf_started
            results = {
                "bm25": (bm25_hits, bm25_seconds, {"bm25": bm25_seconds}),
                "dense": (dense_hits, dense_seconds, {"dense": dense_seconds}),
                "hybrid": (hybrid_hits, bm25_seconds + dense_seconds + rrf_seconds,
                           {"bm25": bm25_seconds, "dense": dense_seconds, "rrf": rrf_seconds}),
            }
            query_retrievals: dict[str, dict] = {}
            for mode in RETRIEVAL_MODES:
                hits, mode_seconds, timing = results[mode]
                parent_ids, orphan_hits = _child_parent_ids(hits, corpus_parent_ids)
                child_hits = [
                    {"paper_id": str(hit.get("paper_id") or ""),
                     "parent_id": str(hit.get("parent_id") or "")}
                    for hit in hits
                ]
                query_retrievals[mode] = {
                    "parent_ids": parent_ids,
                    "candidate_parent_ids": list(dict.fromkeys(
                        hit["parent_id"] for hit in child_hits if hit["parent_id"]
                    )),
                    "child_hits": child_hits,
                    "orphan_child_hits": orphan_hits,
                    "latency_seconds": {**timing, "mode_total": mode_seconds},
                    "child_hit_count": len(hits),
                }
            checkpoint_record = {"record_type": "query", "query_id": query_id,
                                 "retrievals": query_retrievals}
            _append_jsonl(checkpoint, checkpoint_record)
            completed[query_id] = query_retrievals
            if progress:
                progress(f"完成 {query_id}：BM25 {bm25_seconds:.2f}s，Dense {dense_seconds:.2f}s，"
                         f"总计 {time.perf_counter() - query_started:.2f}s")

    if len(completed) != len(queries):
        raise RuntimeError(f"检查点查询数量不完整：{len(completed)}/{len(queries)}")
    report = summarize_fulltext_eval(queries, completed, corpus_parent_ids, source_metadata, ks=(5, 10, 20))
    report.update({
        "status": "complete",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all official LitSearch query/qrel rows; full-text parents only",
        "checkpoint": {"path": str(checkpoint_path.resolve()),
                        "sha256": file_sha256(checkpoint_path, cache=validation_cache)},
        "initialization_seconds": {
            "bm25_load": round(bm25_load_seconds, 3),
            "bm25_numpy_length_cache": round(bm25_numpy_cache_seconds, 3),
            "dense_load": round(dense_load_seconds, 3),
            "embedding_model_load": round(model_load_seconds, 3),
            "parent_id_offset_scan": round(parent_index_seconds, 3),
        },
        "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 3),
    })
    _write_json(report_path, report)
    if progress:
        progress(f"全量离线评测报告已写入：{report_path}")

    connection = bm25_index.get("connection")
    if connection is not None:
        connection.close()
    connection = dense_index.get("docs_connection")
    if connection is not None:
        connection.close()
    return report
