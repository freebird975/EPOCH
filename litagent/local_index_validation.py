"""Read-only validation of relocated full-text indexes, without embeddings or API calls."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import file_sha256, file_signature, verify_sha256
from .dense import DEFAULT_MODEL, load_dense_index
from .retrieval import load_index


def validate_local_indexes(parent_path: Path, child_path: Path, bm25_path: Path,
                           dense_path: Path, *, dense_docs_path: Path,
                           report_path: Path | None = None, progress=None) -> dict:
    """Load both existing indexes and check the exact manifest-bound local corpus.

    This function never builds an index or loads an embedding model. Connections
    and the in-memory FAISS object are released before returning the report.
    """
    paths = {"parents": Path(parent_path).resolve(), "chunks": Path(child_path).resolve(),
             "bm25": Path(bm25_path).resolve(), "dense": Path(dense_path).resolve(),
             "dense_docs": Path(dense_docs_path).resolve()}
    paths["chunk_manifest"] = paths["chunks"].with_suffix(".manifest.json")
    paths["dense_metadata"] = paths["dense"].with_suffix(".meta.json")
    if report_path is not None and Path(report_path).resolve() in paths.values():
        raise ValueError("验证报告不能覆盖语料、索引、映射库或元数据")
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"验证所需的产物不存在：{path}")
    signatures = {name: file_signature(path) for name, path in paths.items()}
    cache: dict = {}
    timings: dict = {}
    started = time.perf_counter()
    bm25 = dense = None

    def announce(message):
        if progress is not None:
            progress(message)

    try:
        announce("校验父语料、子块及 manifest…")
        phase = time.perf_counter()
        manifest = json.loads(paths["chunk_manifest"].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or type(manifest.get("chunks")) is not int or manifest["chunks"] < 1:
            raise ValueError("子块 manifest 缺少有效数量")
        verify_sha256(paths["parents"], manifest.get("parent_sha256"), cache=cache)
        verify_sha256(paths["chunks"], manifest.get("chunk_sha256"), cache=cache)
        timings["corpus_validation_seconds"] = round(time.perf_counter() - phase, 3)

        announce("只读加载 BM25，核对文档数量…")
        phase = time.perf_counter()
        bm25 = load_index(paths["bm25"], corpus_path=paths["chunks"], validation_cache=cache)
        bm25_count = bm25.get("documents", len(bm25.get("papers", [])))
        if bm25.get("evidence_scope") != "full_text_chunk" or bm25_count != manifest["chunks"]:
            raise ValueError("BM25 的数量或证据范围与全文子块 manifest 不一致")
        timings["bm25_load_seconds"] = round(time.perf_counter() - phase, 3)

        announce("校验 FAISS 和 Dense 映射库，并加载向量索引…")
        phase = time.perf_counter()
        dense = load_dense_index(paths["dense"], corpus_path=paths["chunks"],
                                 docs_path=paths["dense_docs"], validation_cache=cache)
        dense_count = dense.get("documents", len(dense.get("papers") or []))
        if (dense.get("evidence_scope") != "full_text_chunk" or dense_count != bm25_count
                or dense.get("model_name") != DEFAULT_MODEL or dense.get("dimension") != 384):
            raise ValueError("全文 BGE/FAISS 的模型、维度、数量或证据范围不一致")
        timings["dense_load_seconds"] = round(time.perf_counter() - phase, 3)

        announce("记录产物哈希并确认文件未改变…")
        phase = time.perf_counter()
        artifacts = {name: {"path": str(path), "bytes": signatures[name][1],
                            "mtime_ns": signatures[name][2],
                            "sha256": file_sha256(path, cache=cache)}
                     for name, path in paths.items()}
        if any(file_signature(path) != signatures[name] for name, path in paths.items()):
            raise ValueError("验证期间产物发生变化，验证结果不可使用")
        timings["artifact_recording_seconds"] = round(time.perf_counter() - phase, 3)
        report = {
            "status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation": {"read_only": True, "artifacts_unchanged": True,
                           "corpus_manifest_hashes": "matched", "index_rebuild_allowed": False,
                           "embedding_model_required": False, "api_required": False,
                           "dense_mapping": dense["mapping_validation"]},
            "corpus": {"chunks": manifest["chunks"], "parents": manifest.get("parents"),
                       "parser_version": manifest.get("parser_version")},
            "bm25": {key: bm25.get(key) for key in
                     ("version", "backend", "evidence_scope", "stored_corpus_path", "corpus_path")},
            "dense": {key: dense.get(key) for key in
                      ("version", "index_backend", "index_type", "evidence_scope", "dimension",
                       "model_name", "model_snapshot", "stored_corpus_path", "corpus_path",
                       "stored_docs_path", "docs_path")},
            "artifacts": artifacts, "timings": timings,
        }
        report["bm25"]["documents"] = bm25_count
        report["dense"]["documents"] = int(dense["faiss_index"].ntotal)
        report["validation"]["bm25_query_only"] = (bool(bm25["connection"].execute("PRAGMA query_only").fetchone()[0])
                                                   if bm25.get("connection") is not None else None)
        report["validation"]["dense_docs_query_only"] = (bool(dense["docs_connection"].execute("PRAGMA query_only").fetchone()[0])
                                                         if dense.get("docs_connection") is not None else None)
        timings["total_seconds"] = round(time.perf_counter() - started, 3)
        if report_path is not None:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_name(f".{report_path.name}-{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(report_path)
            finally:
                temporary.unlink(missing_ok=True)
        return report
    finally:
        for index, field in ((bm25, "connection"), (dense, "docs_connection")):
            if index is not None and index.get(field) is not None:
                index[field].close()
