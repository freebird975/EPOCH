"""Local embeddings and a corpus-bound FAISS cosine index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import faiss
import numpy as np

from .corpus import load_papers
from .artifacts import file_signature, verify_sha256

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE = Path(".rag/models")
INDEX_VERSION = 3
STREAMING_INDEX_VERSION = 4


def _embedding_backend_version() -> str:
    try:
        return version("fastembed")
    except PackageNotFoundError:
        # The CUDA-enabled distribution installs the same `fastembed` module
        # under the distinct `fastembed-gpu` distribution name.
        return version("fastembed-gpu")


def _embedding_batch_size(default: int = 16) -> int:
    batch_size = int(os.getenv("LIT_EMBED_BATCH_SIZE", str(default)))
    if batch_size < 1 or batch_size > 64:
        raise ValueError("LIT_EMBED_BATCH_SIZE 必须在 1 到 64 之间")
    return batch_size


def load_model(model_name: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_CACHE, *, offline: bool = True):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("缺少 FastEmbed；请运行 python -m pip install -r requirements.txt") from exc
    threads = int(os.getenv("LIT_EMBED_THREADS", str(min(os.cpu_count() or 2, 8))))
    if threads < 1:
        raise ValueError("LIT_EMBED_THREADS 必须大于 0")
    device = os.getenv("LIT_EMBED_DEVICE", "cpu").lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("LIT_EMBED_DEVICE 必须是 cpu 或 cuda")
    providers = ["CUDAExecutionProvider"] if device == "cuda" else None
    try:
        return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), local_files_only=offline,
                             threads=threads, providers=providers)
    except Exception as exc:
        if offline:
            raise RuntimeError("本地 embedding 模型不可用；请先运行 python lit.py index-dense 下载并建索引") from exc
        raise RuntimeError(f"无法加载 embedding 模型 {model_name}: {exc}") from exc


def _snapshot(model) -> str:
    """FastEmbed's resolved Hugging Face snapshot identifier."""
    if getattr(model, "snapshot", None):
        return model.snapshot
    model_dir = getattr(getattr(model, "model", None), "_model_dir", None)
    return Path(model_dir).name if model_dir else "unknown"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("embedding 向量形状或数值无效")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths <= 0):
        raise ValueError("embedding 模型返回零向量")
    return np.ascontiguousarray(vectors / lengths, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta_path(index_path: Path) -> Path:
    if index_path.suffix.lower() != ".faiss":
        raise ValueError("FAISS 索引路径必须以 .faiss 结尾")
    return index_path.with_suffix(".meta.json")


def _validate_corpus(metadata: dict, *, validation_cache: dict | None = None) -> list[dict]:
    try:
        corpus_path = Path(metadata["corpus_path"])
    except (KeyError, TypeError) as exc:
        raise ValueError("向量索引缺少语料路径，请重建") from exc
    if not corpus_path.is_file():
        raise ValueError(f"向量索引对应语料不存在: {corpus_path}")
    verify_sha256(corpus_path, metadata.get("corpus_sha256"), cache=validation_cache)
    papers = load_papers(corpus_path)
    if [paper["paper_id"] for paper in papers] != metadata.get("paper_ids"):
        raise ValueError("向量索引的论文 ID 与语料不一致，请重建")
    return papers


def _write_index(index_path: Path, vectors: np.ndarray, metadata: dict) -> dict:
    meta_path = _meta_path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    index_fd, temp_index_name = tempfile.mkstemp(prefix=f".{index_path.stem}-", suffix=".faiss.tmp", dir=index_path.parent)
    meta_fd, temp_meta_name = tempfile.mkstemp(prefix=f".{index_path.stem}-", suffix=".meta.tmp", dir=index_path.parent)
    os.close(index_fd)
    os.close(meta_fd)
    temp_index = Path(temp_index_name)
    temp_meta = Path(temp_meta_name)
    try:
        faiss.write_index(faiss_index, str(temp_index))
        metadata = {
            **metadata,
            "version": INDEX_VERSION,
            "index_backend": "faiss",
            "index_type": "IndexFlatIP",
            "faiss_version": faiss.__version__,
            "index_sha256": _sha256(temp_index),
        }
        temp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_index, index_path)
        os.replace(temp_meta, meta_path)
    finally:
        temp_index.unlink(missing_ok=True)
        temp_meta.unlink(missing_ok=True)
    return {
        "papers": len(metadata["paper_ids"]),
        "dimension": metadata["dimension"],
        "model_name": metadata["model_name"],
        "model_snapshot": metadata["model_snapshot"],
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
    }


def _dense_docs_path(index_path: Path) -> Path:
    return index_path.with_suffix(".docs.sqlite3")


def _write_streamed_index(index_path: Path, faiss_index, metadata: dict,
                          temporary_docs: Path) -> dict:
    meta_path = _meta_path(index_path)
    docs_path = _dense_docs_path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_fd, temp_index_name = tempfile.mkstemp(
        prefix=f".{index_path.stem}-", suffix=".faiss.tmp", dir=index_path.parent)
    meta_fd, temp_meta_name = tempfile.mkstemp(
        prefix=f".{index_path.stem}-", suffix=".meta.tmp", dir=index_path.parent)
    os.close(index_fd)
    os.close(meta_fd)
    temp_index = Path(temp_index_name)
    temp_meta = Path(temp_meta_name)
    try:
        faiss.write_index(faiss_index, str(temp_index))
        metadata = {
            **metadata,
            "version": STREAMING_INDEX_VERSION,
            "index_backend": "faiss",
            "index_type": "IndexFlatIP",
            "faiss_version": faiss.__version__,
            "index_sha256": _sha256(temp_index),
            "docs_path": docs_path.as_posix(),
            "docs_sha256": _sha256(temporary_docs),
        }
        temp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_docs, docs_path)
        os.replace(temp_index, index_path)
        os.replace(temp_meta, meta_path)
    finally:
        temporary_docs.unlink(missing_ok=True)
        temp_index.unlink(missing_ok=True)
        temp_meta.unlink(missing_ok=True)
    return {
        "papers": metadata["documents"],
        "dimension": metadata["dimension"],
        "model_name": metadata["model_name"],
        "model_snapshot": metadata["model_snapshot"],
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
        "docs_path": str(docs_path),
    }


def build_dense_index(papers_path: Path, index_path: Path, model_name: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_CACHE) -> dict:
    _meta_path(index_path)
    if not papers_path.is_file():
        raise FileNotFoundError(f"Dense 子块语料不存在: {papers_path}")
    model = load_model(model_name, cache_dir, offline=False)
    batch_size = _embedding_batch_size()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path = _dense_docs_path(index_path)
    temporary_docs = docs_path.with_name(docs_path.name + ".tmp")
    temporary_docs.unlink(missing_ok=True)
    docs_connection = sqlite3.connect(temporary_docs)
    docs_connection.execute("PRAGMA journal_mode=OFF")
    docs_connection.execute("PRAGMA synchronous=OFF")
    docs_connection.execute("CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, paper_id TEXT NOT NULL, byte_offset INTEGER NOT NULL)")

    corpus_digest = hashlib.sha256()
    scope: str | None = None
    document_count = 0
    faiss_index = None
    texts: list[str] = []
    paper_ids: list[str] = []
    byte_offsets: list[int] = []

    def encode_batch() -> None:
        nonlocal faiss_index, document_count, texts, paper_ids, byte_offsets
        if not texts:
            return
        vectors = _normalize(np.asarray(
            list(model.passage_embed(texts, batch_size=batch_size)), dtype=np.float32))
        if vectors.shape[0] != len(texts):
            raise ValueError("embedding 数量与当前批次文档数不一致")
        if faiss_index is None:
            faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        elif faiss_index.d != vectors.shape[1]:
            raise ValueError("不同批次的 embedding 维度不一致")
        first_doc_id = document_count
        faiss_index.add(vectors)
        docs_connection.executemany(
            "INSERT INTO docs(doc_id, paper_id, byte_offset) VALUES (?, ?, ?)",
            ((first_doc_id + i, paper_ids[i], byte_offsets[i]) for i in range(len(texts))),
        )
        document_count += len(texts)
        if document_count % 10000 < len(texts):
            docs_connection.commit()
            print(f"Dense 编码进度：{document_count} 个向量", flush=True)
        texts, paper_ids, byte_offsets = [], [], []

    try:
        with papers_path.open("rb") as source:
            while True:
                byte_offset = source.tell()
                line = source.readline()
                if not line:
                    break
                corpus_digest.update(line)
                if not line.strip():
                    continue
                paper = json.loads(line)
                current_scope = paper.get("evidence_scope", "abstract")
                if current_scope not in {"abstract", "full_text_chunk"}:
                    raise ValueError("Dense 语料证据范围不一致")
                if scope is None:
                    scope = current_scope
                elif current_scope != scope:
                    raise ValueError("Dense 语料证据范围不一致")
                texts.append(str(paper.get("title") or "") + "\n" + str(paper.get("abstract") or ""))
                paper_ids.append(str(paper.get("paper_id", document_count + len(texts) - 1)))
                byte_offsets.append(byte_offset)
                if len(texts) >= batch_size:
                    encode_batch()
        encode_batch()
        if not document_count or faiss_index is None or scope is None:
            raise ValueError("没有可建立 Dense 索引的文档")
        docs_connection.commit()
        docs_connection.close()
    except Exception:
        docs_connection.close()
        temporary_docs.unlink(missing_ok=True)
        raise

    metadata = {
        "evidence_scope": "full_text_chunk" if scope == "full_text_chunk" else "abstract_only",
        "corpus_path": papers_path.as_posix(),
        "corpus_sha256": corpus_digest.hexdigest(),
        "model_name": model_name,
        "model_snapshot": _snapshot(model),
        "embedding_mode": "passage_embed/query_embed",
        "embedding_backend": getattr(model, "backend_name", "fastembed"),
        "backend_version": _embedding_backend_version(),
        "dimension": faiss_index.d,
        "documents": document_count,
        "batch_size": batch_size,
    }
    return _write_streamed_index(index_path, faiss_index, metadata, temporary_docs)


def migrate_legacy_index(legacy_path: Path, index_path: Path) -> dict:
    """Convert a validated v2 JSON index without running the embedding model again."""
    _meta_path(index_path)
    if not legacy_path.is_file():
        raise ValueError(f"旧版向量索引不存在: {legacy_path}")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    if legacy.get("version") != 2 or legacy.get("evidence_scope") != "abstract_only" or legacy.get("embedding_mode") != "passage_embed/query_embed":
        raise ValueError("旧版向量索引版本或证据范围不兼容，请重建")
    papers = _validate_corpus(legacy)
    dimension = legacy.get("dimension")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("旧版向量索引维度无效，请重建")
    vectors = np.asarray(legacy.get("vectors"), dtype=np.float32)
    if vectors.shape != (len(papers), dimension) or not np.isfinite(vectors).all():
        raise ValueError("旧版向量索引维度或数值无效，请重建")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(lengths - 1) > 1e-3):
        raise ValueError("旧版向量未归一化，无法安全迁移；请重建")
    metadata = {key: value for key, value in legacy.items() if key != "vectors"}
    metadata["migrated_from"] = legacy_path.as_posix()
    metadata["legacy_sha256"] = _sha256(legacy_path)
    return _write_index(index_path, vectors, metadata)


def _validate_docs_mapping(connection, corpus_path: Path, docs_path: Path,
                           count: int, *, validation_cache: dict | None = None) -> str:
    """Bind legacy v4 mappings without a published digest to every corpus row."""
    key = ("dense_mapping", str(docs_path), str(corpus_path))
    signature = (file_signature(corpus_path), file_signature(docs_path), count)
    if validation_cache is not None and validation_cache.get(key) == signature:
        return "full_mapping_scan_reused"
    rows = iter(connection.execute("SELECT doc_id, paper_id, byte_offset FROM docs ORDER BY doc_id"))
    seen = 0
    with corpus_path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = next(rows, None)
            if row is None or row[0] != seen or row[2] != offset:
                raise ValueError(f"Dense 映射的向量 ID/子块偏移不一致：doc_id={seen}")
            paper = json.loads(line)
            if not isinstance(paper, dict) or str(paper.get("paper_id", "")) != row[1]:
                raise ValueError(f"Dense 映射的子块 ID 与原文不一致：doc_id={seen}")
            seen += 1
    if seen != count or next(rows, None) is not None:
        raise ValueError("Dense 映射/原文子块数量不一致")
    if (file_signature(corpus_path), file_signature(docs_path), count) != signature:
        raise ValueError("Dense 映射校验期间文件发生变化")
    if validation_cache is not None:
        validation_cache[key] = signature
    return "full_mapping_scan"


def load_dense_index(path: Path, *, corpus_path: Path | None = None,
                     docs_path: Path | None = None,
                     validation_cache: dict | None = None) -> dict:
    """Read existing FAISS artifacts; override relocation paths only in memory."""
    path = Path(path)
    meta_path = _meta_path(path)
    if not path.is_file() or not meta_path.is_file():
        raise ValueError(f"FAISS 向量索引或元数据不存在: {path}、{meta_path}。请运行 python lit.py index-dense 或 migrate-dense")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("FAISS 元数据必须是 JSON 对象")
    if (metadata.get("version") not in {INDEX_VERSION, STREAMING_INDEX_VERSION}
            or metadata.get("index_backend") != "faiss"
            or metadata.get("index_type") != "IndexFlatIP" or metadata.get("evidence_scope") not in {"abstract_only", "full_text_chunk"}
            or metadata.get("embedding_mode") != "passage_embed/query_embed"):
        raise ValueError("FAISS 向量索引版本或证据范围不兼容，请重建")
    dimension = metadata.get("dimension")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("FAISS 元数据缺少有效维度")
    for field in ("model_name", "model_snapshot"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"FAISS 元数据缺少模型信息：{field}")
    stored_corpus = metadata.get("corpus_path")
    if corpus_path is None and (not isinstance(stored_corpus, str) or not stored_corpus):
        raise ValueError("FAISS 元数据缺少语料路径")
    corpus_path = Path(corpus_path if corpus_path is not None else stored_corpus).resolve()
    metadata = {**metadata, "stored_corpus_path": stored_corpus, "corpus_path": corpus_path.as_posix()}
    signatures = {"index": file_signature(path), "metadata": file_signature(meta_path)}
    docs_connection = None
    try:
        verify_sha256(path, metadata.get("index_sha256"), cache=validation_cache)
        verify_sha256(corpus_path, metadata.get("corpus_sha256"), cache=validation_cache)
        signatures["corpus"] = file_signature(corpus_path)
        mapping_validation = None
        if metadata["version"] == STREAMING_INDEX_VERSION:
            expected_count = metadata.get("documents")
            if type(expected_count) is not int or expected_count < 1:
                raise ValueError("Dense 元数据缺少有效文档数量")
            stored_docs = metadata.get("docs_path")
            if docs_path is None and (not isinstance(stored_docs, str) or not stored_docs):
                raise ValueError("Dense 元数据缺少映射库路径")
            docs_path = Path(docs_path if docs_path is not None else stored_docs).resolve()
            if not docs_path.is_file():
                raise ValueError(f"Dense 文档偏移数据库不存在: {docs_path}")
            metadata.update({"stored_docs_path": stored_docs, "docs_path": docs_path.as_posix()})
            signatures["docs"] = file_signature(docs_path)
            docs_connection = sqlite3.connect(docs_path.as_uri() + "?mode=ro", uri=True)
            docs_connection.execute("PRAGMA query_only=ON")
            schema = list(docs_connection.execute("PRAGMA table_info(docs)"))
            columns = {row[1] for row in schema}
            if not {"doc_id", "paper_id", "byte_offset"}.issubset(columns):
                raise ValueError("Dense 映射库表结构不兼容")
            primary_keys = [row for row in schema if row[5]]
            if (len(primary_keys) != 1 or primary_keys[0][1] != "doc_id"
                    or primary_keys[0][2].upper() != "INTEGER"):
                raise ValueError("Dense 映射库需要唯一的整数向量 ID")
            count, minimum, maximum = docs_connection.execute("SELECT COUNT(*), MIN(doc_id), MAX(doc_id) FROM docs").fetchone()
            if (count != expected_count or minimum != 0 or maximum != expected_count - 1):
                raise ValueError("Dense 文档偏移数据库的数量/向量 ID 不一致")
            if metadata.get("docs_sha256"):
                verify_sha256(docs_path, metadata["docs_sha256"], cache=validation_cache)
                mapping_validation = "docs_sha256"
            else:
                mapping_validation = _validate_docs_mapping(docs_connection, corpus_path, docs_path,
                                                            expected_count, validation_cache=validation_cache)
            papers = None
        else:
            papers = _validate_corpus(metadata, validation_cache=validation_cache)
            expected_count = len(papers)
        try:
            faiss_index = faiss.read_index(str(path))
        except RuntimeError as exc:
            raise ValueError("FAISS 文件无法读取，加载已停止") from exc
        if (not isinstance(faiss_index, faiss.IndexFlatIP)
                or faiss_index.metric_type != faiss.METRIC_INNER_PRODUCT
                or faiss_index.ntotal != expected_count or faiss_index.d != dimension):
            raise ValueError("FAISS 索引类型、数量或维度与语料不一致")
        if any(file_signature(Path(signature[0])) != signature for signature in signatures.values()):
            raise ValueError("Dense 加载期间产物发生变化")
        return {**metadata, "papers": papers, "docs_connection": docs_connection,
                "faiss_index": faiss_index, "mapping_validation": mapping_validation,
                "artifact_signatures": signatures}
    except (sqlite3.Error, json.JSONDecodeError, UnicodeError) as exc:
        if docs_connection is not None:
            docs_connection.close()
        raise ValueError("Dense 映射库/语料损坏，加载已停止") from exc
    except BaseException:
        if docs_connection is not None:
            docs_connection.close()
        raise


def dense_search(index: dict, question: str, model, top_k: int = 5) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if not question.strip():
        return []
    if any(file_signature(Path(signature[0])) != signature
           for signature in index.get("artifact_signatures", {}).values()):
        raise ValueError("Dense 产物在加载后发生变化，请重新验证")
    if model.model_name != index["model_name"] or _snapshot(model) != index["model_snapshot"]:
        raise ValueError("查询模型与向量索引的模型版本不一致，请重建索引")
    query = _normalize(np.asarray(list(model.query_embed(question)), dtype=np.float32))
    if query.shape != (1, index["dimension"]):
        raise ValueError("查询向量维度与索引不一致")
    document_count = index.get("documents", len(index.get("papers") or []))
    if not document_count:
        return []
    scores, doc_ids = index["faiss_index"].search(query, min(top_k, document_count))
    if index.get("version") == STREAMING_INDEX_VERSION:
        candidates = [(float(score), int(doc_id)) for score, doc_id in zip(scores[0], doc_ids[0])
                     if int(doc_id) >= 0]
        connection = index["docs_connection"]
        id_list = [doc_id for _, doc_id in candidates]
        placeholders = ",".join("?" for _ in id_list)
        metadata_rows = connection.execute(
            f"SELECT doc_id, paper_id, byte_offset FROM docs WHERE doc_id IN ({placeholders})",
            id_list,
        ).fetchall()
        by_id = {row[0]: (row[1], row[2]) for row in metadata_rows}
        ranked = sorted(candidates, key=lambda item: (-item[0], by_id[item[1]][0]))
        results = []
        corpus_path = Path(index["corpus_path"])
        with corpus_path.open("rb") as source:
            for score, doc_id in ranked:
                source.seek(by_id[doc_id][1])
                paper = json.loads(source.readline())
                if str(paper.get("paper_id", "")) != by_id[doc_id][0]:
                    raise ValueError("Dense 命中偏移对应的子块 ID 不一致")
                results.append({**paper, "score": round(score, 5), "retrieval_method": "dense"})
        return results
    papers = index["papers"]
    if not papers:
        return []
    ranked = sorted(zip(scores[0], doc_ids[0]), key=lambda item: (-float(item[0]), papers[int(item[1])]["paper_id"]))
    return [{**papers[int(doc_id)], "score": round(float(score), 5), "retrieval_method": "dense"} for score, doc_id in ranked]
