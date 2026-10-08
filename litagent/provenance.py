"""Collect reproducibility metadata without reading secrets or loading models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from .artifacts import file_sha256


CODE_MODULES = (
    "litagent/corpus.py", "litagent/litsearch_data_preparation.py",
    "litagent/retrieval.py", "litagent/dense.py", "litagent/reranker.py",
    "litagent/agentic_retrieval.py", "litagent/fulltext_answer.py",
    "litagent/evidence.py", "litagent/runtime.py", "litsearch_fulltext.py",
    "litagent/fulltext_structure.py", "litagent/paper_understanding.py",
    "litagent/related_work.py",
    "litsearch_p1.py", "litagent/performance.py", "litsearch_maintenance.py",
    "rag/generate.py", "litsearch_e2e.py", "litsearch_benchmark.py",
    "litagent/artifacts.py", "litagent/local_index_validation.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for(data_path: Path, kind: str = "parent") -> tuple[dict | None, Path | None]:
    candidates = [data_path.with_suffix(".manifest.json")]
    if kind == "parent":
        candidates.extend((data_path.parent / "corpus_fulltext_manifest.json", data_path.parent / "manifest.json"))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and candidate != data_path:
            try:
                return json.loads(candidate.read_text(encoding="utf-8")), candidate
            except (OSError, json.JSONDecodeError):
                return {"_error": "unreadable manifest"}, candidate
    return None, None


def _safe_base_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "configured (redacted)"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = host + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit((parsed.scheme, netloc, "", "", ""))
    except ValueError:
        return "configured (redacted)"


def _snapshot(cache: Path, model_name: str) -> str | None:
    slug = "models--" + model_name.replace("/", "--")
    refs = cache / slug / "refs"
    try:
        for ref in (refs / "main", refs / "master"):
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    snapshots = cache / slug / "snapshots"
    try:
        found = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        return found[-1] if found else None
    except OSError:
        return None


def _version(*names: str) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def collect_provenance(parent_path, child_path, bm25_path=None, dense_path=None,
                       rerank=False, parameters=None, *, dense_docs_path=None,
                       validation_cache=None) -> dict:
    """Return paths, content hashes, manifests, model configuration and runtime versions."""
    parent, child = Path(parent_path), Path(child_path)
    files: dict[str, dict | None] = {}
    for name, path in (("parent", parent), ("child", child), ("bm25", bm25_path), ("dense", dense_path)):
        files[name] = ({"path": str(Path(path).resolve()), "exists": Path(path).is_file(),
                        "sha256": file_sha256(path, cache=validation_cache) if Path(path).is_file() else None} if path else None)
    dense_meta_path = Path(dense_path).with_suffix(".meta.json") if dense_path else None
    files["dense_meta"] = ({"path": str(dense_meta_path.resolve()), "exists": dense_meta_path.is_file(),
                            "sha256": sha256_file(dense_meta_path) if dense_meta_path.is_file() else None}
                           if dense_meta_path else None)
    manifests = {}
    for name, path in (("parent", parent), ("child", child)):
        content, manifest_path = _manifest_for(path, name)
        manifests[name] = {"path": str(manifest_path.resolve()), "metadata": content} if manifest_path else None
    dense_meta = {}
    if dense_meta_path and dense_meta_path.is_file():
        try:
            dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dense_meta = {}
    docs_path = dense_docs_path or (dense_meta.get("docs_path") if dense_meta.get("version") == 4 else None)
    files["dense_docs"] = ({"path": str(Path(docs_path).resolve()), "exists": Path(docs_path).is_file(),
                            "sha256": file_sha256(docs_path, cache=validation_cache) if Path(docs_path).is_file() else None}
                           if docs_path else None)
    try:
        from litagent.dense import DEFAULT_MODEL, HARRIER_MODEL, DEFAULT_CACHE
        from litagent.reranker import DEFAULT_RERANK_MODEL
    except ImportError:
        DEFAULT_MODEL, HARRIER_MODEL, DEFAULT_CACHE, DEFAULT_RERANK_MODEL = "BAAI/bge-small-en-v1.5", "microsoft/harrier-oss-v1-0.6b", Path(".rag/models"), "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_model = dense_meta.get("model_name") or DEFAULT_MODEL
    rerank_model = DEFAULT_RERANK_MODEL
    cache = Path(os.getenv("LIT_MODEL_CACHE", str(DEFAULT_CACHE)))
    code = {}
    for relative in CODE_MODULES:
        path = Path(relative)
        if path.is_file():
            code[relative] = sha256_file(path)
    return {
        "files": files,
        "manifests": manifests,
        "models": {
            "chat": {"model": os.getenv("RAG_CHAT_MODEL", "deepseek-flash"),
                     "base_url": _safe_base_url(os.getenv("RAG_API_BASE_URL", "https://api.deepseek.com"))},
            "dense": {"model": dense_model, "snapshot": dense_meta.get("model_snapshot") or _snapshot(cache, dense_model)},
            "reranker": ({"model": rerank_model, "snapshot": _snapshot(cache, rerank_model)} if rerank else None),
        },
        "software": {"python": platform.python_version(), "dependencies": _version("numpy", "faiss-cpu", "fastembed", "sentence-transformers", "openai")},
        "code": code,
        "parameters": parameters or {},
    }
