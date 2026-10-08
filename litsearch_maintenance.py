"""Audit and explicitly rebuild the LitSearch full-text retrieval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_rows(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} 无效 JSON: {exc}") from exc


def _manifest_path(data: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    for candidate in (data.with_suffix(".manifest.json"), data.parent / "corpus_fulltext_manifest.json", data.parent / "manifest.json"):
        if candidate.is_file() and candidate != data:
            return candidate
    return None


def audit(parent: Path, child: Path, *, parent_manifest: Path | None = None,
          child_manifest: Path | None = None, bm25: Path | None = None,
          dense: Path | None = None) -> dict:
    checks: list[dict] = []
    def add(name: str, status: str, detail: str = ""):
        checks.append({"check": name, "status": status, "detail": detail})

    if not parent.is_file():
        add("parent", "missing", str(parent))
    if not child.is_file():
        add("child", "missing", str(child))
    parents = chunks = []
    if parent.is_file():
        try:
            parents = list(_read_rows(parent))
            ids = [str(row.get("paper_id", row.get("corpusid", ""))) for row in parents]
            add("parent_rows", "ok" if parents and all(ids) and len(ids) == len(set(ids)) else "corrupt",
                f"rows={len(parents)}")
        except (OSError, ValueError, TypeError) as exc:
            add("parent_rows", "corrupt", str(exc))
    if child.is_file():
        try:
            chunks = list(_read_rows(child))
            ids = [str(row.get("chunk_id", row.get("paper_id", ""))) for row in chunks]
            add("child_rows", "ok" if chunks and all(ids) and len(ids) == len(set(ids)) else "corrupt",
                f"rows={len(chunks)}")
            parent_ids = {str(row.get("paper_id", row.get("corpusid", ""))) for row in parents}
            missing = sorted({str(row.get("parent_id", "")) for row in chunks} - parent_ids)
            add("child_parent_ids", "ok" if not missing else "stale", f"missing parents={len(missing)}")
        except (OSError, ValueError, TypeError) as exc:
            add("child_rows", "corrupt", str(exc))
    for label, data, explicit, fields in (
        ("parent_manifest", parent, parent_manifest, ("parent_sha256", "corpus_sha256")),
        ("child_manifest", child, child_manifest, ("chunk_sha256", "child_sha256", "corpus_sha256")),
    ):
        manifest_file = _manifest_path(data, explicit)
        if not manifest_file:
            add(label, "missing", "manifest not found")
            continue
        try:
            info = json.loads(manifest_file.read_text(encoding="utf-8"))
            expected = next((info.get(key) for key in fields if info.get(key)), None)
            actual = digest(data) if data.is_file() else None
            if not expected:
                add(label, "missing", f"no hash field in {manifest_file}")
            else:
                add(label, "ok" if expected == actual else "stale", f"expected={expected} actual={actual}")
            if label == "child_manifest" and info.get("parent_sha256"):
                parent_hash = digest(parent) if parent.is_file() else None
                add("child_parent_hash", "ok" if info["parent_sha256"] == parent_hash else "stale",
                    "child manifest must bind the current parent corpus")
        except (OSError, json.JSONDecodeError) as exc:
            add(label, "corrupt", str(exc))

    bm25_hash = dense_hash = None
    if bm25:
        if not bm25.is_file():
            add("bm25", "missing", str(bm25))
        else:
            try:
                from litagent.retrieval import load_index
                index = load_index(bm25)
                bm25_hash = index.get("corpus_sha256")
                actual_child = digest(child) if child.is_file() else None
                ids_expected = [str(row.get("paper_id")) for row in chunks]
                ids_index = [str(row.get("paper_id")) for row in index.get("papers", [])]
                good = (bm25_hash == actual_child and ids_expected == ids_index
                        and index.get("evidence_scope") == "full_text_chunk")
                add("bm25", "ok" if good else "stale",
                    f"papers={len(ids_index)} scope={index.get('evidence_scope')}")
            except Exception as exc:
                add("bm25", "corrupt", str(exc))
    if dense:
        meta = dense.with_suffix(".meta.json")
        if not dense.is_file() or not meta.is_file():
            add("dense", "missing", f"index={dense.is_file()} metadata={meta.is_file()}")
        else:
            try:
                from litagent.dense import load_dense_index
                index = load_dense_index(dense)
                dense_hash = index.get("corpus_sha256")
                actual_child = digest(child) if child.is_file() else None
                ids_expected = [str(row.get("paper_id")) for row in chunks]
                ids_index = [str(row) for row in index.get("paper_ids", [])]
                if not ids_index:
                    ids_index = [str(row.get("paper_id")) for row in index.get("papers", [])]
                good = dense_hash == actual_child and ids_expected == ids_index and index.get("evidence_scope") == "full_text_chunk"
                add("dense", "ok" if good else "stale",
                    f"papers={len(ids_index)} scope={index.get('evidence_scope')} model={index.get('model_name')} snapshot={index.get('model_snapshot')}")
            except Exception as exc:
                add("dense", "corrupt", str(exc))
    if bm25 and dense and bm25_hash and dense_hash:
        add("index_corpus_consistency", "ok" if bm25_hash == dense_hash else "stale",
            f"bm25={bm25_hash} dense={dense_hash}")
    return {"ok": all(item["status"] == "ok" for item in checks), "checks": checks}


def _rewrite_corpus_path(index_file: Path, final_child: Path, *, dense: bool) -> None:
    if dense:
        meta_file = index_file.with_suffix(".meta.json")
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        data["corpus_path"] = final_child.resolve().as_posix()
        meta_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        data["corpus_path"] = final_child.resolve().as_posix()
        index_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def rebuild(parent: Path, child: Path, *, bm25: Path | None = None, dense: Path | None = None,
            model: str = "BAAI/bge-small-en-v1.5", model_cache: Path = Path(".rag/models"),
            chunk_size: int = 1800, overlap: int = 240) -> dict:
    if bm25 is None and dense is None:
        raise ValueError("请显式指定 --bm25 和/或 --dense")
    targets = [child, child.with_suffix(".manifest.json")]
    if bm25:
        targets.append(bm25)
    if dense:
        targets.extend((dense, dense.with_suffix(".meta.json")))
    resolved = [target.resolve() for target in targets]
    if parent.resolve() in resolved or len(resolved) != len(set(resolved)):
        raise ValueError("父文档、子块、manifest 和各索引路径必须互不相同")
    child.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".litsearch-rebuild-", dir=child.parent))
    stage_child = temp_root / child.name
    stage_bm25 = temp_root / "bm25_index.json" if bm25 else None
    stage_dense = temp_root / "dense_index.faiss" if dense else None
    try:
        from litsearch_fulltext import build_chunks
        prepared = build_chunks(parent, stage_child, chunk_size, overlap, structured=True)
        stage_manifest = stage_child.with_suffix(".manifest.json")
        manifest_data = json.loads(stage_manifest.read_text(encoding="utf-8"))
        manifest_data["parent_path"] = parent.resolve().as_posix()
        manifest_data["output"] = child.resolve().as_posix()
        stage_manifest.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
        if bm25:
            from litagent.retrieval import build_index, load_index
            build_index(stage_child, stage_bm25)
            load_index(stage_bm25)
        if dense:
            from litagent.dense import build_dense_index, load_dense_index
            build_dense_index(stage_child, stage_dense, model, model_cache)
            load_dense_index(stage_dense)
        if stage_bm25:
            _rewrite_corpus_path(stage_bm25, child, dense=False)
        if stage_dense:
            _rewrite_corpus_path(stage_dense, child, dense=True)
        pairs = [(stage_child, child), (stage_manifest, child.with_suffix(".manifest.json"))]
        if stage_bm25:
            pairs.append((stage_bm25, bm25))
        if stage_dense:
            pairs.extend(((stage_dense, dense), (stage_dense.with_suffix(".meta.json"), dense.with_suffix(".meta.json"))))
        backup_dir = temp_root / "backup"
        backup_dir.mkdir()
        backed_up, published = [], []
        try:
            for _, target in pairs:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    backup = backup_dir / f"{len(backed_up)}-{target.name}"
                    os.replace(target, backup)
                    backed_up.append((backup, target))
            for source, target in pairs:
                os.replace(source, target)
                published.append(target)
        except Exception:
            for target in reversed(published):
                target.unlink(missing_ok=True)
            for backup, target in reversed(backed_up):
                os.replace(backup, target)
            raise
        return {"chunks": prepared["chunks"], "parents": prepared["parents"],
                "child": str(child), "bm25": str(bm25) if bm25 else None,
                "dense": str(dense) if dense else None,
                "note": "完整重建；Dense 会对所选语料重新编码"}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def write_lock(path: Path = Path(".rag/runtime-lock.txt")) -> None:
    packages = ("pypdf", "fastembed", "fastembed-gpu", "faiss-cpu", "numpy",
                "onnxruntime", "onnxruntime-gpu", "huggingface-hub", "pyarrow", "fsspec")
    rows = []
    for package in packages:
        try:
            rows.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def source_snapshot(path: Path) -> dict:
    """Save executable source and dependency files; credentials/data are excluded."""
    root = Path(__file__).resolve().parent
    paths = [*root.glob("*.py"), *root.glob("requirements*.txt")]
    for package in ("litagent", "rag", "tests"):
        paths.extend((root / package).glob("*.py"))
    files = {item.relative_to(root).as_posix(): digest(item) for item in sorted(paths)}
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"source_sha256": signature, "files": files,
                "scope": "Python source/tests and dependency lists; no credentials, model weights or paper text"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(paths):
            archive.write(item, item.relative_to(root).as_posix())
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest["archive_sha256"] = digest(path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit")
    check.add_argument("--parent", type=Path, required=True)
    check.add_argument("--child", type=Path, required=True)
    check.add_argument("--parent-manifest", type=Path)
    check.add_argument("--child-manifest", type=Path)
    check.add_argument("--bm25", type=Path)
    check.add_argument("--dense", type=Path)
    build = sub.add_parser("rebuild")
    build.add_argument("--parent", type=Path, required=True)
    build.add_argument("--child", type=Path, required=True)
    build.add_argument("--bm25", type=Path)
    build.add_argument("--dense", type=Path)
    build.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    build.add_argument("--model-cache", type=Path, default=Path(".rag/models"))
    build.add_argument("--chunk-size", type=int, default=1800)
    build.add_argument("--overlap", type=int, default=240)
    lock = sub.add_parser("lock")
    lock.add_argument("--output", type=Path, default=Path(".rag/runtime-lock.txt"))
    models = sub.add_parser("cache-models", help="显式下载 BGE 和 Cross Encoder 到本地缓存")
    models.add_argument("--model-cache", type=Path, default=Path(".rag/models"))
    snapshot = sub.add_parser("source-snapshot", help="归档当前源码和依赖文件，供按哈希恢复提示词与实现")
    snapshot.add_argument("--output", type=Path, default=Path(".rag/repro/p2_source.zip"))
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit(args.parent, args.child, parent_manifest=args.parent_manifest,
                           child_manifest=args.child_manifest, bm25=args.bm25, dense=args.dense)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "rebuild":
            print(json.dumps(rebuild(args.parent, args.child, bm25=args.bm25, dense=args.dense,
                                     model=args.model, model_cache=args.model_cache,
                                     chunk_size=args.chunk_size, overlap=args.overlap), ensure_ascii=False, indent=2))
            return 0
        if args.command == "cache-models":
            from litagent.dense import load_model
            from litagent.reranker import load_reranker, reranker_metadata
            embedding = load_model("BAAI/bge-small-en-v1.5", args.model_cache, offline=False)
            reranker = load_reranker(cache_dir=args.model_cache, offline=False)
            print(json.dumps({"embedding_model": embedding.model_name,
                              **reranker_metadata(reranker)}, ensure_ascii=False))
            return 0
        if args.command == "source-snapshot":
            result = source_snapshot(args.output)
            print(json.dumps({"archive": str(args.output), "source_sha256": result["source_sha256"],
                              "files": len(result["files"])}, ensure_ascii=False))
            return 0
        write_lock(args.output)
        print(f"依赖锁已写入：{args.output}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
