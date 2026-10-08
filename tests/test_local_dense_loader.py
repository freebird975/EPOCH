from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from litagent.artifacts import file_sha256
from litagent.dense import DEFAULT_MODEL, dense_search, load_dense_index
from litagent.local_index_validation import validate_local_indexes
from litagent.retrieval import build_index
from litsearch_fulltext import close_retrieval_resources, retrieve_parents


class QueryModel:
    model_name = DEFAULT_MODEL
    snapshot = "fixture-snapshot"

    def query_embed(self, question):
        result = np.zeros((1, 384), dtype=np.float32)
        result[0, 0] = 1
        return result


class LocalDenseLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "relocated # indexes"
        self.root.mkdir()
        self.parent = self.root / "parents.jsonl"
        self.child = self.root / "chunks.jsonl"
        self.path = self.root / "dense index #1.faiss"
        self.docs = self.root / "docs # local.sqlite3"
        self.meta_path = self.path.with_suffix(".meta.json")
        self.bm25 = self.root / "bm25.sqlite3"
        parents = [{"paper_id": key, "title": title, "abstract": "", "full_text": text,
                    "evidence_scope": "full_text", "source_url": f"https://example.org/{key}"}
                   for key, title, text in [("p1", "Graph retrieval", "Graph retrieval evidence"),
                                             ("p2", "Language models", "Language model evidence")]]
        chunks = [{**parent, "paper_id": f"{parent['paper_id']}#0", "parent_id": parent["paper_id"],
                   "chunk_id": f"{parent['paper_id']}#0", "chunk_index": 0,
                   "abstract": parent["full_text"], "text": parent["full_text"],
                   "section": "Body", "evidence_scope": "full_text_chunk"}
                  for parent in parents]
        self.parent.write_text("".join(json.dumps(row) + "\n" for row in parents), encoding="utf-8")
        offsets = []
        with self.child.open("wb") as stream:
            for row in chunks:
                offsets.append(stream.tell())
                stream.write((json.dumps(row) + "\n").encode())
        connection = sqlite3.connect(self.docs)
        connection.execute("CREATE TABLE docs(doc_id INTEGER PRIMARY KEY, paper_id TEXT NOT NULL, byte_offset INTEGER NOT NULL)")
        connection.executemany("INSERT INTO docs VALUES (?, ?, ?)",
                               [(i, row["paper_id"], offsets[i]) for i, row in enumerate(chunks)])
        connection.commit()
        connection.close()
        vectors = np.zeros((2, 384), dtype=np.float32)
        vectors[0, 0] = vectors[1, 1] = 1
        index = faiss.IndexFlatIP(384)
        index.add(vectors)
        faiss.write_index(index, str(self.path))
        self.meta = {"version": 4, "index_backend": "faiss", "index_type": "IndexFlatIP",
                     "evidence_scope": "full_text_chunk", "embedding_mode": "passage_embed/query_embed",
                     "model_name": DEFAULT_MODEL, "model_snapshot": "fixture-snapshot",
                     "corpus_path": "/root/autodl-tmp/unavailable/chunks.jsonl",
                     "docs_path": "/root/autodl-tmp/unavailable/docs.sqlite3",
                     "corpus_sha256": file_sha256(self.child), "index_sha256": file_sha256(self.path),
                     "dimension": 384, "documents": 2}
        self.write_metadata()
        self.child.with_suffix(".manifest.json").write_text(json.dumps({
            "parents": 2, "chunks": 2, "parent_sha256": file_sha256(self.parent),
            "chunk_sha256": file_sha256(self.child), "parser_version": "fixture"}), encoding="utf-8")
        build_index(self.child, self.bm25)
        connection = sqlite3.connect(self.bm25)
        connection.execute("UPDATE metadata SET value=? WHERE key='corpus_path'",
                           (json.dumps("/root/autodl-tmp/unavailable/chunks.jsonl"),))
        connection.commit()
        connection.close()

    def write_metadata(self):
        self.meta_path.write_text(json.dumps(self.meta), encoding="utf-8")

    def load(self, **kwargs):
        return load_dense_index(self.path, corpus_path=self.child, docs_path=self.docs, **kwargs)

    def sql(self, statement):
        connection = sqlite3.connect(self.docs)
        try:
            connection.execute(statement)
            connection.commit()
        finally:
            connection.close()

    def test_relocated_legacy_mapping_loads_readonly_and_preserves_artifacts(self):
        paths = (self.path, self.meta_path, self.docs, self.child)
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
        with patch("litagent.dense.build_dense_index", side_effect=AssertionError("rebuild")), \
                patch("litagent.dense.load_model", side_effect=AssertionError("model")):
            index = self.load()
        try:
            self.assertEqual(index["stored_corpus_path"], self.meta["corpus_path"])
            self.assertEqual(index["stored_docs_path"], self.meta["docs_path"])
            self.assertEqual(index["corpus_path"], self.child.as_posix())
            self.assertEqual(index["mapping_validation"], "full_mapping_scan")
            self.assertEqual((index["faiss_index"].ntotal, index["faiss_index"].d), (2, 384))
            self.assertEqual(dense_search(index, "graph", QueryModel(), 1)[0]["paper_id"], "p1#0")
            with self.assertRaises(sqlite3.OperationalError):
                index["docs_connection"].execute("UPDATE docs SET byte_offset=0")
        finally:
            index["docs_connection"].close()
        self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in paths], before)

    def test_missing_override_and_missing_sidecar_fail_without_creating_files(self):
        with self.assertRaises(ValueError):
            load_dense_index(self.path)
        missing = self.root / "missing.sqlite3"
        with self.assertRaises(ValueError):
            load_dense_index(self.path, corpus_path=self.child, docs_path=missing)
        self.assertFalse(missing.exists())

    def test_wrong_corpus_and_modified_faiss_are_rejected(self):
        wrong = self.root / "wrong.jsonl"
        wrong.write_text("different", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_dense_index(self.path, corpus_path=wrong, docs_path=self.docs)
        self.path.write_bytes(self.path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load()

    def test_invalid_mapping_id_offset_and_count_are_rejected(self):
        original = self.docs.read_bytes()
        for sql in ["UPDATE docs SET paper_id='wrong' WHERE doc_id=1",
                    "UPDATE docs SET byte_offset=1 WHERE doc_id=1", "DELETE FROM docs WHERE doc_id=1"]:
            with self.subTest(sql=sql):
                self.sql(sql)
                with self.assertRaisesRegex(ValueError, "映射|偏移"):
                    self.load()
                self.docs.write_bytes(original)

    def test_v4_with_docs_digest_rejects_tampering(self):
        self.meta["docs_sha256"] = file_sha256(self.docs)
        self.write_metadata()
        index = self.load()
        self.assertEqual(index["mapping_validation"], "docs_sha256")
        index["docs_connection"].close()
        self.sql("UPDATE docs SET paper_id='wrong' WHERE doc_id=1")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load()

    def test_composite_primary_key_is_rejected(self):
        connection = sqlite3.connect(self.docs)
        connection.execute("ALTER TABLE docs RENAME TO old_docs")
        connection.execute("CREATE TABLE docs(doc_id INTEGER, paper_id TEXT, byte_offset INTEGER, PRIMARY KEY(doc_id,paper_id))")
        connection.execute("INSERT INTO docs SELECT * FROM old_docs")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(ValueError, "唯一"):
            self.load()

    def test_dimension_count_model_and_index_type_mismatch_fail(self):
        for key, bad in [("dimension", 385), ("documents", 3), ("model_snapshot", "")]:
            with self.subTest(key=key):
                original = self.meta[key]
                self.meta[key] = bad
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.load()
                self.meta[key] = original
        self.write_metadata()
        other = faiss.IndexFlatL2(384)
        other.add(np.zeros((2, 384), dtype=np.float32))
        faiss.write_index(other, str(self.path))
        self.meta["index_sha256"] = file_sha256(self.path)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "类型"):
            self.load()

    def test_mapping_scan_cache_reused_only_while_artifacts_unchanged(self):
        cache = {}
        first = self.load(validation_cache=cache)
        first["docs_connection"].close()
        second = self.load(validation_cache=cache)
        self.assertEqual(second["mapping_validation"], "full_mapping_scan_reused")
        second["docs_connection"].close()
        self.sql("UPDATE docs SET byte_offset=1 WHERE doc_id=1")
        with self.assertRaisesRegex(ValueError, "偏移"):
            self.load(validation_cache=cache)

    def test_changed_artifact_after_loading_is_rejected_before_query_embedding(self):
        index = self.load()
        try:
            self.child.write_bytes(self.child.read_bytes() + b"\n")
            model = QueryModel()
            with patch.object(model, "query_embed", side_effect=AssertionError("embedding")):
                with self.assertRaisesRegex(ValueError, "发生变化"):
                    dense_search(index, "graph", model)
        finally:
            index["docs_connection"].close()

    def test_validator_loads_both_without_builds_or_model_and_releases_connections(self):
        report_path = self.root / "validation.json"
        with patch("litagent.retrieval.build_index", side_effect=AssertionError("BM25 build")), \
                patch("litagent.dense.build_dense_index", side_effect=AssertionError("Dense build")), \
                patch("litagent.dense.load_model", side_effect=AssertionError("model")):
            report = validate_local_indexes(self.parent, self.child, self.bm25, self.path,
                                             dense_docs_path=self.docs, report_path=report_path)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["bm25"]["documents"], report["dense"]["documents"])
        self.assertTrue(report["validation"]["bm25_query_only"])
        self.assertTrue(report["validation"]["dense_docs_query_only"])
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "passed")
        # Windows would reject unlink if either SQLite handle were still open.
        self.docs.unlink()
        self.bm25.unlink()

    def test_validator_refuses_report_overwriting_artifact(self):
        before = self.child.read_bytes()
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            validate_local_indexes(self.parent, self.child, self.bm25, self.path,
                                   dense_docs_path=self.docs, report_path=self.child)
        self.assertEqual(self.child.read_bytes(), before)

    def test_query_entry_passes_explicit_paths_reuses_handles_and_never_repairs_existing_index(self):
        resources = {}
        params = dict(parent_path=self.parent, child_path=self.child, index_path=self.bm25,
                      dense_index_path=self.path, dense_docs_path=self.docs, retriever="hybrid",
                      pipeline="classic", resources=resources)
        try:
            with patch("litagent.dense.load_model", return_value=QueryModel()), \
                    patch("litsearch_fulltext.build_index", side_effect=AssertionError("rebuild")), \
                    patch("litagent.dense.build_dense_index", side_effect=AssertionError("rebuild")):
                first, _ = retrieve_parents("graph", **params)
                connection = resources["dense"][1]["docs_connection"]
                second, _ = retrieve_parents("graph", **params)
                self.assertIs(resources["dense"][1]["docs_connection"], connection)
                self.assertEqual(first[0]["paper_id"], second[0]["paper_id"])
                self.sql("UPDATE docs SET byte_offset=1 WHERE doc_id=1")
                with self.assertRaisesRegex(ValueError, "偏移"):
                    retrieve_parents("graph", **params, build_missing_indexes=True)
        finally:
            close_retrieval_resources(resources)


class ArtifactCacheTests(unittest.TestCase):
    def test_streamed_hash_cache_invalidates_when_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"
            path.write_bytes(b"first")
            cache = {}
            self.assertEqual(file_sha256(path, cache=cache), hashlib.sha256(b"first").hexdigest())
            with patch("litagent.artifacts.hashlib.sha256", side_effect=AssertionError("rehash")):
                file_sha256(path, cache=cache)
            path.write_bytes(b"changed")
            self.assertEqual(file_sha256(path, cache=cache), hashlib.sha256(b"changed").hexdigest())


if __name__ == "__main__":
    unittest.main()
