from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("numpy"):
    import numpy as np
else:
    np = None

from litagent.hybrid import fuse_rrf

ROOT = Path(__file__).resolve().parents[1]


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_deduplicates_and_preserves_component_ranks(self):
        sparse = [{"paper_id": "a"}, {"paper_id": "b"}]
        dense = [{"paper_id": "b"}, {"paper_id": "c"}]
        hits = fuse_rrf(sparse, dense, top_k=3)
        self.assertEqual([hit["paper_id"] for hit in hits], ["b", "a", "c"])
        self.assertEqual((hits[0]["bm25_rank"], hits[0]["dense_rank"]), (2, 1))
        self.assertEqual([hit["rrf_rank"] for hit in hits], [1, 2, 3])
        self.assertEqual(hits[0]["rrf_score"], round(1 / 62 + 1 / 61, 8))

    @unittest.skipUnless(importlib.util.find_spec("faiss") and np is not None, "FAISS 或 numpy 未安装")
    def test_faiss_index_load_and_top_k_search(self):
        import faiss
        from litagent.dense import dense_search, load_dense_index

        class QueryModel:
            model_name = "test/model"
            snapshot = "test-snapshot"

            def query_embed(self, _question):
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "papers.jsonl"
            papers = [
                {"paper_id": "best", "title": "Best", "abstract": "closest", "source_url": "https://arxiv.org/abs/a", "evidence_scope": "abstract"},
                {"paper_id": "second", "title": "Second", "abstract": "next", "source_url": "https://arxiv.org/abs/b", "evidence_scope": "abstract"},
                {"paper_id": "last", "title": "Last", "abstract": "far", "source_url": "https://arxiv.org/abs/c", "evidence_scope": "abstract"},
            ]
            corpus_path.write_text("".join(json.dumps(row) + "\n" for row in papers), encoding="utf-8")
            index_path = root / "dense.faiss"
            raw_index = faiss.IndexFlatIP(2)
            raw_index.add(np.asarray([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32))
            faiss.write_index(raw_index, str(index_path))
            metadata = {
                "version": 3,
                "evidence_scope": "abstract_only",
                "embedding_mode": "passage_embed/query_embed",
                "index_backend": "faiss",
                "index_type": "IndexFlatIP",
                "corpus_path": str(corpus_path),
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "model_name": "test/model",
                "model_snapshot": "test-snapshot",
                "dimension": 2,
                "paper_ids": [paper["paper_id"] for paper in papers],
            }
            index_path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")

            index = load_dense_index(index_path)
            self.assertEqual(index["faiss_index"].ntotal, 3)
            hits = dense_search(index, "query", QueryModel(), top_k=2)
            self.assertEqual([hit["paper_id"] for hit in hits], ["best", "second"])

    @unittest.skipUnless(importlib.util.find_spec("faiss") and np is not None, "FAISS 或 numpy 未安装")
    def test_faiss_index_rejects_changed_corpus(self):
        from litagent.dense import load_dense_index

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "papers.jsonl"
            corpus_path.write_text('{"paper_id":"a","title":"A","abstract":"text","source_url":"https://arxiv.org/abs/a","evidence_scope":"abstract"}\n', encoding="utf-8")
            index_path = root / "dense.faiss"
            index_path.write_bytes(b"placeholder")
            metadata = {
                "version": 3,
                "evidence_scope": "abstract_only",
                "embedding_mode": "passage_embed/query_embed",
                "index_backend": "faiss",
                "index_type": "IndexFlatIP",
                "corpus_path": str(corpus_path),
                "corpus_sha256": hashlib.sha256(b"old corpus").hexdigest(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "model_name": "test/model",
                "model_snapshot": "test-snapshot",
                "dimension": 2,
                "paper_ids": ["a"],
            }
            index_path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 不匹配"):
                load_dense_index(index_path)


if __name__ == "__main__":
    unittest.main()
