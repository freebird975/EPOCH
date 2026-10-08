from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from litsearch_fulltext import build_chunks, ensure_index, retrieve_parents


class FakeBGE:
    model_name = "BAAI/bge-small-en-v1.5"
    snapshot = "test-snapshot"

    def passage_embed(self, texts, batch_size=16):
        return np.asarray([
            [1.0, 0.0] if "Graph retrieval" in text else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)

    def query_embed(self, question):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FulltextHybridTests(unittest.TestCase):
    def test_bm25_dense_rrf_on_child_chunks_then_parent_dedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = root / "parents.jsonl"
            chunks = root / "chunks.jsonl"
            bm25 = root / "bm25.json"
            dense = root / "dense.faiss"
            rows = [
                {"paper_id": "graph", "title": "Graph retrieval", "abstract": "",
                 "full_text": "Graph retrieval finds related papers in a network.",
                 "source_url": "https://example.org/graph", "evidence_scope": "full_text"},
                {"paper_id": "vector", "title": "Vector search", "abstract": "",
                 "full_text": "Vector search finds related papers using embeddings.",
                 "source_url": "https://example.org/vector", "evidence_scope": "full_text"},
            ]
            parents.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            build_chunks(parents, chunks, chunk_size=500, overlap=50)

            bm25_parents, _ = retrieve_parents(
                "graph retrieval", parent_path=parents, child_path=chunks,
                index_path=bm25, dense_index_path=dense, retriever="bm25",
                build_missing_indexes=True,
            )
            self.assertEqual(bm25_parents[0]["paper_id"], "graph")
            self.assertFalse(dense.exists())

            old_index = json.loads(bm25.read_text(encoding="utf-8"))
            old_index["evidence_scope"] = "abstract_only"
            bm25.write_text(json.dumps(old_index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不一致"):
                retrieve_parents("graph retrieval", parent_path=parents, child_path=chunks,
                                 index_path=bm25, dense_index_path=dense, retriever="hybrid")
            ensure_index(chunks, bm25)

            with patch("litagent.dense.load_model", return_value=FakeBGE()):
                hybrid_parents, hits = retrieve_parents(
                    "graph retrieval", parent_path=parents, child_path=chunks,
                    index_path=bm25, dense_index_path=dense, retriever="hybrid",
                    build_missing_indexes=True,
                )
                dense_parents, _ = retrieve_parents(
                    "graph retrieval", parent_path=parents, child_path=chunks,
                    index_path=bm25, dense_index_path=dense, retriever="dense",
                )
            self.assertEqual(hybrid_parents[0]["paper_id"], "graph")
            self.assertEqual(dense_parents[0]["paper_id"], "graph")
            self.assertEqual(hits[0]["retrieval_method"], "hybrid")
            self.assertEqual((hits[0]["bm25_rank"], hits[0]["dense_rank"]), (1, 1))
            self.assertEqual(json.loads(bm25.read_text(encoding="utf-8"))["evidence_scope"], "full_text_chunk")
            self.assertTrue(dense.with_suffix(".meta.json").is_file())

            parents.write_text(parents.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "父文档.*manifest"):
                retrieve_parents(
                    "graph retrieval", parent_path=parents, child_path=chunks,
                    index_path=bm25, dense_index_path=dense, retriever="hybrid",
                )


if __name__ == "__main__":
    unittest.main()
