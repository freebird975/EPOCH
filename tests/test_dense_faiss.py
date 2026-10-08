from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from litagent.dense import build_dense_index, dense_search, load_dense_index, migrate_legacy_index


class FakeModel:
    model_name = "test/model"
    snapshot = "test-snapshot"

    def passage_embed(self, texts, batch_size=16):
        return np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)

    def query_embed(self, question):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class DenseFaissTests(unittest.TestCase):
    def test_build_and_migrate_return_same_rank_and_reject_modified_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "papers.jsonl"
            papers = [
                {"paper_id": "first", "title": "First", "abstract": "one", "source_url": "https://arxiv.org/abs/1", "evidence_scope": "abstract"},
                {"paper_id": "second", "title": "Second", "abstract": "two", "source_url": "https://arxiv.org/abs/2", "evidence_scope": "abstract"},
            ]
            corpus.write_text("".join(json.dumps(paper) + "\n" for paper in papers), encoding="utf-8")
            built_path = root / "built.faiss"
            with patch("litagent.dense.load_model", return_value=FakeModel()):
                build_dense_index(corpus, built_path, model_name="test/model")
            built = load_dense_index(built_path)
            self.assertEqual(built["index_type"], "IndexFlatIP")
            self.assertEqual([hit["paper_id"] for hit in dense_search(built, "question", FakeModel(), 2)], ["first", "second"])

            legacy_path = root / "old.json"
            legacy = {
                "version": 2,
                "evidence_scope": "abstract_only",
                "embedding_mode": "passage_embed/query_embed",
                "corpus_path": str(corpus),
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "paper_ids": [paper["paper_id"] for paper in papers],
                "dimension": 2,
                "model_name": "test/model",
                "model_snapshot": "test-snapshot",
                "vectors": [[1.0, 0.0], [0.8, 0.6]],
            }
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            migrated_path = root / "migrated.faiss"
            migrate_legacy_index(legacy_path, migrated_path)
            migrated = load_dense_index(migrated_path)
            self.assertEqual(
                [hit["paper_id"] for hit in dense_search(built, "question", FakeModel(), 2)],
                [hit["paper_id"] for hit in dense_search(migrated, "question", FakeModel(), 2)],
            )

            migrated_path.write_bytes(migrated_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_dense_index(migrated_path)
            built["docs_connection"].close()


if __name__ == "__main__":
    unittest.main()
