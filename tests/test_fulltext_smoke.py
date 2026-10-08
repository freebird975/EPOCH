from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from litagent.fulltext_smoke import run_fulltext_smoke


class FulltextSmokeTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing two-document fixture, including its full-text
        # manifest, SQLite BM25 index, FAISS index, and mapping database.
        from tests.test_local_dense_loader import LocalDenseLoaderTests, QueryModel

        self.QueryModel = QueryModel
        self.fixture = LocalDenseLoaderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.queries = self.root / "queries.jsonl"
        self.report = self.root / "smoke-report.json"
        self.log = self.root / "smoke-run.jsonl"
        self.write_query("graph retrieval evidence", ["p1"])

    def write_query(self, question: str, gold: list[str]) -> None:
        self.queries.write_text(json.dumps({
            "id": "smoke-1",
            "question": question,
            "query_set": "fixture",
            "gold_paper_ids": gold,
            "label_status": "fixture_official",
        }) + "\n", encoding="utf-8")

    def run_smoke(self, **overrides):
        params = dict(
            parent_path=self.fixture.parent,
            child_path=self.fixture.child,
            bm25_path=self.fixture.bm25,
            dense_path=self.fixture.path,
            dense_docs_path=self.fixture.docs,
            queries_path=self.queries,
            report_path=self.report,
            log_path=self.log,
            top_k=2,
            candidate_k=2,
            deadline_seconds=30,
        )
        params.update(overrides)
        with patch("litagent.dense.load_model", return_value=self.QueryModel()) as model_load:
            self.model_load_mock = model_load
            return run_fulltext_smoke(**params)

    def test_success_runs_all_modes_persists_outputs_and_releases_handles(self):
        report = self.run_smoke()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["runs"]), 3)
        self.assertEqual({run.get("mode", run.get("retriever")) for run in report["runs"]},
                         {"bm25", "dense", "hybrid"})
        self.assertEqual(report["checks"]["source_integrity"]["status"], "passed")
        self.assertEqual(report["checks"]["rrf_provenance"]["status"], "passed")
        self.assertTrue(report["checks"]["artifacts_unchanged"])
        self.assertEqual(len(report["negative_checks"]), 5)
        self.assertEqual(report["query_gold"]["smoke-1"]["gold_with_full_text"], ["p1"])
        self.assertTrue(self.report.is_file())
        self.assertEqual(json.loads(self.report.read_text(encoding="utf-8"))["status"], "passed")
        self.assertTrue(self.log.is_file())
        self.assertTrue(self.log.read_text(encoding="utf-8").strip())
        # Windows refuses to remove open SQLite files.
        self.fixture.docs.unlink()
        self.fixture.bm25.unlink()

    def test_same_id_bm25_payload_tampering_fails_source_integrity(self):
        connection = sqlite3.connect(self.fixture.bm25)
        payload = connection.execute("SELECT payload FROM docs WHERE paper_id='p1#0'").fetchone()[0]
        row = json.loads(payload)
        row["abstract"] = "tampered payload under an unchanged paper id"
        connection.execute("UPDATE docs SET payload=? WHERE paper_id='p1#0'",
                           (json.dumps(row, ensure_ascii=False),))
        connection.commit()
        connection.close()

        report = self.run_smoke()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["type"], "ValueError")
        self.assertIn("原始语料", report["failure"]["reason"])

    def test_missing_dense_index_fails_without_implicit_build(self):
        missing = self.root / "missing.faiss"
        with patch("litagent.dense.build_dense_index", side_effect=AssertionError("implicit build")):
            report = self.run_smoke(dense_path=missing)
        self.assertEqual(report["status"], "failed")
        details = json.dumps(report, ensure_ascii=False).lower()
        self.assertIn("missing.faiss", details)
        self.assertFalse(missing.exists())

    def test_corrupted_hybrid_component_rank_fails_rrf_provenance(self):
        from litagent import fulltext_smoke

        retrieve = fulltext_smoke.retrieve_parents

        def corrupt_hybrid(*args, **kwargs):
            parents, chunks = retrieve(*args, **kwargs)
            if kwargs.get("retriever") == "hybrid" and chunks:
                chunks[0]["dense_rank"] = 999
            return parents, chunks

        with patch("litagent.fulltext_smoke.retrieve_parents", side_effect=corrupt_hybrid):
            report = self.run_smoke()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["type"], "ValueError")
        self.assertIn("RRF", report["failure"]["reason"])

    def test_report_path_cannot_overwrite_input(self):
        before = self.fixture.child.read_bytes()
        with self.assertRaises(ValueError):
            self.run_smoke(report_path=self.fixture.child)
        self.assertEqual(self.fixture.child.read_bytes(), before)

    def test_chinese_question_is_rejected_before_model_loading(self):
        self.write_query("图检索证据", ["p1"])
        with self.assertRaises(ValueError):
            self.run_smoke()
        self.model_load_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
