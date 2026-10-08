from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from litsearch import _digest, run_bm25


class LitSearchBenchmarkTests(unittest.TestCase):
    def test_full_corpus_evaluation_keeps_official_ids_and_empty_abstracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            papers = [
                {"paper_id": "101", "title": "Graph retrieval", "abstract": "Graph search for papers.",
                 "source_url": "https://example.org/dataset", "evidence_scope": "abstract"},
                {"paper_id": "202", "title": "Vector search", "abstract": "",
                 "source_url": "https://example.org/dataset", "evidence_scope": "abstract"},
                {"paper_id": "303", "title": "Unrelated work", "abstract": "A separate topic.",
                 "source_url": "https://example.org/dataset", "evidence_scope": "abstract"},
            ]
            items = [
                {"id": "q1", "question": "graph retrieval", "gold_paper_ids": ["101"],
                 "query_set": "inline_acl", "specificity": 1, "quality": 2,
                 "label_status": "LitSearch_official"},
                {"id": "q2", "question": "vector search", "gold_paper_ids": ["202"],
                 "query_set": "author", "specificity": 0, "quality": 2,
                 "label_status": "LitSearch_official"},
            ]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in papers), encoding="utf-8")
            queries.write_text("".join(json.dumps(row) + "\n" for row in items), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "revision": "test", "corpus_count": 3, "query_count": 2,
                "corpus_sha256": _digest(corpus), "queries_sha256": _digest(queries),
            }), encoding="utf-8")
            report_path = root / "report.json"
            report = run_bm25(root, report_path, top_k=5)
            self.assertEqual(report["run"]["corpus_count"], 3)
            self.assertEqual(report["results"]["5"]["recall_at_k"], 1.0)
            self.assertEqual(set(report["slices"]["query_set"]), {"inline_acl", "author"})
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
