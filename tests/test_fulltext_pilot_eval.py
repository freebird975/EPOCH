from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from litsearch_fulltext import _digest, _full_text, build_chunks, run_pilot_eval


class FulltextPilotEvalTests(unittest.TestCase):
    def test_s2orc_nested_text_field(self):
        self.assertEqual(_full_text({"content": {"text": "paper body"}}), "paper body")

    def test_parent_metrics_use_only_gold_present_in_100_paper_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents_path = root / "corpus_fulltext.jsonl"
            rows = []
            for number in range(100):
                paper_id = f"p{number:03d}"
                topic = "graph retrieval" if number == 0 else "vector search" if number == 1 else "unrelated filler"
                rows.append({
                    "paper_id": paper_id, "title": topic, "abstract": "",
                    "full_text": f"This paper studies {topic}.",
                    "source_url": "https://example.org/paper", "evidence_scope": "full_text",
                })
            parents_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "corpus_fulltext_manifest.json").write_text(json.dumps({
                "pilot_100": True, "counts": {"with_full_text": 100},
                "paper_ids": [row["paper_id"] for row in rows],
                "parent_sha256": _digest(parents_path),
                "selection_method": "test fixture", "source_row_groups": 1,
                "eligible_query_count": 2,
            }), encoding="utf-8")
            queries = [
                {"id": "q1", "question": "graph retrieval", "gold_paper_ids": ["p000"],
                 "query_set": "manual_acl", "specificity": 1, "label_status": "LitSearch_official"},
                {"id": "q2", "question": "vector search", "gold_paper_ids": ["p001", "outside"],
                 "query_set": "manual_acl", "specificity": 1, "label_status": "LitSearch_official"},
                {"id": "q3", "question": "missing", "gold_paper_ids": ["outside"],
                 "query_set": "manual_acl", "specificity": 1, "label_status": "LitSearch_official"},
            ]
            (root / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in queries), encoding="utf-8")
            build_chunks(parents_path, root / "fulltext_chunks.jsonl", chunk_size=500, overlap=50)
            report_path = root / "report.json"
            report = run_pilot_eval(root, "bm25", top_k=5, candidate_k=10, report_path=report_path)
            self.assertEqual(report["run"]["eligible_query_count"], 2)
            self.assertEqual(report["run"]["partial_gold_query_count"], 1)
            self.assertEqual(report["run"]["orphan_child_hits"], 0)
            self.assertEqual(report["results"]["5"]["recall_at_k"], 1.0)
            self.assertEqual(report["results"]["5"]["rows"][1]["gold_paper_ids"], ["p001"])
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
