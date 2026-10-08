import unittest

from litagent.p1_review import report_to_rows, score_rows


class P1ReviewTests(unittest.TestCase):
    def test_paper_fields_export_claims_with_evidence_and_stable_ids(self):
        report = {"paper_id": "p1", "fields": {"methods": [{"claim": "Uses X", "evidence": [
            {"paper_id": "p1", "chunk_id": "c1", "section": "Method", "quote": "We use X."}]}],
            "limitations": []}}
        first = report_to_rows(report)
        again = report_to_rows(report)
        self.assertEqual(first, again)
        self.assertEqual(first[0]["kind"], "paper_field")
        self.assertEqual(first[0]["source_references"][0]["chunk_id"], "c1")

    def test_related_work_exports_paper_fields_groups_and_comparison_rows(self):
        ref = {"paper_id": "p1", "chunk_id": "c1", "section": "Results"}
        report = {"papers": [{"paper_id": "p1", "method": {"text": "Method A", "status": "evidence_linked", "evidence": [ref]}}],
                  "groups": [{"label": "A family", "text": "Shared pattern", "evidence": [ref, {"paper_id": "p2", "chunk_id": "c2", "section": "Method"}]}],
                  "comparisons": [{"dimension": "Speed", "synthesis": "A is faster", "evidence": [ref],
                                   "entries": [{"paper_id": "p1", "value": "Fast", "evidence": [ref]}]}]}
        rows = report_to_rows(report)
        self.assertEqual([r["kind"] for r in rows], ["related_paper_field", "group", "comparison", "comparison_entry"])
        self.assertTrue(all(r["source_references"] for r in rows))

        wrapped = report_to_rows({"task": "related_work", "topic": "retrieval", "result": report})
        self.assertEqual([r["kind"] for r in wrapped], [r["kind"] for r in rows])
        self.assertTrue(all(r["source_references"] for r in wrapped))

    def test_scoring_excludes_unlabeled_rows_from_correct_denominator(self):
        rows = [{"label": "correct", "evidence_support": "yes"}, {"label": "", "evidence_support": ""},
                {"label": "uncertain", "evidence_support": "no"}]
        result = score_rows(rows)
        self.assertEqual(result["total_rows"], 3)
        self.assertEqual(result["labeled_rows"], 2)
        self.assertEqual(result["labels"]["correct"], 1)
        self.assertEqual(result["decided_rows"], 1)
        self.assertEqual(result["correctness_rate_among_decided"], 1.0)
        self.assertAlmostEqual(result["label_coverage"], 2 / 3)
        self.assertEqual(result["evidence_labeled_rows"], 2)

    def test_empty_rows_have_no_accuracy_score(self):
        result = score_rows([])
        self.assertEqual(result["total_rows"], 0)
        self.assertIsNone(result["correctness_rate_among_decided"])


if __name__ == "__main__":
    unittest.main()
