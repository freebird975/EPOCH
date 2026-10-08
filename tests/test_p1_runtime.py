import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from litagent.runtime import RuntimePolicy, current_runtime
from litsearch_p1 import run_paper_understanding, run_related_work


class P1RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parents = self.root / "parents.jsonl"
        self.chunks = self.root / "chunks.jsonl"
        self.out = self.root / "report.json"
        parent = {"paper_id": "p1", "title": "A Paper", "full_text": "text"}
        self.parents.write_text(json.dumps(parent) + "\n", encoding="utf-8")
        rows = [
            {"paper_id": "c1", "parent_id": "p1", "chunk_id": "c1",
             "chunk_index": 0, "section": "Introduction", "text": "A" * 400},
            {"paper_id": "c2", "parent_id": "p1", "chunk_id": "c2",
             "chunk_index": 1, "section": "Results", "text": "B" * 400},
        ]
        self.chunks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.chunks.with_suffix(".manifest.json").write_text(json.dumps({
            "parent_sha256": hashlib.sha256(self.parents.read_bytes()).hexdigest(),
            "chunk_sha256": hashlib.sha256(self.chunks.read_bytes()).hexdigest()}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_has_runtime_provenance_and_no_api_calls(self):
        report = run_paper_understanding("p1", parent_path=self.parents,
                                         child_path=self.chunks, window_chars=500,
                                         dry_run=True, output_path=self.out)
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["api_calls"], [])
        self.assertIn("runtime", report)
        self.assertIn("timings", report)
        self.assertIn("provenance", report)
        self.assertIn("litsearch_p1.py", report["provenance"]["code_sha256"])
        saved = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(saved["prompt_version"], "paper-understanding-v1")

    def test_api_budget_stops_windows_and_reports_partial_coverage(self):
        def completion(messages, *, stage, max_tokens):
            runtime = current_runtime()
            runtime.claim_api(30, max_tokens)
            if stage == "paper_understanding_extract":
                return json.dumps({"methods": [{"claim": "A method is introduced.",
                    "evidence": [{"chunk_id": "c1", "quote": "A" * 20}]}]})
            return json.dumps({"verdicts": [{"id": "0", "verdict": "support"}]})

        report = run_paper_understanding(
            "p1", parent_path=self.parents, child_path=self.chunks,
            window_chars=500, completion_fn=completion, output_path=self.out,
            runtime_policy=RuntimePolicy(max_api_calls=1, max_completion_tokens=3000),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runtime"]["api_attempts"], 1)
        self.assertEqual(report["omitted_chunks"], 2)
        self.assertEqual(report["failed_windows"], [1])
        self.assertEqual(report["completed_windows"], 0)
        self.assertEqual(report["failure"]["type"], "BudgetExceeded")

    def test_related_dry_run_records_runtime_and_retrieval_timing(self):
        def retrieve(*args, **kwargs):
            return ([{"paper_id": "p1", "title": "A Paper", "matched_chunks": []}], [])

        report = run_related_work("topic", parent_path=self.parents, child_path=self.chunks,
                                  output_path=self.out, dry_run=True, retrieve_fn=retrieve,
                                  retriever="bm25")
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["api_calls"], [])
        self.assertEqual([row["stage"] for row in report["timings"]], ["related_work_retrieval"])
        self.assertIn("provenance", report)
        self.assertEqual(report["prompt_version"], "related-work-v1")


if __name__ == "__main__":
    unittest.main()
