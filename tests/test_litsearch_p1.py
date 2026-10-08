from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from litsearch_p1 import (load_paper_and_chunks, partition_paper_chunks,
                          run_paper_understanding, run_related_work)


class P1WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parents = self.root / "parents.jsonl"
        self.chunks = self.root / "chunks.jsonl"
        self.parents.write_text(json.dumps({"paper_id": "p1", "title": "Example",
                                            "full_text": "A graph method improves retrieval."}) + "\n",
                                encoding="utf-8")
        rows = [
            {"paper_id": "c1", "parent_id": "p1", "chunk_id": "c1", "chunk_index": 0,
             "section": "Methods", "abstract": "We use graph expansion for retrieval."},
            {"paper_id": "c2", "parent_id": "p1", "chunk_id": "c2", "chunk_index": 1,
             "section": "Results", "abstract": "The method improves retrieval precision."},
        ]
        self.chunks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.chunks.with_suffix(".manifest.json").write_text(json.dumps({
            "parent_sha256": hashlib.sha256(self.parents.read_bytes()).hexdigest(),
            "chunk_sha256": hashlib.sha256(self.chunks.read_bytes()).hexdigest()}), encoding="utf-8")

    def test_understanding_rejects_changed_parent_before_generation(self):
        self.parents.write_text(self.parents.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_paper_and_chunks("p1", self.parents, self.chunks)

    def test_understanding_workflow_uses_real_child_shape_and_saves_card(self):
        seen = []

        def complete(messages, *, stage, max_tokens):
            seen.append(stage)
            return json.dumps({"methods": [{"claim": "The paper uses graph expansion.",
                                             "evidence": [{"chunk_id": "c1",
                                                           "quote": "We use graph expansion"}]}]})

        report = run_paper_understanding(
            "p1", parent_path=self.parents, child_path=self.chunks,
            output_path=self.root / "card.json", completion_fn=complete,
            verifier_fn=lambda flat, chunks: {"0": {"verdict": "support"}},
        )
        self.assertEqual(seen, ["paper_understanding_extract"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["selected_chunks"], 2)
        self.assertEqual(report["fields"]["methods"][0]["evidence"][0]["section"], "Methods")
        self.assertTrue((self.root / "card.json").is_file())

    def test_window_cap_is_explicit_and_dry_run_avoids_api(self):
        _, chunks = load_paper_and_chunks("p1", self.parents, self.chunks)
        windows, omitted = partition_paper_chunks(chunks, window_chars=500, max_windows=1)
        self.assertEqual(omitted, 0)
        self.assertEqual(len(windows), 1)
        report = run_paper_understanding("p1", parent_path=self.parents, child_path=self.chunks,
                                         dry_run=True, output_path=self.root / "dry.json")
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["api_calls"], [])

    def test_all_failed_understanding_windows_are_reported_failed(self):
        def broken_complete(messages, *, stage, max_tokens):
            raise RuntimeError("simulated API failure")

        report = run_paper_understanding(
            "p1", parent_path=self.parents, child_path=self.chunks,
            completion_fn=broken_complete, output_path=self.root / "failed-card.json")
        self.assertEqual(report["status"], "failed")
        self.assertTrue((self.root / "failed-card.json").is_file())

    def test_related_work_retrieval_and_verification_are_injectable(self):
        def retrieve(question, top_k, candidate_k, **kwargs):
            kwargs["trace"]["first_round_queries"] = [question]
            return ([{"paper_id": "p1", "title": "Example", "matched_chunks": [
                {"chunk_id": "c1", "section": "Methods", "text": "We use graph expansion for retrieval."}]}],
                    [{"paper_id": "c1", "parent_id": "p1"}])

        def complete(messages, **kwargs):
            return json.dumps({"papers": [{"paper_id": "p1", "task": "unknown",
                                           "method": {"text": "Graph expansion",
                                                      "evidence": [{"paper_id": "p1", "chunk_id": "c1",
                                                                    "section": "Methods"}]},
                                           "data": "unknown", "findings": "unknown"}]})

        report = run_related_work(
            "retrieval", parent_path=self.parents, child_path=self.chunks,
            output_path=self.root / "related.json", retrieve_fn=retrieve, complete=complete,
            verifier_fn=lambda checks: {"verdicts": [
                {"id": item["id"], "verdict": "support"} for item in checks]},
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["candidate_parents"][0]["paper_id"], "p1")
        self.assertEqual(report["result"]["papers"][0]["method"]["text"], "Graph expansion")
        self.assertEqual(report["retrieval_trace"]["first_round_queries"], ["retrieval"])

    def test_related_work_saves_safe_failure_report(self):
        def retrieve(question, top_k, candidate_k, **kwargs):
            return ([{"paper_id": "p1", "title": "Example", "matched_chunks": [
                {"chunk_id": "c1", "section": "Methods", "text": "Graph expansion."}]}], [])

        def broken_complete(messages, **kwargs):
            raise RuntimeError("private credential must not enter reports")

        report = run_related_work("retrieval", parent_path=self.parents,
                                  child_path=self.chunks, retrieve_fn=retrieve,
                                  complete=broken_complete,
                                  output_path=self.root / "failed.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_type"], "RuntimeError")
        self.assertNotIn("private credential", (self.root / "failed.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
