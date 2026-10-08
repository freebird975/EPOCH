from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import litsearch_fulltext
from litsearch_fulltext import _verify_with_batch_llm, run_fulltext_ask


class FulltextAskTests(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "paper_id": "p1", "title": "A paper", "source_url": "https://example.org/p1",
            "full_text": "This paper uses a world model for planning.",
            "matched_chunks": [{"chunk_id": "c7", "section": "Methods",
                                "text": "This paper uses a world model for planning.", "score": 0.5}],
        }

    def _retrieve(self, *args, **kwargs):
        kwargs["trace"]["first_round_queries"] = [args[0]]
        return [self.parent], [{"paper_id": "c7", "parent_id": "p1"}]

    def test_verified_answer_and_exact_evidence_saved_without_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "run.json"
            result = run_fulltext_ask(
                "Does it use a world model?", report_path=report_path,
                retrieve_fn=self._retrieve,
                generate_fn=lambda q, c, stage: {"content": "It uses a world model [1]."},
                judge_fn=lambda claim, evidence: "support",
            )
            saved = json.loads(report_path.read_text(encoding="utf8"))
            self.assertEqual(result["status"], "verified")
            self.assertEqual(saved["contexts"][0]["chunk_id"], "c7")
            self.assertEqual(saved["verification"]["claims"][0]["evidence"][0]["section"], "Methods")
            self.assertEqual(saved["candidate_parents"][0]["paper_id"], "p1")

    def test_unsupported_draft_revised_once_then_refused(self):
        calls = []

        def revise(question, contexts, draft, failed):
            calls.append(failed)
            return {"content": "It definitely uses a world model [1]."}

        with tempfile.TemporaryDirectory() as temporary:
            result = run_fulltext_ask(
                "Question", report_path=Path(temporary) / "run.json",
                retrieve_fn=self._retrieve,
                generate_fn=lambda q, c, stage: {"content": "Unsupported claim [1]."},
                revise_fn=revise,
                judge_fn=lambda claim, evidence: "insufficient",
            )
            self.assertEqual(result["status"], "refused_unverified")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("Unsupported claim", result["answer"])

    def test_context_failure_writes_failure_report(self):
        self.parent["full_text"] = "x" * 1000
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed.json"
            with self.assertRaisesRegex(ValueError, "超出"):
                run_fulltext_ask(
                    "Question", report_path=path, context_strategy="full",
                    max_context_chars=500, retrieve_fn=self._retrieve,
                )
            saved = json.loads(path.read_text(encoding="utf8"))
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["failure"]["type"], "ValueError")

    def test_cli_outputs_unique_cited_sources_in_numeric_order(self):
        report = {
            "answer": "Claim A [3], claim B [1], and claim C [2][3].",
            "status": "verified",
            "contexts": [
                {"paper_id": f"p{number}", "section": "Methods", "chunk_id": f"c{number}",
                 "source_url": f"https://example.org/{number}"}
                for number in range(1, 4)
            ],
            "context_chars": 100,
            "api_calls": [],
            "report_path": "run.json",
            "degraded": False,
            "degradations": [],
        }
        output = io.StringIO()
        with patch("sys.argv", ["litsearch_fulltext.py", "ask", "test question", "--pipeline", "classic"]), \
             patch("litsearch_fulltext.run_fulltext_ask", return_value=report), \
             redirect_stdout(output):
            self.assertEqual(litsearch_fulltext.main(), 0)

        source_lines = [line for line in output.getvalue().splitlines() if line.startswith("[")]
        self.assertEqual([line.split("]", 1)[0] + "]" for line in source_lines], ["[1]", "[2]", "[3]"])

    def test_batch_judge_checks_two_claims_in_one_model_call(self):
        contexts = [
            {"paper_id": "p1", "chunk_id": "c1", "section": "Methods", "title": "One", "text": "Evidence one"},
            {"paper_id": "p2", "chunk_id": "c2", "section": "Results", "title": "Two", "text": "Evidence two"},
        ]
        output = json.dumps({"items": [
            {"index": 0, "verdict": "support", "reason": "first block"},
            {"index": 1, "verdict": "insufficient", "reason": "second block lacks detail"},
        ]})
        with patch("rag.generate._chat_completion", return_value=output) as llm:
            result = _verify_with_batch_llm("First fact [1]. Second fact [2].", contexts)
        llm.assert_called_once()
        self.assertEqual([claim.verdict for claim in result.claims], ["support", "insufficient"])
        self.assertTrue(result.needs_revision)

    def test_generation_timeout_is_reported_without_exposing_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timeout.json"

            def timeout(_question, _contexts, stage):
                raise TimeoutError("upstream timeout")

            with self.assertRaises(TimeoutError):
                run_fulltext_ask("Question", report_path=path,
                                 retrieve_fn=self._retrieve, generate_fn=timeout)
            saved = json.loads(path.read_text(encoding="utf8"))
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["failure"]["type"], "TimeoutError")
            self.assertEqual(saved["contexts"][0]["chunk_id"], "c7")


if __name__ == "__main__":
    unittest.main()
