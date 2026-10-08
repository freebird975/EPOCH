import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from litagent.fulltext_human_review import (
    _write_jsonl,
    export_e2e_report,
    export_step5_report,
    score_rows,
)


class FulltextHumanReviewTests(unittest.TestCase):
    def _e2e_report(self):
        return {
            "run": {"prompt_version": "p-test", "chat_model": "test-model"},
            "systems": {
                "hybrid": {
                    "rows": [{
                        "id": "q1", "question": "What does paper P show?",
                        "retrieval": {"status": "scored"},
                        "answer": {
                            "text": "Paper P reports result X [2].",
                            "refused": False, "failed": False, "citation_ids": ["p1"],
                            "diagnostics": {
                                "status": "verified",
                                "contexts": [
                                    {"number": 1, "paper_id": "p0", "text": "Uncited context"},
                                    {"number": 2, "paper_id": "p1", "title": "Paper P",
                                     "chunk_id": "c2", "section": "Results", "text": "Result X."},
                                ],
                                "verification": {"claims": [{"claim": "reports result X", "citations": [2],
                                                               "verdict": "supported"}]},
                                "candidate_parents": [{"paper_id": "p1", "title": "Paper P", "best_chunk_rank": 1}],
                            },
                        },
                    }]
                }
            },
        }

    def test_e2e_export_includes_only_cited_evidence_and_human_fields(self):
        rows = export_e2e_report(self._e2e_report())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["result"]["evidence"][0]["number"], 2)
        self.assertEqual(row["result"]["verification_claims"][0]["verdict"], "supported")
        self.assertEqual(row["annotation"]["answer_correctness"], "")
        self.assertEqual(row["source"]["system"], "hybrid")
        self.assertEqual(row["result"]["inline_citation_numbers"], [2])

    def test_e2e_export_joins_answerability_from_question_set_without_exposing_gold_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            question_file = Path(temp) / "questions.jsonl"
            question_file.write_text(json.dumps({"id": "q1", "question": "What does paper P show?",
                                                 "answerability": "answerable",
                                                 "gold_paper_ids": ["p1"]}) + "\n", encoding="utf-8")
            row = export_e2e_report(self._e2e_report(), question_set=question_file)[0]
        self.assertEqual(row["question"]["answerability"], "answerable")
        self.assertNotIn("gold_paper_ids", row["question"])

    def test_e2e_question_set_hash_mismatch_and_missing_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            question_file = Path(temp) / "questions.jsonl"
            question_file.write_text(json.dumps({"id": "q1", "question": "What does paper P show?",
                                                 "answerability": "answerable"}) + "\n", encoding="utf-8")
            report = self._e2e_report()
            report["run"] = {"questions_path": str(question_file), "questions_file_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                export_e2e_report(report)
            report["run"]["questions_file_sha256"] = hashlib.sha256(question_file.read_bytes()).hexdigest()
            report["run"]["questions_path"] = str(Path(temp) / "missing.jsonl")
            with self.assertRaises(FileNotFoundError):
                export_e2e_report(report)

    def test_e2e_question_set_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            question_file = Path(temp) / "questions.jsonl"
            question = {"id": "q1", "question": "What does paper P show?", "answerability": "answerable"}
            question_file.write_text(json.dumps(question) + "\n" + json.dumps(question) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate question id"):
                export_e2e_report(self._e2e_report(), question_set=question_file)

    def test_step5_export_joins_question_summary_to_detail_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "q1.json").write_text(json.dumps({
                "run": {"question": "Question?", "prompt_version": "p2"},
                "status": "verified", "answer": "Answer [1].",
                "contexts": [{"number": 1, "paper_id": "p1", "text": "Evidence."}],
                "verification": {"claims": []}, "candidate_parents": [],
            }), encoding="utf-8")
            rows = export_step5_report({
                "prompt_version": "p2", "parameters": {"pipeline": "classic", "retriever": "hybrid", "rerank": True},
                "questions": [{"id": "q1", "question": "Question?", "answerability": "answerable",
                               "status": "verified"}],
            }, root)
        self.assertEqual(rows[0]["result"]["status"], "verified")
        self.assertEqual(rows[0]["result"]["evidence"][0]["text"], "Evidence.")
        self.assertEqual(rows[0]["question"]["answerability"], "answerable")

    def test_step5_export_rejects_missing_or_mismatched_detail(self):
        summary = {"parameters": {}, "questions": [{"id": "q1", "question": "Question?", "status": "verified"}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                export_step5_report(summary, root)
            (root / "q1.json").write_text(json.dumps({"run": {"question": "Different question"},
                                                      "status": "verified"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "question text mismatch"):
                export_step5_report(summary, root)

    def test_score_reports_per_system_and_ignores_blank_or_uncertain_decisions(self):
        rows = export_e2e_report(self._e2e_report())
        rows[0]["question"]["answerability"] = "answerable"
        rows[0]["annotation"].update({
            "answer_correctness": "partially_correct",
            "citation_support": "all_supported",
            "answerability_handling": "appropriate",
        })
        empty = dict(rows[0])
        empty["review_id"] = "second"
        empty["source"] = {**rows[0]["source"], "question_id": "q2"}
        empty["annotation"] = {"answer_correctness": "uncertain", "citation_support": "unclear",
                               "answerability_handling": "uncertain"}
        score = score_rows([rows[0], empty])
        group = score["by_system"]["hybrid"]
        self.assertEqual(group["row_count"], 2)
        self.assertEqual(group["unique_question_count"], 2)
        self.assertEqual(group["answer_correctness"]["decided_rows"], 1)
        self.assertEqual(group["answer_correctness"]["strict_correct_rate_among_decided"], 0.0)
        self.assertEqual(group["citation_support"]["fully_supported_rate_among_decided"], 1.0)
        self.assertEqual(group["answerability_handling"]["appropriate_rate_among_decided"], 1.0)
        self.assertEqual(group["answerability_refusal_decision"]["accuracy_against_question_labels"], 1.0)

    def test_export_refuses_to_overwrite_existing_manual_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "review.jsonl"
            output.write_text('{"annotation":{"answer_correctness":"correct"}}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _write_jsonl(output, [], force=False)
            self.assertIn('"correct"', output.read_text(encoding="utf-8"))

    def test_invalid_human_label_is_rejected(self):
        rows = export_e2e_report(self._e2e_report())
        rows[0]["annotation"]["answer_correctness"] = "looks good"
        with self.assertRaisesRegex(ValueError, "invalid answer_correctness"):
            score_rows(rows)

    def test_failed_rows_are_excluded_from_refusal_decision_accuracy(self):
        rows = export_e2e_report(self._e2e_report())
        rows[0]["question"]["answerability"] = "answerable"
        rows[0]["result"]["failed"] = True
        score = score_rows(rows)["overall"]["answerability_refusal_decision"]
        self.assertIsNone(score["accuracy_against_question_labels"])
        self.assertEqual(score["scored_rows"], 0)
        self.assertEqual(score["skipped_failed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
