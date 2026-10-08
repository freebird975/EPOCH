import unittest

from litagent.end_to_end_eval import evaluate_e2e, load_questions


class EndToEndEvalTests(unittest.TestCase):
    def test_failure_before_retrieval_is_unscored_and_retains_runtime(self):
        result = evaluate_e2e([{"id": "q", "question": "q", "gold_paper_ids": ["p"]}], {"s": (
            lambda *args: [], lambda *args: {"status": "failed", "answer": "", "runtime": {"api_attempts": 1},
                                             "timings": [{"stage": "planner", "status": "failed", "seconds": 1}]})}, run_metadata={})
        system = result["systems"]["s"]
        self.assertEqual(system["aggregate"]["retrieval_scored_count"], 0)
        self.assertEqual(system["rows"][0]["retrieval"]["status"], "retrieval_failed")
        self.assertEqual(system["rows"][0]["answer"]["diagnostics"]["runtime"]["api_attempts"], 1)

    def setUp(self):
        self.questions = [
            {"id": "q1", "question": "question", "category": "fact",
             "answerability": "answerable", "gold_paper_ids": ["p1"]},
            {"id": "q2", "question": "not answerable", "category": "unanswerable",
             "answerability": "unanswerable", "gold_paper_ids": None},
        ]

    def test_injected_systems_get_row_and_aggregate_retrieval_and_answer_metrics(self):
        def retrieve(question, limit, item):
            return [{"parent_id": "p1", "title": "paper"}] if item["id"] == "q1" else []

        def answer(question, hits, item):
            if item["id"] == "q2":
                return {"answer": "Insufficient evidence", "refused": True, "citations": []}
            return {"answer": "Fact", "refused": False, "citations": ["p1", "outside"]}

        def judge(item, prediction, hits):
            return {"claim_count": 2, "supported_claims": 1, "fact_correctness": 0.5,
                    "citation_support_rate": 0.5}

        report = evaluate_e2e(self.questions, {"classic": (retrieve, answer)},
                              run_metadata={"model": "mock"}, answer_judge=judge)
        system = report["systems"]["classic"]
        self.assertEqual(system["rows"][0]["retrieval"]["recall_at_k"], 1.0)
        self.assertEqual(system["rows"][0]["answer"]["citation_validity"], 0.5)
        self.assertTrue(system["rows"][1]["answer"]["refusal_correct"])
        self.assertAlmostEqual(system["aggregate"]["supported_claim_rate"], 0.5)
        self.assertEqual(system["aggregate"]["retrieval_scored_count"], 1)
        self.assertEqual(system["aggregate"]["refusal_scored_count"], 2)

    def test_partial_pilot_qrels_are_not_scored_as_complete(self):
        report = evaluate_e2e(self.questions[:1], {"pilot": (
            lambda q, k, item: [], lambda q, hits, item: {"answer": ""})},
            run_metadata={}, corpus_paper_ids={"another-paper"})
        row = report["systems"]["pilot"]["rows"][0]["retrieval"]
        self.assertEqual(row["status"], "incomplete_gold_out_of_scope")
        self.assertIsNone(row["recall_at_k"])

    def test_original_and_translated_qrels_are_reported_separately(self):
        questions = [
            {"id": "en", "question": "English", "answerability": "answerable",
             "gold_paper_ids": ["p1"],
             "source": {"type": "official_litsearch_query", "query_id": "litsearch_1"}},
            {"id": "zh", "question": "中文", "answerability": "answerable",
             "gold_paper_ids": ["p1"],
             "source": {"type": "official_litsearch_query_translation", "query_id": "litsearch_1"}},
        ]
        report = evaluate_e2e(questions, {"system": (
            lambda q, k, item: [{"paper_id": "p1"}] if item["id"] == "en" else [],
            lambda q, hits, item: {"answer": ""})}, run_metadata={})
        aggregate = report["systems"]["system"]["aggregate"]
        self.assertEqual(aggregate["retrieval_scored_count"], 2)
        self.assertEqual(aggregate["original_query_count"], 1)
        self.assertEqual(aggregate["translated_query_count"], 1)
        self.assertEqual(aggregate["original_recall_at_k"], 1)
        self.assertEqual(aggregate["translated_recall_at_k"], 0)

    def test_fixed_dataset_has_documented_coverage_and_official_qrels(self):
        questions = load_questions("eval/fulltext_e2e_questions.jsonl")
        categories = {item["category"] for item in questions}
        self.assertTrue({"comparison", "multihop_retrieval", "chinese_fact_retrieval",
                         "unanswerable", "hard_specific"}.issubset(categories))
        official = [item for item in questions if item.get("qrels_status", "").startswith("official")]
        self.assertTrue(official)
        self.assertTrue(all(item["gold_paper_ids"] for item in official))
        self.assertTrue(all("source" in item and "note" in item["source"] for item in questions))
        pilot = [item for item in questions if item["id"].startswith("e2e_pilot_")]
        originals = [item for item in pilot if item["source"]["type"] == "official_litsearch_query"]
        translations = [item for item in pilot
                        if item["source"]["type"] == "official_litsearch_query_translation"]
        self.assertEqual(len(originals), 8)
        self.assertEqual(len({item["source"]["query_id"] for item in originals}), 8)
        self.assertEqual(len(translations), 2)
        by_query = {item["source"]["query_id"]: item["gold_paper_ids"] for item in originals}
        self.assertTrue(all(item["gold_paper_ids"] == by_query[item["source"]["query_id"]]
                            for item in translations))


if __name__ == "__main__":
    unittest.main()
