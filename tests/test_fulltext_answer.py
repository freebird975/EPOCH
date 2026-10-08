from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from litagent.fulltext_answer import (estimate_flash_cost_range_usd, is_refusal_answer,
                                      prompt_context, question_language, refusal_message,
                                      select_contexts)
from rag.generate import capture_api_calls, chat_completion_result


class FulltextAnswerTests(unittest.TestCase):
    def setUp(self):
        self.parents = [
            {"paper_id": "p1", "title": "One", "source_url": "https://example.org/one",
             "full_text": "A" * 2000,
             "matched_chunks": [{"chunk_id": "c1", "section": "Methods", "text": "first evidence"},
                                {"chunk_id": "c2", "section": "Results", "text": "second evidence"}]},
            {"paper_id": "p2", "title": "Two", "source_url": "https://example.org/two",
             "full_text": "B" * 2000,
             "matched_chunks": [{"chunk_id": "c3", "section": "Abstract", "text": "other evidence"}]},
        ]

    def test_chunk_context_has_exact_provenance_and_round_robin_diversity(self):
        contexts = select_contexts(self.parents, max_context_chars=2000)
        self.assertEqual([item["chunk_id"] for item in contexts], ["c1", "c3", "c2"])
        self.assertEqual([item["paper_id"] for item in contexts], ["p1", "p2", "p1"])
        self.assertEqual([item["number"] for item in contexts], [1, 2, 3])
        self.assertIn("chunk_id=c3", prompt_context(contexts))

    def test_numeric_paper_id_adds_record_link_without_replacing_dataset_source(self):
        parent = {"paper_id": "252715594", "corpusid": 252715594, "title": "Phenaki",
                  "source_url": "https://huggingface.co/datasets/princeton-nlp/LitSearch",
                  "matched_chunks": [{"chunk_id": "c1", "section": "Abstract", "text": "evidence"}]}
        context = select_contexts([parent], max_context_chars=1000)[0]
        self.assertEqual(context["source_url"], parent["source_url"])
        self.assertEqual(context["paper_record_api_url"],
                         "https://api.semanticscholar.org/graph/v1/paper/"
                         "CorpusId%3A252715594?fields=title%2Curl%2CexternalIds")

    def test_second_chunk_covers_specific_query_terms(self):
        parents = [{**self.parents[0], "matched_chunks": [
            {"chunk_id": "top", "section": "Intro", "text": "generic paper"},
            {"chunk_id": "middle", "section": "Other", "text": "unrelated detail"},
            {"chunk_id": "focused", "section": "Method", "text": "online occupancy measure for adversarial MDP"},
        ]}]
        selected = select_contexts(parents, question="online occupany adversarial MDP", max_context_chars=2000)
        self.assertEqual([item["chunk_id"] for item in selected], ["top", "focused"])

    def test_full_mode_rejects_oversize_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "超出"):
            select_contexts(self.parents, strategy="full", max_context_chars=2100)

    def test_explicit_refusal_is_separate_from_verification(self):
        self.assertTrue(is_refusal_answer("现有证据不足以回答该问题。[1]"))
        self.assertTrue(is_refusal_answer("可见证据中没有这类论文。[1]"))
        self.assertTrue(is_refusal_answer("There is insufficient evidence in the corpus."))
        self.assertFalse(is_refusal_answer("The paper introduces FSText decomposition [1]."))

    def test_refusal_matches_question_language(self):
        self.assertEqual(question_language("Which paper?"), "English")
        self.assertEqual(question_language("哪篇论文？"), "Chinese")
        self.assertTrue(refusal_message("Which paper?", "unverified").startswith("The evidence"))
        self.assertTrue(refusal_message("哪篇论文？", "unverified").startswith("现有证据"))

    def test_api_telemetry_excludes_prompts_and_credentials(self):
        response = {"model": "deepseek-flash", "choices": [{"message": {"content": "grounded [1]"},
                                                               "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 20,
                              "prompt_cache_miss_tokens": 80, "completion_tokens": 10}}
        with patch("rag.generate.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with capture_api_calls() as calls:
                result = chat_completion_result([{"role": "user", "content": "private query"}],
                                                api_key="test-secret", stage="answer")
        self.assertEqual(result["content"], "grounded [1]")
        self.assertEqual(calls[0]["stage"], "answer")
        self.assertNotIn("test-secret", json.dumps(calls))
        self.assertNotIn("private query", json.dumps(calls))
        cost = estimate_flash_cost_range_usd(calls)
        self.assertEqual(cost["prompt_tokens"], 100)
        self.assertGreater(cost["estimated_usd_peak"], cost["estimated_usd_offpeak"])


if __name__ == "__main__":
    unittest.main()
