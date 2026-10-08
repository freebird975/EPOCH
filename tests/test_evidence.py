from __future__ import annotations

import unittest

from litagent.evidence import citation_numbers, verify_answer


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            {"paper_id": "paper-A", "chunk_id": "c-7", "section": "Results",
             "text": "The method improves accuracy by 4 percent."},
            {"paper_id": "paper-B", "chunk_id": "c-2", "section": "Methods",
             "text": "The study uses a synthetic dataset."},
        ]

    def test_judge_receives_exact_claim_and_provenance_and_supports_claim(self):
        seen = []

        def judge(claim, evidence):
            seen.append((claim, evidence))
            return {"verdict": "support", "reason": "Reported in results."}

        result = verify_answer("Accuracy improves by 4 percent [1].", self.contexts, judge)
        self.assertTrue(result.safe)
        self.assertFalse(result.needs_revision)
        self.assertEqual(result.claims[0].verdict, "support")
        self.assertEqual(seen[0][1][0].paper_id, "paper-A")
        self.assertEqual(seen[0][1][0].section, "Results")
        self.assertEqual(seen[0][1][0].chunk_id, "c-7")
        self.assertIsNone(seen[0][1][0].page)

    def test_valid_but_unsupported_citation_requires_revision(self):
        result = verify_answer(
            "The method improves accuracy by 40 percent [1].",
            self.contexts,
            lambda claim, evidence: "insufficient",
        )
        self.assertEqual(result.claims[0].citations, (1,))
        self.assertEqual(result.claims[0].verdict, "insufficient")
        self.assertTrue(result.needs_revision)
        self.assertFalse(result.safe)

    def test_grouped_citations_map_to_both_exact_blocks(self):
        result = verify_answer("Both studies are relevant [1,2].", self.contexts,
                               lambda claim, evidence: "support")
        self.assertEqual(result.claims[0].citations, (1, 2))
        self.assertEqual([item.chunk_id for item in result.claims[0].evidence], ["c-7", "c-2"])
        self.assertTrue(result.safe)

    def test_bounded_numeric_range_expands_to_exact_blocks(self):
        self.assertEqual(citation_numbers("Evidence [1-2], repeated [2]."), [1, 2])
        result = verify_answer("Both studies [1-2].", self.contexts,
                               lambda claim, evidence: "support")
        self.assertEqual(result.claims[0].citations, (1, 2))

    def test_bibliographic_year_and_explicit_evidence_caveat(self):
        self.assertEqual(citation_numbers("Luo et al. [2021] found this [1]."), [1])
        result = verify_answer("A paper by Luo et al. [2021] reports this [1]. "
                               "However, the blocks do not establish priority.",
                               self.contexts, lambda claim, evidence: "support")
        self.assertEqual(len(result.claims), 1)
        self.assertIn("[2021]", result.claims[0].claim)
        self.assertTrue(result.safe)

    def test_priority_caveat_after_named_paper_does_not_erase_supported_answer(self):
        contexts = [dict(self.contexts[0], title="Efficient Backdoor Attacks")]
        answer = ("The paper Efficient Backdoor Attacks uses CLIP [1]. "
                  "The evidence does not identify which paper first used CLIP for this purpose.")
        result = verify_answer(answer, contexts, lambda claim, evidence: "support")
        self.assertEqual(len(result.claims), 1)
        self.assertTrue(result.safe)

    def test_priority_caveat_without_identified_paper_still_requires_answer(self):
        answer = ("A method uses online estimation [1]. "
                  "The evidence does not identify which paper first derived it.")
        result = verify_answer(answer, self.contexts, lambda claim, evidence: "support")
        self.assertEqual(len(result.claims), 2)
        self.assertFalse(result.safe)

    def test_contradiction_is_explicit_and_unsafe(self):
        result = verify_answer("It uses real-world data [2].", self.contexts,
                               lambda claim, evidence: "contradiction")
        self.assertEqual(result.claims[0].verdict, "contradiction")
        self.assertTrue(result.needs_revision)

    def test_invalid_number_is_detected_without_calling_judge(self):
        calls = []
        result = verify_answer("The claim is certain [3].", self.contexts,
                               lambda *args: calls.append(args) or "support")
        self.assertEqual(result.unsupported_citations, [3])
        self.assertEqual(result.claims[0].verdict, "insufficient")
        self.assertEqual(result.claims[0].reason, "invalid_citation_number")
        self.assertEqual(calls, [])
        self.assertTrue(result.needs_revision)

    def test_uncited_claim_and_empty_answer_are_not_safe(self):
        result = verify_answer("A claim without a reference.", self.contexts)
        self.assertEqual(result.claims[0].reason, "missing_citation")
        self.assertTrue(result.needs_revision)
        empty = verify_answer("", self.contexts)
        self.assertFalse(empty.safe)
        self.assertTrue(empty.needs_revision)

    def test_exact_text_only_fallback_is_conservative(self):
        exact = verify_answer("The study uses a synthetic dataset [2].", self.contexts)
        self.assertEqual(exact.claims[0].verdict, "support")
        paraphrase = verify_answer("Synthetic data was used [2].", self.contexts)
        self.assertEqual(paraphrase.claims[0].verdict, "insufficient")

    def test_supplied_page_is_retained_but_missing_page_not_invented(self):
        contexts = [dict(self.contexts[0], page=12)]
        result = verify_answer("The method improves accuracy by 4 percent [1].", contexts)
        self.assertEqual(result.claims[0].evidence[0].page, 12)
        self.assertNotIn("page", self.contexts[0])

    def test_judge_failure_does_not_leak_exception(self):
        def fail(claim, evidence):
            raise RuntimeError("secret token=do-not-return")

        result = verify_answer("Some result [1].", self.contexts, fail)
        self.assertEqual(result.claims[0].verdict, "insufficient")
        self.assertEqual(result.claims[0].reason, "judge_error")
        self.assertNotIn("secret", result.claims[0].reason)

    def test_context_validation_is_bounded(self):
        with self.assertRaises(ValueError):
            verify_answer("text", [{"paper_id": "p", "chunk_id": "c", "section": "s",
                                    "text": "x" * 100001}])


if __name__ == "__main__":
    unittest.main()
