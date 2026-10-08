import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from litagent.paper_understanding import FIELDS, understand_paper
from litsearch_p1 import run_paper_understanding


PAPER = {"paper_id": "p-17", "title": "A Study"}
CHUNKS = [
    {"paper_id": "p-17", "chunk_id": "c1", "section": "Abstract",
     "text": "We introduce a retrieval method evaluated on the AtlasQA dataset. It improves exact match by 4.2 points."},
    {"paper_id": "p-17", "chunk_id": "c2", "section": "Limitations",
     "text": "Our experiments use one English benchmark and do not evaluate multilingual retrieval."},
]


def proposal(*, quote=None, claim="The method improves exact match by 4.2 points."):
    return {
        "research_problem": [{"claim": "The work studies retrieval.", "evidence": [
            {"chunk_id": "c1", "quote": "We introduce a retrieval method"}]}],
        "methods": [{"claim": "The paper introduces a retrieval method.", "evidence": [
            {"chunk_id": "c1", "quote": "We introduce a retrieval method"}]}],
        "datasets": [{"claim": "AtlasQA is used.", "evidence": [
            {"chunk_id": "c1", "quote": "evaluated on the AtlasQA dataset"}]}],
        "experimental_setup": [],
        "main_findings": [{"claim": claim, "evidence": [
            {"chunk_id": "c1", "quote": quote or "improves exact match by 4.2 points"}]}],
        "limitations": [{"claim": "Multilingual retrieval is not evaluated.", "evidence": [
            {"chunk_id": "c2", "quote": "do not evaluate multilingual retrieval"}]}],
    }


class PaperUnderstandingTests(unittest.TestCase):
    def run_with(self, extraction, verdict_fn):
        calls = []

        def complete(messages, *, stage, max_tokens):
            calls.append(stage)
            return json.dumps(extraction)

        return understand_paper(PAPER, CHUNKS, completion_fn=complete, verifier_fn=verdict_fn), calls

    def test_supported_fields_keep_exact_source_locations(self):
        result, _ = self.run_with(proposal(), lambda flat, chunks: {
            str(i): {"verdict": "support"} for i in range(len(flat))
        })
        self.assertEqual(result["status"], "verified")
        card = result["fields"]
        self.assertEqual(set(card), set(FIELDS))
        self.assertEqual(card["main_findings"][0]["evidence"][0], {
            "paper_id": "p-17", "chunk_id": "c1", "section": "Abstract",
            "quote": "improves exact match by 4.2 points"})
        self.assertNotIn("page", card["main_findings"][0]["evidence"][0])

    def test_accepts_litsearch_child_record_shape(self):
        child = {"paper_id": "c1", "parent_id": "p-17", "chunk_id": "c1",
                 "section": "Abstract", "abstract": CHUNKS[0]["text"],
                 "start_line": 8, "end_line": 11, "location_type": "source_text_line"}

        def complete(messages, *, stage, max_tokens):
            return json.dumps({"methods": [{"claim": "A retrieval method is introduced.",
                                             "evidence": [{"chunk_id": "c1",
                                                           "quote": "We introduce a retrieval method"}]}]})

        result = understand_paper(PAPER, [child], completion_fn=complete,
                                  verifier_fn=lambda flat, chunks: {"0": {"verdict": "support"}})
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["fields"]["methods"][0]["evidence"][0]["chunk_id"], "c1")
        self.assertEqual(result["fields"]["methods"][0]["evidence"][0]["start_line"], 8)

    def test_dry_run_report_contains_paper_record_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = root / "parents.jsonl"
            chunks = root / "chunks.jsonl"
            parent = {"paper_id": "252715594", "corpusid": 252715594, "title": "Phenaki",
                      "source_url": "https://huggingface.co/datasets/princeton-nlp/LitSearch",
                      "full_text": "Evidence text."}
            child = {"paper_id": "chunk-1", "parent_id": "252715594", "chunk_index": 0,
                     "chunk_id": "chunk-1", "section": "Abstract", "abstract": "Evidence text."}
            parent_text = json.dumps(parent) + "\n"
            child_text = json.dumps(child) + "\n"
            parents.write_text(parent_text, encoding="utf-8")
            chunks.write_text(child_text, encoding="utf-8")
            manifest = {"parent_sha256": hashlib.sha256(parents.read_bytes()).hexdigest(),
                        "chunk_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest()}
            chunks.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = run_paper_understanding("252715594", parent_path=parents,
                                             child_path=chunks, output_path=root / "report.json",
                                             dry_run=True)
            self.assertEqual(report["paper_record_api_url"],
                             "https://api.semanticscholar.org/graph/v1/paper/"
                             "CorpusId%3A252715594?fields=title%2Curl%2CexternalIds")

    def test_non_verbatim_quote_is_dropped_before_semantic_judge(self):
        result, _ = self.run_with(proposal(quote="the model improves accuracy"),
                                  lambda flat, chunks: {str(i): {"verdict": "support"}
                                                        for i in range(len(flat))})
        self.assertEqual(result["dropped_claim_count"], 1)
        self.assertEqual(result["fields"]["main_findings"], [])

    def test_unsupported_or_contradictory_claims_are_dropped(self):
        result, _ = self.run_with(proposal(), lambda flat, chunks: {
            str(i): {"verdict": ("contradiction" if flat[i][0] == "methods" else "insufficient")
                     if flat[i][0] in {"methods", "main_findings"} else "support"}
            for i in range(len(flat))
        })
        self.assertEqual(result["fields"]["methods"], [])
        self.assertEqual(result["fields"]["main_findings"], [])
        self.assertEqual(result["dropped_claim_count"], 2)

    def test_numeric_ambiguity_is_left_unknown(self):
        p = proposal(claim="The method improves exact match by 42 points.")
        result, _ = self.run_with(p, lambda flat, chunks: {
            str(i): {"verdict": "insufficient" if "42 points" in item[1]["claim"] else "support"}
            for i, item in enumerate(flat)
        })
        self.assertEqual(result["fields"]["main_findings"], [])

    def test_missing_fields_are_explicit_empty_lists_and_all_unsupported_is_refusal(self):
        result, _ = self.run_with({"main_findings": []}, lambda flat, chunks: {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(all(result["fields"][field] == [] for field in FIELDS))

    def test_semantic_verifier_error_drops_all_claims(self):
        def broken(flat, chunks):
            raise RuntimeError("secret token must not leak")

        result, _ = self.run_with(proposal(), broken)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("secret", json.dumps(result))

    def test_context_over_limit_fails_before_model_call(self):
        called = False

        def completion(*args, **kwargs):
            nonlocal called
            called = True
            return "{}"

        with self.assertRaises(ValueError):
            understand_paper(PAPER, CHUNKS, completion_fn=completion, max_context_chars=12)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
