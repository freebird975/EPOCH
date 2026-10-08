import json
import unittest

from litagent.related_work import synthesize_related_work


def _parents():
    return [
        {"paper_id": "paper-b", "title": "B", "matched_chunks": [
            {"chunk_id": "b2", "section": "Experiments", "text": "Paper B evaluates retrieval on Dataset Y."},
        ]},
        {"paper_id": "paper-a", "title": "A", "matched_chunks": [
            {"chunk_id": "a1", "section": "Method", "text": "Paper A uses graph expansion for retrieval.",
             "start_line": 42, "end_line": 44, "location_type": "source_text_line"},
        ]},
    ]


def _response():
    return {
        "papers": [
            {"paper_id": "paper-b", "task": {"text": "Retrieval evaluation", "evidence": [
                {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]},
             "method": "unknown", "data": {"text": "Dataset Y", "evidence": [
                 {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]},
             "findings": "unknown"},
            {"paper_id": "paper-a", "task": "unknown",
             "method": {"text": "Graph expansion", "evidence": [
                 {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"}]},
             "data": "unknown", "findings": "unknown"},
        ],
        "groups": [{"label": "Retrieval approaches", "paper_ids": ["paper-a", "paper-b"],
                    "rationale": "Both papers study retrieval, with different evidence emphases.",
                    "evidence": [
                        {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"},
                        {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]}],
        "comparisons": [{"dimension": "Evidence emphasis", "entries": [
            {"paper_id": "paper-a", "value": "Graph expansion", "evidence": [
                {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"}]},
            {"paper_id": "paper-b", "value": "Evaluation on Dataset Y", "evidence": [
                {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]}],
            "synthesis": "The retrieved evidence describes different emphases.", "evidence": [
                {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"},
                {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]}],
    }


def _run(response, parents=None, max_context_chars=18_000, verifier_fn=None):
    captured = {}

    def complete(messages, **kwargs):
        captured["kwargs"] = kwargs
        captured["payload"] = json.loads(messages[1]["content"])
        return json.dumps(response)

    result = synthesize_related_work("retrieval approaches", parents or _parents(),
                                     complete=complete, max_context_chars=max_context_chars,
                                     verifier_fn=verifier_fn or _support_all)
    return result, captured


def _support_all(checks):
    return {"verdicts": [{"id": check["id"], "verdict": "support"} for check in checks]}


class RelatedWorkTests(unittest.TestCase):
  def test_paper_outputs_include_record_links_for_corpus_ids(self):
    parents = _parents()
    parents[0]["corpusid"] = 252715594
    result, _ = _run(_response(), parents=parents)
    papers = {paper["paper_id"]: paper for paper in result["papers"]}
    self.assertEqual(papers["paper-b"]["paper_record_api_url"],
                     "https://api.semanticscholar.org/graph/v1/paper/"
                     "CorpusId%3A252715594?fields=title%2Curl%2CexternalIds")
    self.assertEqual(papers["paper-a"]["paper_record_api_url"], "")

  def test_supported_facts_and_multpaper_synthesis_have_exact_citations(self):
    result, captured = _run(_response())
    papers = {paper["paper_id"]: paper for paper in result["papers"]}
    self.assertEqual(papers["paper-a"]["method"]["text"], "Graph expansion")
    self.assertEqual(papers["paper-a"]["method"]["evidence"], [
        {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method",
         "start_line": 42, "end_line": 44, "location_type": "source_text_line"}]
    )
    self.assertTrue(result["groups"][0]["is_inference"])
    self.assertEqual(len({ref["paper_id"] for ref in result["groups"][0]["evidence"]}), 2)
    self.assertTrue(result["comparisons"][0]["is_inference"])
    self.assertEqual(result["coverage"]["scope"], "retrieved_subset")
    self.assertIn("not an exhaustive survey", result["coverage"]["caveat"])
    self.assertTrue(captured["kwargs"]["json_mode"])
    self.assertEqual(captured["kwargs"]["stage"], "related_work")


  def test_invalid_or_cross_paper_fact_citations_become_unknown(self):
    response = _response()
    response["papers"][1]["method"] = {"text": "Graph expansion", "evidence": [
        {"paper_id": "paper-b", "chunk_id": "b2", "section": "Experiments"}]}
    response["papers"][0]["task"] = {"text": "Made up", "evidence": [
        {"paper_id": "paper-b", "chunk_id": "nonexistent", "section": "Experiments"}]}
    result, _ = _run(response)
    papers = {paper["paper_id"]: paper for paper in result["papers"]}
    for paper_id, key in (("paper-a", "method"), ("paper-b", "task")):
        self.assertEqual(papers[paper_id][key]["status"], "unknown")
        self.assertEqual(papers[paper_id][key]["evidence"], [])


  def test_unverifiable_cross_paper_group_is_dropped_and_order_is_stable(self):
    response = _response()
    response["groups"].append({"label": "Unsupported", "paper_ids": ["paper-a", "paper-b"],
                               "rationale": "unsupported comparison", "evidence": [
                                   {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"}]})
    first, _ = _run(response)
    second, _ = _run(response, list(reversed(_parents())))
    self.assertEqual([paper["paper_id"] for paper in first["papers"]], ["paper-a", "paper-b"])
    self.assertEqual(first["groups"], second["groups"])
    self.assertEqual(first["comparisons"], second["comparisons"])
    self.assertEqual([group["label"] for group in first["groups"]], ["Retrieval approaches"])


  def test_context_is_bounded_and_empty_evidence_returns_machine_readable_empty_result(self):
    result, captured = _run(_response(), parents=[{
        "paper_id": "paper-a", "title": "A", "matched_chunks": [
            {"chunk_id": "big", "section": "Body", "text": "x" * 10_000},
        ],
    }], max_context_chars=500)
    self.assertEqual(captured, {})  # Empty evidence returns without making a model call.
    self.assertEqual(result["papers"], [])
    self.assertEqual(result["coverage"]["retrieved_papers"], 0)

  def test_coverage_lists_omitted_candidates(self):
    response = {"papers": [_response()["papers"][1]]}
    result, _ = _run(response)
    self.assertEqual(result["coverage"]["candidate_paper_ids"], ["paper-b", "paper-a"])
    self.assertEqual(result["coverage"]["included_paper_ids"], ["paper-a"])
    self.assertEqual(result["coverage"]["omitted_paper_ids"], ["paper-b"])

  def test_retrieval_rank_gets_one_chunk_per_paper_before_extra_chunks(self):
    parents = [
        {"paper_id": "z", "title": "Higher hit", "matched_chunks": [
            {"chunk_id": "z1", "section": "Methods", "text": "z" * 500},
            {"chunk_id": "z2", "section": "Results", "text": "z" * 500}]},
        {"paper_id": "a", "title": "Lower hit", "matched_chunks": [
            {"chunk_id": "a1", "section": "Methods", "text": "a" * 500}]},
    ]
    result, captured = _run({"papers": []}, parents=parents, max_context_chars=1600)
    self.assertEqual([row["chunk_id"] for row in result["evidence_contexts"]], ["z1", "a1"])
    self.assertEqual(result["coverage"]["retrieved_papers"], 2)

  def test_semantically_unsupported_claim_with_valid_citation_becomes_unknown(self):
    response = _response()

    def verifier(checks):
        verdicts = []
        for check in checks:
            verdict = "insufficient" if check["claim"] == "Fabricated result" else "support"
            verdicts.append({"id": check["id"], "verdict": verdict})
        return {"verdicts": verdicts}

    response["papers"][1]["findings"] = {"text": "Fabricated result", "evidence": [
        {"paper_id": "paper-a", "chunk_id": "a1", "section": "Method"}]}
    result, _ = _run(response, verifier_fn=verifier)
    paper = next(item for item in result["papers"] if item["paper_id"] == "paper-a")
    self.assertEqual(paper["findings"]["text"], "unknown")
    self.assertEqual(paper["findings"]["reason"], "semantic_support_not_confirmed")

  def test_verifier_failure_fails_closed_for_all_claims(self):
    def verifier(_checks):
        raise RuntimeError("simulated verifier outage")

    result, _ = _run(_response(), verifier_fn=verifier)
    papers = {paper["paper_id"]: paper for paper in result["papers"]}
    self.assertEqual(papers["paper-a"]["method"]["status"], "unknown")
    self.assertEqual(result["groups"], [])
    self.assertEqual(result["comparisons"], [])

  def test_semantically_unsupported_cross_paper_inference_is_dropped(self):
    def verifier(checks):
        return {"verdicts": [
            {"id": check["id"],
             "verdict": "insufficient" if check["kind"] == "cross_paper_inference"
             and check["claim"].startswith("Both papers") else "support"}
            for check in checks
        ]}

    result, _ = _run(_response(), verifier_fn=verifier)
    self.assertEqual(result["groups"], [])
    self.assertEqual(len(result["comparisons"]), 1)

  def test_group_or_comparison_cannot_claim_uncited_paper_membership(self):
    response = _response()
    response["papers"].append({"paper_id": "paper-c", "task": "unknown", "method": "unknown",
                               "data": "unknown", "findings": "unknown"})
    response["groups"][0]["paper_ids"].append("paper-c")
    response["comparisons"][0]["entries"][1]["evidence"] = []
    result, _ = _run(response, parents=_parents() + [{"paper_id": "paper-c", "title": "C",
        "matched_chunks": [{"chunk_id": "c1", "section": "Body", "text": "Paper C studies retrieval."}]}])
    self.assertEqual(result["groups"], [])
    self.assertEqual(result["comparisons"], [])

  def test_comparison_is_removed_if_any_entry_fails_semantic_verification(self):
    def verifier(checks):
        return {"verdicts": [
            {"id": check["id"], "verdict": "insufficient" if
             check["kind"] == "single_paper_comparison_value" and
             check["claim"] == "Evaluation on Dataset Y" else "support"}
            for check in checks]}

    result, _ = _run(_response(), verifier_fn=verifier)
    self.assertEqual(result["comparisons"], [])


if __name__ == "__main__":
    unittest.main()

