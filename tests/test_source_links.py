from __future__ import annotations

import unittest

from litagent.source_links import semantic_scholar_record_api_url


class SemanticScholarRecordLinkTests(unittest.TestCase):
    def test_numeric_corpus_id_produces_encoded_record_api_url(self):
        self.assertEqual(
            semantic_scholar_record_api_url({"paper_id": "252715594"}),
            "https://api.semanticscholar.org/graph/v1/paper/"
            "CorpusId%3A252715594?fields=title%2Curl%2CexternalIds",
        )

    def test_corpusid_field_takes_precedence_and_integer_is_supported(self):
        self.assertEqual(
            semantic_scholar_record_api_url({"paper_id": "wrong", "corpusid": 252715594}),
            semantic_scholar_record_api_url(252715594),
        )

    def test_non_corpus_ids_have_no_guessed_url(self):
        self.assertEqual(semantic_scholar_record_api_url("arxiv:2210.02399"), "")
        self.assertEqual(semantic_scholar_record_api_url({"paper_id": ""}), "")


if __name__ == "__main__":
    unittest.main()
