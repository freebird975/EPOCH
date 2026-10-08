from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from litagent.fulltext_structure import audit_corpus_scale, parse_fulltext_structure
from litagent.litsearch_data_preparation import DataPreparationModule


class FulltextStructureTests(unittest.TestCase):
    def test_paragraphs_preserve_hierarchy_types_and_line_locations(self):
        text = """1 Introduction
Intro text here.

1.1 Method
Method text.

| A | B |
|---|---|
| 1 | 2 |

\\begin{equation}
x = y
\\end{equation}
"""
        blocks = parse_fulltext_structure(text)
        self.assertEqual(blocks[0]["section_path"], ["1 Introduction"])
        self.assertEqual(blocks[1]["section_path"], ["1 Introduction", "1.1 Method"])
        self.assertEqual(blocks[2]["paragraph_type"], "table")
        self.assertEqual(blocks[3]["paragraph_type"], "equation")
        self.assertEqual((blocks[0]["start_line"], blocks[0]["end_line"]), (2, 2))
        self.assertNotIn("page", blocks[0])

    def test_author_and_affiliation_lines_do_not_become_headings(self):
        text = """PAPER TITLE\n\nAlice Smith\nUniversity of Somewhere\n\n1 Introduction\nBody.\n"""
        blocks = parse_fulltext_structure(text)
        self.assertEqual(blocks[0]["section_path"], ["全文"])
        self.assertEqual(blocks[1]["section_path"], ["全文"])
        self.assertIn("Alice Smith", blocks[1]["text"])
        self.assertTrue(any(block["section_path"] == ["1 Introduction"] for block in blocks))

    def test_publication_dates_and_arxiv_artifacts_are_not_headings(self):
        text = """PAPER TITLE
17 Oct 2023
17 Oct 20232E097A024D332E65BEA4104F007B7871arXiv:2310.11511v1[cs.CL]

INTRODUCTION
The paper introduces its method.
"""
        blocks = parse_fulltext_structure(text)
        self.assertTrue(all("17 Oct" not in block["section"] for block in blocks))
        self.assertEqual(blocks[0]["section_path"], ["全文"])
        self.assertTrue(any(block["section_path"] == ["INTRODUCTION"] for block in blocks))

    def test_structured_chunks_are_deterministic_and_legacy_default_is_unchanged(self):
        parent = {"paper_id": "p1", "parent_id": "p1", "full_text": "1 Introduction\n\nText.",
                  "source_url": "local", "title": "Paper", "doc_type": "parent"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parents.jsonl"
            path.write_text(json.dumps(parent) + "\n", encoding="utf-8")
            prep = DataPreparationModule(path)
            legacy = list(prep.iter_chunk_documents(chunk_size=220, overlap=0))
            structured_a = list(prep.iter_chunk_documents(chunk_size=220, overlap=0, structured=True))
            structured_b = list(prep.iter_chunk_documents(chunk_size=220, overlap=0, structured=True))
        self.assertEqual(legacy[0]["section"], "1 Introduction")
        self.assertEqual(structured_a, structured_b)
        self.assertEqual(structured_a[0]["parent_id"], "p1")
        self.assertEqual(structured_a[0]["section_path"], ["1 Introduction"])
        self.assertEqual(structured_a[0]["location_type"], "source_text_line")

    def test_scale_audit_counts_bytes_and_detects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / "parents.jsonl"
            child = Path(td) / "children.jsonl"
            manifest = Path(td) / "manifest.json"
            parent.write_text(json.dumps({"paper_id": "p", "full_text": "abc"}) + "\n", encoding="utf-8")
            child.write_text(json.dumps({"chunk_id": "c"}) + "\n", encoding="utf-8")
            manifest.write_text(json.dumps({"parent_sha256": "wrong"}), encoding="utf-8")
            report = audit_corpus_scale(parent, child, target_documents=10, manifest_path=manifest)
        self.assertEqual(report["parent_documents"], 1)
        self.assertEqual(report["child_chunks"], 1)
        self.assertFalse(report["manifest_checksum_matches"])
        self.assertEqual(report["projection"]["target_documents"], 10)
        self.assertEqual(report["license_metadata"], "not_verified")


if __name__ == "__main__":
    unittest.main()
