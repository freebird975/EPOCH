from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from litagent.retrieval import build_index
from litsearch_fulltext import _digest
from litsearch_scale_eval import run_local_eval


class LocalScaleEvalTests(unittest.TestCase):
    def test_complete_qrels_only_and_index_bound_to_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = root / "corpus_fulltext.jsonl"
            children = root / "chunks.jsonl"
            queries = root / "queries.jsonl"
            index = root / "bm25.json"
            parent_rows = [
                {"paper_id": "p1", "title": "Graph Retrieval", "full_text": "Graph expansion helps retrieval."},
                {"paper_id": "p2", "title": "Other", "full_text": "Unrelated work."},
            ]
            child_rows = [
                {"paper_id": "c1", "parent_id": "p1", "chunk_id": "c1", "chunk_index": 0,
                 "title": "Graph Retrieval", "abstract": "Graph expansion helps retrieval.",
                 "section": "Methods", "source_url": "example", "evidence_scope": "full_text_chunk"},
                {"paper_id": "c2", "parent_id": "p2", "chunk_id": "c2", "chunk_index": 0,
                 "title": "Other", "abstract": "Unrelated work.",
                 "section": "Body", "source_url": "example", "evidence_scope": "full_text_chunk"},
            ]
            question_rows = [
                {"id": "q1", "question": "graph expansion retrieval", "query_set": "manual_iclr",
                 "gold_paper_ids": ["p1"]},
                {"id": "q2", "question": "other", "query_set": "manual_iclr",
                 "gold_paper_ids": ["p2", "outside"]},
            ]
            for path, rows in ((parents, parent_rows), (children, child_rows), (queries, question_rows)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "corpus_fulltext_manifest.json").write_text(
                json.dumps({"parent_sha256": _digest(parents)}), encoding="utf-8")
            children.with_suffix(".manifest.json").write_text(
                json.dumps({"parent_sha256": _digest(parents), "chunk_sha256": _digest(children),
                            "chunks": 2}), encoding="utf-8")
            build_index(children, index)
            report = run_local_eval(parent_path=parents, child_path=children, query_path=queries,
                                    bm25_path=index, top_k=1, candidate_k=2)
            self.assertEqual(report["run"]["complete_qrel_queries"], 1)
            self.assertEqual(report["run"]["partial_qrel_queries_excluded"], 1)
            self.assertEqual(report["scores"]["recall_at_k"], 1.0)
            self.assertEqual(report["rows"][0]["retrieved_paper_ids"], ["p1"])


if __name__ == "__main__":
    unittest.main()
