import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litagent.provenance import collect_provenance, sha256_file


class ProvenanceTests(unittest.TestCase):
    def test_hash_and_missing_artifacts_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child = root / "parents.jsonl", root / "chunks.jsonl"
            parent.write_text('{"paper_id":"p1"}\n', encoding="utf-8")
            child.write_text('{"paper_id":"c1"}\n', encoding="utf-8")
            result = collect_provenance(parent, child, root / "absent.json", root / "absent.faiss")
            self.assertEqual(result["files"]["parent"]["sha256"], sha256_file(parent))
            self.assertFalse(result["files"]["bm25"]["exists"])
            self.assertFalse(result["files"]["dense_meta"]["exists"])

    def test_base_url_drops_credentials_query_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child = root / "p", root / "c"
            parent.write_text("", encoding="utf-8")
            child.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"RAG_API_BASE_URL": "https://user:secret@example.test/v1?key=secret"}):
                result = collect_provenance(parent, child)
            self.assertEqual(result["models"]["chat"]["base_url"], "https://example.test")
            self.assertNotIn("secret", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
