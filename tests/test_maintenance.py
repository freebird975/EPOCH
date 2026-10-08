import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litsearch_maintenance import audit, rebuild


class MaintenanceAuditTests(unittest.TestCase):
    def test_rebuild_rejects_collision_with_parent_source(self):
        with self.assertRaisesRegex(ValueError, "互不相同"):
            rebuild(Path("parents.jsonl"), Path("parents.jsonl"), bm25=Path("index.json"))

    def test_manifest_and_parent_child_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child = root / "parent.jsonl", root / "child.jsonl"
            parent.write_text(json.dumps({"paper_id": "p1", "full_text": "text"}) + "\n", encoding="utf-8")
            child.write_text(json.dumps({"paper_id": "c1", "parent_id": "p1"}) + "\n", encoding="utf-8")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            pm, cm = root / "parent.manifest.json", root / "child.manifest.json"
            pm.write_text(json.dumps({"corpus_sha256": sha(parent)}), encoding="utf-8")
            cm.write_text(json.dumps({"chunk_sha256": sha(child)}), encoding="utf-8")
            result = audit(parent, child)
            self.assertTrue(result["ok"], result)

    def test_stale_chunk_hash_and_missing_dense_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child = root / "parent.jsonl", root / "child.jsonl"
            parent.write_text('{"paper_id":"p1"}\n', encoding="utf-8")
            child.write_text('{"paper_id":"p1","parent_id":"p1"}\n', encoding="utf-8")
            (root / "parent.manifest.json").write_text(json.dumps({"corpus_sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}), encoding="utf-8")
            (root / "child.manifest.json").write_text(json.dumps({"chunk_sha256": "wrong"}), encoding="utf-8")
            result = audit(parent, child, dense=root / "missing.faiss")
            statuses = {row["check"]: row["status"] for row in result["checks"]}
            self.assertEqual(statuses["child_manifest"], "stale")
            self.assertEqual(statuses["dense"], "missing")
            self.assertFalse(result["ok"])

    def test_rebuild_publishes_manifest_and_is_retrievable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child, bm25 = root / "parents.jsonl", root / "chunks.jsonl", root / "bm25.json"
            row = {"paper_id": "p1", "title": "Retrieval Systems", "abstract": "retrieval augmented generation",
                   "full_text": "Introduction\nRetrieval systems retrieve useful documents.\n\nMethods\nGeneration uses retrieved evidence.",
                   "source_url": "https://example.test/p1"}
            parent.write_text(json.dumps(row) + "\n", encoding="utf-8")
            parent_manifest = root / "corpus_fulltext_manifest.json"
            parent_manifest.write_text(json.dumps({"parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}), encoding="utf-8")
            rebuilt = rebuild(parent, child, bm25=bm25, chunk_size=300, overlap=20)
            self.assertGreater(rebuilt["chunks"], 0)
            manifest_path = child.with_suffix(".manifest.json")
            self.assertTrue(manifest_path.is_file())
            audit_result = audit(parent, child, bm25=bm25)
            self.assertTrue(audit_result["ok"], audit_result)
            from litsearch_fulltext import retrieve_parents
            parents, hits = retrieve_parents("retrieval generation", top_k=1, candidate_k=5,
                                             parent_path=parent, child_path=child, index_path=bm25,
                                             retriever="bm25")
            self.assertEqual(parents[0]["paper_id"], "p1")
            self.assertTrue(hits)

    def test_failed_publication_restores_child_manifest_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child, bm25 = root / "parents.jsonl", root / "chunks.jsonl", root / "bm25.json"
            parent.write_text(json.dumps({"paper_id": "p1", "title": "Title", "abstract": "abstract",
                                          "full_text": "Retrieval content for indexing.", "source_url": "https://example.test"}) + "\n", encoding="utf-8")
            manifest = child.with_suffix(".manifest.json")
            child.write_text("old child\n", encoding="utf-8")
            manifest.write_text("old manifest\n", encoding="utf-8")
            bm25.write_text("old index\n", encoding="utf-8")
            before = {p: p.read_bytes() for p in (child, manifest, bm25)}
            import litsearch_maintenance
            real_replace = litsearch_maintenance.os.replace
            failed = False
            def fail_index_publish(source, target):
                nonlocal failed
                if not failed and Path(target) == bm25 and ".litsearch-rebuild-" in str(source):
                    failed = True
                    raise OSError("injected publish failure")
                return real_replace(source, target)
            with patch.object(litsearch_maintenance.os, "replace", side_effect=fail_index_publish):
                with self.assertRaises(OSError):
                    rebuild(parent, child, bm25=bm25, chunk_size=300, overlap=20)
            self.assertTrue(failed)
            self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == "__main__":
    unittest.main()
