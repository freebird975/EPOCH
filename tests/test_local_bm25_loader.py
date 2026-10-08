import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from litagent.retrieval import build_index, load_index, search


def _corpus(path: Path) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in [
        {"paper_id": "a", "source_url": "https://example.test/a", "title": "Graph neural retrieval", "abstract": "graph methods", "evidence_scope": "full_text_chunk"},
        {"paper_id": "b", "source_url": "https://example.test/b", "title": "Language models", "abstract": "retrieval systems", "evidence_scope": "full_text_chunk"},
    ]) + "\n", encoding="utf-8")
    return path


class LocalBm25LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_sqlite_relocated_corpus_searches_read_only_and_preserves_file(self):
        corpus = _corpus(self.root / "chunks # one.jsonl")
        index_path = self.root / "bm25 index #1.sqlite3"
        build_index(corpus, index_path)
        moved = self.root / "relocated" / corpus.name
        moved.parent.mkdir()
        moved.write_bytes(corpus.read_bytes())
        before = (index_path.read_bytes(), index_path.stat().st_mtime_ns)

        index = load_index(index_path, corpus_path=moved)
        self.addCleanup(index["connection"].close)
        self.assertEqual(index["documents"], 2)
        self.assertEqual(index["stored_corpus_path"], corpus.resolve().as_posix())
        self.assertEqual(index["corpus_path"], str(moved.resolve()))
        self.assertEqual(search(index, "graph neural", 2)[0]["paper_id"], "a")
        with self.assertRaises(sqlite3.OperationalError):
            index["connection"].execute("INSERT INTO docs VALUES (2, 'x', 0, '{}')")
        self.assertEqual((index_path.read_bytes(), index_path.stat().st_mtime_ns), before)

    def test_sqlite_corpus_override_must_validate(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        index_path = self.root / "index.sqlite3"
        build_index(corpus, index_path)
        wrong = self.root / "wrong.jsonl"
        wrong.write_text("different\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_index(index_path, corpus_path=wrong)
        with self.assertRaises(ValueError):
            load_index(index_path, corpus_path=self.root / "missing.jsonl")

    def test_sqlite_rejects_bad_metadata_version_or_document_count(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        for key, value in [("version", 2), ("documents", 3)]:
            with self.subTest(key=key):
                index_path = self.root / f"{key}.sqlite3"
                build_index(corpus, index_path)
                connection = sqlite3.connect(index_path)
                connection.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(value), key))
                connection.commit()
                connection.close()
                with self.assertRaises(ValueError):
                    load_index(index_path, corpus_path=corpus)

    def test_sqlite_rejects_bool_count_and_invalid_bm25_parameters(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        for key, value in [("documents", True), ("k1", 0), ("b", 1.5), ("avg_length", float("inf"))]:
            with self.subTest(key=key):
                index_path = self.root / f"invalid-{key}.sqlite3"
                build_index(corpus, index_path)
                connection = sqlite3.connect(index_path)
                connection.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(value), key))
                connection.commit()
                connection.close()
                with self.assertRaises(ValueError):
                    load_index(index_path, corpus_path=corpus)

    def test_explicit_override_handles_malformed_stored_path(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        index_path = self.root / "index.sqlite3"
        build_index(corpus, index_path)
        connection = sqlite3.connect(index_path)
        connection.execute("UPDATE metadata SET value='42' WHERE key='corpus_path'")
        connection.commit()
        connection.close()
        with self.assertRaises(ValueError):
            load_index(index_path)
        index = load_index(index_path, corpus_path=corpus)
        self.addCleanup(index["connection"].close)
        self.assertEqual(index["stored_corpus_path"], 42)
        self.assertEqual(index["corpus_path"], str(corpus.resolve()))

    def test_sqlite_rejects_non_primary_doc_id(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        index_path = self.root / "index.sqlite3"
        build_index(corpus, index_path)
        connection = sqlite3.connect(index_path)
        connection.execute("CREATE TABLE docs_copy AS SELECT * FROM docs")
        connection.execute("DROP TABLE docs")
        connection.execute("ALTER TABLE docs_copy RENAME TO docs")
        connection.commit()
        connection.close()
        with self.assertRaises(ValueError):
            load_index(index_path, corpus_path=corpus)

    def test_corrupt_sqlite_fails_without_rebuild(self):
        path = self.root / "broken.sqlite3"
        path.write_bytes(b"not a database")
        with self.assertRaises(ValueError):
            load_index(path, corpus_path=self.root / "unused.jsonl")
        self.assertEqual(path.read_bytes(), b"not a database")

    def test_sqlite_rejects_malformed_metadata_json(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        index_path = self.root / "index.sqlite3"
        build_index(corpus, index_path)
        connection = sqlite3.connect(index_path)
        connection.execute("UPDATE metadata SET value='{' WHERE key='documents'")
        connection.commit()
        connection.close()
        with self.assertRaises(ValueError):
            load_index(index_path, corpus_path=corpus)

    def test_json_loader_accepts_explicit_relocated_corpus(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        build_index(corpus, self.root / "index.json")
        target = self.root / "copy" / "chunks.jsonl"
        target.parent.mkdir()
        target.write_bytes(corpus.read_bytes())
        index = load_index(self.root / "index.json", corpus_path=target)
        self.assertEqual(index["stored_corpus_path"], corpus.resolve().as_posix())
        self.assertEqual(index["corpus_path"], str(target.resolve()))
        self.assertEqual(search(index, "graph neural", 1)[0]["paper_id"], "a")

    def test_loader_and_search_reject_changed_corpus(self):
        corpus = _corpus(self.root / "chunks.jsonl")
        index_path = self.root / "index.sqlite3"
        build_index(corpus, index_path)
        index = load_index(index_path, corpus_path=corpus)
        self.addCleanup(index["connection"].close)
        with corpus.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            search(index, "graph neural")


if __name__ == "__main__":
    unittest.main()
