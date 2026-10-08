from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from litagent.corpus import load_papers, parse_atom
from litagent.evaluation import load_questions
from litagent.retrieval import build_index, load_index, search
import lit
from rag.generate import answer, resolve_api_key, translate_query_to_english

ROOT = Path(__file__).resolve().parents[1]


class LiteraturePipelineTests(unittest.TestCase):
    def test_saved_snapshot_and_evidence_quotes_are_consistent(self):
        raw = (ROOT / "data/raw/arxiv.xml").read_bytes()
        papers, metadata = parse_atom(raw)
        saved = load_papers(ROOT / "data/papers.jsonl")
        self.assertEqual(len(papers), 100)
        self.assertEqual([paper["paper_id"] for paper in papers],
                         [paper["paper_id"] for paper in saved])
        self.assertEqual(metadata["normalized_count"], 100)
        questions = load_questions(
            ROOT / "eval/seed_questions.jsonl",
            {paper["paper_id"]: paper for paper in saved},
        )
        self.assertEqual(len(questions), 10)
        chinese = load_questions(
            ROOT / "eval/zh_questions.jsonl",
            {paper["paper_id"]: paper for paper in saved},
        )
        self.assertEqual(len(chinese), len(questions))
        self.assertEqual(
            [item["gold_paper_ids"] for item in chinese],
            [item["gold_paper_ids"] for item in questions],
        )

    def test_index_detects_corpus_change(self):
        papers = [
            {
                "paper_id": "1", "title": "Corrective retrieval", "abstract": "A retrieval evaluator judges evidence quality.",
                "source_url": "https://arxiv.org/abs/1", "evidence_scope": "abstract",
            },
            {
                "paper_id": "2", "title": "Graph search", "abstract": "Graph structure supports relational search.",
                "source_url": "https://arxiv.org/abs/2", "evidence_scope": "abstract",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "papers.jsonl"
            index_path = root / "index.json"
            corpus_path.write_text(
                "".join(json.dumps(paper) + "\n" for paper in papers), encoding="utf-8"
            )
            build_index(corpus_path, index_path)
            hits = search(load_index(index_path), "evaluator evidence quality")
            self.assertEqual(hits[0]["paper_id"], "1")
            corpus_path.write_text(corpus_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 不匹配"):
                load_index(index_path)

    def test_ask_passes_abstracts_with_arxiv_sources(self):
        output = io.StringIO()
        question = "How does corrective retrieval evaluate evidence?"
        with (
            patch("sys.argv", ["lit.py", "ask", question, "--retriever", "bm25", "--index", str(ROOT / "data/bm25_index.json"), "--top-k", "5"]),
            patch("lit.answer", return_value="摘要中描述了检索质量评估。[2] 复现工作也讨论了这一点。[1]") as generate,
            redirect_stdout(output),
        ):
            self.assertEqual(lit.main(), 0)
        sent_question, context = generate.call_args.args
        self.assertEqual(sent_question, question)
        self.assertEqual(len(context), 5)
        self.assertTrue(all(hit["section"] == "Abstract / 摘要" for hit in context))
        self.assertTrue(all("https://arxiv.org/abs/" in hit["source"] for hit in context))
        self.assertIn("仅基于检索到的论文摘要", output.getvalue())
        source_lines = [line for line in output.getvalue().splitlines() if "arXiv:" in line]
        self.assertEqual(len(source_lines), 2)
        self.assertTrue(source_lines[0].startswith("[1]"))
        self.assertTrue(source_lines[1].startswith("[2]"))

    def test_ask_flags_invalid_citation_numbers(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["lit.py", "ask", "corrective retrieval", "--retriever", "bm25", "--index", str(ROOT / "data/bm25_index.json"), "--top-k", "2"]),
            patch("lit.answer", return_value="A claim with a missing source [9]."),
            redirect_stdout(output),
        ):
            self.assertEqual(lit.main(), 0)
        self.assertIn("没有有效的来源引用", output.getvalue())
        self.assertIn("未提供的编号 [9]", output.getvalue())
        self.assertNotIn("arXiv:", output.getvalue())

    def test_citation_parser_accepts_grouped_numbers(self):
        self.assertEqual(lit._cited_numbers("First [2, 1]; repeated [2] and [3；4]."), [2, 1, 3, 4])

    def test_chinese_ask_uses_english_query_but_keeps_original_question(self):
        question = "纠错式 RAG 如何处理低质量检索结果？"
        translated = "How does corrective RAG handle poor retrieval?"
        output = io.StringIO()
        with (
            patch("sys.argv", ["lit.py", "ask", question, "--retriever", "bm25", "--top-k", "3"]),
            patch("lit.resolve_api_key", return_value="test-only") as key,
            patch("lit.translate_query_to_english", return_value=translated) as translate,
            patch("lit.answer", return_value="它评估检索质量。[1]") as generate,
            redirect_stdout(output),
        ):
            self.assertEqual(lit.main(), 0)
        key.assert_called_once()
        translate.assert_called_once_with(question, api_key="test-only")
        self.assertEqual(generate.call_args.args[0], question)
        self.assertEqual(generate.call_args.kwargs, {"api_key": "test-only"})
        self.assertIn("英文检索查询：" + translated, output.getvalue())

    def test_no_translate_skips_api_for_chinese_search(self):
        with (
            patch("sys.argv", ["lit.py", "search", "纠错式 RAG", "--retriever", "bm25", "--no-translate"]),
            patch("lit.resolve_api_key") as key,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(lit.main(), 0)
        key.assert_not_called()

    def test_auto_chinese_uses_bge_with_translation(self):
        question = "纠错式 RAG 如何评估检索结果？"
        translated = "How does corrective RAG evaluate retrieval results?"
        seen = []
        searchers = {"dense": lambda query, limit: seen.append((query, limit)) or []}
        metadata = {"dense_model": "BAAI/bge-small-en-v1.5"}
        with (
            patch("sys.argv", ["lit.py", "search", question]),
            patch("lit._make_searchers", return_value=(searchers, [], metadata)) as make_searchers,
            patch("lit.resolve_api_key", return_value="test-only") as key,
            patch("lit.translate_query_to_english", return_value=translated) as translate,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(lit.main(), 0)
        self.assertEqual(make_searchers.call_args.args[0], {"dense"})
        self.assertEqual(seen, [(translated, 5)])
        key.assert_called_once()
        translate.assert_called_once_with(question, api_key="test-only")

    def test_auto_chinese_ask_keeps_original_question_for_generation(self):
        question = "纠错式 RAG 如何处理低质量检索？"
        hit = {
            "paper_id": "2401.15884", "title": "Corrective Retrieval Augmented Generation",
            "source_url": "https://arxiv.org/abs/2401.15884v3", "abstract": "A retrieval evaluator assesses quality.",
        }
        with (
            patch("sys.argv", ["lit.py", "ask", question]),
            patch("lit._make_searchers", return_value=({"dense": lambda _query, _limit: [hit]}, [], {"dense_model": "BAAI/bge-small-en-v1.5"})),
            patch("lit.resolve_api_key", return_value="test-only"),
            patch("lit.translate_query_to_english", return_value="How does corrective RAG handle poor retrieval?") as translate,
            patch("lit.answer", return_value="它评估检索质量。[1]") as generate,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(lit.main(), 0)
        translate.assert_called_once_with(question, api_key="test-only")
        self.assertEqual(generate.call_args.args[0], question)
        self.assertEqual(generate.call_args.kwargs, {"api_key": "test-only"})

    def test_translation_request_returns_only_search_query(self):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["payload"] = json.loads(request.data)
            response = {"choices": [{"message": {"content": "Corrective RAG retrieval quality evaluation"}}]}
            return io.BytesIO(json.dumps(response).encode("utf-8"))

        with patch("rag.generate.urllib.request.urlopen", side_effect=fake_urlopen):
            translated = translate_query_to_english("纠错式 RAG 如何评价检索质量？", api_key="test-only")
        self.assertEqual(translated, "Corrective RAG retrieval quality evaluation")
        self.assertEqual(seen["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(seen["payload"]["model"], "deepseek-flash")
        self.assertEqual(seen["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(seen["payload"]["max_tokens"], 256)

    def test_answer_prompt_preserves_question_language_and_sources(self):
        context = [{"source": "arXiv:2401.15884", "section": "Abstract / 摘要", "text": "A retrieval evaluator assesses evidence."}]
        with patch("rag.generate._chat_completion", return_value="根据摘要，评估器检查证据。[1]") as completion:
            self.assertIn("[1]", answer("纠错式 RAG 如何评估证据？", context, api_key="test-only"))
        messages = completion.call_args.args[0]
        self.assertIn("中文问题用中文，英文问题用英文", messages[0]["content"])
        self.assertIn("arXiv:2401.15884", messages[1]["content"])
        self.assertIn("纠错式 RAG 如何评估证据？", messages[1]["content"])

    def test_project_env_key_loads_with_environment_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text("DEEPSEEK_API_KEY=test-project-key\n", encoding="utf-8")
            with patch("rag.generate.PROJECT_ENV", env_file), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_api_key(), "test-project-key")
                os.environ["DEEPSEEK_API_KEY"] = "test-process-key"
                self.assertEqual(resolve_api_key(), "test-process-key")


if __name__ == "__main__":
    unittest.main()
