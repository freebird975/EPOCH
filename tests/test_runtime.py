import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litagent.agentic_retrieval import QueryAgents, _queries
from litagent.runtime import BudgetExceeded, RunRuntime, RuntimePolicy, use_runtime
from litsearch_fulltext import build_chunks, ensure_index, retrieve_parents, run_fulltext_ask
from rag.generate import chat_completion_result


class RuntimeTests(unittest.TestCase):
    def test_call_and_token_limits_stop_before_network_and_bound_timeout(self):
        runtime = RunRuntime(RuntimePolicy(max_api_calls=1, max_completion_tokens=100, api_timeout_seconds=3))
        self.assertLessEqual(runtime.claim_api(60, 60), 3)
        with use_runtime(runtime), patch("urllib.request.urlopen") as network:
            with self.assertRaises(BudgetExceeded):
                chat_completion_result([], api_key="unit-test", max_tokens=10)
            network.assert_not_called()
        with self.assertRaises(BudgetExceeded):
            RunRuntime(RuntimePolicy(max_completion_tokens=5)).claim_api(60, 6)

    def test_deadline_records_failed_stage_and_budget_never_degrades(self):
        now = [0.0]
        runtime = RunRuntime(RuntimePolicy(deadline_seconds=1, failure_policy="degrade"), clock=lambda: now[0])
        with self.assertRaises(BudgetExceeded):
            with runtime.span("dense_recall"):
                now[0] = 2.0
        self.assertEqual(runtime.timings[0]["status"], "failed")
        self.assertFalse(runtime.degrade("dense", "bm25", BudgetExceeded("expired")))

    def test_decomposer_failure_has_explicit_fallback_and_english_queries(self):
        def complete(*args, **kwargs):
            if kwargs["stage"] == "planner":
                return '{"queries":["retrieval evidence"]}'
            raise RuntimeError("unavailable")
        runtime = RunRuntime(RuntimePolicy(failure_policy="degrade"))
        with use_runtime(runtime):
            plan = QueryAgents(complete).plan("检索证据")
        self.assertEqual(plan["queries"], ["retrieval evidence"])
        self.assertEqual(runtime.degradations[0]["fallback"], "main_query_only")
        self.assertEqual(_queries(["中文 RAG", "English RAG"], 4), ["English RAG"])

    def test_missing_dense_optional_but_stale_index_never_degrades(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, child, index = root / "parent.jsonl", root / "child.jsonl", root / "bm25.json"
            parent.write_text(json.dumps({"paper_id":"p", "title":"retrieval", "full_text":"Retrieval evidence.",
                                          "source_url":"https://example.org", "evidence_scope":"full_text"}) + "\n", encoding="utf-8")
            build_chunks(parent, child)
            args = dict(parent_path=parent, child_path=child, index_path=index,
                        dense_index_path=root / "absent.faiss", retriever="hybrid")
            with self.assertRaises(FileNotFoundError):
                retrieve_parents("retrieval", **args)
            self.assertFalse(index.exists())
            ensure_index(child, index)
            runtime = RunRuntime(RuntimePolicy(failure_policy="degrade"))
            trace = {}
            with use_runtime(runtime):
                parents, _ = retrieve_parents("retrieval", trace=trace, **args)
            self.assertEqual(parents[0]["paper_id"], "p")
            self.assertEqual(trace["effective_retriever"], "bm25")
            self.assertEqual(runtime.degradations[0]["stage"], "dense")
            class FailedPlanner:
                def plan(self, question):
                    raise RuntimeError("offline")
            degraded = RunRuntime(RuntimePolicy(failure_policy="degrade"))
            with use_runtime(degraded), patch("litagent.reranker.load_reranker", side_effect=RuntimeError("missing")):
                trace = {}
                retrieve_parents("retrieval", pipeline="agentic", query_agents=FailedPlanner(),
                                 rerank=True, trace=trace, **args)
            self.assertEqual(trace["effective_pipeline"], "classic")
            self.assertFalse(trace["effective_rerank"])
            self.assertEqual([d["stage"] for d in degraded.degradations], ["dense", "planner", "reranker"])
            with use_runtime(RunRuntime(RuntimePolicy(failure_policy="degrade"))):
                with self.assertRaises(RuntimeError):
                    retrieve_parents("中文问题", pipeline="agentic", query_agents=FailedPlanner(), **args)
            data = json.loads(index.read_text(encoding="utf-8"))
            data["corpus_sha256"] = "stale"
            index.write_text(json.dumps(data), encoding="utf-8")
            with use_runtime(RunRuntime(RuntimePolicy(failure_policy="degrade"))):
                with self.assertRaises(ValueError):
                    retrieve_parents("retrieval", **args)

    def test_ask_report_retains_stage_failures_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            def retrieve(*args, **kwargs):
                return [{"paper_id":"p", "matched_chunks":[{"chunk_id":"c", "section":"Methods", "text":"Evidence", "start_line":7, "end_line":8}]}], []
            def fail(*args, **kwargs):
                raise TimeoutError("unavailable")
            with self.assertRaises(TimeoutError):
                run_fulltext_ask("Question", report_path=path, retrieve_fn=retrieve, generate_fn=fail)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["contexts"][0]["start_line"], 7)
            self.assertTrue(any(s["stage"] == "generation" and s["status"] == "failed" for s in report["timings"]))
            self.assertIn("litagent/fulltext_answer.py", report["provenance"]["code"])


if __name__ == "__main__":
    unittest.main()
