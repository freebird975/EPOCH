import json
import tempfile
import unittest
from pathlib import Path

from litagent.performance import analyze_reports, percentile


class PerformanceReportTests(unittest.TestCase):
    def test_same_questions_from_two_versions_do_not_share_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version, seconds in (("v1", 2), ("v2", 9)):
                self.write(root, f"{version}/e2e.json", {
                    "run": {"prompt_version": version, "parent_sha256": "p", "chunk_sha256": "c", "top_k": 5},
                    "systems": {"classic_hybrid": {"rows": [{"id": "q", "question": "same q",
                        "answer": {"failed": False, "refused": False}, "retrieval": {"recall_at_k": 1}}]}}})
                self.write(root, f"{version}/details/q.json", {
                    "run": {"question": "same q", "pipeline": "classic", "parent_sha256": "p", "child_sha256": "c", "top_k": 5},
                    "status": "verified", "elapsed_seconds": seconds})
            groups = analyze_reports([root])["groups"]
            self.assertEqual(len(groups), 2)
            self.assertEqual({g["metadata"]["prompt_version"]: g["latency"]["successful"]["p50_seconds"]
                              for g in groups}, {"v1": 2, "v2": 9})
            self.assertTrue(all(g["run_count"] == 1 for g in groups))

    def test_aggregate_only_retains_nested_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, "aggregate.json", {"systems": {"classic_hybrid": {"rows": [{
                "id": "q", "question": "q", "answer": {"failed": False, "refused": False,
                    "diagnostics": {"status": "verified", "elapsed_seconds": 3,
                        "timings": [{"stage": "retrieval", "seconds": 1, "status": "ok"}],
                        "api_calls": [{"stage": "answer", "latency_seconds": 2,
                                       "usage": {"prompt_tokens": 10, "completion_tokens": 1}}]}},
                "retrieval": {"status": "scored", "recall_at_k": 1}}]}}})
            group = analyze_reports([root])["groups"][0]
            self.assertEqual(group["latency"]["successful"]["p50_seconds"], 3)
            self.assertEqual(group["stages"]["retrieval"]["p50_seconds"], 1)
            self.assertEqual(group["api_calls_by_stage"]["answer"]["prompt_tokens"], 10)

    def write(self, root: Path, name: str, data: dict) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_percentile_uses_type_7_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.5), 5)
        self.assertEqual(percentile([0, 10], 0.95), 9.5)
        self.assertIsNone(percentile([], 0.5))

    def test_outcomes_timings_api_metrics_and_prompt_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {"parent_sha256": "corpus-a", "child_sha256": "chunks-a", "pipeline": "classic"}
            self.write(root, "s.json", {"run": {**common, "prompt_version": "v1"}, "status": "verified",
                "elapsed_seconds": 10, "timings": [{"stage": "answer", "seconds": 8, "status": "ok"}],
                "api_calls": [{"stage": "answer", "latency_seconds": 2, "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}]})
            self.write(root, "r.json", {"run": {**common, "prompt_version": "v1"}, "status": "refused_unverified", "elapsed_seconds": 20,
                "timings": [{"stage": "answer", "seconds": 16, "status": "refused"}]})
            self.write(root, "f.json", {"run": {**common, "prompt_version": "v1"}, "status": "failed", "elapsed_seconds": 100})
            self.write(root, "u.json", {"run": {**common, "prompt_version": "v1"}, "elapsed_seconds": 4})
            self.write(root, "v2.json", {"run": {**common, "prompt_version": "v2"}, "status": "verified", "elapsed_seconds": 12})
            result = analyze_reports([root])
            self.assertEqual(len(result["groups"]), 2)
            v1 = next(g for g in result["groups"] if g["metadata"]["prompt_version"] == "v1")
            self.assertEqual(v1["outcome_counts"], {"successful": 1, "refused": 1, "failed": 1, "unlabeled": 1})
            self.assertEqual(v1["latency"]["successful"]["p50_seconds"], 10)
            self.assertEqual(v1["latency"]["failed"]["sample_count"], 1)
            self.assertEqual(v1["stages"]["answer"]["p50_seconds"], 12)
            self.assertEqual(v1["api_calls_by_stage"]["answer"]["total_tokens"], 7)

    def test_nested_e2e_rows_group_quality_and_dedupe_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"prompt_version": "p0-v1", "parent_sha256": "p", "chunk_sha256": "c", "top_k": 5}
            aggregate = {"schema_version": 1, "dataset": {"sha256": "ds"}, "run": metadata,
                "systems": {"classic_hybrid": {"rows": [
                    {"id": "q1", "question": "original q", "category": "answerable", "answer": {"failed": False, "refused": False},
                     "retrieval": {"recall_at_k": 1, "mrr_at_k": 1, "ndcg_at_k": 1}},
                    {"id": "q2_zh", "question": "translated q", "category": "answerable", "answer": {"failed": False, "refused": False},
                     "retrieval": {"recall_at_k": 0, "mrr_at_k": 0, "ndcg_at_k": 0}},
                ]}}}
            self.write(root, "aggregate.json", aggregate)
            # Same query/config as a nested detail report: use the detail metrics
            # and avoid counting the aggregate row a second time.
            self.write(root, "details/classic/q1.json", {"run": {"question": "original q", "pipeline": "classic",
                "parent_sha256": "p", "child_sha256": "c", "top_k": 5}, "status": "verified", "elapsed_seconds": 2})
            result = analyze_reports([root])
            self.assertEqual(len(result["groups"]), 1)
            group = result["groups"][0]
            self.assertEqual(group["metadata"]["prompt_version"], "p0-v1")
            self.assertEqual(group["run_count"], 2)
            self.assertEqual(group["latency"]["successful"]["sample_count"], 1)
            self.assertEqual(group["retrieval_quality"]["original:answerable"]["recall_at_k"]["mean"], 1)
            self.assertEqual(group["retrieval_quality"]["translated:answerable"]["recall_at_k"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
