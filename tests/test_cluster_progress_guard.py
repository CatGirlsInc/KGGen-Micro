import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_pilot


class ClusterProgressGuardTests(unittest.TestCase):
    def test_attempt_summary_distinguishes_recovery_from_first_success(self):
        summary = run_pilot.summarize_attempts(
            [
                {"status": "failed"},
                {"status": "succeeded"},
            ]
        )

        self.assertEqual(summary["attempts_used"], 2)
        self.assertEqual(summary["record_retry_count"], 1)
        self.assertFalse(summary["first_attempt_success"])
        self.assertTrue(summary["final_success"])

    def test_attempt_summary_keeps_unrecovered_record_failed(self):
        summary = run_pilot.summarize_attempts(
            [{"status": "failed"}, {"status": "failed"}, {"status": "failed"}]
        )

        self.assertEqual(summary["record_retry_count"], 2)
        self.assertFalse(summary["first_attempt_success"])
        self.assertFalse(summary["final_success"])

    def test_confirmation_cohort_loader_accepts_unique_records(self):
        cohort = {
            "record_ids": [24, 7],
            "records": [
                {"id": 24, "topic": "first", "text": "first source"},
                {"id": 7, "topic": "second", "text": "second source"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps(cohort), encoding="utf-8")
            loaded = run_pilot.load_cohort(path)

        self.assertEqual([record["id"] for record in loaded["records"]], [24, 7])

    def test_confirmation_cohort_loader_rejects_duplicate_ids(self):
        cohort = {
            "record_ids": [24, 24],
            "records": [
                {"id": 24, "topic": "first", "text": "first source"},
                {"id": 24, "topic": "duplicate", "text": "duplicate source"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps(cohort), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                run_pilot.load_cohort(path)

    def test_stale_suggestion_does_not_reset_no_progress_counter(self):
        cluster_module = importlib.import_module("kg_gen.steps._3_cluster_graph")
        original_cluster_items = cluster_module.cluster_items
        was_guard_installed = getattr(
            cluster_module, "_kggen_micro_progress_guard", False
        )
        calls = []

        class FakeDSPy:
            def Predict(self, signature):
                if signature == "extract":
                    def extract(items, context):
                        calls.append(set(items))
                        return SimpleNamespace(cluster={"already-clustered"})

                    return extract

                return lambda cluster, context: SimpleNamespace(
                    validated_items=set(cluster)
                )

        def process_remaining(batch, clusters, context, validate):
            for item in batch:
                clusters.append(
                    cluster_module.Cluster(representative=item, members={item})
                )

        try:
            with (
                patch.object(cluster_module, "LOOP_N", 1),
                patch.object(
                    cluster_module,
                    "get_extract_cluster_sig",
                    return_value=("extract", str),
                ),
                patch.object(
                    cluster_module,
                    "get_validate_cluster_sig",
                    return_value=("validate", str),
                ),
                patch.object(
                    cluster_module,
                    "choose_rep",
                    return_value=SimpleNamespace(representative="already-clustered"),
                ),
                patch.object(cluster_module, "_process_batch", process_remaining),
            ):
                if was_guard_installed:
                    del cluster_module._kggen_micro_progress_guard
                run_pilot.install_kggen_cluster_progress_guard()
                representatives, clusters = cluster_module.cluster_items(
                    FakeDSPy(),
                    {"already-clustered", "still-remaining"},
                    context="test",
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], {"already-clustered", "still-remaining"})
            self.assertEqual(calls[1], {"still-remaining"})
            self.assertEqual(representatives, {"already-clustered", "still-remaining"})
            self.assertEqual(
                clusters["still-remaining"], {"still-remaining"}
            )
        finally:
            cluster_module.cluster_items = original_cluster_items
            if was_guard_installed:
                cluster_module._kggen_micro_progress_guard = True
            elif hasattr(cluster_module, "_kggen_micro_progress_guard"):
                del cluster_module._kggen_micro_progress_guard


if __name__ == "__main__":
    unittest.main()