import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_pilot


class ClusterProgressGuardTests(unittest.TestCase):
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