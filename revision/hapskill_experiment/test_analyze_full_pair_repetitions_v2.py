#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_full_pair_repetitions_v2.py")
SPEC = importlib.util.spec_from_file_location("full_pair_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class FullPairAnalysisTests(unittest.TestCase):
    def test_failed_condition_gets_unchanged_source_itt_metrics(self) -> None:
        metrics = analysis.itt_metrics(17, {"status": "failed"})
        self.assertEqual(metrics["final_alerts"], 17)
        self.assertEqual(metrics["remaining_alerts"], 17)
        self.assertEqual(metrics["net_reduction"], 0)
        self.assertEqual(metrics["introduced_alerts"], 0)
        self.assertEqual(analysis.metric_invariants(metrics), [])

    def test_completed_condition_preserves_observed_metrics(self) -> None:
        observed = {
            "initial_alerts": 10,
            "final_alerts": 2,
            "eliminated_alerts": 9,
            "remaining_alerts": 1,
            "introduced_alerts": 1,
            "net_reduction": 8,
        }
        self.assertEqual(
            analysis.itt_metrics(10, {"status": "completed", "alert_metrics": observed}),
            observed,
        )

    def test_failure_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            category, _ = analysis.classify_failure(
                {
                    "status": "failed",
                    "failure": "RuntimeError: Round 1 did not reach HomeCheck validation after 6 agent turns",
                },
                run_dir,
            )
            self.assertEqual(category, "coverage_exhaustion")
            category, _ = analysis.classify_failure(
                {
                    "status": "failed",
                    "failure": "RuntimeError: Extra data: line 5 column 1",
                },
                run_dir,
            )
            self.assertEqual(category, "completion_evidence_parse_failure")

    def test_count_json_documents_detects_concatenated_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            path.write_text('{"first": 1}\n{"second": 2}\n', encoding="utf-8")
            self.assertEqual(analysis.count_json_documents(path), 2)

    def test_real_campaign_reconciles(self) -> None:
        rows, audit = analysis.build_rows_and_audit()
        self.assertEqual(audit["status"], "passed", audit["problems"])
        self.assertEqual(audit["task_count"], 140)
        self.assertEqual(len(rows), 70)
        self.assertEqual({row["project"] for row in rows}.__len__(), 35)
        self.assertEqual({row["repeat"] for row in rows}, {1, 2})


if __name__ == "__main__":
    unittest.main()
