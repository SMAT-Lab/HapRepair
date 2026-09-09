#!/usr/bin/env python3

from __future__ import annotations

import unittest

import reconcile_v14_e1 as reconcile


class MetricTests(unittest.TestCase):
    def test_valid_multiset_metrics(self) -> None:
        metrics = {
            "initial_alerts": 10,
            "final_alerts": 3,
            "eliminated_alerts": 8,
            "remaining_alerts": 2,
            "introduced_alerts": 1,
            "net_reduction": 7,
        }
        self.assertEqual(reconcile.metric_invariants(metrics), [])

    def test_invalid_multiset_metrics_report_each_relation(self) -> None:
        metrics = {
            "initial_alerts": 10,
            "final_alerts": 4,
            "eliminated_alerts": 9,
            "remaining_alerts": 2,
            "introduced_alerts": 1,
            "net_reduction": 3,
        }
        self.assertEqual(len(reconcile.metric_invariants(metrics)), 3)


class AggregateTests(unittest.TestCase):
    def project(
        self, name: str, initial: int, eliminated: int, introduced: int
    ) -> dict:
        remaining = initial - eliminated
        return {
            "project": name,
            "initial_alerts": initial,
            "final_alerts": remaining + introduced,
            "eliminated_alerts": eliminated,
            "remaining_alerts": remaining,
            "introduced_alerts": introduced,
            "net_reduction": eliminated - introduced,
            "validation_scans": 1,
            "input_tokens": 10,
            "cached_input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 12,
            "wall_clock_seconds": 1.5,
            "per_rule_alert_deltas": [
                {
                    "rule": "@performance/example",
                    "eliminated_alerts": eliminated,
                    "remaining_alerts": remaining,
                    "introduced_alerts": introduced,
                }
            ],
        }

    def test_aggregate_preserves_gross_and_net_metrics(self) -> None:
        observed = reconcile.aggregate_rows(
            [self.project("one", 10, 8, 1), self.project("two", 5, 5, 0)]
        )
        self.assertEqual(observed["initial_alerts"], 15)
        self.assertEqual(observed["eliminated_alerts"], 13)
        self.assertEqual(observed["introduced_alerts"], 1)
        self.assertEqual(observed["net_reduction"], 12)

    def test_rule_aggregate_keeps_introduced_only_rule(self) -> None:
        row = self.project("one", 1, 1, 0)
        row["per_rule_alert_deltas"].append(
            {
                "rule": "@performance/introduced",
                "eliminated_alerts": 0,
                "remaining_alerts": 0,
                "introduced_alerts": 1,
            }
        )
        rules = {item["rule"]: item for item in reconcile.aggregate_rules([row])}
        self.assertEqual(rules["@performance/introduced"]["initial_alerts"], 0)
        self.assertEqual(rules["@performance/introduced"]["final_alerts"], 1)
        self.assertIsNone(rules["@performance/introduced"]["gross_elimination_rate"])


class StratumTests(unittest.TestCase):
    def test_alert_volume_boundaries(self) -> None:
        self.assertEqual(reconcile.alert_volume(99), "lt_100")
        self.assertEqual(reconcile.alert_volume(100), "100_to_499")
        self.assertEqual(reconcile.alert_volume(499), "100_to_499")
        self.assertEqual(reconcile.alert_volume(500), "ge_500")


class ResidualTests(unittest.TestCase):
    def test_same_rule_introduced_alert_uses_delta_identity(self) -> None:
        row = {"project": "example", "initial_per_rule": {"@performance/example": 1}}
        finding = {
            "relative_path": "new.ets",
            "line": 2,
            "column": 1,
            "severity": "warn",
            "rule": "@performance/example",
            "message": "same rule, new identity",
        }
        rows = reconcile.residual_rows(
            [row],
            {
                "example": {
                    "findings": [finding],
                    "deltas": {"remaining": [], "introduced": [finding]},
                }
            },
        )
        self.assertEqual(rows[0]["classification"], "introduced")


if __name__ == "__main__":
    unittest.main()
