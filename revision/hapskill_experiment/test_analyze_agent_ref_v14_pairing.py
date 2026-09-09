#!/usr/bin/env python3

from __future__ import annotations

import unittest

import analyze_agent_ref_v14_pairing as analysis


def condition(initial: int, eliminated: int, introduced: int, tokens: int = 10) -> dict:
    remaining = initial - eliminated
    return {
        "initial_alerts": initial,
        "final_alerts": remaining + introduced,
        "eliminated_alerts": eliminated,
        "remaining_alerts": remaining,
        "introduced_alerts": introduced,
        "net_reduction": eliminated - introduced,
        "validation_scans": 1,
        "total_tokens": tokens,
        "wall_clock_seconds": 2.0,
    }


class MetricTests(unittest.TestCase):
    def test_metric_invariants_accept_multiset_contract(self) -> None:
        self.assertEqual(analysis.metric_invariants(condition(10, 8, 1)), [])

    def test_metric_invariants_report_all_broken_relations(self) -> None:
        metrics = condition(10, 8, 1)
        metrics.update(final_alerts=9, net_reduction=1)
        self.assertEqual(len(analysis.metric_invariants(metrics)), 2)


class AggregateTests(unittest.TestCase):
    def test_paired_aggregate_preserves_condition_denominators(self) -> None:
        rows = [
            {"skill": condition(10, 10, 0), "reference": condition(10, 9, 1)},
            {"skill": condition(5, 4, 0), "reference": condition(5, 5, 0)},
        ]
        observed = analysis.paired_aggregate(rows)
        self.assertEqual(observed["skill"]["initial_alerts"], 15)
        self.assertEqual(observed["reference"]["initial_alerts"], 15)
        self.assertEqual(observed["reference_minus_skill"]["introduced_alerts"], 1)

    def test_wifi_sensitivity_exclusion_is_exact(self) -> None:
        rows = [
            {
                "project": "wifi_testapp",
                "skill": condition(10, 10, 0),
                "reference": condition(10, 10, 0),
            },
            {
                "project": "other",
                "skill": condition(5, 5, 0),
                "reference": condition(5, 4, 0),
            },
        ]
        selected = [row for row in rows if row["project"] != "wifi_testapp"]
        observed = analysis.paired_aggregate(selected)
        self.assertEqual(observed["skill"]["project_count"], 1)
        self.assertEqual(observed["skill"]["initial_alerts"], 5)


class PerRuleTests(unittest.TestCase):
    def row(self, project: str, skill: dict, reference: dict) -> dict:
        return {
            "project": project,
            "skill_per_rule": skill,
            "reference_per_rule": reference,
        }

    def test_common_support_and_introduced_only_are_separate(self) -> None:
        observed = analysis.build_rules(
            [
                self.row(
                    "one",
                    {
                        "@performance/base": {
                            "eliminated_alerts": 2,
                            "remaining_alerts": 0,
                            "introduced_alerts": 0,
                        }
                    },
                    {
                        "@performance/base": {
                            "eliminated_alerts": 1,
                            "remaining_alerts": 1,
                            "introduced_alerts": 0,
                        },
                        "@performance/new": {
                            "eliminated_alerts": 0,
                            "remaining_alerts": 0,
                            "introduced_alerts": 1,
                        },
                    },
                )
            ]
        )
        self.assertEqual(len(observed["common_support"]), 1)
        self.assertEqual(len(observed["introduced_only"]), 1)
        self.assertEqual(observed["common_support"][0]["initial_alerts"], 2)

    def test_mismatched_initial_rule_support_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            analysis.build_rules(
                [
                    self.row(
                        "one",
                        {
                            "@performance/base": {
                                "eliminated_alerts": 2,
                                "remaining_alerts": 0,
                                "introduced_alerts": 0,
                            }
                        },
                        {
                            "@performance/base": {
                                "eliminated_alerts": 1,
                                "remaining_alerts": 0,
                                "introduced_alerts": 0,
                            }
                        },
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
