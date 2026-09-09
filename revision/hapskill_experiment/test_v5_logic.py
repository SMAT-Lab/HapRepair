#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_condition_v5 as condition


class V5SchedulingTests(unittest.TestCase):
    def test_retrieval_runtime_preflight_records_dependencies(self) -> None:
        identity = condition.validate_retrieval_runtime(Path(sys.executable))
        self.assertEqual(
            identity["requested_executable"], str(Path(sys.executable).resolve())
        )
        self.assertTrue(identity["torch"])
        self.assertTrue(identity["transformers"])

    @patch("run_condition_v5.subprocess.Popen")
    def test_broker_is_forced_offline(self, popen) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            process = popen.return_value
            process.poll.return_value = 1
            with self.assertRaises(RuntimeError):
                condition.start_broker(
                    root / "run", root / "state", "cpu", Path(sys.executable)
                )
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_round_plan_includes_all_disjoint_rules(self) -> None:
        findings = [
            {"rule": "a", "relative_path": "a.ets", "line": 1},
            {"rule": "a", "relative_path": "b.ets", "line": 2},
            {"rule": "b", "relative_path": "z.ets", "line": 50},
        ]
        plan = condition.build_round_plan(findings)
        self.assertEqual(set(plan["required_rules"]), {"a", "b"})
        self.assertEqual(set(plan["rule_groups"]["a"]["files"]), {"a.ets", "b.ets"})
        self.assertEqual(set(plan["rule_groups"]["b"]["files"]), {"z.ets"})

    def test_recovery_prompt_forbids_new_alert_work(self) -> None:
        prompt = condition.prompt_for_round(
            project={"name": "Image", "commit": "c", "tree_oid": "t"},
            condition="hapskill",
            round_number=2,
            max_rounds=5,
            localization_path=Path("/tmp/workspace/.exp_agent/findings.json"),
            previous_feedback="structural gate failed",
            build_available=False,
            recovery_only=True,
            preflight_evidence="/workspace/.exp_agent/preflight.json",
        )
        self.assertIn("gate-recovery-only", prompt)
        self.assertIn("Do not work on a new HomeCheck rule", prompt)
        self.assertIn("preflight.json", prompt)

    def test_alias_is_treated_as_corpus_covered(self) -> None:
        covered = condition.covered_target_rules(
            {"@performance/init-list-component"},
            {"@hw-ets-eslint/init-list-component"},
        )
        self.assertEqual(covered, {"@performance/init-list-component"})

    def test_retrieval_audit_records_each_successful_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "retrieval.json"
            output.write_text(
                json.dumps(
                    {
                        "backend": {"backend": "stella", "paper_comparable": True},
                        "results": [
                            {
                                "requested_rule": "a",
                                "available": True,
                                "examples": [{"pair_id": "p"}],
                            },
                            {
                                "requested_rule": "b",
                                "available": False,
                                "examples": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            trace = root / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "operation": "retrieve-repairs",
                        "status": "ok",
                        "output_path": str(output),
                        "output_sha256": condition.sha256_file(output),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            audit = condition.audit_retrieval_trace(trace)
        self.assertEqual(audit["successful_rules"], ["a"])


if __name__ == "__main__":
    unittest.main()
