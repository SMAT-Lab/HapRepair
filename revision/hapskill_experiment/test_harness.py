#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_formal as formal
import run_condition as condition
import run_paired as paired
import skill_broker as broker


class BrokerTests(unittest.TestCase):
    def test_retrieval_request_requires_top_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "Top-1"):
            broker.validate_requests(
                [{"rule": "@performance/example", "context": "code", "top_k": 2}]
            )

    def test_retrieval_request_requires_agent_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "context"):
            broker.validate_requests([{"rule": "@performance/example", "context": ""}])


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            condition.DEFAULT_PROTOCOL.read_text(encoding="utf-8")
        )

    def test_formal_order_alternates_within_project(self) -> None:
        tasks = paired.tasks(self.protocol, pilot=False)
        projects: dict[str, list[str]] = {}
        for task in tasks:
            projects.setdefault(task["project"], []).append(task["condition"])
        self.assertEqual(len(projects), 10)
        self.assertTrue(
            all(
                sorted(values) == ["hapskill", "vanilla"]
                for values in projects.values()
            )
        )
        first_conditions = [values[0] for values in projects.values()]
        self.assertEqual(first_conditions, ["vanilla", "hapskill"] * 5)

    def test_skill_protocol_hash_is_pinned(self) -> None:
        self.assertEqual(
            condition.sha256_file(condition.SKILL_PROTOCOL),
            self.protocol["retrieval"]["skill_protocol_sha256"],
        )

    def test_frozen_unavailable_build_gate_is_static_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = condition.prepare_frozen_validation_gate(
                {
                    "name": "p",
                    "validation_availability": {
                        "build": "not_configured",
                        "test": "not_configured",
                        "scope": "statically validated only",
                    },
                },
                root / "workspace",
                root / "gate",
            )
            self.assertFalse(setup["available"])
            self.assertEqual(setup["validation_scope"], "statically validated only")
            self.assertTrue((root / "gate" / "setup.json").is_file())

    def test_frozen_runtime_rejects_image_name_drift(self) -> None:
        with patch.object(
            condition,
            "docker_image_identity",
            return_value={"image": "wrong", "image_id": "sha256:x"},
        ):
            with self.assertRaisesRegex(RuntimeError, "image name"):
                condition.validate_frozen_runtime(
                    {
                        "runtime": {
                            "container_image": "expected",
                            "container_image_id": "sha256:x",
                        }
                    },
                    "wrong",
                )

    def test_run_skill_prepends_frozen_deveco_node(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"ok": true}', stderr=""
        )
        with patch.object(condition.subprocess, "run", return_value=completed) as mocked:
            self.assertEqual(condition.run_skill(None, "status"), {"ok": True})
        environment = mocked.call_args.kwargs["env"]
        self.assertEqual(
            environment["PATH"].split(":", 1)[0],
            str(condition.DEVECO_NODE_BIN),
        )

    def test_vanilla_prompt_does_not_expose_broker(self) -> None:
        prompt = condition.prompt_for_round(
            project={"name": "p", "commit": "c", "tree_oid": "t"},
            condition="vanilla",
            round_number=1,
            max_rounds=5,
            localization_path=Path("/tmp/workspace/.exp_hapskill/findings.json"),
            previous_feedback="none",
            build_available=False,
        )
        self.assertNotIn("/hapskill-client.py", prompt)
        self.assertIn("No HapRepair retrieval capability", prompt)
        self.assertIn("high-confidence homogeneous rule clusters", prompt)
        self.assertIn("not a limit on the number of files", prompt)

    def test_hapskill_prompt_reuses_one_retrieval_per_rule(self) -> None:
        prompt = condition.prompt_for_round(
            project={"name": "p", "commit": "c", "tree_oid": "t"},
            condition="hapskill",
            round_number=2,
            max_rounds=5,
            localization_path=Path("/tmp/workspace/.exp_hapskill/findings.json"),
            previous_feedback="No source diff; budget unchanged.",
            build_available=False,
            edit_attempt=2,
        )
        self.assertIn("edit attempt 2", prompt)
        self.assertIn("do not retrieve per alert", prompt)
        self.assertIn("Only report retrieval", prompt)

    def test_round_plan_closes_over_colocated_rules(self) -> None:
        findings = [
            {"rule": "a", "relative_path": "x.ets", "line": line}
            for line in range(1, 5)
        ] + [
            {"rule": "b", "relative_path": "x.ets", "line": line}
            for line in range(1, 4)
        ]
        plan = condition.build_round_plan(findings)
        self.assertEqual(plan["required_rules"], ["a", "b"])

    def test_round_completion_rejects_partial_file_coverage(self) -> None:
        plan = {
            "required_rules": ["a"],
            "rule_groups": {"a": {"files": {"x.ets": 2, "y.ets": 1}}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "completion.json"
            report.write_text(
                json.dumps(
                    {
                        "selected_rules": ["a"],
                        "completed_files": {"a": ["x.ets"]},
                        "blocked": [],
                    }
                ),
                encoding="utf-8",
            )
            audit = condition.audit_round_completion(plan, report)
        self.assertFalse(audit["complete"])
        self.assertIn("1 unaccounted files", audit["feedback"])

    def test_validation_wrapper_passes_non_shell_environment_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            base_environment = {
                "PATH": os.environ["PATH"],
                "npm_config_@ohos:registry": "https://repo.harmonyos.com/npm/",
            }
            build_setup = {
                "available": True,
                "environment": {
                    "NODE_OPTIONS": "--require=/tmp/arkts_ignore_diagnostics.js",
                },
                "build_command": ["/usr/bin/env"],
            }
            with patch.object(
                condition, "build_environment", return_value=base_environment
            ):
                wrapper = condition.validation_wrapper(run_dir, "project", build_setup)

            self.assertIsNotNone(wrapper)
            result = subprocess.run(
                [str(wrapper)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
            )
            self.assertEqual(
                observed["npm_config_@ohos:registry"],
                "https://repo.harmonyos.com/npm/",
            )
            self.assertEqual(
                observed["NODE_OPTIONS"],
                "--require=/tmp/arkts_ignore_diagnostics.js",
            )


class PairSummaryTests(unittest.TestCase):
    def test_pair_summary_rejects_input_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition_name, tree_hash in (("vanilla", "a"), ("hapskill", "b")):
                path = root / "run" / condition_name / "p" / "run_manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "input_tree_sha256": tree_hash,
                            "initial_target_findings_sha256": "same",
                            "model": {"requested_id": "gpt-5.6-luna"},
                            "max_validation_scans": 5,
                            "protocol_sha256": "protocol",
                            "alert_metrics": {
                                "initial_alerts": 1,
                                "final_alerts": 0,
                                "eliminated_alerts": 1,
                                "remaining_alerts": 0,
                                "introduced_alerts": 0,
                                "net_reduction": 1,
                            },
                            "validation_scan_count": 1,
                            "repair_required_round_count": 0,
                            "final_candidate_abandonment": None,
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                            "wall_clock_seconds": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
            summary = paired.paired_summary(root, "run", "p")
            self.assertEqual(summary["status"], "not_comparable")

    def test_pair_summary_reports_gross_and_net_metrics_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition_name, eliminated, introduced in (
                ("vanilla", 8, 0),
                ("hapskill", 10, 2),
            ):
                path = root / "run" / condition_name / "p" / "run_manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "input_tree_sha256": "same",
                            "initial_target_findings_sha256": "same",
                            "model": {"requested_id": "gpt-5.6-luna"},
                            "max_validation_scans": 5,
                            "protocol_sha256": "protocol",
                            "paper_facing": False,
                            "alert_metrics": {
                                "initial_alerts": 10,
                                "final_alerts": 2,
                                "eliminated_alerts": eliminated,
                                "remaining_alerts": 10 - eliminated,
                                "introduced_alerts": introduced,
                                "net_reduction": 8,
                            },
                            "validation_scan_count": 5,
                            "no_diff_retry_count": 0,
                            "repair_required_round_count": 0,
                            "final_candidate_abandonment": None,
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                            "wall_clock_seconds": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
            summary = paired.paired_summary(root, "run", "p")
            text = summary["evaluation_summary"]["outcome_summary"]
            self.assertIn("Vanilla eliminated 8", text)
            self.assertIn("HapRepair Skill eliminated 10", text)
            self.assertIn("Their net reductions were 8 and 8", text)
            self.assertIn(
                "used +0 validation scans",
                summary["evaluation_summary"]["baseline_relation"],
            )


class FormalV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent
        self.protocol_35 = self.root / "protocol-hapskill-35-v4.json"
        self.protocol_10 = self.root / "protocol-agent-ref-10-v4.json"
        self.inputs_35 = self.root / "formal_inputs_hapskill_35_v4.json"
        self.inputs_10 = self.root / "formal_inputs_agent_ref_10_v4.json"

    def test_independent_formal_contracts(self) -> None:
        protocol_35, inputs_35, tasks_35 = formal.validate_contract(
            self.protocol_35, self.inputs_35
        )
        protocol_10, inputs_10, tasks_10 = formal.validate_contract(
            self.protocol_10, self.inputs_10
        )
        self.assertEqual(protocol_35["formal_run"]["condition"], "hapskill")
        self.assertEqual(protocol_10["formal_run"]["condition"], "vanilla")
        self.assertEqual(len(tasks_35), inputs_35["project_count"])
        self.assertEqual(len(tasks_10), inputs_10["project_count"])
        self.assertEqual({item["condition"] for item in tasks_35}, {"hapskill"})
        self.assertEqual({item["condition"] for item in tasks_10}, {"vanilla"})

    def test_reference_membership_is_unchanged_and_nested_in_35(self) -> None:
        inputs_35 = json.loads(self.inputs_35.read_text(encoding="utf-8"))
        inputs_10 = json.loads(self.inputs_10.read_text(encoding="utf-8"))
        selected = json.loads(
            (self.root.parent / "coding_agent_baseline" / "selected_projects.json").read_text(
                encoding="utf-8"
            )
        )
        selected_names = [item["name"] for item in selected["projects"]]
        reference_names = [item["name"] for item in inputs_10["projects"]]
        self.assertEqual(reference_names, selected_names)
        self.assertIn("wifi_testapp", reference_names)
        by_name = {item["name"]: item for item in inputs_35["projects"]}
        self.assertTrue(
            all(item == by_name[item["name"]] for item in inputs_10["projects"])
        )

    def test_formal_04_is_not_authorized(self) -> None:
        for path in (self.protocol_35, self.protocol_10):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(
                payload["formal_run"]["run_id"],
                "exp_hapskill_10_luna_formal_04",
            )
            self.assertTrue(payload["execution"]["formal_04_forbidden"])


if __name__ == "__main__":
    unittest.main()
