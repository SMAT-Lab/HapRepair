#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_agent_baseline as runner
import build_gate as build
import public_api_guard as public_api
import run_formal as formal
import run_haprepair_condition as haprepair
import summarize_pair as summary
import validation_gate as gate


class ValidationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.binary = self.root / "codelinter"
        self.binary.write_text("#!/bin/sh\n", encoding="utf-8")
        self.config = self.root / "config.json5"
        self.config.write_text("{}\n", encoding="utf-8")
        self.state_dir = self.root / "gate"
        gate.initialize_gate(
            self.state_dir,
            self.workspace,
            max_validation_scans=2,
            codelinter=self.binary,
            config=self.config,
            codelinter_identity={"test": True},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fake_scan(self, command, report_path, workspace):
        report_path.write_text("[]\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", ""), 0.01, [], ""

    @patch("validation_gate._execute_scan")
    def test_failed_attempt_still_consumes_budget(self, execute_scan) -> None:
        execute_scan.side_effect = RuntimeError("scanner crashed")
        first = gate.run_scan(self.state_dir, "validation")
        self.assertEqual(first["status"], "scan_failed")
        self.assertEqual(
            gate.load_state(self.state_dir)["validation_attempts_consumed"], 1
        )

        execute_scan.side_effect = None
        execute_scan.side_effect = self.fake_scan
        second = gate.run_scan(self.state_dir, "validation")
        self.assertEqual(second["status"], "scanned")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            gate.run_scan(self.state_dir, "validation")

    @patch("validation_gate._execute_scan")
    def test_initial_and_final_do_not_consume_budget(self, execute_scan) -> None:
        execute_scan.side_effect = self.fake_scan
        gate.run_scan(self.state_dir, "initial")
        gate.run_scan(self.state_dir, "final")
        state = gate.load_state(self.state_dir)
        self.assertEqual(state["validation_attempts_consumed"], 0)
        with self.assertRaisesRegex(RuntimeError, "already been attempted"):
            gate.run_scan(self.state_dir, "final")


class MetricTests(unittest.TestCase):
    def test_multiset_matching_ignores_line_shifts(self) -> None:
        initial = [
            {"relative_path": "a.ets", "line": 3, "rule": "r1", "message": "Use  x"},
            {"relative_path": "a.ets", "line": 8, "rule": "r1", "message": "Use x"},
            {"relative_path": "b.ets", "line": 4, "rule": "r2", "message": "Fix y"},
        ]
        final = [
            {"relative_path": "a.ets", "line": 30, "rule": "r1", "message": "Use x"},
            {"relative_path": "c.ets", "line": 2, "rule": "r3", "message": "New z"},
        ]
        metrics, _ = runner.compute_alert_metrics(initial, final)
        self.assertEqual(
            metrics,
            {
                "initial_alerts": 3,
                "final_alerts": 2,
                "eliminated_alerts": 2,
                "remaining_alerts": 1,
                "introduced_alerts": 1,
                "net_reduction": 1,
            },
        )

    def test_trace_parsing(self) -> None:
        raw = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "abc"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "./hvigorw build",
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 3,
                            "output_tokens": 4,
                        },
                    }
                ),
            ]
        )
        events = runner.parse_codex_events(raw)
        self.assertEqual(runner.extract_thread_id(events), "abc")
        self.assertEqual(runner.extract_usage(events)["total_tokens"], 14)
        commands = runner.extract_commands(events)
        self.assertEqual(commands[0]["category"], "build")
        self.assertEqual(runner.command_status(commands, "build"), "passed")

    def test_search_for_build_tools_is_not_a_build(self) -> None:
        command = (
            "/bin/bash -lc \"find . -type f -name 'hvigorw' -o "
            "-name 'build.gradle'; rg precompileJavaScript /rules\""
        )
        self.assertEqual(runner.classify_command(command), "other")
        self.assertEqual(runner.classify_command("./hvigorw buildHap"), "build")
        self.assertEqual(runner.classify_command("cd app && npm test"), "test")

    def test_restricted_data_access_is_flagged(self) -> None:
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "rg foo /tmp/HapRepair/revision/knowledge_base",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ]
        self.assertTrue(runner.extract_commands(events)[0]["restricted_path_reference"])

    def test_source_diff_includes_resource_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            repaired = root / "repaired"
            (original / "src").mkdir(parents=True)
            (repaired / "src").mkdir(parents=True)
            (original / "src" / "a.ets").write_text("let a = 1\n", encoding="utf-8")
            (repaired / "src" / "a.ets").write_text("let a = 2\n", encoding="utf-8")
            (repaired / "color.json").write_text("{}\n", encoding="utf-8")
            result = runner.source_diff(original, repaired, root / "changes.patch")
            self.assertEqual(result["changed_source_files"], ["src/a.ets"])
            self.assertEqual(result["changed_resource_or_config_files"], ["color.json"])

    def test_summary_validation_scope_reports_buildable_projects(self) -> None:
        manifest = {"build_status": "passed", "test_status": "not_run"}
        self.assertEqual(
            summary.validation_scope(manifest),
            "build passed; existing tests not run",
        )

    def test_summary_validation_scope_reports_static_only_projects(self) -> None:
        manifest = {"build_status": "not_available", "test_status": "not_available"}
        self.assertEqual(summary.validation_scope(manifest), "statically validated only")

    def test_haprepair_skips_localizations_without_existing_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "entry" / "src" / "main" / "ets" / "Index.ets"
            source.parent.mkdir(parents=True)
            source.write_text("let value = 1\n", encoding="utf-8")
            findings = [
                {
                    "relative_path": "entry/src/main/ets/Index.ets",
                    "line": 1,
                    "column": 1,
                    "rule": "@performance/example",
                    "message": "existing",
                },
                {
                    "relative_path": "entry/src/main/module.json5",
                    "line": 0,
                    "column": 0,
                    "rule": "@performance/dark-color-mode-check",
                    "message": "synthetic localization",
                },
            ]
            grouped, skipped = haprepair.partition_pipeline_findings(
                workspace, findings
            )
            self.assertEqual(list(grouped), [source.resolve()])
            self.assertEqual(len(skipped), 1)
            self.assertIn("does not exist", skipped[0]["reason"])

    def test_haprepair_rejects_narration_before_unfenced_source(self) -> None:
        candidate, record = haprepair.extract_validated_code(
            "I will edit the file.\nimport value from './value'\n",
            "Index.ets",
        )
        self.assertIsNone(candidate)
        self.assertEqual(record["status"], "rejected")

    def test_haprepair_accepts_single_fenced_arkts_source(self) -> None:
        candidate, record = haprepair.extract_validated_code(
            "```arkts\nexport class Value {\n  run(): void {}\n}\n```",
            "Value.ets",
        )
        self.assertEqual(
            candidate,
            "export class Value {\n  run(): void {}\n}",
        )
        self.assertEqual(record["status"], "accepted")

    @patch("run_haprepair_condition.call_model")
    @patch("run_haprepair_condition.combine_repair_results", return_value="merge")
    @patch("run_haprepair_condition.generate_fix_prompt", return_value="repair")
    @patch("run_haprepair_condition.render_rag_demo", return_value="demo")
    @patch("run_haprepair_condition.build_merged_blocks_for_file")
    def test_haprepair_rejected_merge_preserves_exact_source(
        self,
        build_blocks,
        _render_demo,
        _generate_prompt,
        _combine_results,
        call_model,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "Index.ets"
            original = "let value = 1\n"
            source.write_text(original, encoding="utf-8")
            build_blocks.return_value = (
                [
                    {
                        "surrounding_context": ["let value = 1"],
                        "defects": [
                            {
                                "rule": "example-rule",
                                "message": "example",
                                "code": "let value = 1",
                            }
                        ],
                    }
                ],
                ["let value = 1"],
            )
            call_model.side_effect = [
                {"text": "candidate", "usage": {}},
                {"text": "I will edit the file.\nlet value = 2", "usage": {}},
            ]

            class Retriever:
                def retrieve(self, rule, context):
                    return {"pair_id": "PAIR-TEST"}, 1.0

            haprepair.repair_round(
                workspace,
                [
                    {
                        "relative_path": "Index.ets",
                        "line": 1,
                        "column": 1,
                        "rule": "@performance/example-rule",
                        "message": "example",
                    }
                ],
                Retriever(),
                None,
                workspace / "round",
                model="gpt-5.6-luna",
                effort="high",
            )
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_evaluator_feedback_reports_public_api_rejection(self) -> None:
        feedback = runner.format_evaluator_feedback(
            {
                "status": "failed",
                "changes": {
                    "changed": ["library/Index.ets::Api"],
                    "removed": [],
                    "added": [],
                },
            },
            {"status": "skipped_public_api_guard_failed"},
        )
        self.assertIn("library/Index.ets::Api", feedback)
        self.assertIn("rolled back", feedback)

    def test_only_empty_auth_placeholder_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            codex_home = Path(temporary)
            placeholder = codex_home / "auth.json"
            placeholder.touch()
            runner.remove_empty_auth_mount_placeholder(codex_home)
            self.assertFalse(placeholder.exists())
            placeholder.write_text("credential", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                runner.remove_empty_auth_mount_placeholder(codex_home)

    @patch("run_agent_baseline.subprocess.run")
    def test_codex_container_uses_host_uid_and_gid(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps({"type": "thread.started", "thread_id": "t"}), ""
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            codex_home = root / "codex"
            workspace.mkdir()
            codex_home.mkdir()
            runner.run_codex_turn(
                workspace,
                codex_home,
                "prompt",
                root / "trace.jsonl",
                root / "stderr.log",
                model="gpt-5.6-luna",
                provider="xsp",
                effort="high",
                thread_id=None,
                container_image="image",
            )
        command = run.call_args.args[0]
        user_index = command.index("--user")
        self.assertEqual(command[user_index + 1], f"{os.getuid()}:{os.getgid()}")


class BuildGateTests(unittest.TestCase):
    def test_arkts_hook_matches_global_and_cached_hvigor_paths(self) -> None:
        script = """
const hook = require(process.argv[1])
const source = 'const e=await this.initDefaultArkCompileConfig();next()'
const paths = [
  '/tools/hvigor/hvigor-ohos-plugin/src/tasks/ark-compile.js',
  '/cache/node_modules/@ohos/hvigor-ohos-plugin/src/tasks/ark-compile.js'
]
const result = paths.map((path) => ({
  matched: hook.shouldInject(path),
  injected: hook.injectIgnoreWarning(source, path).includes('e.ignoreWarning=!0;')
}))
process.stdout.write(JSON.stringify(result))
"""
        result = subprocess.run(
            ["node", "-e", script, str(build.ARKTS_DIAGNOSTIC_HOOK)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [
                {"matched": True, "injected": True},
                {"matched": True, "injected": True},
            ],
        )

    def test_arkts_diagnostic_environment_preserves_node_options(self) -> None:
        result = build.arkts_diagnostic_environment(
            {"NODE_OPTIONS": "--max-old-space-size=4096", "EXISTING": "value"}
        )
        self.assertEqual(result["EXISTING"], "value")
        self.assertIn("--max-old-space-size=4096", result["NODE_OPTIONS"])
        self.assertIn("--require=", result["NODE_OPTIONS"])
        self.assertIn("arkts_ignore_diagnostics.js", result["NODE_OPTIONS"])

    def test_unavailable_preflight_does_not_execute_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            preflight = root / "preflight" / "project"
            preflight.mkdir(parents=True)
            (preflight / "result.json").write_text(
                json.dumps({"build_status": "failed", "test_status": "not_run"}),
                encoding="utf-8",
            )
            result = build.prepare_build_gate(
                "project",
                workspace,
                root / "output",
                preflight_root=root / "preflight",
            )
            self.assertFalse(result["available"])
            self.assertIsNone(result["initial_build_check"])

    def test_setup_restores_dependency_lock_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            lock = workspace / "oh-package-lock.json5"
            lock.write_text("original\n", encoding="utf-8")
            preflight = root / "preflight" / "project"
            preflight.mkdir(parents=True)
            (preflight / "result.json").write_text(
                json.dumps(
                    {
                        "build_status": "passed",
                        "test_status": "not_run",
                        "dependency_install": {"command": ["dep"]},
                        "build": {"command": ["build"]},
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(command, cwd, env, log_path, timeout):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("ok\n", encoding="utf-8")
                lock.write_text("generated\n", encoding="utf-8")
                return {
                    "command": command,
                    "cwd": str(cwd),
                    "duration_seconds": 0.1,
                    "timeout_seconds": timeout,
                    "timed_out": False,
                    "exit_code": 0,
                    "log_path": str(log_path),
                }

            with (
                patch("build_gate.dependency_command", return_value=["dep"]),
                patch("build_gate.build_command", return_value=["build"]),
                patch("build_gate.build_environment", return_value={}),
                patch("build_gate.run_logged", side_effect=fake_run),
            ):
                result = build.prepare_build_gate(
                    "project",
                    workspace,
                    root / "output",
                    preflight_root=root / "preflight",
                )
            self.assertTrue(result["initial_build_passed"])
            self.assertEqual(lock.read_text(encoding="utf-8"), "original\n")


class PublicApiGuardTests(unittest.TestCase):
    def test_exported_interface_signature_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            module = workspace / "library"
            source = module / "src" / "Api.ets"
            source.parent.mkdir(parents=True)
            (module / "oh-package.json5").write_text(
                '{ "main": "Index.ets" }\n', encoding="utf-8"
            )
            (module / "Index.ets").write_text(
                "import { Api } from './src/Api'\nexport { Api }\n",
                encoding="utf-8",
            )
            source.write_text(
                "export interface Api { run(value: string): void }\n",
                encoding="utf-8",
            )
            setup = public_api.prepare_public_api_guard(
                workspace, root / "guard"
            )
            source.write_text(
                "export interface Api { run(value: number): void }\n",
                encoding="utf-8",
            )
            result = public_api.run_public_api_guard(
                workspace, root / "guard" / "rounds", setup, label="round_01"
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["changes"]["changed"], ["library/Index.ets::Api"])


class FormalOrchestrationTests(unittest.TestCase):
    def test_frozen_task_order_parses_all_paired_conditions(self) -> None:
        protocol = json.loads(formal.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        tasks = formal.parse_tasks(protocol)
        self.assertEqual(len(tasks), 20)
        pairs = {(item["project"], item["condition"]) for item in tasks}
        self.assertEqual(len(pairs), 20)
        self.assertEqual(tasks[0]["project"], "ohos_cordova")
        self.assertEqual(tasks[0]["condition"], "coding_agent")


if __name__ == "__main__":
    unittest.main()
