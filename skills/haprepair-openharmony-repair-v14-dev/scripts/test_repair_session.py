#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import homecheck
import repair_session


class SnapshotTests(unittest.TestCase):
    def test_snapshot_restore_preserves_and_removes_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "Index.ets"
            source.write_text("function original() {}\n", encoding="utf-8")
            snapshot = repair_session.snapshot_workspace(workspace, root / "snapshot")
            source.write_text("function changed() {}\n", encoding="utf-8")
            added = workspace / "Added.ets"
            added.write_text("function added() {}\n", encoding="utf-8")
            repair_session.restore_snapshot(workspace, snapshot)
            self.assertEqual(
                source.read_text(encoding="utf-8"), "function original() {}\n"
            )
            self.assertFalse(added.exists())

    def test_source_diff_flags_large_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            repaired = root / "repaired"
            original.mkdir()
            repaired.mkdir()
            (original / "Index.ets").write_text(
                "".join(f"const value{i} = {i};\n" for i in range(30)),
                encoding="utf-8",
            )
            (repaired / "Index.ets").write_text("const value0 = 0;\n", encoding="utf-8")
            result = repair_session.source_diff(
                original, repaired, root / "changes.patch"
            )
        self.assertEqual(result["changed_source_file_count"], 1)
        self.assertEqual(len(result["large_deletion_flags"]), 1)


class CoverageTests(unittest.TestCase):
    def plan(self) -> dict[str, object]:
        guide = homecheck.guide_record("@performance/foreach-args-check")
        semantic_spec = homecheck.semantic_record("@performance/foreach-args-check")
        return {
            "rules": [
                {
                    "rule": "@performance/foreach-args-check",
                    "guide": guide,
                    "semantic_spec": semantic_spec,
                    "files": [
                        {
                            "relative_path": "A.ets",
                            "locations": [{"line": 1, "column": 2}],
                        },
                        {
                            "relative_path": "B.ets",
                            "locations": [{"line": 3, "column": 4}],
                        },
                    ],
                }
            ]
        }

    def repair(self, path: str, line: int, column: int) -> dict[str, object]:
        return {
            "rule": "@performance/foreach-args-check",
            "relative_path": path,
            "locations": [{"line": line, "column": column}],
            "entities": [f"ForEach in {path}"],
            "invariants": ["callback parameter meaning"],
            "evidence": ["all callback uses inspected"],
            "transformation": "Aligned the callback signature with its uses",
            "status": "repaired",
        }

    def test_complete_report_requires_specs_evidence_and_all_locations(self) -> None:
        semantic_spec = homecheck.semantic_record("@performance/foreach-args-check")
        paths = [semantic_spec["core_spec_path"]] + [
            family["spec_path"] for family in semantic_spec["families"]
        ]
        result = repair_session.audit_completion(
            self.plan(),
            {
                "selected_rules": ["@performance/foreach-args-check"],
                "consulted_specs": {
                    "@performance/foreach-args-check": paths
                },
                "entity_repairs": [
                    self.repair("A.ets", 1, 2),
                    self.repair("B.ets", 3, 4),
                ],
                "unresolved_external": [],
            },
        )
        self.assertTrue(result["complete"])

    def test_blocked_or_partial_work_is_rejected(self) -> None:
        semantic_spec = homecheck.semantic_record("@performance/foreach-args-check")
        paths = [semantic_spec["core_spec_path"]] + [
            family["spec_path"] for family in semantic_spec["families"]
        ]
        result = repair_session.audit_completion(
            self.plan(),
            {
                "selected_rules": ["@performance/foreach-args-check"],
                "consulted_specs": {
                    "@performance/foreach-args-check": paths
                },
                "entity_repairs": [self.repair("A.ets", 1, 2)],
                "blocked": [
                    {
                        "rule": "@performance/foreach-args-check",
                        "relative_path": "B.ets",
                        "reason": "defer until later because there are too many files",
                    }
                ],
            },
        )
        self.assertFalse(result["complete"])
        self.assertTrue(any("blocked" in problem for problem in result["problems"]))
        self.assertTrue(any("unrepaired file" in problem for problem in result["problems"]))


class GateTests(unittest.TestCase):
    def test_agent_mode_rejects_controller_mutations(self) -> None:
        with patch.dict(os.environ, {repair_session.AGENT_MODE_ENV: "1"}):
            with self.assertRaises(PermissionError):
                repair_session.enforce_agent_boundary("validate-round")

    def test_operator_mode_allows_controller_mutations(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(repair_session.AGENT_MODE_ENV, None)
            repair_session.enforce_agent_boundary("validate-round")

    def test_semantic_guard_rejects_new_indexof_workaround(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            workspace = root / "workspace"
            baseline.mkdir()
            workspace.mkdir()
            (baseline / "Index.ets").write_text(
                "const index = 0;\n", encoding="utf-8"
            )
            (workspace / "Index.ets").write_text(
                "const index = items.indexOf(item);\n", encoding="utf-8"
            )
            result = repair_session.semantic_guard(baseline, workspace, {})
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("indexOf" in item for item in result["failures"]))

    def test_semantic_guard_requires_reuse_and_state_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            workspace = root / "workspace"
            baseline.mkdir()
            workspace.mkdir()
            (baseline / "Index.ets").write_text(
                "@State value: number = 0;\n", encoding="utf-8"
            )
            (workspace / "Index.ets").write_text(
                "@Reusable\nvalue: number = 0;\n", encoding="utf-8"
            )
            result = repair_session.semantic_guard(baseline, workspace, {})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["failures"]), 2)
    def test_structural_guard_detects_removed_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            workspace = root / "workspace"
            baseline.mkdir()
            workspace.mkdir()
            (baseline / "Index.ets").write_text(
                "function keepMe(): void {}\n", encoding="utf-8"
            )
            (workspace / "Index.ets").write_text("const value = 1;\n", encoding="utf-8")
            result = repair_session.structural_guard(
                baseline, workspace, root / "guard"
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("function:keepMe", result["removed_declarations"]["Index.ets"])

    def test_public_api_guard_detects_changed_export(self) -> None:
        baseline = {"api": {"Index.ets::run": {"signature": "():void"}}}
        observed = {"api": {"Index.ets::run": {"signature": "(x:number):void"}}}
        result = repair_session.compare_public_api(baseline, observed)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["changes"]["changed"], ["Index.ets::run"])

    def test_namespace_export_guard_rejects_named_reexport_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            workspace = root / "workspace"
            baseline.mkdir()
            workspace.mkdir()
            (baseline / "Index.ets").write_text(
                "export * as Utils from './utils';\n", encoding="utf-8"
            )
            (workspace / "Index.ets").write_text(
                "export { isNull } from './utils';\n", encoding="utf-8"
            )
            result = repair_session.namespace_export_guard(baseline, workspace)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("namespace export identity" in item for item in result["failures"]))

    def test_build_failure_is_reported_without_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = repair_session.run_validation_command(
                [sys.executable, "-c", "raise SystemExit(7)"],
                root,
                root / "commands",
                label="build",
                timeout_seconds=10,
                ignore_arkts_diagnostics=False,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)

    def test_build_failure_retains_active_round_without_homecheck_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            state_dir = root / "state"
            workspace.mkdir()
            state_dir.mkdir()
            source = workspace / "Index.ets"
            source.write_text("const value = 1;\n", encoding="utf-8")
            fingerprint = repair_session.source_fingerprint(workspace)
            session = {
                "schema_version": 1,
                "status": "round_active",
                "workspace": str(workspace),
                "state_dir": str(state_dir),
                "homecheck_state_dir": str(root / "homecheck"),
                "validation": {
                    "build_command": [sys.executable, "-c", "raise SystemExit(7)"],
                    "test_command": None,
                    "timeout_seconds": 10,
                    "ignore_arkts_diagnostics": False,
                },
                "active_round": {
                    "round": 1,
                    "round_dir": str(state_dir / "rounds" / "round_01"),
                    "preflight": {
                        "status": "preflight_passed",
                        "candidate_source_fingerprint": fingerprint,
                    },
                    "validation_attempts": [],
                },
            }
            repair_session.save_session(state_dir, session)
            with (
                patch.object(homecheck, "scan_session") as scan,
                patch.object(
                    repair_session,
                    "validation_budget",
                    return_value={"consumed": 0, "maximum": 5},
                ),
            ):
                result = repair_session.validate_round(
                    argparse.Namespace(state_dir=state_dir)
                )
            observed = repair_session.load_session(state_dir)
        scan.assert_not_called()
        self.assertEqual(result["status"], "repair_required")
        self.assertFalse(result["scan_consumed"])
        self.assertIsNotNone(observed["active_round"])
        self.assertIsNone(observed["active_round"]["preflight"])


class CandidateTests(unittest.TestCase):
    def test_related_same_file_alert_exchange_is_unresolved(self) -> None:
        record = repair_session.interaction_record(
            1,
            {
                "eliminated": [
                    {
                        "relative_path": "Index.ets",
                        "rule": "@performance/hp-arkui-use-reusable-component",
                        "message": "old",
                    }
                ],
                "remaining": [],
                "introduced": [
                    {
                        "relative_path": "Index.ets",
                        "rule": "@performance/avoid-overusing-custom-component-check",
                        "message": "new",
                    }
                ],
            },
            provisional=False,
        )
        self.assertIsNotNone(record)
        self.assertTrue(record["has_unresolved_sibling_exchange"])

    def test_candidate_score_orders_alerts_then_introduced_then_scope(self) -> None:
        better = repair_session.candidate_score(
            {"final_alerts": 2, "introduced_alerts": 1}, 5, 2
        )
        worse = repair_session.candidate_score(
            {"final_alerts": 3, "introduced_alerts": 0}, 1, 1
        )
        self.assertLess(better, worse)

    def test_finalize_restores_best_valid_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "Index.ets"
            source.write_text("const best = true;\n", encoding="utf-8")
            best = repair_session.snapshot_workspace(workspace, root / "best")
            initial = repair_session.snapshot_workspace(workspace, root / "initial")
            source.write_text("const broken = true;\n", encoding="utf-8")
            state_dir = root / "state"
            state_dir.mkdir()
            session = {
                "schema_version": 1,
                "status": "repair_required",
                "workspace": str(workspace),
                "state_dir": str(state_dir),
                "homecheck_state_dir": str(root / "homecheck"),
                "initial_snapshot": initial,
                "last_valid_snapshot": best,
                "last_valid_round": 1,
                "best_valid_snapshot": best,
                "best_valid_round": 1,
                "best_valid_score": [0, 0, 1, 1],
                "best_valid_metrics": {},
                "active_round": None,
                "rounds": [],
                "rule_interactions": [],
                "final_candidate_selection": None,
                "final_scan": None,
            }
            repair_session.save_session(state_dir, session)
            with (
                patch.object(
                    homecheck,
                    "scan_session",
                    return_value={"status": "scanned", "metrics": {"final_alerts": 0}},
                ),
                patch.object(
                    repair_session,
                    "validation_budget",
                    return_value={"consumed": 1, "maximum": 5},
                ),
            ):
                result = repair_session.finalize(
                    argparse.Namespace(state_dir=state_dir)
                )
            self.assertEqual(source.read_text(encoding="utf-8"), "const best = true;\n")
        self.assertEqual(
            result["final_candidate_selection"]["status"],
            "restored_best_valid_candidate",
        )


class ArchitectureTests(unittest.TestCase):
    def test_initial_scan_reports_outer_lifecycle_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_dir = root / "state"
            state_dir.mkdir()
            repair_session.save_session(
                state_dir,
                {
                    "status": "initialized",
                    "homecheck_state_dir": str(root / "homecheck"),
                },
            )
            with (
                patch.object(
                    homecheck,
                    "scan_session",
                    return_value={
                        "operation": "scan",
                        "status": "scanned",
                        "target_finding_count": 0,
                        "metrics": {"final_alerts": 0},
                    },
                ),
                patch.object(
                    repair_session, "current_plan", return_value={"rules": []}
                ),
            ):
                result = repair_session.scan_initial(
                    argparse.Namespace(state_dir=state_dir)
                )
        self.assertEqual(result["operation"], "scan_initial")

    def test_operations_cover_full_candidate_lifecycle(self) -> None:
        parser = repair_session.build_parser()
        subparsers = next(
            action for action in parser._actions if action.dest == "operation"
        )
        self.assertEqual(
            set(subparsers.choices),
            {
                "init-session",
                "scan-initial",
                "begin-round",
                "record-completion",
                "preflight-round",
                "validate-round",
                "finalize",
                "status",
            },
        )


if __name__ == "__main__":
    unittest.main()
