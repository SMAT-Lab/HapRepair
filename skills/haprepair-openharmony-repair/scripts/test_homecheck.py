#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import homecheck


class RepairGuideTests(unittest.TestCase):
    def test_frozen_static_guide_bundle(self) -> None:
        manifest = homecheck.validate_guides()
        self.assertEqual(manifest["pair_count"], 383)
        self.assertEqual(manifest["rule_count"], 63)
        self.assertEqual(len(manifest["rules"]), 63)

    def test_exact_rule_maps_to_one_static_guide(self) -> None:
        result = homecheck.guide_record("@performance/foreach-args-check")
        self.assertTrue(result["covered"])
        self.assertEqual(result["example_count"], 23)
        self.assertTrue(
            result["guide_path"].endswith("performance--foreach-args-check.md")
        )

    def test_current_init_list_alias_maps_to_historical_guide(self) -> None:
        result = homecheck.guide_record("@performance/init-list-component")
        self.assertEqual(result["canonical_rule"], "@hw-ets-eslint/init-list-component")
        self.assertTrue(result["covered"])

    def test_uncovered_rule_does_not_fabricate_a_reference(self) -> None:
        result = homecheck.guide_record("@performance/not-in-corpus")
        self.assertFalse(result["covered"])
        self.assertIsNone(result["guide_path"])

    def test_semantic_specs_cover_corpus_and_observed_rules(self) -> None:
        manifest = homecheck.validate_specs()
        rules = [rule for family in manifest["families"] for rule in family["rules"]]
        self.assertEqual(len(rules), 71)
        self.assertEqual(len(set(rules)), 71)
        result = homecheck.semantic_record(
            "@performance/hp-arkui-use-onAnimationStart-for-swiper-preload"
        )
        self.assertTrue(result["covered"])
        self.assertEqual(result["families"][0]["id"], "collections-reuse")

    def test_pan_gesture_aliases_separate_guide_and_runtime_spec(self) -> None:
        guide = homecheck.guide_record(
            "@performance/hp-arkui-reduce-pangesture-distance"
        )
        spec = homecheck.semantic_record(
            "@performance/hp-arkui-reduce-pan-gesture-distance"
        )
        self.assertTrue(guide["covered"])
        self.assertEqual(
            spec["canonical_rule"],
            "@performance/hp-arkui-reduce-pangesture-distance",
        )
        self.assertTrue(spec["covered"])


class FindingTests(unittest.TestCase):
    def test_normalization_and_target_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            source = project / "Index.ets"
            source.write_text("@Entry struct Index {}\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    [
                        {
                            "filePath": str(source),
                            "messages": [
                                {
                                    "line": 1,
                                    "column": 1,
                                    "rule": "@performance/foreach-args-check",
                                    "message": "repair me",
                                },
                                {
                                    "line": 1,
                                    "column": 2,
                                    "rule": "@parsing-error/parsing-error",
                                    "message": "parser",
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            all_findings = homecheck.normalize_findings(report, project)
        targets = homecheck.target_findings(all_findings)
        self.assertEqual(len(all_findings), 2)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["relative_path"], "Index.ets")

    def test_multiset_alert_metrics(self) -> None:
        finding = {
            "relative_path": "Index.ets",
            "rule": "@performance/example",
            "message": "same  message",
        }
        introduced = {
            "relative_path": "Other.ets",
            "rule": "@security/example",
            "message": "new",
        }
        metrics, deltas = homecheck.alert_metrics(
            [finding, dict(finding)], [finding, introduced]
        )
        self.assertEqual(metrics["eliminated_alerts"], 1)
        self.assertEqual(metrics["remaining_alerts"], 1)
        self.assertEqual(metrics["introduced_alerts"], 1)
        self.assertEqual(metrics["net_reduction"], 0)
        self.assertEqual(len(deltas["eliminated"]), 1)

    def test_interaction_clusters_related_rules_in_one_file(self) -> None:
        findings = [
            {
                "relative_path": "Index.ets",
                "line": 10,
                "column": 1,
                "rule": "@performance/hp-arkui-set-cache-count-for-lazyforeach-grid",
                "message": "cache",
            },
            {
                "relative_path": "Index.ets",
                "line": 12,
                "column": 1,
                "rule": "@performance/hp-arkui-use-reusable-component",
                "message": "reuse",
            },
        ]
        clusters = homecheck.interaction_clusters(findings)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["alert_count"], 2)


class SessionTests(unittest.TestCase):
    def synthetic_scan(
        self, state_dir: Path, attempt: int, findings: list[dict[str, object]]
    ) -> dict[str, object]:
        scan_dir = state_dir / "synthetic" / str(attempt)
        scan_dir.mkdir(parents=True)
        findings_path = scan_dir / "findings.json"
        homecheck.write_json(findings_path, findings)
        return {
            "status": "scanned",
            "exit_code": 0,
            "elapsed_seconds": 0.1,
            "command": ["synthetic-homecheck"],
            "raw_finding_count": len(findings),
            "target_finding_count": len(findings),
            "excluded_non_target_finding_count": 0,
            "findings_path": str(findings_path),
            "findings_sha256": homecheck.sha256_file(findings_path),
            "all_findings_path": str(findings_path),
            "report_path": None,
            "stdout_path": None,
            "stderr_path": None,
            "parse_error": None,
        }

    def initialized_state(self, root: Path, maximum: int = 1) -> Path:
        workspace = root / "workspace"
        workspace.mkdir()
        state_dir = root / "state"
        state_dir.mkdir()
        homecheck.write_json(
            state_dir / "state.json",
            {
                "schema_version": 1,
                "workspace": str(workspace),
                "codelinter": "/synthetic/codelinter",
                "codelinter_identity": {"verified": True},
                "config": str(homecheck.DEFAULT_CONFIG),
                "config_sha256": homecheck.sha256_file(homecheck.DEFAULT_CONFIG),
                "guide_manifest": str(homecheck.GUIDE_MANIFEST),
                "guide_manifest_sha256": homecheck.sha256_file(
                    homecheck.GUIDE_MANIFEST
                ),
                "guide_pair_count": 383,
                "guide_rule_count": 63,
                "spec_manifest": str(homecheck.SPEC_MANIFEST),
                "spec_manifest_sha256": homecheck.sha256_file(
                    homecheck.SPEC_MANIFEST
                ),
                "spec_rule_count": 71,
                "max_validation_scans": maximum,
                "validation_scans_consumed": 0,
                "initial_scan": None,
                "validation_scans": [],
                "final_scan": None,
                "current_scan": None,
            },
        )
        return state_dir

    def test_validation_budget_and_complete_plan(self) -> None:
        initial = [
            {
                "relative_path": "A.ets",
                "line": 1,
                "column": 1,
                "rule": "@performance/foreach-args-check",
                "message": "a",
            },
            {
                "relative_path": "B.ets",
                "line": 2,
                "column": 1,
                "rule": "@security/no-cycle",
                "message": "b",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_dir = self.initialized_state(root)
            scans = [
                self.synthetic_scan(state_dir, 0, initial),
                self.synthetic_scan(state_dir, 1, initial[:1]),
            ]
            with patch.object(homecheck, "execute_scan", side_effect=scans):
                homecheck.scan_session(
                    argparse.Namespace(state_dir=state_dir, kind="initial")
                )
                plan = homecheck.make_plan(
                    argparse.Namespace(state_dir=state_dir, output=None)
                )
                validation = homecheck.scan_session(
                    argparse.Namespace(state_dir=state_dir, kind="validation")
                )
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    homecheck.scan_session(
                        argparse.Namespace(state_dir=state_dir, kind="validation")
                    )
        self.assertEqual(plan["required_rule_count"], 2)
        self.assertEqual(plan["required_rule_file_groups"], 2)
        self.assertTrue(all(item["guide"]["covered"] for item in plan["rules"]))
        self.assertTrue(
            all(item["semantic_spec"]["covered"] for item in plan["rules"])
        )
        self.assertEqual(validation["metrics"]["eliminated_alerts"], 1)
        self.assertEqual(validation["validation_budget"]["consumed"], 1)


class ArchitectureTests(unittest.TestCase):
    def test_tool_has_no_dynamic_retrieval_stack(self) -> None:
        source = Path(homecheck.__file__).read_text(encoding="utf-8").lower()
        for forbidden in (
            "stella",
            "transformers",
            "torch",
            "embedding_cache",
            "retrieve-repairs",
            "vector database",
        ):
            self.assertNotIn(forbidden, source)

    def test_parser_exposes_homecheck_not_retrieval(self) -> None:
        parser = homecheck.build_parser()
        subparsers = next(
            action for action in parser._actions if action.dest == "operation"
        )
        self.assertEqual(
            set(subparsers.choices),
            {"verify", "init-session", "scan", "make-plan", "guide", "spec", "status"},
        )


if __name__ == "__main__":
    unittest.main()
