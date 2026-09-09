#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_condition_v6 as condition


class V6SchedulingTests(unittest.TestCase):
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

    def test_hapskill_prompt_uses_static_guides_without_rag(self) -> None:
        prompt = condition.prompt_for_round(
            project={"name": "Image", "commit": "c", "tree_oid": "t"},
            condition="hapskill",
            round_number=1,
            max_rounds=5,
            localization_path=Path("/tmp/workspace/.exp_agent/findings.json"),
            previous_feedback="none",
            build_available=False,
        )
        self.assertIn("$haprepair-openharmony-repair", prompt)
        self.assertIn("references/repair-guides/index.md", prompt)
        self.assertNotIn("retrieve-repairs", prompt)
        self.assertNotIn("Stella", prompt)

    def test_alias_is_treated_as_guide_covered(self) -> None:
        covered = condition.covered_target_rules(
            {"@performance/init-list-component"},
            {"@hw-ets-eslint/init-list-component"},
        )
        self.assertEqual(covered, {"@performance/init-list-component"})

    def test_guide_access_uses_command_trace(self) -> None:
        plan = {
            "guide_paths": {
                "a": "/codex-home/skills/haprepair/references/repair-guides/a.md",
                "b": "/codex-home/skills/haprepair/references/repair-guides/b.md",
            }
        }
        audit = condition.audit_guide_access(
            [{"command": "sed -n '1,80p' /x/a.md"}], plan
        )
        self.assertEqual(audit["accessed_rules"], ["a"])
        self.assertEqual(audit["missing_rules"], ["b"])

    def test_completion_rejects_deferred_work(self) -> None:
        plan = {
            "required_rules": ["a"],
            "rule_groups": {"a": {"files": {"x.ets": 1}}},
            "guide_paths": {"a": "/codex-home/skills/haprepair/a.md"},
        }
        report = {
            "selected_rules": ["a"],
            "completed_files": {},
            "blocked": [
                {
                    "rule": "a",
                    "relative_path": "x.ets",
                    "reason": "Deferred until another pass",
                }
            ],
            "consulted_guides": {"a": "/codex-home/skills/haprepair/a.md"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "completion.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            audit = condition.audit_round_completion(plan, path)
        self.assertFalse(audit["complete"])
        self.assertIn("not deferred work", audit["feedback"])

    def test_install_skill_copies_guide_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = condition.install_skill(root)
            installed = root / "skills" / "haprepair-openharmony-repair"
            self.assertTrue((installed / "SKILL.md").is_file())
            self.assertEqual(result["guide_pair_count"], 383)
            self.assertEqual(result["guide_rule_count"], 63)


if __name__ == "__main__":
    unittest.main()
