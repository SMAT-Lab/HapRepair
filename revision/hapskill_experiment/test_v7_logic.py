#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_condition_v7 as condition


class PathTests(unittest.TestCase):
    def test_container_paths_map_to_isolated_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = condition.container_to_host(
                "/run-state/homecheck/state.json",
                workspace=root / "workspace",
                state_dir=root / "state",
                codex_home=root / "codex-home",
            )
        self.assertEqual(observed, root / "state/homecheck/state.json")

    def test_mounts_overlay_only_codelinter_result_as_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mounts = condition.common_mounts(
                root / "workspace",
                root / "state",
                root / "codex-home",
                root / "result",
            )
        joined = "\n".join(mounts)
        self.assertIn(f"{condition.COMMAND_LINE_TOOLS}:ro", joined)
        self.assertIn(f"{condition.CONTAINER_RESULT_DIR}:rw", joined)


class GuideAuditTests(unittest.TestCase):
    def plan(self) -> dict[str, object]:
        return {
            "rules": [
                {
                    "rule": "@performance/example",
                    "guide": {
                        "covered": True,
                        "guide_path": (
                            "/codex-home/skills/haprepair-openharmony-repair/"
                            "references/repair-guides/performance--example.md"
                        ),
                    },
                },
                {
                    "rule": "@performance/uncovered",
                    "guide": {"covered": False, "guide_path": None},
                },
            ]
        }

    def test_exact_static_guide_access_passes(self) -> None:
        audit = condition.audit_guide_access(
            [
                {
                    "command": (
                        "sed -n '1,200p' /codex-home/skills/"
                        "haprepair-openharmony-repair/references/repair-guides/"
                        "performance--example.md"
                    )
                }
            ],
            self.plan(),
        )
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["required_rules"], ["@performance/example"])

    def test_claimed_but_unobserved_guide_access_fails(self) -> None:
        audit = condition.audit_guide_access([], self.plan())
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["missing_rules"], ["@performance/example"])


class PromptTests(unittest.TestCase):
    def test_prompt_requires_complete_evaluator_controlled_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "workspace/.exp_agent"
            prompt = condition.prompt_for_attempt(
                project={"name": "Example", "commit": "abc"},
                round_number=1,
                maximum_scans=5,
                attempt=1,
                findings_path=control / "findings.json",
                plan_path=control / "plan.json",
                completion_path=control / "completion.json",
                feedback="none",
                gate_recovery=False,
                build_available=False,
            )
        self.assertIn("every rule/file group", prompt)
        self.assertIn("do not invoke HomeCheck", prompt)
        self.assertIn("completion report", prompt)
        self.assertIn("Static guides are not RAG", prompt)


if __name__ == "__main__":
    unittest.main()
