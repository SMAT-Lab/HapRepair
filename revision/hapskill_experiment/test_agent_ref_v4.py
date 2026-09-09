from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import agent_ref_session as session
import run_agent_ref_v4 as runner
import run_formal_agent_ref_v4 as formal


FINDINGS = [
    {
        "relative_path": "a.ets",
        "line": 2,
        "column": 3,
        "rule": "@performance/a",
        "message": "one",
    },
    {
        "relative_path": "a.ets",
        "line": 7,
        "column": 1,
        "rule": "@security/b",
        "message": "two",
    },
    {
        "relative_path": "b.ts",
        "line": 4,
        "column": 2,
        "rule": "@performance/a",
        "message": "three",
    },
]


class ReferenceSessionTests(unittest.TestCase):
    def test_agent_paths_preserve_hidden_control_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            artifact = workspace / ".exp_agent/round_01/completion.json"
            self.assertEqual(
                session.workspace_container_path(artifact, workspace),
                "/workspace/.exp_agent/round_01/completion.json",
            )

    def test_completion_schema_requires_exact_evidence_array(self) -> None:
        item = session.completion_schema()["properties"]["entity_repairs"]["items"]
        self.assertIn("evidence", item["required"])
        self.assertEqual(item["properties"]["evidence"]["type"], "array")
        self.assertNotIn("repository_evidence", item["properties"])

    def test_prompt_round_trip_uses_exact_workspace_paths_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            control = workspace / ".exp_agent/round_01"
            prompt = runner.prompt_for_attempt(
                {"name": "P", "commit": "C"},
                1,
                1,
                workspace,
                control / "plan.json",
                control / "completion.json",
                control / "completion-schema.json",
                "feedback",
                False,
            )
        self.assertIn("/workspace/.exp_agent/round_01/plan.json", prompt)
        self.assertIn("/workspace/.exp_agent/round_01/completion.json", prompt)
        self.assertIn("/workspace/.exp_agent/round_01/completion-schema.json", prompt)
        self.assertNotIn("/workspace/round_01/", prompt)
        self.assertIn('exact key "evidence"', prompt)

    def test_plan_is_sanitized_and_groups_all_targets(self) -> None:
        plan = session.make_reference_plan(FINDINGS)
        self.assertEqual(plan["finding_count"], 3)
        self.assertEqual(plan["rule_count"], 2)
        self.assertEqual(
            plan["same_file_interaction_clusters"],
            [{"relative_path": "a.ets", "rules": ["@performance/a", "@security/b"]}],
        )
        encoded = str(plan).lower()
        for marker in session.FORBIDDEN_PLAN_MARKERS:
            self.assertNotIn(marker, encoded)

    def test_completion_requires_exact_rule_file_location_coverage(self) -> None:
        plan = session.make_reference_plan(FINDINGS)
        report = {
            "selected_rules": ["@performance/a", "@security/b"],
            "blocked": [],
            "unresolved_external": [],
            "entity_repairs": [
                {
                    "rule": "@performance/a",
                    "relative_path": "a.ets",
                    "locations": [{"line": 2, "column": 3}],
                    "entities": ["A"],
                    "evidence": ["call site"],
                    "transformation": "cache value",
                    "status": "repaired",
                },
                {
                    "rule": "@security/b",
                    "relative_path": "a.ets",
                    "locations": [{"line": 7, "column": 1}],
                    "entities": ["B"],
                    "evidence": ["import graph"],
                    "transformation": "move type",
                    "status": "repaired",
                },
                {
                    "rule": "@performance/a",
                    "relative_path": "b.ts",
                    "locations": [{"line": 4, "column": 2}],
                    "entities": ["C"],
                    "evidence": ["uses"],
                    "transformation": "reuse object",
                    "status": "repaired",
                },
            ],
        }
        self.assertTrue(session.audit_completion(plan, report)["complete"])
        report["entity_repairs"][0]["locations"][0]["line"] = 99
        audit = session.audit_completion(plan, report)
        self.assertFalse(audit["complete"])
        self.assertIn("location mismatch", audit["feedback"])

    def test_candidate_score_is_lexicographic(self) -> None:
        better = session.candidate_score(
            {"final_alerts": 1, "introduced_alerts": 1}, 5, 2
        )
        worse = session.candidate_score(
            {"final_alerts": 2, "introduced_alerts": 0}, 1, 1
        )
        self.assertLess(better, worse)

    def test_restricted_command_audit_allows_sdk_but_blocks_method_state(self) -> None:
        self.assertFalse(
            session.is_restricted_command(
                "rg State /home/zhihao/hdd/haprepair/baseline_data/openharmony_sdk"
            )
        )
        self.assertTrue(
            session.is_restricted_command("cat /run/skill_state/session.json")
        )
        self.assertTrue(session.is_restricted_command("find exp_hapskill -type f"))

    def test_scheduler_resume_classification_never_reuses_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = {"project": "P"}
            self.assertEqual(formal.classify_existing(root, "run", task), "untouched")
            directory = formal.condition_dir(root, "run", "P")
            directory.mkdir(parents=True)
            self.assertEqual(
                formal.classify_existing(root, "run", task), "incomplete_requires_audit"
            )
            formal.write_json(directory / "run_manifest.json", {"status": "completed"})
            self.assertEqual(formal.classify_existing(root, "run", task), "completed")


if __name__ == "__main__":
    unittest.main()
