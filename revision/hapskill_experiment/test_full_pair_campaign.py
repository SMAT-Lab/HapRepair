from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import prepare_full_pair_campaign as prepare
import resume_condition_v14 as resume
import run_agent_ref_v4 as baseline
import run_full_pair_repetitions as scheduler


class FullPairCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = prepare.read_json(prepare.INPUTS)

    def test_task_expansion_is_complete_and_deterministic(self) -> None:
        first = prepare.task_order(self.inputs)
        second = prepare.task_order(self.inputs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 140)
        self.assertEqual([item["ordinal"] for item in first], list(range(1, 141)))
        self.assertEqual(len({item["task_id"] for item in first}), 140)
        for project in (item["name"] for item in self.inputs["projects"]):
            observed = [item for item in first if item["project"] == project]
            self.assertEqual(len(observed), 4)
            self.assertEqual({item["repeat"] for item in observed}, {1, 2})
            self.assertEqual(
                {item["condition"] for item in observed}, {"hapskill", "vanilla"}
            )

    def test_strata_are_frozen_from_initial_counts_only(self) -> None:
        strata = prepare.strata_manifest(self.inputs)
        self.assertEqual(strata["group_sizes"], {"low": 12, "middle": 11, "high": 12})
        flattened = [
            item
            for name in ("low", "middle", "high")
            for item in strata["groups"][name]
        ]
        self.assertEqual(len(flattened), 35)
        ordered = [(item["initial_alerts"], item["project"]) for item in flattened]
        self.assertEqual(ordered, sorted(ordered))

    def test_condition_protocols_have_equal_common_budget(self) -> None:
        skill = prepare.skill_protocol(self.inputs)
        reference = prepare.baseline_protocol(self.inputs)
        self.assertEqual(skill["common_contract"], reference["common_contract"])
        self.assertEqual(skill["scheduling"], reference["scheduling"])
        self.assertEqual(skill["model"], reference["model"])
        self.assertEqual(skill["common_contract"]["maximum_validation_scans"], 5)
        self.assertEqual(skill["scheduling"]["maximum_agent_turns_per_round"], 6)

    def test_baseline_contract_accepts_all_35_shared_projects(self) -> None:
        protocol = prepare.baseline_protocol(self.inputs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            observed, project = baseline.load_contract(
                path, prepare.INPUTS, "CanvasTest"
            )
        self.assertEqual(observed["dataset"]["project_count"], 35)
        self.assertEqual(project["name"], "CanvasTest")

    def test_resume_never_refreshes_active_round_allowance(self) -> None:
        self.assertEqual(
            list(
                resume.remaining_attempt_range(existing_attempts=0, maximum_attempts=6)
            ),
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            list(
                resume.remaining_attempt_range(existing_attempts=4, maximum_attempts=6)
            ),
            [5, 6],
        )
        self.assertEqual(
            list(
                resume.remaining_attempt_range(existing_attempts=6, maximum_attempts=6)
            ),
            [],
        )
        self.assertEqual(
            list(
                resume.remaining_attempt_range(existing_attempts=8, maximum_attempts=6)
            ),
            [],
        )

    def test_scheduler_preserves_terminal_failures_and_blocks_incomplete_state(
        self,
    ) -> None:
        task = {
            "run_id": "run",
            "condition": "vanilla",
            "project": "P",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(scheduler.classify_existing(root, task), "untouched")
            directory = scheduler.condition_dir(root, task)
            directory.mkdir(parents=True)
            self.assertEqual(
                scheduler.classify_existing(root, task),
                "incomplete_requires_audit",
            )
            scheduler.write_json(directory / "run_manifest.json", {"status": "failed"})
            self.assertEqual(scheduler.classify_existing(root, task), "failed")


if __name__ == "__main__":
    unittest.main()
