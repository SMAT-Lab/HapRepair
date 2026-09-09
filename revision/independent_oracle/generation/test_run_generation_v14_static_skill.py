from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_generation_v14_static_skill.py")
SPEC = importlib.util.spec_from_file_location("run_generation_v14_static_skill", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FinalV14RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            Path(__file__)
            .with_name("protocol_v14_static_skill.json")
            .read_text(encoding="utf-8")
        )

    def test_protocol_disables_retrieval_and_feedback(self) -> None:
        method = self.protocol["method"]
        self.assertIs(method["dynamic_retrieval"], False)
        self.assertIs(method["similarity_ranking"], False)
        self.assertIsNone(method["embedding_model"])
        self.assertIsNone(method["top_k"])
        self.assertIs(method["knowledge_base_mounted"], False)
        self.assertIs(method["homecheck_controller_available_for_feedback"], False)
        self.assertIs(self.protocol["benchmark"]["reference_project_mounted"], False)

    def test_prompt_does_not_inject_demonstration_or_reference(self) -> None:
        prompt = runner.DEFAULT_PROMPT.read_text(encoding="utf-8").lower()
        forbidden = (
            "<demonstration>",
            "demo_problem_code",
            "demo_repair_code",
            "retrieved_pair_id",
            "top-1",
        )
        self.assertTrue(all(marker not in prompt for marker in forbidden))
        self.assertIn("human reference repair", prompt)
        self.assertIn("unavailable", prompt)

    def test_blind_ids_are_stable_unique_and_new(self) -> None:
        cases = [{"case_id": f"case-{index}"} for index in range(63)]
        first = runner.make_blind_ids(cases, 20260807)
        second = runner.make_blind_ids(list(reversed(cases)), 20260807)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 63)
        self.assertTrue(all(value.startswith("V14-") for value in first.values()))

    def test_task_excludes_reference_and_feedback(self) -> None:
        case = {
            "case_id": "case-x",
            "rule": "@performance/example",
            "rule_description": "Example rule.",
            "defective_files": ["a.ets"],
        }
        task = runner.render_task(case=case, blind_id="V14-001", findings=[])
        encoded = json.dumps(task).lower()
        self.assertNotIn("repaired_project", encoded)
        self.assertNotIn("reference_code", encoded)
        self.assertIs(task["human_reference_available"], False)
        self.assertIs(task["homecheck_scan_available"], False)
        self.assertEqual(task["accepted_candidates"], 1)

    def test_restricted_command_detection(self) -> None:
        self.assertTrue(
            runner.is_restricted_command("python repair_session.py scan-initial")
        )
        self.assertTrue(runner.is_restricted_command("rg foo /x/knowledge_base"))
        self.assertTrue(runner.is_restricted_command("codelinter --format json ."))
        self.assertFalse(runner.is_restricted_command("sed -n 1,80p a.ets"))

    def test_source_diff_records_edit_and_deletion(self) -> None:
        before = {"a.ets": b"const a = 1;\n", "b.ets": b"const b = 2;\n"}
        after = {"a.ets": b"const a = 3;\n"}
        diff = runner.source_diff(before, after)
        self.assertIn("a/a.ets", diff)
        self.assertIn("b/b.ets", diff)
        self.assertIn("-const b = 2;", diff)

    def test_docker_mounts_exclude_repository_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = runner.build_docker_command(
                protocol=self.protocol,
                workspace=root / "workspace",
                codex_home=root / "codex_home",
                run_state=root / "run_state",
            )
        rendered = " ".join(command)
        self.assertIn(":/workspace", rendered)
        self.assertIn(":/codex-home", rendered)
        self.assertIn(":/run-state", rendered)
        self.assertNotIn("repaired_project", rendered)
        self.assertNotIn("knowledge_base", rendered)
        self.assertNotIn(str(runner.REPO_ROOT) + ":", rendered)

    def test_frozen_identities_match_files(self) -> None:
        method = self.protocol["method"]
        skill = runner.resolve_repo_path(method["skill_source"])
        self.assertEqual(
            runner.tree_digest(runner.tree_manifest(skill)),
            method["skill_tree_sha256"],
        )
        self.assertEqual(
            runner.sha256_file(runner.DEFAULT_PROMPT), method["prompt_template_sha256"]
        )
        self.assertEqual(runner.sha256_file(SCRIPT), method["runner_sha256"])


if __name__ == "__main__":
    unittest.main()
