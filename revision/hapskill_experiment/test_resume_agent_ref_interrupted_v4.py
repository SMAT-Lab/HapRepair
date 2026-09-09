from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import resume_agent_ref_interrupted_v4 as resume


guards = resume.guards


class RebootResumeTests(unittest.TestCase):
    def make_interrupted_condition(self, root: Path) -> tuple[Path, str, str]:
        run_dir = root / "condition"
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True)
        source = workspace / "entry/src/main/ets/Main.ets"
        source.parent.mkdir(parents=True)
        source.write_text(
            "export class Main {\n  value: number = 1;\n}\n", encoding="utf-8"
        )
        control = workspace / ".exp_agent/round_01"
        control.mkdir(parents=True)
        (control / "plan.json").write_text("{}\n", encoding="utf-8")
        (control / "completion-schema.json").write_text("{}\n", encoding="utf-8")

        round_dir = run_dir / "evaluator_state/rounds/round_01"
        snapshot = guards.snapshot_workspace(workspace, round_dir / "snapshot")
        source.write_text(
            "export class Main {\n  value: number = 2;\n}\n", encoding="utf-8"
        )
        completed = guards.source_diff(
            Path(snapshot["files_dir"]), workspace, round_dir / "attempt_03.patch"
        )
        source.write_text(
            "export class Main {\n  value: number = 3;\n}\n", encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as temporary:
            partial = guards.source_diff(
                Path(snapshot["files_dir"]),
                workspace,
                Path(temporary) / "partial.patch",
            )
        return run_dir, completed["patch_sha256"], partial["patch_sha256"]

    def test_archive_precedes_exact_completed_candidate_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, completed_sha, partial_sha = self.make_interrupted_condition(
                Path(temporary)
            )
            incident = {
                "interrupted_condition": {
                    "attempt_03_patch_sha256": completed_sha,
                    "partial_attempt_04_patch_sha256": partial_sha,
                }
            }
            result = resume.archive_and_restore(run_dir, incident)
            source = run_dir / "workspace/entry/src/main/ets/Main.ets"
            self.assertIn("value: number = 2", source.read_text(encoding="utf-8"))
            self.assertEqual(
                result["partial_attempt_04_patch"]["patch_sha256"], partial_sha
            )
            self.assertEqual(
                result["reconstructed_attempt_03"]["patch_sha256"], completed_sha
            )
            self.assertTrue(result["byte_exact_patch_reconstruction"])
            archived = (
                run_dir
                / "evaluator_state/interruption_recovery"
                / resume.RECOVERY_ID
                / "partial_attempt_04/files/entry/src/main/ets/Main.ets"
            )
            self.assertIn("value: number = 3", archived.read_text(encoding="utf-8"))

    def test_reconstruction_refuses_changed_partial_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, completed_sha, _ = self.make_interrupted_condition(Path(temporary))
            incident = {
                "interrupted_condition": {
                    "attempt_03_patch_sha256": completed_sha,
                    "partial_attempt_04_patch_sha256": "0" * 64,
                }
            }
            with self.assertRaisesRegex(RuntimeError, "changed after incident audit"):
                resume.archive_and_restore(run_dir, incident)

    def test_normalizes_difflib_missing_eof_newline_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            candidate = root / "candidate"
            restored = root / "restored"
            for directory in (original, candidate, restored):
                directory.mkdir()
            relative = Path("Main.ets")
            (original / relative).write_text("class Main {\n}", encoding="utf-8")
            (candidate / relative).write_text(
                "class Main {\n  value: number = 1;\n}\n", encoding="utf-8"
            )
            (restored / relative).write_text("class Main {\n}", encoding="utf-8")
            malformed = guards.source_diff(
                original, candidate, root / "malformed.patch"
            )
            self.assertIn(
                "-}+  value: number = 1;",
                (root / "malformed.patch").read_text(encoding="utf-8"),
            )
            normalized = resume.normalize_difflib_patch(
                original, root / "malformed.patch", root / "normalized.patch"
            )
            self.assertGreaterEqual(normalized["repaired_no_newline_boundaries"], 1)
            resume.patch_workspace(restored, root / "normalized.patch")
            regenerated = guards.source_diff(
                original, restored, root / "regenerated.patch"
            )
            self.assertEqual(regenerated["patch_sha256"], malformed["patch_sha256"])

    def test_historical_trace_recovery_requires_one_completed_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            traces = Path(temporary) / "condition/traces"
            traces.mkdir(parents=True)
            thread = "019fd7bc-4ec9-7da2-bed1-cb77dfd382cd"
            evaluator = traces.parent / "evaluator_state/rounds/round_01"
            evaluator.mkdir(parents=True)
            for attempt in range(1, 4):
                events = [
                    {"type": "thread.started", "thread_id": thread},
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": attempt,
                            "cached_input_tokens": 0,
                            "output_tokens": attempt,
                        },
                    },
                ]
                (traces / f"round_01_attempt_{attempt:02d}.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in events),
                    encoding="utf-8",
                )
                (traces / f"round_01_attempt_{attempt:02d}.stderr.log").write_text(
                    "", encoding="utf-8"
                )
                (evaluator / f"attempt_{attempt:02d}.patch").write_text(
                    "", encoding="utf-8"
                )
                (evaluator / f"preflight_{attempt:02d}").mkdir()
            observed, commands, usage, records = resume.historical_trace_state(traces)
            self.assertEqual(observed, thread)
            self.assertEqual(commands, [])
            self.assertEqual(usage["input_tokens"], 6)
            self.assertEqual(usage["output_tokens"], 6)
            self.assertEqual(len(records), 3)
            self.assertTrue(all(item["coverage"]["complete"] for item in records))

    def test_scheduler_completion_preserves_protocol_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scheduler = root / resume.RUN_ID / "formal_scheduler/manifest.json"
            scheduler.parent.mkdir(parents=True)
            tasks = [
                {"project": resume.PROJECT, "status": "active"},
                {"project": "guard_failure", "status": "failed"},
            ]
            resume.write_json(scheduler, {"tasks": tasks, "status": "running"})
            manifest_path = (
                root / resume.RUN_ID / "vanilla" / resume.PROJECT / "run_manifest.json"
            )
            resume.write_json(manifest_path, {"status": "completed"})
            resume.complete_scheduler(root, {"status": "completed"})
            observed = resume.read_json(scheduler)
            self.assertEqual(observed["status"], "completed_with_protocol_failures")
            self.assertEqual(
                observed["final_status_counts"], {"completed": 1, "failed": 1}
            )
            self.assertEqual(observed["tasks"][1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
