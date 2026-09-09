from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_generation_v14_static_skill.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_generation_v14_static_skill", SCRIPT
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FinalV14ArtifactAuditTest(unittest.TestCase):
    def test_repair_snapshot_includes_resources_and_excludes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "entry/resources/base").mkdir(parents=True)
            (root / "entry/resources/base/color.json").write_text("{}\n")
            (root / "entry/module.json5").write_text("{}\n")
            (root / "HAPREPAIR_TASK.json").write_text("{}\n")
            (root / "completion.json").write_text("{}\n")
            snapshot = audit.repair_snapshot(root)
        self.assertEqual(
            sorted(snapshot),
            ["entry/module.json5", "entry/resources/base/color.json"],
        )

    def test_unified_diff_records_added_resource_file(self) -> None:
        patch = audit.unified_diff({}, {"entry/resources/dark/color.json": b"{}\n"})
        self.assertIn("a/entry/resources/dark/color.json", patch)
        self.assertIn("b/entry/resources/dark/color.json", patch)
        self.assertIn("+{}", patch)

    def test_max_concurrency_counts_overlapping_intervals(self) -> None:
        def stamp(second: int) -> str:
            return datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc).isoformat()

        events = [
            {"event": "request_started", "timestamp": stamp(0)},
            {"event": "request_started", "timestamp": stamp(1)},
            {"event": "request_finished", "timestamp": stamp(2)},
            {"event": "request_finished", "timestamp": stamp(3)},
        ]
        self.assertEqual(audit.max_concurrency(events), 2)

    def test_acceptance_consistency_preserves_restricted_rejection(self) -> None:
        result = {
            "status": "rejected",
            "exit_code": 0,
            "event_count": 2,
            "source_diff_present": True,
            "deleted_input_paths": [],
            "missing_required_paths": [],
            "task_unchanged": True,
            "restricted_commands": ["codelinter --format json ."],
        }
        self.assertTrue(audit.result_acceptance_consistent(result))


if __name__ == "__main__":
    unittest.main()
