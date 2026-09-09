from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_generation_v14_reference_access.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_generation_v14_reference_access", SCRIPT
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FinalV14ReferenceAccessAuditTest(unittest.TestCase):
    def write_trace(self, events: list[dict]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trace.jsonl"
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        return path

    def completed_command(self, command: str, output: str = "") -> dict:
        return {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": command,
                "aggregated_output": output,
                "exit_code": 0,
                "status": "completed",
            },
        }

    def test_direct_sed_is_content_access(self) -> None:
        events = audit.command_events(
            self.write_trace([self.completed_command("sed -n '1,80p' /skill/guide.md")])
        )
        self.assertTrue(
            audit.content_access_evidence(events, relative_path="references/guide.md")
        )

    def test_filename_listing_is_not_content_access(self) -> None:
        events = audit.command_events(
            self.write_trace(
                [
                    self.completed_command(
                        "rg --files /skill/references", "/skill/references/guide.md\n"
                    ),
                    self.completed_command(
                        "find /skill -type f", "/skill/references/guide.md\n"
                    ),
                ]
            )
        )
        self.assertFalse(
            audit.content_access_evidence(events, relative_path="references/guide.md")
        )

    def test_broad_ripgrep_with_content_prefix_is_access(self) -> None:
        events = audit.command_events(
            self.write_trace(
                [
                    self.completed_command(
                        "rg -n 'target' /skill/references",
                        "/skill/references/guide.md:4:target text\n",
                    )
                ]
            )
        )
        evidence = audit.content_access_evidence(
            events, relative_path="references/guide.md"
        )
        self.assertEqual(evidence[0]["basis"], "ripgrep_content_output")


if __name__ == "__main__":
    unittest.main()
