#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_range_sidecar import build_sidecar


SAMPLE = {
    "blind_id": "RQ1-001",
    "project": "project",
    "relative_path": "entry/src/main/ets/Page.ets",
    "rule": "@performance/rule",
    "line": 10,
    "column": 5,
    "message": "Exact message",
}


class RangeSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.report = Path(self.temp.name) / "issuesReport.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_report(self, defects: list[dict]) -> None:
        self.report.write_text(
            json.dumps(
                [
                    {
                        "filePath": "/checkout/entry/src/main/ets/Page.ets",
                        "defects": defects,
                    }
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def defect(**overrides: object) -> dict:
        value = {
            "reportLine": 10,
            "reportColumn": 5,
            "rangeStartLine": 10,
            "rangeStartColumn": 5,
            "endLine": 14,
            "endColumn": 3,
            "description": "Exact message",
            "ruleId": "@performance/rule",
            "rangeSource": "ir-statement",
        }
        value.update(overrides)
        return value

    def build(self) -> dict:
        return build_sidecar([SAMPLE], {"project": self.report}, "abc123")

    def test_strict_identity_match_adds_annotation_only_range(self) -> None:
        self.write_report([self.defect()])

        sidecar = self.build()

        self.assertEqual(sidecar["matched_count"], 1)
        self.assertEqual(
            sidecar["ranges"]["RQ1-001"],
            {
                "start_line": 10,
                "start_column": 5,
                "end_line": 14,
                "end_column": 3,
                "source": "ir-statement",
            },
        )
        self.assertEqual(sidecar["detector_unchanged"], "CodeLinter 6.0.240")

    def test_message_mismatch_is_not_aligned(self) -> None:
        self.write_report([self.defect(description="Different message")])

        sidecar = self.build()

        self.assertEqual(sidecar["ranges"], {})
        self.assertEqual(sidecar["unmatched"][0]["reason"], "no_native_match")

    def test_native_range_may_start_before_reported_line(self) -> None:
        self.write_report(
            [
                self.defect(
                    rangeStartLine=4,
                    rangeStartColumn=1,
                    endLine=14,
                    rangeSource="comment-block",
                )
            ]
        )

        sidecar = self.build()

        self.assertEqual(sidecar["ranges"]["RQ1-001"]["start_line"], 4)
        self.assertEqual(
            sidecar["ranges"]["RQ1-001"]["source"], "comment-block"
        )

    def test_native_range_may_start_after_reported_container_line(self) -> None:
        self.write_report(
            [self.defect(rangeStartLine=11, rangeStartColumn=7, endLine=14)]
        )

        sidecar = self.build()

        self.assertEqual(sidecar["ranges"]["RQ1-001"]["start_line"], 11)

    def test_duplicate_native_identity_is_rejected_as_ambiguous(self) -> None:
        self.write_report([self.defect(), self.defect(endLine=16)])

        sidecar = self.build()

        self.assertEqual(sidecar["ranges"], {})
        self.assertEqual(sidecar["unmatched"][0]["reason"], "ambiguous_native_match")

    def test_legacy_range_is_skipped(self) -> None:
        self.write_report([self.defect(rangeSource="legacy")])

        sidecar = self.build()

        self.assertEqual(sidecar["ranges"], {})
        self.assertEqual(sidecar["skipped_legacy_count"], 1)


if __name__ == "__main__":
    unittest.main()
