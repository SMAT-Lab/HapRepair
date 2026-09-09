#!/usr/bin/env python3
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from summarize_adjudication_v14_static_skill import (
    ADJUDICATION_COLUMNS,
    AUTHOR_COLUMNS,
    build_summary,
    format_markdown,
)


SOURCE_PACKAGE = (
    Path(__file__).resolve().parent
    / "adjudication_packages"
    / "exp_indep_final_static_blind_01"
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class FinalV14SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        shutil.copytree(SOURCE_PACKAGE, self.package)
        self.bundle = self.package / "judge_bundle"

    def tearDown(self):
        self.temp.cleanup()

    def fill_author(self, role, label="Correct", rationale=""):
        path = self.bundle / f"labels_{role}.csv"
        rows = read_csv(path)
        for row in rows:
            row["label"] = label
            row["rationale"] = rationale
        write_csv(path, AUTHOR_COLUMNS, rows)

    def test_60_human_candidates_and_3_failures_use_distinct_denominators(self):
        self.fill_author("author_1")
        self.fill_author("author_2")
        summary, rows = build_summary(self.package)

        self.assertEqual(len(rows), 63)
        self.assertEqual(
            summary["denominators"],
            {
                "overall_cases": 63,
                "human_adjudication_candidates": 60,
                "automatic_generation_failures": 3,
                "inter_rater_agreement_cases": 60,
            },
        )
        self.assertEqual(summary["final_results"]["correct"], 60)
        self.assertEqual(summary["final_results"]["not_correct"], 3)
        self.assertEqual(summary["inter_rater_agreement"]["agreement_count"], 60)
        self.assertEqual(summary["inter_rater_agreement"]["denominator"], 60)
        by_category = summary["final_results"]["by_frozen_category"]
        self.assertEqual(
            (
                by_category["performance"]["correct"],
                by_category["performance"]["total"],
            ),
            (40, 42),
        )
        self.assertEqual(
            (
                by_category["arkts_eslint"]["correct"],
                by_category["arkts_eslint"]["total"],
            ),
            (1, 1),
        )
        self.assertEqual(
            (by_category["security"]["correct"], by_category["security"]["total"]),
            (19, 20),
        )
        markdown = format_markdown(summary)
        self.assertNotIn("RAG-augmented", markdown)
        self.assertIn("60/63", markdown)
        self.assertIn("60/60", markdown)

    def test_disagreement_requires_exact_adjudication(self):
        self.fill_author("author_1")
        self.fill_author("author_2")
        second_path = self.bundle / "labels_author_2.csv"
        second = read_csv(second_path)
        blind_id = second[0]["blind_id"]
        second[0]["label"] = "Incorrect"
        second[0]["rationale"] = "Behavior differs"
        write_csv(second_path, AUTHOR_COLUMNS, second)

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            build_summary(self.package)

        write_csv(
            self.package / "third_author_adjudication.csv",
            ADJUDICATION_COLUMNS,
            [
                {
                    "blind_id": blind_id,
                    "author_1_label": "Correct",
                    "author_2_label": "Incorrect",
                    "adjudicated_label": "Incorrect",
                    "rationale": "Candidate changes behavior",
                }
            ],
        )
        summary, _ = build_summary(self.package)
        self.assertEqual(summary["final_results"]["correct"], 59)
        self.assertEqual(summary["inter_rater_agreement"]["agreement_count"], 59)
        self.assertEqual(summary["adjudication"]["completed_case_count"], 1)

    def test_incomplete_author_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing or invalid label"):
            build_summary(self.package)


if __name__ == "__main__":
    unittest.main()
