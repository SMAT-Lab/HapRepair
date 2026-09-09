#!/usr/bin/env python3

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from summarize_adjudication import (
    ADJUDICATION_COLUMNS,
    AUTHOR_COLUMNS,
    build_summary,
)


SOURCE_PACKAGE = (
    Path(__file__).resolve().parent
    / "adjudication_packages"
    / "exp_indep_63_blind_01"
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class SummarizeAdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        shutil.copytree(SOURCE_PACKAGE, self.package)

    def tearDown(self):
        self.temp.cleanup()

    @property
    def bundle(self):
        return self.package / "judge_bundle"

    def test_complete_unanimous_annotations(self):
        summary, rows = build_summary(self.package)
        self.assertEqual(len(rows), 63)
        self.assertEqual(summary["final_results"]["correct"], 63)
        self.assertEqual(summary["inter_rater_agreement"]["raw_agreement"], 1.0)
        kappa = summary["inter_rater_agreement"]["cohen_kappa"]
        self.assertIsNone(kappa["value"])
        self.assertEqual(kappa["status"], "not_estimable_no_marginal_variation")
        self.assertTrue(summary["adjudication"]["no_third_author_decisions_required"])

    def test_disagreement_requires_complete_adjudication(self):
        second_path = self.bundle / "labels_author_2.csv"
        second = read_csv(second_path)
        blind_id = second[0]["blind_id"]
        second[0]["label"] = "Incorrect"
        second[0]["rationale"] = "The candidate changes required behavior."
        write_csv(second_path, AUTHOR_COLUMNS, second)
        with self.assertRaisesRegex(ValueError, "exactly cover author disagreements"):
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
                    "rationale": "The public behavior differs from the reference.",
                }
            ],
        )
        summary, _ = build_summary(self.package)
        self.assertEqual(summary["final_results"]["correct"], 62)
        self.assertEqual(summary["final_results"]["not_correct"], 1)
        self.assertEqual(summary["adjudication"]["completed_case_count"], 1)
        self.assertIsNotNone(
            summary["inter_rater_agreement"]["cohen_kappa"]["value"]
        )

    def test_non_correct_author_label_requires_rationale(self):
        first_path = self.bundle / "labels_author_1.csv"
        first = read_csv(first_path)
        first[0]["label"] = "Suspicious"
        write_csv(first_path, AUTHOR_COLUMNS, first)
        with self.assertRaisesRegex(ValueError, "requires a rationale"):
            build_summary(self.package)


if __name__ == "__main__":
    unittest.main()
