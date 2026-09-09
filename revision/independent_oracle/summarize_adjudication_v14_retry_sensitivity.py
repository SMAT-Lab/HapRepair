#!/usr/bin/env python3
"""Summarize the three-case retry sensitivity without replacing the main result."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import summarize_adjudication_v14_static_skill as main_summary


ORACLE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = (
    ORACLE_DIR / "adjudication_packages" / "exp_indep_3_v14_retry_sensitivity_blind_01"
)
MAIN_PACKAGE_DIR = (
    ORACLE_DIR / "adjudication_packages" / "exp_indep_final_static_blind_01"
)
CASE_COLUMNS = (
    "order",
    "blind_id",
    "category",
    "rule",
    "author_1_label",
    "author_2_label",
    "agreement",
    "adjudicated_label",
    "final_label",
    "strict_correct",
)


def build_summary() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = PACKAGE_DIR / "judge_bundle"
    manifest_path = bundle / "package_manifest.json"
    coordinator_path = PACKAGE_DIR / "coordinator_manifest.json"
    first_path = bundle / "labels_author_1.csv"
    second_path = bundle / "labels_author_2.csv"
    adjudication_path = PACKAGE_DIR / "third_author_adjudication.csv"
    manifest = main_summary.read_json(manifest_path)
    coordinator = main_summary.read_json(coordinator_path)
    entries = manifest["case_files"]
    expected_ids = [entry["blind_id"] for entry in entries]
    if manifest.get("case_count") != len(entries) or len(entries) != 3:
        raise ValueError("Retry package must contain exactly three cases")

    cases = {}
    for entry in entries:
        path = (bundle / entry["path"]).resolve()
        if bundle.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Invalid case path for {entry['blind_id']}")
        if main_summary.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Case hash mismatch for {entry['blind_id']}")
        case = main_summary.read_json(path)
        if case.get("blind_id") != entry["blind_id"]:
            raise ValueError(f"Case identity mismatch for {entry['blind_id']}")
        cases[entry["blind_id"]] = case

    first = main_summary.validate_author_rows(
        main_summary.read_csv(first_path, main_summary.AUTHOR_COLUMNS),
        expected_ids,
        "author_1",
    )
    second = main_summary.validate_author_rows(
        main_summary.read_csv(second_path, main_summary.AUTHOR_COLUMNS),
        expected_ids,
        "author_2",
    )
    disagreement_ids = [
        blind_id
        for blind_id in expected_ids
        if first[blind_id]["label"] != second[blind_id]["label"]
    ]
    adjudication = main_summary.validate_adjudication_rows(
        main_summary.read_csv(adjudication_path, main_summary.ADJUDICATION_COLUMNS),
        disagreement_ids,
        first,
        second,
    )

    rows = []
    for order, blind_id in enumerate(expected_ids, start=1):
        left = first[blind_id]["label"]
        right = second[blind_id]["label"]
        agreed = left == right
        adjudicated = "" if agreed else adjudication[blind_id]["adjudicated_label"]
        final_label = left if agreed else adjudicated
        case = cases[blind_id]
        rows.append(
            {
                "order": order,
                "blind_id": blind_id,
                "category": case["category"],
                "rule": case["rule"],
                "author_1_label": left,
                "author_2_label": right,
                "agreement": agreed,
                "adjudicated_label": adjudicated,
                "final_label": final_label,
                "strict_correct": final_label == "Correct",
            }
        )

    first_labels = [row["author_1_label"] for row in rows]
    second_labels = [row["author_2_label"] for row in rows]
    final_labels = [row["final_label"] for row in rows]
    agreements = len(rows) - len(disagreement_ids)
    by_category = {}
    for category in ("performance", "security"):
        values = [row["final_label"] for row in rows if row["category"] == category]
        by_category[category] = {
            **main_summary.strict_result(values),
            "final_label_counts": main_summary.count_values(
                values, main_summary.LABELS
            ),
        }

    parent = main_summary.read_json(MAIN_PACKAGE_DIR / "adjudication_summary.json")
    parent_results = parent["final_results"]
    retry_correct = sum(row["strict_correct"] for row in rows)
    combined_correct = parent_results["correct"] + retry_correct
    if parent_results["total"] != 63 or parent_results["correct"] != 60:
        raise ValueError("Parent intention-to-treat summary has drifted")
    combined_categories = {
        category: {
            "total": parent_results["by_frozen_category"][category]["total"],
            "correct": parent_results["by_frozen_category"][category]["correct"]
            + sum(row["strict_correct"] for row in rows if row["category"] == category),
        }
        for category in ("performance", "arkts_eslint", "security")
    }
    for values in combined_categories.values():
        values["not_correct"] = values["total"] - values["correct"]
        values["strict_correctness_rate"] = values["correct"] / values["total"]
        values["strict_correctness_percentage"] = (
            100.0 * values["correct"] / values["total"]
        )

    summary = {
        "schema_version": 1,
        "experiment": coordinator["experiment"],
        "package_id": coordinator["package_id"],
        "status": "complete",
        "classification": coordinator["classification"],
        "claim_scope": (
            "Author-requested sensitivity for retrying three parent protocol "
            "rejections; not an API retry and not a replacement for the main result."
        ),
        "annotation_validation": {
            "author_1_complete": True,
            "author_2_complete": True,
            "required_adjudications_complete": True,
            "case_hashes_verified": True,
        },
        "retry_results": {
            **main_summary.strict_result(final_labels),
            "final_label_counts": main_summary.count_values(
                final_labels, main_summary.LABELS
            ),
            "by_category": by_category,
        },
        "inter_rater_agreement": {
            "denominator": len(rows),
            "agreement_count": agreements,
            "disagreement_count": len(disagreement_ids),
            "raw_agreement": agreements / len(rows),
            "raw_agreement_percentage": 100.0 * agreements / len(rows),
            "cohen_kappa": main_summary.cohen_kappa(first_labels, second_labels),
        },
        "adjudication": {
            "required_case_count": len(disagreement_ids),
            "completed_case_count": len(adjudication),
            "no_third_author_decisions_required": not disagreement_ids,
        },
        "parent_main_result": {
            "correct": parent_results["correct"],
            "total": parent_results["total"],
            "strict_correctness_percentage": parent_results[
                "strict_correctness_percentage"
            ],
            "replace_parent_main_result": False,
        },
        "retry_completed_sensitivity": {
            "correct": combined_correct,
            "total": 63,
            "not_correct": 63 - combined_correct,
            "strict_correctness_rate": combined_correct / 63,
            "strict_correctness_percentage": 100.0 * combined_correct / 63,
            "by_frozen_category": combined_categories,
            "interpretation": (
                "Combines the 60 main human candidates with the three separately "
                "regenerated candidates; this is not the single-invocation estimate."
            ),
        },
        "evaluation_summary": {
            "outcome": "supported_as_sensitivity",
            "evaluation_summary": (
                "All three regenerated candidates were independently labeled Correct "
                "by both authors, yielding a retry-completed sensitivity of 63/63."
            ),
            "claim_update": (
                "The final-v14 controlled cases are all semantically correct when the "
                "three lexical protocol rejections receive a fresh generation."
            ),
            "baseline_relation": (
                "Supplementary to, and not a replacement for, the 60/63 parent "
                "single-invocation intention-to-treat result."
            ),
            "failure_mode": (
                "Parent failures were harness-level command-string lexical rejections, "
                "not API or semantic repair failures."
            ),
            "next_action": "Proceed to cross-experiment reconciliation.",
        },
        "source_sha256": {
            "package_manifest": main_summary.sha256_file(manifest_path),
            "coordinator_manifest": main_summary.sha256_file(coordinator_path),
            "labels_author_1": main_summary.sha256_file(first_path),
            "labels_author_2": main_summary.sha256_file(second_path),
            "third_author_adjudication": main_summary.sha256_file(adjudication_path),
            "parent_main_summary": main_summary.sha256_file(
                MAIN_PACKAGE_DIR / "adjudication_summary.json"
            ),
        },
    }
    return summary, rows


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    (PACKAGE_DIR / "sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (PACKAGE_DIR / "sensitivity_case_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    retry = summary["retry_results"]
    combined = summary["retry_completed_sensitivity"]
    agreement = summary["inter_rater_agreement"]
    (PACKAGE_DIR / "sensitivity_results.md").write_text(
        f"""# Final-v14 Retry Sensitivity

- Retry candidates: {retry["correct"]}/{retry["total"]} Correct
- Retry-package agreement: {agreement["agreement_count"]}/{agreement["denominator"]}
- Third-author adjudications: {summary["adjudication"]["required_case_count"]}
- Parent single-invocation result: 60/63 (95.2%)
- Retry-completed sensitivity: {combined["correct"]}/63 ({combined["strict_correctness_percentage"]:.1f}%)

The parent outcomes were restricted-command protocol rejections, not API
failures. This sensitivity does not replace the parent single-invocation result.
""",
        encoding="utf-8",
    )


def main() -> None:
    summary, rows = build_summary()
    write_outputs(summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
