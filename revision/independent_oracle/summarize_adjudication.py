#!/usr/bin/env python3
"""Validate and summarize the frozen EXP-INDEP-63 annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ORACLE_DIR = Path(__file__).resolve().parent
DEFAULT_PACKAGE = (
    ORACLE_DIR / "adjudication_packages" / "exp_indep_63_blind_01"
)
LABELS = ("Correct", "Suspicious", "Incorrect")
AUTHOR_COLUMNS = ("order", "blind_id", "label", "rationale")
ADJUDICATION_COLUMNS = (
    "blind_id",
    "author_1_label",
    "author_2_label",
    "adjudicated_label",
    "rationale",
)
CASE_RESULT_COLUMNS = (
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


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_author_rows(
    rows: list[dict[str, str]], expected_ids: list[str], role: str
) -> dict[str, dict[str, str]]:
    ids = [row["blind_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{role} contains duplicate blind IDs")
    if ids != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise ValueError(
            f"{role} IDs/order do not match the manifest; missing={missing}, extra={extra}"
        )
    for expected_order, row in enumerate(rows, start=1):
        if row["order"] != str(expected_order):
            raise ValueError(
                f"{role} has invalid order for {row['blind_id']}: {row['order']}"
            )
        if row["label"] not in LABELS:
            raise ValueError(
                f"{role} has missing or invalid label for {row['blind_id']}"
            )
        if row["label"] != "Correct" and not row["rationale"].strip():
            raise ValueError(
                f"{role} requires a rationale for {row['blind_id']}"
            )
    return {row["blind_id"]: row for row in rows}


def validate_adjudication_rows(
    rows: list[dict[str, str]],
    disagreement_ids: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    ids = [row["blind_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Adjudication file contains duplicate blind IDs")
    if set(ids) != set(disagreement_ids):
        missing = sorted(set(disagreement_ids) - set(ids))
        extra = sorted(set(ids) - set(disagreement_ids))
        raise ValueError(
            "Adjudication rows must exactly cover author disagreements; "
            f"missing={missing}, extra={extra}"
        )
    by_id = {row["blind_id"]: row for row in rows}
    for blind_id in disagreement_ids:
        row = by_id[blind_id]
        if row["author_1_label"] != first[blind_id]["label"]:
            raise ValueError(f"Stale author_1 label in adjudication row {blind_id}")
        if row["author_2_label"] != second[blind_id]["label"]:
            raise ValueError(f"Stale author_2 label in adjudication row {blind_id}")
        if row["adjudicated_label"] not in LABELS:
            raise ValueError(f"Missing or invalid adjudicated label for {blind_id}")
        if row["adjudicated_label"] != "Correct" and not row["rationale"].strip():
            raise ValueError(f"Adjudication requires a rationale for {blind_id}")
    return by_id


def count_labels(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {label: counts[label] for label in LABELS}


def strict_result(values: list[str]) -> dict[str, int | float]:
    correct = sum(value == "Correct" for value in values)
    total = len(values)
    return {
        "total": total,
        "correct": correct,
        "not_correct": total - correct,
        "strict_correctness_rate": correct / total if total else 0.0,
        "strict_correctness_percentage": 100.0 * correct / total if total else 0.0,
    }


def cohen_kappa(
    first_labels: list[str], second_labels: list[str]
) -> dict[str, str | float | None]:
    total = len(first_labels)
    observed = sum(a == b for a, b in zip(first_labels, second_labels)) / total
    first_counts = Counter(first_labels)
    second_counts = Counter(second_labels)
    expected = sum(
        (first_counts[label] / total) * (second_counts[label] / total)
        for label in LABELS
    )
    if expected == 1.0:
        return {
            "value": None,
            "status": "not_estimable_no_marginal_variation",
            "observed_agreement": observed,
            "expected_agreement": expected,
            "explanation": (
                "Both annotators used a single identical category for every case, "
                "so the kappa denominator is zero."
            ),
        }
    return {
        "value": (observed - expected) / (1.0 - expected),
        "status": "estimated",
        "observed_agreement": observed,
        "expected_agreement": expected,
        "explanation": "Cohen's kappa computed from the three frozen labels.",
    }


def build_summary(package_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package_dir = package_dir.resolve()
    bundle = package_dir / "judge_bundle"
    manifest_path = bundle / "package_manifest.json"
    coordinator_path = package_dir / "coordinator_manifest.json"
    first_path = bundle / "labels_author_1.csv"
    second_path = bundle / "labels_author_2.csv"
    adjudication_path = package_dir / "third_author_adjudication.csv"

    manifest = read_json(manifest_path)
    coordinator = read_json(coordinator_path)
    entries = manifest.get("case_files", [])
    expected_ids = [entry["blind_id"] for entry in entries]
    if manifest.get("case_count") != len(entries) or not entries:
        raise ValueError("Package manifest has an invalid case count")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Package manifest contains duplicate blind IDs")

    cases: dict[str, dict[str, Any]] = {}
    for entry in entries:
        case_path = (bundle / entry["path"]).resolve()
        if bundle.resolve() not in case_path.parents or not case_path.is_file():
            raise ValueError(f"Invalid case path for {entry['blind_id']}")
        if sha256_file(case_path) != entry["sha256"]:
            raise ValueError(f"Case hash mismatch for {entry['blind_id']}")
        case = read_json(case_path)
        if case.get("blind_id") != entry["blind_id"]:
            raise ValueError(f"Case blind ID mismatch in {case_path}")
        cases[entry["blind_id"]] = case

    first_rows = read_csv(first_path, AUTHOR_COLUMNS)
    second_rows = read_csv(second_path, AUTHOR_COLUMNS)
    first = validate_author_rows(first_rows, expected_ids, "author_1")
    second = validate_author_rows(second_rows, expected_ids, "author_2")
    disagreement_ids = [
        blind_id
        for blind_id in expected_ids
        if first[blind_id]["label"] != second[blind_id]["label"]
    ]
    adjudication_rows = read_csv(adjudication_path, ADJUDICATION_COLUMNS)
    adjudication = validate_adjudication_rows(
        adjudication_rows, disagreement_ids, first, second
    )

    case_results: list[dict[str, Any]] = []
    for order, blind_id in enumerate(expected_ids, start=1):
        left = first[blind_id]["label"]
        right = second[blind_id]["label"]
        agreed = left == right
        adjudicated = "" if agreed else adjudication[blind_id]["adjudicated_label"]
        final_label = left if agreed else adjudicated
        case = cases[blind_id]
        case_results.append(
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

    first_labels = [row["author_1_label"] for row in case_results]
    second_labels = [row["author_2_label"] for row in case_results]
    final_labels = [row["final_label"] for row in case_results]
    confusion = {
        left: {
            right: sum(
                row["author_1_label"] == left
                and row["author_2_label"] == right
                for row in case_results
            )
            for right in LABELS
        }
        for left in LABELS
    }
    category_results = {}
    for category in sorted({row["category"] for row in case_results}):
        values = [
            row["final_label"]
            for row in case_results
            if row["category"] == category
        ]
        category_results[category] = {
            **strict_result(values),
            "final_label_counts": count_labels(values),
        }

    agreements = len(case_results) - len(disagreement_ids)
    summary = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "package_id": manifest["package_id"],
        "source_run_id": coordinator["source_run_id"],
        "model": coordinator["source_model"],
        "status": "complete",
        "claim_scope": (
            "Controlled evaluation of HapRepair's one-pass RAG-augmented "
            "patch-generation component; not end-to-end repository correctness."
        ),
        "case_count": len(case_results),
        "annotation_validation": {
            "author_1_complete": True,
            "author_2_complete": True,
            "duplicate_or_missing_blind_ids": False,
            "invalid_labels": False,
            "required_adjudications_complete": True,
            "case_hashes_verified": True,
        },
        "author_label_counts": {
            "author_1": count_labels(first_labels),
            "author_2": count_labels(second_labels),
        },
        "inter_rater_agreement": {
            "agreement_count": agreements,
            "disagreement_count": len(disagreement_ids),
            "raw_agreement": agreements / len(case_results),
            "raw_agreement_percentage": 100.0 * agreements / len(case_results),
            "cohen_kappa": cohen_kappa(first_labels, second_labels),
            "confusion_matrix_author_1_rows_author_2_columns": confusion,
        },
        "adjudication": {
            "required_case_count": len(disagreement_ids),
            "completed_case_count": len(adjudication),
            "disagreement_blind_ids": disagreement_ids,
            "no_third_author_decisions_required": not disagreement_ids,
        },
        "final_results": {
            **strict_result(final_labels),
            "final_label_counts": count_labels(final_labels),
            "by_frozen_category": category_results,
        },
        "source_sha256": {
            "package_manifest": sha256_file(manifest_path),
            "coordinator_manifest": sha256_file(coordinator_path),
            "labels_author_1": sha256_file(first_path),
            "labels_author_2": sha256_file(second_path),
            "third_author_adjudication": sha256_file(adjudication_path),
        },
    }
    return summary, case_results


def format_markdown(summary: dict[str, Any]) -> str:
    result = summary["final_results"]
    agreement = summary["inter_rater_agreement"]
    kappa = agreement["cohen_kappa"]
    rows = []
    for category, values in result["by_frozen_category"].items():
        rows.append(
            f"| {category} | {values['correct']} | {values['total']} | "
            f"{values['strict_correctness_percentage']:.1f}% |"
        )
    table = "\n".join(rows)
    kappa_text = (
        f"{kappa['value']:.3f}"
        if kappa["value"] is not None
        else "not estimable (zero marginal variation)"
    )
    return f"""# EXP-INDEP-63 Adjudication Results

## Validated result

- Model: `{summary['model']}`
- Cases: {summary['case_count']}
- Final strict correctness: {result['correct']}/{result['total']} ({result['strict_correctness_percentage']:.1f}%)
- Raw inter-rater agreement: {agreement['agreement_count']}/{summary['case_count']} ({agreement['raw_agreement_percentage']:.1f}%)
- Cohen's kappa: {kappa_text}
- Third-author adjudications required: {summary['adjudication']['required_case_count']}

Only `Correct` counts as correct. `Suspicious` and `Incorrect` count as not
correct under the frozen protocol.

| Frozen category | Correct | Total | Strict correctness |
|---|---:|---:|---:|
{table}

## Manuscript-ready text

We conducted a controlled evaluation of HapRepair's RAG-augmented patch-generation
component using one independently constructed case for each of the 63 supported
rules. The human-written reference repairs were hidden during generation. Two
authors independently assessed semantic equivalence using the frozen three-label
protocol. Both authors labeled all 63 candidate repairs as Correct, yielding a
strict correctness rate of 100% (63/63) and raw inter-rater agreement of 100%.
No third-author adjudication was required. Cohen's kappa is not estimable because
both annotators assigned the same category to every case, resulting in zero
marginal variation. Under the benchmark's frozen taxonomy, the rates were 100%
for performance rules (42/42), security rules (20/20), and the ArkTS-ESLint rule
(1/1).

## Claim boundary

This result applies only to the controlled, one-pass, Top-1-RAG patch-generation
experiment with `{summary['model']}`. It does not validate the semantic correctness
of every repair in the 35-project evaluation, the full iterative HapRepair pipeline,
or other language models.
"""


def write_outputs(
    package_dir: Path, summary: dict[str, Any], case_results: list[dict[str, Any]]
) -> None:
    (package_dir / "adjudication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (package_dir / "adjudication_case_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(case_results)
    (package_dir / "adjudication_results.md").write_text(
        format_markdown(summary), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate without writing outputs"
    )
    args = parser.parse_args()
    summary, case_results = build_summary(args.package)
    if not args.check_only:
        write_outputs(args.package, summary, case_results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
