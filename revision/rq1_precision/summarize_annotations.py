#!/usr/bin/env python3
"""Validate and summarize EXP-RQ1-PRECISION independent annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PACKAGE = (
    WORKSPACE_ROOT / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1"
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_author(
    rows: list[dict[str, str]], expected: list[str], role: str
) -> dict[str, dict[str, str]]:
    if [row["blind_id"] for row in rows] != expected:
        raise ValueError(f"{role} IDs or order differ from the frozen manifest")
    for order, row in enumerate(rows, start=1):
        if row["order"] != str(order):
            raise ValueError(f"{role} has invalid order for {row['blind_id']}")
        if row["label"] not in LABELS:
            raise ValueError(f"{role} has missing/invalid label for {row['blind_id']}")
        if row["label"] != "Correct" and not row["rationale"].strip():
            raise ValueError(f"{role} requires rationale for {row['blind_id']}")
    return {row["blind_id"]: row for row in rows}


def validate_adjudication(
    rows: list[dict[str, str]],
    disagreements: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    if len({row["blind_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate adjudication IDs")
    if {row["blind_id"] for row in rows} != set(disagreements):
        raise ValueError("Adjudication rows must exactly cover disagreements")
    by_id = {row["blind_id"]: row for row in rows}
    for blind_id in disagreements:
        row = by_id[blind_id]
        if row["author_1_label"] != first[blind_id]["label"]:
            raise ValueError(f"Stale author 1 label for {blind_id}")
        if row["author_2_label"] != second[blind_id]["label"]:
            raise ValueError(f"Stale author 2 label for {blind_id}")
        if row["adjudicated_label"] not in LABELS:
            raise ValueError(f"Missing adjudication for {blind_id}")
        if row["adjudicated_label"] != "Correct" and not row["rationale"].strip():
            raise ValueError(f"Adjudication requires rationale for {blind_id}")
    return by_id


def cohen_kappa(left: list[str], right: list[str]) -> dict[str, Any]:
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / total * right_counts[label] / total
        for label in LABELS
    )
    if math.isclose(expected, 1.0):
        return {
            "value": None,
            "status": "not_estimable_no_marginal_variation",
            "observed_agreement": observed,
            "expected_agreement": expected,
        }
    return {
        "value": (observed - expected) / (1 - expected),
        "status": "estimated",
        "observed_agreement": observed,
        "expected_agreement": expected,
    }


def estimate_strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot estimate an empty group")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["sampling_stratum"]].append(row)
    population = 0
    estimated_correct = 0.0
    variance_total = 0.0
    strata_results = []
    for stratum, items in sorted(by_stratum.items()):
        N = int(items[0]["stratum_population"])
        n = int(items[0]["stratum_sample"])
        if len(items) != n:
            raise ValueError(f"Incomplete sample in stratum {stratum}")
        if any(int(item["stratum_population"]) != N for item in items):
            raise ValueError(f"Inconsistent population in stratum {stratum}")
        values = [1.0 if item["final_label"] == "Correct" else 0.0 for item in items]
        mean = sum(values) / n
        if n == N:
            variance = 0.0
        elif n < 2:
            raise ValueError(f"Variance is not estimable for singleton stratum {stratum}")
        else:
            sample_variance = sum((value - mean) ** 2 for value in values) / (n - 1)
            variance = N * N * (1 - n / N) * sample_variance / n
        population += N
        estimated_correct += N * mean
        variance_total += variance
        strata_results.append(
            {
                "stratum": stratum,
                "population": N,
                "sample": n,
                "strict_correct": sum(values),
                "sample_precision": mean,
                "estimated_correct_total": N * mean,
                "estimated_total_variance": variance,
            }
        )
    estimate = estimated_correct / population
    standard_error = math.sqrt(variance_total) / population
    return {
        "population": population,
        "sample": len(rows),
        "estimated_strict_precision": estimate,
        "standard_error": standard_error,
        "ci95": [max(0.0, estimate - 1.96 * standard_error), min(1.0, estimate + 1.96 * standard_error)],
        "estimated_correct_total": estimated_correct,
        "strata": strata_results,
    }


def build_summary(package: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package = package.resolve()
    bundle = package / "judge_bundle"
    manifest = read_json(bundle / "package_manifest.json")
    protocol = read_json(package / "protocol.json")
    entries = manifest["case_files"]
    expected_ids = [entry["blind_id"] for entry in entries]
    if manifest["case_count"] != 231 or len(entries) != 231:
        raise ValueError("Frozen package must contain 231 cases")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate blind IDs in manifest")
    for entry in entries:
        path = (bundle / entry["path"]).resolve()
        if bundle not in path.parents or not path.is_file():
            raise ValueError(f"Invalid case path for {entry['blind_id']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Case hash mismatch for {entry['blind_id']}")

    first_rows = read_csv(bundle / "labels_author_1.csv", AUTHOR_COLUMNS)
    second_rows = read_csv(bundle / "labels_author_2.csv", AUTHOR_COLUMNS)
    first = validate_author(first_rows, expected_ids, "author_1")
    second = validate_author(second_rows, expected_ids, "author_2")
    disagreements = [
        blind_id
        for blind_id in expected_ids
        if first[blind_id]["label"] != second[blind_id]["label"]
    ]
    adjudication = validate_adjudication(
        read_csv(package / "third_author_adjudication.csv", ADJUDICATION_COLUMNS),
        disagreements,
        first,
        second,
    )
    sample_by_id = {
        item["blind_id"]: item for item in read_jsonl(package / "frozen_sample.jsonl")
    }
    if set(sample_by_id) != set(expected_ids):
        raise ValueError("Frozen sample IDs differ from package manifest")

    case_results = []
    for entry in entries:
        blind_id = entry["blind_id"]
        left = first[blind_id]["label"]
        right = second[blind_id]["label"]
        final = left if left == right else adjudication[blind_id]["adjudicated_label"]
        case_results.append(
            {
                **sample_by_id[blind_id],
                "order": entry["order"],
                "blind_id": blind_id,
                "author_1_label": left,
                "author_2_label": right,
                "agreement": left == right,
                "adjudicated_label": "" if left == right else final,
                "final_label": final,
                "strict_correct": final == "Correct",
            }
        )

    performance = [row for row in case_results if row["category"] == "performance"]
    security = [row for row in case_results if row["category"] == "security"]
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        by_rule[row["rule"]].append(row)
    rule_results = {
        rule: estimate_strata(rows) for rule, rows in sorted(by_rule.items())
    }
    performance_result = estimate_strata(performance)
    security_result = estimate_strata(security)
    overall_result = estimate_strata(case_results)
    performance_rule_values = [
        value["estimated_strict_precision"]
        for rule, value in rule_results.items()
        if rule.startswith("@performance/")
    ]
    security_rule_values = [
        value["estimated_strict_precision"]
        for rule, value in rule_results.items()
        if rule.startswith("@security/")
    ]
    left_labels = [row["author_1_label"] for row in case_results]
    right_labels = [row["author_2_label"] for row in case_results]
    agreements = sum(a == b for a, b in zip(left_labels, right_labels))
    confusion = {
        left: {
            right: sum(a == left and b == right for a, b in zip(left_labels, right_labels))
            for right in LABELS
        }
        for left in LABELS
    }
    summary = {
        "schema_version": 1,
        "experiment": "EXP-RQ1-PRECISION",
        "package_id": manifest["package_id"],
        "status": "complete",
        "protocol_sha256": sha256_file(package / "protocol.json"),
        "sample_size": len(case_results),
        "performance_sample_size": len(performance),
        "security_sample_size": len(security),
        "label_counts": dict(Counter(row["final_label"] for row in case_results)),
        "agreement": {
            "count": agreements,
            "disagreement_count": len(disagreements),
            "raw_agreement": agreements / len(case_results),
            "cohen_kappa": cohen_kappa(left_labels, right_labels),
            "confusion_matrix_author_1_rows_author_2_columns": confusion,
        },
        "adjudication": {
            "required": len(disagreements),
            "completed": len(adjudication),
        },
        "estimates": {
            "overall": overall_result,
            "performance": performance_result,
            "security": security_result,
            "performance_macro_precision": sum(performance_rule_values) / len(performance_rule_values),
            "security_macro_precision": sum(security_rule_values) / len(security_rule_values),
            "per_rule": rule_results,
        },
        "claim_scope": protocol["claim_scope"],
    }
    return summary, case_results


def write_case_results(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "order",
        "blind_id",
        "alert_id",
        "category",
        "rule",
        "project",
        "relative_path",
        "line",
        "sampling_stratum",
        "stratum_population",
        "stratum_sample",
        "inclusion_probability",
        "analysis_weight",
        "author_1_label",
        "author_2_label",
        "agreement",
        "adjudicated_label",
        "final_label",
        "strict_correct",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    summary, rows = build_summary(args.package)
    write_json(args.package / "results.json", summary)
    write_case_results(args.package / "case_results.csv", rows)
    print(json.dumps({"status": "complete", "results": str(args.package / "results.json")}, indent=2))


if __name__ == "__main__":
    main()
