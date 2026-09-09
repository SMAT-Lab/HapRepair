#!/usr/bin/env python3
"""Validate and summarize the final-v14 independent annotations."""

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
    ORACLE_DIR / "adjudication_packages" / "exp_indep_final_static_blind_01"
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
    "result_kind",
    "order",
    "judge_blind_id",
    "source_blind_id",
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
        raise ValueError(f"{role} IDs/order do not match the manifest")
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
            raise ValueError(f"{role} requires a rationale for {row['blind_id']}")
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
        raise ValueError("Adjudication rows must exactly cover author disagreements")
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


def count_values(values: list[str], labels: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(values)
    return {label: counts[label] for label in labels}


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
    if total == 0 or total != len(second_labels):
        raise ValueError(
            "Cohen's kappa requires two non-empty equal-length label lists"
        )
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
                "Both annotators used a single identical category for every human "
                "candidate, so the kappa denominator is zero."
            ),
        }
    return {
        "value": (observed - expected) / (1.0 - expected),
        "status": "estimated",
        "observed_agreement": observed,
        "expected_agreement": expected,
        "explanation": "Cohen's kappa over the 60 human-adjudication candidates.",
    }


def _validate_count_contract(
    manifest: dict[str, Any],
    coordinator: dict[str, Any],
    failures: dict[str, Any],
    source_map: list[dict[str, Any]],
) -> None:
    candidate_count = manifest.get("case_count")
    failure_count = failures.get("count")
    overall_count = coordinator.get("overall_case_denominator")
    if candidate_count != 60 or failure_count != 3 or overall_count != 63:
        raise ValueError("Final-v14 package must preserve the frozen 60+3=63 contract")
    if coordinator.get("human_adjudication_candidate_count") != candidate_count:
        raise ValueError("Coordinator human-candidate count does not match manifest")
    if coordinator.get("automatic_generation_failure_count") != failure_count:
        raise ValueError("Coordinator failure count does not match failure records")
    if candidate_count + failure_count != overall_count:
        raise ValueError(
            "Candidate and failure counts do not match overall denominator"
        )
    if len(source_map) != candidate_count:
        raise ValueError("Source map does not exactly cover the human candidates")

    candidate_categories = Counter(row["category"] for row in source_map)
    failure_rows = failures.get("rows", [])
    failure_categories = Counter(row["category"] for row in failure_rows)
    if len(failure_rows) != failure_count:
        raise ValueError("Failure rows do not match the frozen failure count")
    categories = ("performance", "arkts_eslint", "security")
    normalized_candidates = {
        category: candidate_categories[category] for category in categories
    }
    normalized_failures = {
        category: failure_categories[category] for category in categories
    }
    if normalized_candidates != coordinator.get("human_candidate_counts_by_category"):
        raise ValueError("Human-candidate category counts do not match coordinator")
    if normalized_failures != coordinator.get("automatic_failure_counts_by_category"):
        raise ValueError("Failure category counts do not match coordinator")
    combined = candidate_categories + failure_categories
    if combined != Counter({"performance": 42, "arkts_eslint": 1, "security": 20}):
        raise ValueError("Combined category counts do not match the frozen taxonomy")
    for row in failure_rows:
        if row.get("automatic_final_label") != "GenerationFailure":
            raise ValueError("Automatic failure has an invalid final label")
        if row.get("strict_correct") is not False:
            raise ValueError("Automatic generation failures must count as not correct")
        if row.get("rejection_reasons") != ["restricted_command_observed"]:
            raise ValueError(
                "Automatic failure is not a frozen restricted-command rejection"
            )

    source_ids = [row["source_blind_id"] for row in source_map]
    failure_ids = [row["source_blind_id"] for row in failure_rows]
    judge_ids = [row["judge_blind_id"] for row in source_map]
    if len(source_ids) != len(set(source_ids)) or len(judge_ids) != len(set(judge_ids)):
        raise ValueError("Source map contains duplicate IDs")
    if len(failure_ids) != len(set(failure_ids)):
        raise ValueError("Failure records contain duplicate IDs")
    if set(source_ids) & set(failure_ids):
        raise ValueError("Human candidates and automatic failures overlap")


def build_summary(package_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package_dir = package_dir.resolve()
    bundle = package_dir / "judge_bundle"
    manifest_path = bundle / "package_manifest.json"
    coordinator_path = package_dir / "coordinator_manifest.json"
    failures_path = package_dir / "automatic_generation_failures.json"
    source_map_path = package_dir / "source_to_judge_blind_map.json"
    verification_path = package_dir / "package_verification.json"
    first_path = bundle / "labels_author_1.csv"
    second_path = bundle / "labels_author_2.csv"
    adjudication_path = package_dir / "third_author_adjudication.csv"

    manifest = read_json(manifest_path)
    coordinator = read_json(coordinator_path)
    failures = read_json(failures_path)
    source_map = read_json(source_map_path)
    verification = read_json(verification_path)
    if not verification.get("all_passed"):
        raise ValueError("Annotation package verification is not passing")
    if manifest.get("package_id") != coordinator.get("package_id"):
        raise ValueError("Judge and coordinator package IDs do not match")
    _validate_count_contract(manifest, coordinator, failures, source_map)

    entries = manifest.get("case_files", [])
    expected_ids = [entry["blind_id"] for entry in entries]
    if len(entries) != manifest["case_count"] or len(expected_ids) != len(
        set(expected_ids)
    ):
        raise ValueError("Package manifest has an invalid case list")
    map_by_judge_id = {row["judge_blind_id"]: row for row in source_map}
    if set(map_by_judge_id) != set(expected_ids):
        raise ValueError("Source map IDs do not match judge manifest")

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
        mapped = map_by_judge_id[entry["blind_id"]]
        if (case.get("category"), case.get("rule")) != (
            mapped.get("category"),
            mapped.get("rule"),
        ):
            raise ValueError(f"Source map content mismatch for {entry['blind_id']}")
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
    adjudication = validate_adjudication_rows(
        read_csv(adjudication_path, ADJUDICATION_COLUMNS),
        disagreement_ids,
        first,
        second,
    )

    candidate_results: list[dict[str, Any]] = []
    for order, blind_id in enumerate(expected_ids, start=1):
        left = first[blind_id]["label"]
        right = second[blind_id]["label"]
        agreed = left == right
        adjudicated = "" if agreed else adjudication[blind_id]["adjudicated_label"]
        final_label = left if agreed else adjudicated
        case = cases[blind_id]
        candidate_results.append(
            {
                "result_kind": "human_candidate",
                "order": order,
                "judge_blind_id": blind_id,
                "source_blind_id": map_by_judge_id[blind_id]["source_blind_id"],
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

    failure_results = [
        {
            "result_kind": "automatic_generation_failure",
            "order": len(candidate_results) + offset,
            "judge_blind_id": "",
            "source_blind_id": row["source_blind_id"],
            "category": row["category"],
            "rule": row["rule"],
            "author_1_label": "",
            "author_2_label": "",
            "agreement": "",
            "adjudicated_label": "",
            "final_label": "GenerationFailure",
            "strict_correct": False,
        }
        for offset, row in enumerate(failures["rows"], start=1)
    ]
    case_results = candidate_results + failure_results

    first_labels = [row["author_1_label"] for row in candidate_results]
    second_labels = [row["author_2_label"] for row in candidate_results]
    final_labels = [row["final_label"] for row in case_results]
    result_labels = (*LABELS, "GenerationFailure")
    category_results: dict[str, Any] = {}
    for category in ("performance", "arkts_eslint", "security"):
        values = [
            row["final_label"] for row in case_results if row["category"] == category
        ]
        category_results[category] = {
            **strict_result(values),
            "final_label_counts": count_values(values, result_labels),
        }

    agreements = len(candidate_results) - len(disagreement_ids)
    summary = {
        "schema_version": 1,
        "experiment": coordinator["experiment"],
        "package_id": manifest["package_id"],
        "source_run_id": coordinator["source_run_id"],
        "model": coordinator["source_model"],
        "status": "complete",
        "claim_scope": (
            "Controlled semantic-correctness evaluation of the final-v14 Skill-based "
            "repair component; not end-to-end repository correctness and not a causal "
            "estimate of static-reference benefit."
        ),
        "denominators": {
            "overall_cases": len(case_results),
            "human_adjudication_candidates": len(candidate_results),
            "automatic_generation_failures": len(failure_results),
            "inter_rater_agreement_cases": len(candidate_results),
        },
        "annotation_validation": {
            "package_verification_passed": True,
            "author_1_complete": True,
            "author_2_complete": True,
            "required_adjudications_complete": True,
            "case_hashes_verified": True,
            "count_partition_verified": True,
        },
        "author_label_counts": {
            "author_1": count_values(first_labels, LABELS),
            "author_2": count_values(second_labels, LABELS),
        },
        "inter_rater_agreement": {
            "denominator": len(candidate_results),
            "agreement_count": agreements,
            "disagreement_count": len(disagreement_ids),
            "raw_agreement": agreements / len(candidate_results),
            "raw_agreement_percentage": 100.0 * agreements / len(candidate_results),
            "cohen_kappa": cohen_kappa(first_labels, second_labels),
        },
        "adjudication": {
            "required_case_count": len(disagreement_ids),
            "completed_case_count": len(adjudication),
            "disagreement_blind_ids": disagreement_ids,
            "no_third_author_decisions_required": not disagreement_ids,
        },
        "final_results": {
            **strict_result(final_labels),
            "final_label_counts": count_values(final_labels, result_labels),
            "by_frozen_category": category_results,
        },
        "source_sha256": {
            "package_manifest": sha256_file(manifest_path),
            "coordinator_manifest": sha256_file(coordinator_path),
            "package_verification": sha256_file(verification_path),
            "automatic_generation_failures": sha256_file(failures_path),
            "source_to_judge_blind_map": sha256_file(source_map_path),
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
    table = "\n".join(
        f"| {category} | {values['correct']} | {values['total']} | "
        f"{values['strict_correctness_percentage']:.1f}% |"
        for category, values in result["by_frozen_category"].items()
    )
    kappa_text = (
        f"{kappa['value']:.3f}"
        if kappa["value"] is not None
        else "not estimable (zero marginal variation)"
    )
    return f"""# Final-v14 Independent Adjudication Results

## Validated result

- Model: `{summary["model"]}`
- Overall cases: {summary["denominators"]["overall_cases"]}
- Human-adjudication candidates: {summary["denominators"]["human_adjudication_candidates"]}
- Automatic generation failures: {summary["denominators"]["automatic_generation_failures"]}
- Overall strict correctness: {result["correct"]}/{result["total"]} ({result["strict_correctness_percentage"]:.1f}%)
- Raw inter-rater agreement: {agreement["agreement_count"]}/{agreement["denominator"]} ({agreement["raw_agreement_percentage"]:.1f}%)
- Cohen's kappa: {kappa_text}
- Third-author adjudications required: {summary["adjudication"]["required_case_count"]}

Only `Correct` counts as correct. `Suspicious`, `Incorrect`, and the three
automatic `GenerationFailure` outcomes count as not correct. The automatic
failures are excluded from agreement and kappa because they were not shown to
the authors.

| Frozen category | Correct | Total | Strict correctness |
|---|---:|---:|---:|
{table}

## Claim boundary

This result evaluates the controlled final-v14 Skill-based repair component
with `{summary["model"]}`. It does not establish whole-repository correctness,
cross-model robustness, or a causal benefit from consulting the bundled static
references.
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
