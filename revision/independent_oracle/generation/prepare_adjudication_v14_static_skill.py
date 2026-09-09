#!/usr/bin/env python3
"""Build and verify the blind final-v14 human-adjudication package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
REVISION_DIR = ORACLE_DIR.parent
REPO_ROOT = REVISION_DIR.parent
RUNS_DIR = ORACLE_DIR / "generation_runs"
PACKAGES_DIR = ORACLE_DIR / "adjudication_packages"
REFERENCE_PROJECT = ORACLE_DIR / "benchmark" / "repaired_project"
CASES_MANIFEST = ORACLE_DIR / "cases_manifest.json"
ORDER_SEED = 20260807
EXPECTED_RUN_ID = "exp_indep_63_v14_static_skill_luna_01"
EXPECTED_RECOVERY_IDS = {"V14-016"}
EXPECTED_FAILURE_IDS = {"V14-030", "V14-001", "V14-029"}
EXPECTED_TOTAL_COUNTS = {"performance": 42, "arkts_eslint": 1, "security": 20}
EXPECTED_CANDIDATE_COUNTS = {"performance": 40, "arkts_eslint": 1, "security": 19}
EXPECTED_FAILURE_COUNTS = {"performance": 2, "arkts_eslint": 0, "security": 1}
METADATA_FILENAMES = {
    "HAPREPAIR_TASK.json",
    "HAPREPAIR_COMPLETION.json",
    "completion.json",
}
AUTHOR_COLUMNS = ("order", "blind_id", "label", "rationale")
ADJUDICATION_COLUMNS = (
    "blind_id",
    "author_1_label",
    "author_2_label",
    "adjudicated_label",
    "rationale",
)
FORBIDDEN_JUDGE_KEYS = {
    "case_id",
    "source_blind_id",
    "source_run_id",
    "requested_model",
    "thread_id",
    "generation_status",
    "recovery_status",
    "rejection_reasons",
    "restricted_commands",
    "commands",
    "usage",
    "reference_access",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def category(rule: str) -> str:
    if rule.startswith("@performance/"):
        return "performance"
    if rule.startswith("@security/"):
        return "security"
    return "arkts_eslint"


def repair_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in METADATA_FILENAMES
        and ".git" not in path.relative_to(root).parts
    }


def file_records_from_snapshot(snapshot: dict[str, bytes]) -> list[dict[str, str]]:
    return [
        {"path": relative, "content": content.decode("utf-8", errors="replace")}
        for relative, content in sorted(snapshot.items())
    ]


def file_records(root: Path, paths: list[str]) -> list[dict[str, str]]:
    return [
        {
            "path": relative,
            "content": (root / relative).read_text(encoding="utf-8"),
        }
        for relative in paths
    ]


def deep_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(deep_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(deep_keys(item) for item in value))
    return set()


def immutable_tree_digest(paths: list[Path], root: Path) -> str:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def resolve_evidence(decision_path: Path, record: dict[str, str]) -> Path:
    path = (decision_path.parent / record["path"]).resolve()
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Decision evidence mismatch: {path}")
    return path


def validate_decision(run_dir: Path) -> tuple[dict[str, Any], Path]:
    decision_path = run_dir / "decisions/e3_05d_capture_recovery_selected.json"
    decision = read_json(decision_path)
    if decision.get("verdict") != "selected":
        raise ValueError("E3-05D is not resolved")
    if decision.get("selected_option") != "transparent_capture_recovery":
        raise ValueError("Transparent capture recovery was not selected")
    policy = decision["counting_policy"]
    expected = {
        "overall_case_denominator": 63,
        "human_adjudication_candidate_count": 60,
        "automatic_generation_failure_count": 3,
        "human_agreement_denominator": 60,
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("E3-05D counting policy mismatch")
    for record in decision["evidence"].values():
        resolve_evidence(decision_path, record)
    return decision, decision_path


def source_population(
    run_dir: Path, decision: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = read_json(run_dir / "input_manifest.json")
    recoveries = set(decision["counting_policy"]["capture_recovery_blind_ids"])
    expected_failures = set(
        decision["counting_policy"]["automatic_generation_failure_blind_ids"]
    )
    candidates = []
    failures = []
    for item in prepared:
        result_path = run_dir / item["case_dir"] / "result.json"
        result = read_json(result_path)
        record = {"item": item, "result": result, "result_path": result_path}
        if result["status"] == "accepted" or item["blind_id"] in recoveries:
            candidates.append(record)
        else:
            failures.append(record)
    candidate_ids = {record["item"]["blind_id"] for record in candidates}
    failure_ids = {record["item"]["blind_id"] for record in failures}
    if len(candidates) != 60 or len(failures) != 3:
        raise ValueError("Decision does not partition the 63 terminal cases as 60+3")
    if not recoveries <= candidate_ids or failure_ids != expected_failures:
        raise ValueError(
            "Decision candidate/failure identities do not match run results"
        )
    if recoveries != EXPECTED_RECOVERY_IDS or failure_ids != EXPECTED_FAILURE_IDS:
        raise ValueError("Unexpected recovery or failure identity")
    return candidates, failures


def annotation_guide() -> str:
    return """# Independent Repair Adjudication Guide

Evaluate each candidate against the defective input, target rule, finding location,
and human reference. Do not consult the other annotator or any generation trace.

Allowed labels:

- `Correct`: The candidate resolves the target defect, preserves intended behavior
  and interfaces, is valid ArkTS/configuration for the shown case, and is semantically
  equivalent to the reference even when the implementation differs.
- `Suspicious`: The candidate is plausible, but the shown evidence is insufficient
  to establish semantic equivalence or contains a material unresolved ambiguity.
- `Incorrect`: The candidate leaves the target unresolved, is invalid, removes
  required behavior, changes an interface without justification, introduces an
  evident regression, or repairs a different problem.

Use `Suspicious` only for genuine evidentiary uncertainty. Only `Correct` counts as
strictly correct. `Suspicious` and `Incorrect` both count as not correct.

Each author completes only their own CSV. A third author adjudicates every label
disagreement after both author files are frozen.
"""


def build_package(*, run_id: str, package_dir: Path) -> dict[str, Any]:
    if run_id != EXPECTED_RUN_ID:
        raise ValueError(f"Unexpected final-v14 run ID: {run_id}")
    run_dir = RUNS_DIR / run_id
    if package_dir.exists():
        raise FileExistsError(f"Package directory already exists: {package_dir}")
    decision, decision_path = validate_decision(run_dir)
    artifact_audit = read_json(run_dir / "artifact_audit_v1/audit.json")
    if not artifact_audit.get("artifact_audit_passed"):
        raise ValueError("Source run artifact audit has not passed")
    candidates, failures = source_population(run_dir, decision)
    cases = read_json(CASES_MANIFEST)["cases"]
    cases_by_id = {case["case_id"]: case for case in cases}

    random.Random(ORDER_SEED).shuffle(candidates)
    judge_dir = package_dir / "judge_bundle"
    cases_dir = judge_dir / "cases"
    cases_dir.mkdir(parents=True)
    blind_map = []
    case_hashes = []
    candidate_counts: Counter[str] = Counter()
    for order, record in enumerate(candidates, start=1):
        item = record["item"]
        case = cases_by_id[item["case_id"]]
        source_blind_id = item["blind_id"]
        judge_blind_id = f"JDG-{order:03d}"
        case_dir = run_dir / item["case_dir"]
        task = read_json(case_dir / "workspace/HAPREPAIR_TASK.json")
        candidate_snapshot = repair_snapshot(case_dir / "workspace")
        defective_paths = [entry["path"] for entry in item["input_files"]]
        judge_case = {
            "schema_version": 1,
            "blind_id": judge_blind_id,
            "category": category(item["rule"]),
            "rule": item["rule"],
            "rule_description": case["rule_description"],
            "target_findings": task["target_findings"],
            "defective_files": file_records(case_dir / "baseline", defective_paths),
            "human_reference_files": file_records(
                REFERENCE_PROJECT, case["repaired_files"]
            ),
            "candidate_repair_files": file_records_from_snapshot(candidate_snapshot),
        }
        path = cases_dir / f"{judge_blind_id}.json"
        write_json(path, judge_case)
        case_hashes.append(
            {
                "blind_id": judge_blind_id,
                "path": path.relative_to(judge_dir).as_posix(),
                "sha256": sha256_file(path),
            }
        )
        blind_map.append(
            {
                "order": order,
                "judge_blind_id": judge_blind_id,
                "source_blind_id": source_blind_id,
                "case_id": item["case_id"],
                "category": judge_case["category"],
                "rule": item["rule"],
                "source_result_sha256": sha256_file(record["result_path"]),
                "candidate_source": (
                    "capture_recovery_overlay"
                    if source_blind_id in EXPECTED_RECOVERY_IDS
                    else "frozen_runner_accepted_workspace"
                ),
                "candidate_file_hashes": [
                    {
                        "path": relative,
                        "sha256": sha256_bytes(content),
                        "size_bytes": len(content),
                    }
                    for relative, content in sorted(candidate_snapshot.items())
                ],
            }
        )
        candidate_counts[judge_case["category"]] += 1

    label_rows = [
        {"order": index, "blind_id": entry["blind_id"], "label": "", "rationale": ""}
        for index, entry in enumerate(case_hashes, start=1)
    ]
    write_csv(judge_dir / "labels_author_1.csv", AUTHOR_COLUMNS, label_rows)
    write_csv(judge_dir / "labels_author_2.csv", AUTHOR_COLUMNS, label_rows)
    write_csv(package_dir / "third_author_adjudication.csv", ADJUDICATION_COLUMNS, [])
    (judge_dir / "ANNOTATION_GUIDE.md").write_text(annotation_guide(), encoding="utf-8")
    package_manifest = {
        "schema_version": 1,
        "experiment": "EXP-INDEP-63-FINAL-STATIC",
        "package_id": package_dir.name,
        "case_count": len(case_hashes),
        "category_counts": dict(sorted(candidate_counts.items())),
        "order_seed": ORDER_SEED,
        "labels": ["Correct", "Suspicious", "Incorrect"],
        "strict_correct_labels": ["Correct"],
        "case_files": case_hashes,
    }
    write_json(judge_dir / "package_manifest.json", package_manifest)

    failure_rows = []
    failure_counts: Counter[str] = Counter()
    for record in failures:
        item = record["item"]
        result = record["result"]
        item_category = category(item["rule"])
        failure_counts[item_category] += 1
        failure_rows.append(
            {
                "source_blind_id": item["blind_id"],
                "case_id": item["case_id"],
                "category": item_category,
                "rule": item["rule"],
                "source_result_sha256": sha256_file(record["result_path"]),
                "rejection_reasons": result["rejection_reasons"],
                "restricted_commands": result["restricted_commands"],
                "automatic_final_label": "GenerationFailure",
                "strict_correct": False,
            }
        )
    automatic_failures = {
        "schema_version": 1,
        "source_run_id": run_id,
        "count": len(failure_rows),
        "category_counts": dict(sorted(failure_counts.items())),
        "rows": failure_rows,
    }
    write_json(package_dir / "automatic_generation_failures.json", automatic_failures)
    write_json(package_dir / "source_to_judge_blind_map.json", blind_map)

    immutable_paths = [
        judge_dir / "ANNOTATION_GUIDE.md",
        judge_dir / "package_manifest.json",
        *sorted(cases_dir.glob("*.json")),
    ]
    protocol = read_json(run_dir / "protocol_snapshot.json")
    coordinator = {
        "schema_version": 1,
        "experiment": "EXP-INDEP-63-FINAL-V14",
        "package_id": package_dir.name,
        "source_run_id": run_id,
        "source_model": protocol["model"]["requested_id"],
        "source_protocol_sha256": sha256_file(run_dir / "protocol_snapshot.json"),
        "source_metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "source_verification_sha256": sha256_file(run_dir / "verification.json"),
        "source_artifact_audit_sha256": sha256_file(
            run_dir / "artifact_audit_v1/audit.json"
        ),
        "counting_decision": decision_path.relative_to(run_dir).as_posix(),
        "counting_decision_sha256": sha256_file(decision_path),
        "overall_case_denominator": 63,
        "human_adjudication_candidate_count": 60,
        "automatic_generation_failure_count": 3,
        "human_candidate_counts_by_category": EXPECTED_CANDIDATE_COUNTS,
        "automatic_failure_counts_by_category": EXPECTED_FAILURE_COUNTS,
        "judge_bundle": "judge_bundle/",
        "judge_bundle_immutable_tree_sha256": immutable_tree_digest(
            immutable_paths, package_dir
        ),
        "judge_bundle_contains_source_or_generation_identity": False,
        "judge_bundle_excluded_fields": sorted(FORBIDDEN_JUDGE_KEYS),
        "reference_access_stage": "post_generation_only",
        "order_seed": ORDER_SEED,
        "source_to_judge_blind_map": "source_to_judge_blind_map.json",
        "automatic_generation_failures": "automatic_generation_failures.json",
        "next_action": "Two authors independently label all 60 candidate repairs; a third author adjudicates disagreements only.",
    }
    write_json(package_dir / "coordinator_manifest.json", coordinator)
    verification = verify_package(package_dir)
    write_json(package_dir / "package_verification.json", verification)
    if not verification["all_passed"]:
        raise ValueError("Generated package failed verification")
    return verification


def verify_package(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    judge_dir = package_dir / "judge_bundle"
    manifest = read_json(judge_dir / "package_manifest.json")
    coordinator = read_json(package_dir / "coordinator_manifest.json")
    blind_map = read_json(package_dir / "source_to_judge_blind_map.json")
    failures = read_json(package_dir / "automatic_generation_failures.json")
    run_dir = RUNS_DIR / coordinator["source_run_id"]
    decision, decision_path = validate_decision(run_dir)
    prepared = read_json(run_dir / "input_manifest.json")
    by_source_blind = {item["blind_id"]: item for item in prepared}
    entries = manifest["case_files"]
    judge_ids = [entry["blind_id"] for entry in entries]
    source_candidate_ids = {entry["source_blind_id"] for entry in blind_map}
    failure_ids = {entry["source_blind_id"] for entry in failures["rows"]}
    case_hashes_match = True
    case_identity_matches = True
    forbidden_keys_absent = True
    source_contents_match = True
    for entry in entries:
        path = (judge_dir / entry["path"]).resolve()
        if judge_dir.resolve() not in path.parents or not path.is_file():
            case_hashes_match = False
            continue
        if sha256_file(path) != entry["sha256"]:
            case_hashes_match = False
        case = read_json(path)
        if case.get("blind_id") != entry["blind_id"]:
            case_identity_matches = False
        if deep_keys(case) & FORBIDDEN_JUDGE_KEYS:
            forbidden_keys_absent = False
        mapping = next(
            (row for row in blind_map if row["judge_blind_id"] == entry["blind_id"]),
            None,
        )
        if mapping is None:
            source_contents_match = False
            continue
        item = by_source_blind[mapping["source_blind_id"]]
        case_dir = run_dir / item["case_dir"]
        expected_candidate = file_records_from_snapshot(
            repair_snapshot(case_dir / "workspace")
        )
        expected_defective = file_records(
            case_dir / "baseline", [record["path"] for record in item["input_files"]]
        )
        case_meta = next(
            record
            for record in read_json(CASES_MANIFEST)["cases"]
            if record["case_id"] == item["case_id"]
        )
        expected_reference = file_records(
            REFERENCE_PROJECT, case_meta["repaired_files"]
        )
        source_contents_match &= (
            case["candidate_repair_files"] == expected_candidate
            and case["defective_files"] == expected_defective
            and case["human_reference_files"] == expected_reference
        )

    author_1 = read_csv(judge_dir / "labels_author_1.csv", AUTHOR_COLUMNS)
    author_2 = read_csv(judge_dir / "labels_author_2.csv", AUTHOR_COLUMNS)
    adjudication = read_csv(
        package_dir / "third_author_adjudication.csv", ADJUDICATION_COLUMNS
    )
    category_counts = Counter(row["category"] for row in blind_map)
    failure_counts = Counter(row["category"] for row in failures["rows"])
    judge_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            judge_dir / "ANNOTATION_GUIDE.md",
            judge_dir / "package_manifest.json",
            judge_dir / "labels_author_1.csv",
            judge_dir / "labels_author_2.csv",
            *sorted((judge_dir / "cases").glob("*.json")),
        ]
    )
    forbidden_literals = [
        coordinator["source_run_id"],
        coordinator["source_model"],
        "V14-",
        "capture_recovery",
        "GenerationFailure",
        "restricted_command",
    ]
    immutable_paths = [
        judge_dir / "ANNOTATION_GUIDE.md",
        judge_dir / "package_manifest.json",
        *sorted((judge_dir / "cases").glob("*.json")),
    ]
    checks = {
        "decision_is_selected_and_hash_bound": decision["verdict"] == "selected"
        and coordinator["counting_decision_sha256"] == sha256_file(decision_path),
        "judge_case_count_60": manifest["case_count"] == len(entries) == 60,
        "judge_ids_unique_and_remapped": len(judge_ids) == len(set(judge_ids)) == 60
        and all(re.fullmatch(r"JDG-\d{3}", blind_id) for blind_id in judge_ids),
        "case_hashes_match": case_hashes_match,
        "case_identity_matches_manifest": case_identity_matches,
        "candidate_defective_reference_contents_match_sources": source_contents_match,
        "judge_case_forbidden_keys_absent": forbidden_keys_absent,
        "judge_bundle_forbidden_literals_absent": all(
            literal not in judge_text for literal in forbidden_literals
        ),
        "source_map_exactly_covers_60_candidates": len(blind_map) == 60
        and len(source_candidate_ids) == 60,
        "automatic_failure_count_3": failures["count"] == len(failures["rows"]) == 3,
        "candidate_failure_partition_exactly_63": len(
            source_candidate_ids | failure_ids
        )
        == 63
        and not (source_candidate_ids & failure_ids)
        and source_candidate_ids | failure_ids == set(by_source_blind),
        "recovery_and_failure_ids_match_decision": EXPECTED_RECOVERY_IDS
        <= source_candidate_ids
        and failure_ids == EXPECTED_FAILURE_IDS,
        "candidate_taxonomy_40_1_19": dict(category_counts)
        == EXPECTED_CANDIDATE_COUNTS,
        "failure_taxonomy_2_0_1": {
            name: failure_counts[name] for name in EXPECTED_FAILURE_COUNTS
        }
        == EXPECTED_FAILURE_COUNTS,
        "combined_taxonomy_42_1_20": {
            name: category_counts[name] + failure_counts[name]
            for name in EXPECTED_TOTAL_COUNTS
        }
        == EXPECTED_TOTAL_COUNTS,
        "automatic_failures_are_frozen_restricted_rejections": all(
            row["rejection_reasons"] == ["restricted_command_observed"]
            and row["restricted_commands"]
            and row["automatic_final_label"] == "GenerationFailure"
            and row["strict_correct"] is False
            for row in failures["rows"]
        ),
        "author_csvs_match_order_and_are_blank": all(
            [row["blind_id"] for row in rows] == judge_ids
            and all(not row["label"] and not row["rationale"] for row in rows)
            for rows in (author_1, author_2)
        ),
        "adjudication_csv_is_empty": not adjudication,
        "immutable_judge_tree_hash_matches": coordinator[
            "judge_bundle_immutable_tree_sha256"
        ]
        == immutable_tree_digest(immutable_paths, package_dir),
    }
    return {
        "schema_version": 1,
        "package_id": package_dir.name,
        "checks": checks,
        "all_passed": all(checks.values()),
        "human_candidate_count": 60,
        "automatic_generation_failure_count": 3,
        "overall_case_denominator": 63,
        "next_action": "Open the role-isolated annotation frontend for two independent authors.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--package-id", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--package-id", required=True)
    args = parser.parse_args()
    package_dir = PACKAGES_DIR / args.package_id
    if args.command == "prepare":
        result = build_package(run_id=args.run_id, package_dir=package_dir)
    else:
        result = verify_package(package_dir)
        write_json(package_dir / "package_verification.json", result)
        if not result["all_passed"]:
            raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
