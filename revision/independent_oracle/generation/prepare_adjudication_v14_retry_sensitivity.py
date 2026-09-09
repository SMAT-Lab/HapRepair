#!/usr/bin/env python3
"""Build and verify the three-case retry-sensitivity blind package."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import prepare_adjudication_v14_static_skill as base


RUN_ID = "exp_indep_3_v14_protocol_rejection_retry_luna_01"
PACKAGE_ID = "exp_indep_3_v14_retry_sensitivity_blind_01"
ORDER_SEED = 20260808
FORBIDDEN_LITERALS = (
    RUN_ID,
    "RTRY-",
    "V14-",
    "retry",
    "sensitivity",
    "restricted_command",
    "gpt-5.6-luna",
)


def load_sources() -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = base.RUNS_DIR / RUN_ID
    verification = base.read_json(run_dir / "verification.json")
    if not verification.get("all_passed"):
        raise ValueError("Retry-sensitivity run verification is not passing")
    prepared = base.read_json(run_dir / "input_manifest.json")
    if len(prepared) != 3:
        raise ValueError("Retry-sensitivity run does not contain exactly three inputs")
    records = []
    for item in prepared:
        result_path = run_dir / item["case_dir"] / "result.json"
        result = base.read_json(result_path)
        if result.get("status") != "accepted" or result.get("restricted_commands"):
            raise ValueError(f"Retry output is not accepted: {item['blind_id']}")
        records.append({"item": item, "result": result, "result_path": result_path})
    return run_dir, verification, records


def build_package(package_dir: Path) -> dict[str, Any]:
    if package_dir.exists():
        raise FileExistsError(f"Package directory already exists: {package_dir}")
    run_dir, verification, records = load_sources()
    cases = base.read_json(base.CASES_MANIFEST)["cases"]
    cases_by_id = {case["case_id"]: case for case in cases}
    random.Random(ORDER_SEED).shuffle(records)

    judge_dir = package_dir / "judge_bundle"
    cases_dir = judge_dir / "cases"
    cases_dir.mkdir(parents=True)
    blind_map = []
    case_hashes = []
    category_counts: Counter[str] = Counter()
    for order, record in enumerate(records, start=1):
        item = record["item"]
        case = cases_by_id[item["case_id"]]
        case_dir = run_dir / item["case_dir"]
        task = base.read_json(case_dir / "workspace/HAPREPAIR_TASK.json")
        judge_blind_id = f"RSJ-{order:03d}"
        candidate_snapshot = base.repair_snapshot(case_dir / "workspace")
        defective_paths = [entry["path"] for entry in item["input_files"]]
        judge_case = {
            "schema_version": 1,
            "blind_id": judge_blind_id,
            "category": base.category(item["rule"]),
            "rule": item["rule"],
            "rule_description": case["rule_description"],
            "target_findings": task["target_findings"],
            "defective_files": base.file_records(
                case_dir / "baseline", defective_paths
            ),
            "human_reference_files": base.file_records(
                base.REFERENCE_PROJECT, case["repaired_files"]
            ),
            "candidate_repair_files": base.file_records_from_snapshot(
                candidate_snapshot
            ),
        }
        path = cases_dir / f"{judge_blind_id}.json"
        base.write_json(path, judge_case)
        case_hashes.append(
            {
                "blind_id": judge_blind_id,
                "path": path.relative_to(judge_dir).as_posix(),
                "sha256": base.sha256_file(path),
            }
        )
        blind_map.append(
            {
                "order": order,
                "judge_blind_id": judge_blind_id,
                "retry_blind_id": item["blind_id"],
                "case_id": item["case_id"],
                "category": judge_case["category"],
                "rule": item["rule"],
                "source_result_sha256": base.sha256_file(record["result_path"]),
                "candidate_file_hashes": [
                    {
                        "path": relative,
                        "sha256": base.sha256_bytes(content),
                        "size_bytes": len(content),
                    }
                    for relative, content in sorted(candidate_snapshot.items())
                ],
            }
        )
        category_counts[judge_case["category"]] += 1

    label_rows = [
        {"order": index, "blind_id": row["blind_id"], "label": "", "rationale": ""}
        for index, row in enumerate(case_hashes, start=1)
    ]
    base.write_csv(judge_dir / "labels_author_1.csv", base.AUTHOR_COLUMNS, label_rows)
    base.write_csv(judge_dir / "labels_author_2.csv", base.AUTHOR_COLUMNS, label_rows)
    base.write_csv(
        package_dir / "third_author_adjudication.csv", base.ADJUDICATION_COLUMNS, []
    )
    (judge_dir / "ANNOTATION_GUIDE.md").write_text(
        base.annotation_guide(), encoding="utf-8"
    )
    base.write_json(
        judge_dir / "package_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-INDEP-3-BLIND-REPAIR-ADJUDICATION",
            "package_id": "exp_indep_3_blind_01",
            "case_count": 3,
            "category_counts": dict(sorted(category_counts.items())),
            "order_seed": ORDER_SEED,
            "labels": ["Correct", "Suspicious", "Incorrect"],
            "strict_correct_labels": ["Correct"],
            "case_files": case_hashes,
        },
    )
    base.write_json(package_dir / "source_to_judge_blind_map.json", blind_map)
    immutable_paths = [
        judge_dir / "ANNOTATION_GUIDE.md",
        judge_dir / "package_manifest.json",
        *sorted(cases_dir.glob("*.json")),
    ]
    protocol = base.read_json(run_dir / "protocol_snapshot.json")
    base.write_json(
        package_dir / "coordinator_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-INDEP-63-FINAL-V14-REJECTION-RETRY-SENSITIVITY",
            "package_id": package_dir.name,
            "source_run_id": RUN_ID,
            "source_model": protocol["model"]["requested_id"],
            "source_verification_sha256": base.sha256_file(
                run_dir / "verification.json"
            ),
            "source_metrics_sha256": base.sha256_file(run_dir / "metrics.json"),
            "human_adjudication_candidate_count": 3,
            "category_counts": dict(sorted(category_counts.items())),
            "replace_parent_main_result": False,
            "classification": (
                "protocol-rejection retry sensitivity; not an API or endpoint retry"
            ),
            "judge_bundle_immutable_tree_sha256": base.immutable_tree_digest(
                immutable_paths, package_dir
            ),
            "judge_bundle_contains_source_or_generation_identity": False,
            "source_to_judge_blind_map": "source_to_judge_blind_map.json",
            "next_action": (
                "Two authors independently label all three retry candidates; a third "
                "author adjudicates disagreements only."
            ),
        },
    )
    result = verify_package(package_dir)
    base.write_json(package_dir / "package_verification.json", result)
    if not result["all_passed"]:
        raise ValueError("Generated retry package failed verification")
    return result


def verify_package(package_dir: Path) -> dict[str, Any]:
    run_dir, run_verification, records = load_sources()
    judge_dir = package_dir / "judge_bundle"
    manifest = base.read_json(judge_dir / "package_manifest.json")
    coordinator = base.read_json(package_dir / "coordinator_manifest.json")
    blind_map = base.read_json(package_dir / "source_to_judge_blind_map.json")
    entries = manifest["case_files"]
    judge_ids = [entry["blind_id"] for entry in entries]
    by_retry_id = {record["item"]["blind_id"]: record for record in records}
    case_hashes_match = True
    identities_match = True
    source_contents_match = True
    forbidden_keys_absent = True
    for entry in entries:
        path = (judge_dir / entry["path"]).resolve()
        if judge_dir.resolve() not in path.parents or not path.is_file():
            case_hashes_match = False
            continue
        case_hashes_match &= base.sha256_file(path) == entry["sha256"]
        case = base.read_json(path)
        identities_match &= case.get("blind_id") == entry["blind_id"]
        forbidden_keys_absent &= not (base.deep_keys(case) & base.FORBIDDEN_JUDGE_KEYS)
        mapping = next(
            (row for row in blind_map if row["judge_blind_id"] == entry["blind_id"]),
            None,
        )
        if mapping is None:
            source_contents_match = False
            continue
        record = by_retry_id[mapping["retry_blind_id"]]
        item = record["item"]
        case_dir = run_dir / item["case_dir"]
        case_meta = next(
            row
            for row in base.read_json(base.CASES_MANIFEST)["cases"]
            if row["case_id"] == item["case_id"]
        )
        source_contents_match &= (
            case["candidate_repair_files"]
            == base.file_records_from_snapshot(
                base.repair_snapshot(case_dir / "workspace")
            )
            and case["defective_files"]
            == base.file_records(
                case_dir / "baseline", [row["path"] for row in item["input_files"]]
            )
            and case["human_reference_files"]
            == base.file_records(base.REFERENCE_PROJECT, case_meta["repaired_files"])
        )

    author_rows = [
        base.read_csv(judge_dir / f"labels_author_{index}.csv", base.AUTHOR_COLUMNS)
        for index in (1, 2)
    ]
    adjudication = base.read_csv(
        package_dir / "third_author_adjudication.csv", base.ADJUDICATION_COLUMNS
    )
    judge_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            judge_dir / "ANNOTATION_GUIDE.md",
            judge_dir / "package_manifest.json",
            *sorted((judge_dir / "cases").glob("*.json")),
        ]
    )
    immutable_paths = [
        judge_dir / "ANNOTATION_GUIDE.md",
        judge_dir / "package_manifest.json",
        *sorted((judge_dir / "cases").glob("*.json")),
    ]
    checks = {
        "source_run_verification_passed": run_verification["all_passed"],
        "judge_case_count_3": manifest["case_count"] == len(entries) == 3,
        "judge_ids_unique_and_remapped": len(judge_ids) == len(set(judge_ids)) == 3
        and all(re.fullmatch(r"RSJ-\d{3}", blind_id) for blind_id in judge_ids),
        "source_map_exactly_covers_3": len(blind_map) == 3
        and {row["retry_blind_id"] for row in blind_map} == set(by_retry_id),
        "case_hashes_match": case_hashes_match,
        "case_identities_match": identities_match,
        "candidate_defective_reference_contents_match_sources": source_contents_match,
        "judge_case_forbidden_keys_absent": forbidden_keys_absent,
        "judge_bundle_forbidden_literals_absent": all(
            literal.lower() not in judge_text.lower() for literal in FORBIDDEN_LITERALS
        ),
        "author_csvs_match_order_and_are_blank": all(
            [row["blind_id"] for row in rows] == judge_ids
            and all(not row["label"] and not row["rationale"] for row in rows)
            for rows in author_rows
        ),
        "adjudication_csv_is_empty": not adjudication,
        "immutable_judge_tree_hash_matches": coordinator[
            "judge_bundle_immutable_tree_sha256"
        ]
        == base.immutable_tree_digest(immutable_paths, package_dir),
        "parent_main_result_not_replaced": coordinator["replace_parent_main_result"]
        is False,
    }
    return {
        "schema_version": 1,
        "package_id": package_dir.name,
        "checks": checks,
        "all_passed": all(checks.values()),
        "human_candidate_count": 3,
        "next_action": "Open a role-isolated frontend for the three retry candidates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("verify")
    args = parser.parse_args()
    package_dir = base.PACKAGES_DIR / PACKAGE_ID
    if args.command == "prepare":
        result = build_package(package_dir)
    else:
        result = verify_package(package_dir)
        base.write_json(package_dir / "package_verification.json", result)
        if not result["all_passed"]:
            raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
