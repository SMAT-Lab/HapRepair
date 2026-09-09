#!/usr/bin/env python3
"""Build the blinded human-adjudication package after formal generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
REVISION_DIR = ORACLE_DIR.parent
REPO_ROOT = REVISION_DIR.parent
RUNS_DIR = ORACLE_DIR / "generation_runs"
PACKAGES_DIR = ORACLE_DIR / "adjudication_packages"
REFERENCE_PROJECT = ORACLE_DIR / "benchmark" / "repaired_project"
ORDER_SEED = 20260803


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(rule: str) -> str:
    if rule.startswith("@performance/"):
        return "performance"
    if rule.startswith("@security/"):
        return "security"
    return "arkts-eslint"


def read_files(project: Path, paths: list[str]) -> list[dict[str, str]]:
    return [
        {"path": relative, "content": (project / relative).read_text(encoding="utf-8")}
        for relative in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--package-id", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    verification = read_json(run_dir / "verification.json")
    metrics = read_json(run_dir / "metrics.json")
    protocol = read_json(run_dir / "protocol_snapshot.json")
    if not verification.get("all_passed") or metrics.get("status") != "complete":
        raise SystemExit("Generation run has not passed complete verification")
    if metrics.get("semantic_correctness_status") != "not_adjudicated":
        raise SystemExit("Generation run is already marked as adjudicated")

    package_dir = PACKAGES_DIR / args.package_id
    if package_dir.exists():
        raise SystemExit(f"Package directory already exists: {package_dir}")
    judge_dir = package_dir / "judge_bundle"
    cases_dir = judge_dir / "cases"
    cases_dir.mkdir(parents=True)

    manifest = read_json(ORACLE_DIR / "cases_manifest.json")
    cases_by_id = {case["case_id"]: case for case in manifest["cases"]}
    prepared = read_json(run_dir / "input_manifest.json")
    order = [item["blind_id"] for item in prepared]
    random.Random(ORDER_SEED).shuffle(order)
    by_blind = {item["blind_id"]: item for item in prepared}

    case_hashes = []
    for blind_id in order:
        item = by_blind[blind_id]
        case = cases_by_id[item["case_id"]]
        prompt = read_json(run_dir / item["prompt"])
        generated = read_json(run_dir / "parsed_outputs" / f"{blind_id}.json")
        judge_case = {
            "schema_version": 1,
            "blind_id": blind_id,
            "category": category(item["rule"]),
            "rule": item["rule"],
            "rule_description": case["rule_description"],
            "target_findings": prompt["target_findings"],
            "defective_files": prompt["target_files"],
            "human_reference_files": read_files(
                REFERENCE_PROJECT, case["repaired_files"]
            ),
            "candidate_repair_files": generated["files"],
        }
        path = cases_dir / f"{blind_id}.json"
        write_json(path, judge_case)
        case_hashes.append(
            {"blind_id": blind_id, "path": path.relative_to(judge_dir).as_posix(), "sha256": sha256_file(path)}
        )

    label_fields = ("order", "blind_id", "label", "rationale")
    for filename in ("labels_author_1.csv", "labels_author_2.csv"):
        with (judge_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=label_fields)
            writer.writeheader()
            for index, blind_id in enumerate(order, start=1):
                writer.writerow(
                    {"order": index, "blind_id": blind_id, "label": "", "rationale": ""}
                )

    with (package_dir / "third_author_adjudication.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "blind_id",
            "author_1_label",
            "author_2_label",
            "adjudicated_label",
            "rationale",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

    guide = """# Independent Repair Adjudication Guide

Evaluate the candidate repair against the defective input, target rule, finding
location, and human reference. Do not consult the other annotator.

Allowed labels:

- `Correct`: The candidate resolves the target defect, preserves the intended
  behavior and interfaces, is valid ArkTS/configuration for the shown case, and
  is semantically equivalent to the reference even if the text differs.
- `Suspicious`: The candidate is plausible but the shown evidence is
  insufficient to establish semantic equivalence, or it contains a material
  ambiguity that is neither a demonstrated failure nor clearly correct.
- `Incorrect`: The candidate leaves the target defect unresolved, is invalid,
  removes required behavior, changes an interface without justification,
  introduces an evident regression, or repairs a different problem.

Use `Suspicious` only for genuine evidentiary uncertainty, not as a softer form
of `Incorrect`. In the strict correctness rate, only `Correct` counts as
correct; both `Suspicious` and `Incorrect` count as not correct.

Each author completes only their own CSV. A third author adjudicates every
label disagreement after the two files are frozen.
"""
    (judge_dir / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    write_json(
        judge_dir / "package_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-INDEP-63",
            "package_id": args.package_id,
            "case_count": len(order),
            "order_seed": ORDER_SEED,
            "labels": ["Correct", "Suspicious", "Incorrect"],
            "strict_correct_labels": ["Correct"],
            "case_files": case_hashes,
        },
    )
    write_json(
        package_dir / "coordinator_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-INDEP-63",
            "package_id": args.package_id,
            "source_run_id": args.run_id,
            "source_protocol_sha256": sha256_file(run_dir / "protocol_snapshot.json"),
            "source_verification_sha256": sha256_file(run_dir / "verification.json"),
            "source_metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "source_model": protocol["model"]["requested_id"],
            "judge_bundle": "judge_bundle/",
            "judge_bundle_contains_model_or_retrieval_identity": False,
            "judge_bundle_excluded_fields": [
                "case_id",
                "requested_model",
                "reported_model",
                "retrieved_pair_id",
                "retrieval_score",
                "response_id",
            ],
            "reference_access_stage": "post_generation_only",
            "order_seed": ORDER_SEED,
        },
    )
    print(
        json.dumps(
            {
                "package_id": args.package_id,
                "case_count": len(order),
                "judge_bundle": str(judge_dir),
                "status": "ready_for_two_independent_authors",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
