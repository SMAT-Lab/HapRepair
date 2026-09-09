#!/usr/bin/env python3
"""Validate the frozen EXP-RQ1-PRECISION sampling and annotation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from freeze_sample import ADJUDICATION_COLUMNS, AUTHOR_COLUMNS


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PACKAGE = (
    WORKSPACE_ROOT / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1"
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate(package: Path) -> dict[str, Any]:
    package = package.resolve()
    bundle = package / "judge_bundle"
    protocol = read_json(package / "protocol.json")
    allocation = read_json(package / "allocation.json")
    coordinator = read_json(package / "coordinator_manifest.json")
    manifest = read_json(bundle / "package_manifest.json")
    sample = read_jsonl(package / "frozen_sample.jsonl")
    entries = manifest["case_files"]
    expected_ids = [entry["blind_id"] for entry in entries]
    checks = [
        check("protocol_status", protocol["status"] == "frozen_ready_for_independent_annotation"),
        check("population_hash", sha256_file(Path(protocol["population_path"])) == protocol["population_sha256"]),
        check("protocol_hash", sha256_file(package / "protocol.json") == coordinator["protocol_sha256"]),
        check("allocation_hash", sha256_file(package / "allocation.json") == coordinator["allocation_sha256"]),
        check("sample_hash", sha256_file(package / "frozen_sample.jsonl") == coordinator["frozen_sample_sha256"]),
        check("case_count", len(entries) == manifest["case_count"] == protocol["sample_size"] == 231),
        check("blind_ids_unique", len(expected_ids) == len(set(expected_ids))),
        check("sample_ids_match", set(expected_ids) == {item["blind_id"] for item in sample}),
        check("sample_alert_ids_unique", len(sample) == len({item["alert_id"] for item in sample})),
    ]
    categories = Counter(item["category"] for item in sample)
    checks.extend(
        [
            check("performance_count", categories["performance"] == 170, categories["performance"]),
            check("security_count", categories["security"] == 61, categories["security"]),
            check("performance_rule_coverage", len({item["rule"] for item in sample if item["category"] == "performance"}) == 38),
            check("security_rule_coverage", len({item["rule"] for item in sample if item["category"] == "security"}) == 3),
        ]
    )
    sample_strata = Counter(item["sampling_stratum"] for item in sample)
    allocation_strata = {
        (f"{row['category']}\0{row['rule']}" if row["project"] is None else f"{row['category']}\0{row['rule']}\0{row['project']}"): row
        for row in allocation["strata"]
    }
    checks.append(check("strata_match", set(sample_strata) == set(allocation_strata)))
    checks.append(
        check(
            "stratum_sample_counts",
            all(sample_strata[key] == row["sample"] for key, row in allocation_strata.items()),
        )
    )
    checks.append(
        check(
            "no_noncensus_singletons",
            all(row["sample"] > 1 or row["sample"] == row["population"] for row in allocation["strata"]),
        )
    )

    case_problems = []
    for entry in entries:
        case_path = (bundle / entry["path"]).resolve()
        if bundle not in case_path.parents or not case_path.is_file():
            case_problems.append(f"invalid case path: {entry['blind_id']}")
            continue
        if sha256_file(case_path) != entry["sha256"]:
            case_problems.append(f"case hash mismatch: {entry['blind_id']}")
            continue
        case = read_json(case_path)
        source = (bundle / case["source_file"]["path"]).resolve()
        if bundle not in source.parents or not source.is_file():
            case_problems.append(f"invalid source path: {entry['blind_id']}")
        elif sha256_file(source) != case["source_file"]["sha256"]:
            case_problems.append(f"source hash mismatch: {entry['blind_id']}")
        for path in case["rule_evidence"]["semantic_spec_paths"]:
            evidence = (bundle / path).resolve()
            if bundle not in evidence.parents or not evidence.is_file():
                case_problems.append(f"missing semantic evidence: {entry['blind_id']}:{path}")
        reference = case["rule_evidence"]["static_reference_path"]
        if reference:
            evidence = (bundle / reference).resolve()
            if bundle not in evidence.parents or not evidence.is_file():
                case_problems.append(f"missing static reference: {entry['blind_id']}:{reference}")
    checks.append(check("case_and_evidence_hashes", not case_problems, case_problems))

    for role in ("author_1", "author_2"):
        rows = read_csv(bundle / f"labels_{role}.csv", AUTHOR_COLUMNS)
        checks.append(check(f"{role}_order", [row["blind_id"] for row in rows] == expected_ids))
        checks.append(
            check(
                f"{role}_template_or_valid_labels",
                all(
                    (not row["label"] and not row["rationale"])
                    or row["label"] in manifest["labels"]
                    for row in rows
                ),
            )
        )
    read_csv(package / "third_author_adjudication.csv", ADJUDICATION_COLUMNS)
    checks.append(check("adjudication_schema", True))
    problems = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "experiment": "EXP-RQ1-PRECISION",
        "package_id": manifest["package_id"],
        "status": "passed" if not problems else "failed",
        "ready_for_independent_annotation": not problems,
        "checks": checks,
        "problems": problems,
        "hashes": {
            "protocol": sha256_file(package / "protocol.json"),
            "allocation": sha256_file(package / "allocation.json"),
            "frozen_sample": sha256_file(package / "frozen_sample.jsonl"),
            "package_manifest": sha256_file(bundle / "package_manifest.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    audit = validate(args.package)
    output = args.package / "audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
