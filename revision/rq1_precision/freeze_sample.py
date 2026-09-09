#!/usr/bin/env python3
"""Freeze the rule-aware EXP-RQ1-PRECISION sample and annotation package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
HAPREPAIR_ROOT = WORKSPACE_ROOT / "HapRepair"
POPULATION_DIR = WORKSPACE_ROOT / "paper" / "rebuttal" / "rq1_population"
POPULATION_PATH = POPULATION_DIR / "population_35.jsonl"
POPULATION_SUMMARY = POPULATION_DIR / "population_35_summary.json"
SCAN_MANIFEST = (
    HAPREPAIR_ROOT
    / "revision/coding_agent_baseline/scan_runs/candidate_scan_35_gitcode_01/scan_manifest.json"
)
SKILL_DIR = HAPREPAIR_ROOT / "skills/haprepair-openharmony-repair"
OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1"
)

PERFORMANCE_TARGET = 170
PERFORMANCE_MINIMUM = 3
SECURITY_TARGETS = {
    "@security/no-commented-code": 30,
    "@security/no-cycle": 30,
    "@security/no-unsafe-hash": 1,
}
SECURITY_PROJECT_MINIMUM = 2
PERFORMANCE_SEED = "EXP-RQ1-PRECISION-performance-20260807-v1"
SECURITY_SEED = "EXP-RQ1-PRECISION-security-20260807-v1"
ORDER_SEED = "EXP-RQ1-PRECISION-order-20260807-v1"
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(seed: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{identifier}".encode()).hexdigest()


def safe_name(value: str) -> str:
    return value.replace("@", "").replace("/", "--")


def allocate_with_minimum(
    populations: dict[str, int], target: int, minimum: int
) -> dict[str, int]:
    if target > sum(populations.values()):
        raise ValueError("Sample target exceeds population")
    allocation = {key: min(count, minimum) for key, count in populations.items()}
    base = sum(allocation.values())
    if base > target:
        raise ValueError("Minimum allocation exceeds target")
    remaining = target - base
    capacities = {key: populations[key] - allocation[key] for key in populations}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValueError("Insufficient residual capacity")
    if not remaining:
        return allocation

    quotas = {
        key: remaining * capacities[key] / capacity_total
        for key in sorted(populations)
    }
    floors = {key: min(capacities[key], math.floor(quotas[key])) for key in quotas}
    for key, value in floors.items():
        allocation[key] += value
    left = target - sum(allocation.values())
    order = sorted(
        populations,
        key=lambda key: (-(quotas[key] - floors[key]), key),
    )
    for key in order:
        if not left:
            break
        if allocation[key] < populations[key]:
            allocation[key] += 1
            left -= 1
    if left or sum(allocation.values()) != target:
        raise ValueError("Hamilton allocation did not reach the target")
    return allocation


def choose(
    records: list[dict[str, Any]], count: int, seed: str
) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError("Requested sample exceeds stratum population")
    return sorted(
        records,
        key=lambda item: (stable_rank(seed, item["alert_id"]), item["alert_id"]),
    )[:count]


def load_rule_reference_manifest() -> dict[str, dict[str, Any]]:
    manifest = read_json(SKILL_DIR / "references/repair-guides/manifest.json")
    return {entry["rule"]: entry for entry in manifest["rules"]}


def family_spec_for_rule(rule: str) -> str | None:
    manifest = read_json(SKILL_DIR / "references/rule-specs/manifest.json")
    canonical = manifest.get("aliases", {}).get(rule, rule)
    for family in manifest["families"]:
        if canonical in family["rules"]:
            return family["spec"]
    return None


def copy_rule_evidence(
    rules: list[str], bundle: Path
) -> dict[str, dict[str, Any]]:
    guide_manifest = load_rule_reference_manifest()
    evidence_dir = bundle / "rule_evidence"
    evidence_dir.mkdir(parents=True)
    core_source = SKILL_DIR / "references/rule-specs/core-invariants.md"
    core_target = evidence_dir / core_source.name
    shutil.copyfile(core_source, core_target)
    copied_specs = {core_source.name}
    result: dict[str, dict[str, Any]] = {}
    for rule in rules:
        family = family_spec_for_rule(rule)
        specs = [core_target.relative_to(bundle).as_posix()]
        if family:
            family_source = SKILL_DIR / "references/rule-specs" / family
            family_target = evidence_dir / family
            if family not in copied_specs:
                shutil.copyfile(family_source, family_target)
                copied_specs.add(family)
            specs.append(family_target.relative_to(bundle).as_posix())
        guide = guide_manifest.get(rule)
        guide_path = None
        if guide:
            guide_source = SKILL_DIR / "references/repair-guides" / guide["file"]
            guide_target = evidence_dir / guide["file"]
            shutil.copyfile(guide_source, guide_target)
            guide_path = guide_target.relative_to(bundle).as_posix()
        result[rule] = {
            "semantic_spec_paths": specs,
            "static_reference_path": guide_path,
            "static_reference_available": guide_path is not None,
        }
    return result


def source_roots() -> dict[str, Path]:
    manifest = read_json(SCAN_MANIFEST)
    return {item["name"]: Path(item["source_path"]) for item in manifest["projects"]}


def copy_source_file(
    record: dict[str, Any], roots: dict[str, Path], bundle: Path
) -> dict[str, Any]:
    root = roots[record["project"]].resolve()
    source = (root / record["relative_path"]).resolve()
    if root != source and root not in source.parents:
        raise ValueError(f"Source path escapes project root: {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    content = source.read_text(encoding="utf-8")
    content_sha = hashlib.sha256(content.encode()).hexdigest()
    target = bundle / "source_files" / f"{content_sha}.json"
    if not target.exists():
        write_json(
            target,
            {
                "schema_version": 1,
                "relative_path": record["relative_path"],
                "content_sha256": content_sha,
                "content": content,
            },
        )
    return {
        "path": target.relative_to(bundle).as_posix(),
        "sha256": sha256_file(target),
        "content_sha256": content_sha,
    }


def build_samples(population: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    performance = [item for item in population if item["category"] == "performance"]
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in performance:
        by_rule[item["rule"]].append(item)
    performance_populations = {rule: len(rows) for rule, rows in sorted(by_rule.items())}
    performance_allocation = allocate_with_minimum(
        performance_populations, PERFORMANCE_TARGET, PERFORMANCE_MINIMUM
    )
    sampled: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for rule, rows in sorted(by_rule.items()):
        n = performance_allocation[rule]
        for item in choose(rows, n, f"{PERFORMANCE_SEED}\0{rule}"):
            sampled.append(
                {
                    **item,
                    "sampling_stratum": f"performance\0{rule}",
                    "stratum_population": len(rows),
                    "stratum_sample": n,
                    "inclusion_probability": n / len(rows),
                    "analysis_weight": len(rows) / n,
                }
            )
        allocation_rows.append(
            {
                "category": "performance",
                "rule": rule,
                "project": None,
                "population": len(rows),
                "sample": n,
                "inclusion_probability": n / len(rows),
                "analysis_weight": len(rows) / n,
            }
        )

    security = [item for item in population if item["category"] == "security"]
    security_counts = Counter(item["rule"] for item in security)
    if set(security_counts) != set(SECURITY_TARGETS):
        raise ValueError(
            "Security rule population differs from the frozen contract: "
            f"observed={sorted(security_counts)}, expected={sorted(SECURITY_TARGETS)}"
        )
    for rule, target in SECURITY_TARGETS.items():
        rows = [item for item in security if item["rule"] == rule]
        if not rows:
            raise ValueError(f"Missing security population: {rule}")
        by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in rows:
            by_project[item["project"]].append(item)
        project_populations = {
            project: len(items) for project, items in sorted(by_project.items())
        }
        project_allocation = allocate_with_minimum(
            project_populations, target, SECURITY_PROJECT_MINIMUM
        )
        for project, project_rows in sorted(by_project.items()):
            n = project_allocation[project]
            for item in choose(
                project_rows, n, f"{SECURITY_SEED}\0{rule}\0{project}"
            ):
                sampled.append(
                    {
                        **item,
                        "sampling_stratum": f"security\0{rule}\0{project}",
                        "stratum_population": len(project_rows),
                        "stratum_sample": n,
                        "inclusion_probability": n / len(project_rows),
                        "analysis_weight": len(project_rows) / n,
                    }
                )
            allocation_rows.append(
                {
                    "category": "security",
                    "rule": rule,
                    "project": project,
                    "population": len(project_rows),
                    "sample": n,
                    "inclusion_probability": n / len(project_rows),
                    "analysis_weight": len(project_rows) / n,
                }
            )

    if len(sampled) != PERFORMANCE_TARGET + sum(SECURITY_TARGETS.values()):
        raise ValueError("Unexpected total sample size")
    if len({item["alert_id"] for item in sampled}) != len(sampled):
        raise ValueError("Sample contains duplicate alerts")
    for row in allocation_rows:
        if row["sample"] == 1 and row["population"] > 1:
            raise ValueError(f"Non-census singleton stratum: {row}")
    allocation = {
        "performance": {
            "method": "rule strata; min(3, N_h), then Hamilton allocation proportional to residual capacity",
            "seed": PERFORMANCE_SEED,
            "population": len(performance),
            "sample": PERFORMANCE_TARGET,
            "rule_count": len(by_rule),
        },
        "security": {
            "method": "fixed rule targets; within each rule use project strata, min(2, N_hr), then Hamilton allocation proportional to residual capacity",
            "seed": SECURITY_SEED,
            "population": len(security),
            "sample": sum(SECURITY_TARGETS.values()),
            "rule_targets": SECURITY_TARGETS,
        },
        "strata": allocation_rows,
    }
    return sampled, allocation


def write_csv_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Frozen output already exists: {OUTPUT_DIR}")
    population_summary = read_json(POPULATION_SUMMARY)
    if sha256_file(POPULATION_PATH) != population_summary["population_sha256"]:
        raise SystemExit("Population hash does not match its frozen summary")
    population = read_jsonl(POPULATION_PATH)
    if len(population) != 9884:
        raise SystemExit("Unexpected frozen population size")

    sampled, allocation = build_samples(population)
    sampled.sort(
        key=lambda item: (stable_rank(ORDER_SEED, item["alert_id"]), item["alert_id"])
    )
    bundle = OUTPUT_DIR / "judge_bundle"
    cases_dir = bundle / "cases"
    cases_dir.mkdir(parents=True)
    roots = source_roots()
    rules = sorted({item["rule"] for item in sampled})
    rule_evidence = copy_rule_evidence(rules, bundle)

    case_entries = []
    coordinator_rows = []
    frozen_sample = []
    for order, record in enumerate(sampled, start=1):
        blind_id = f"RQ1-{order:03d}"
        source_file = copy_source_file(record, roots, bundle)
        finding = {
            key: record[key]
            for key in (
                "relative_path",
                "line",
                "column",
                "end_line",
                "end_column",
                "severity",
                "rule",
                "message",
            )
        }
        case = {
            "schema_version": 1,
            "blind_id": blind_id,
            "category": record["category"],
            "rule": record["rule"],
            "finding": finding,
            "source_file": source_file,
            "rule_evidence": rule_evidence[record["rule"]],
        }
        case_path = cases_dir / f"{blind_id}.json"
        write_json(case_path, case)
        case_entries.append(
            {
                "order": order,
                "blind_id": blind_id,
                "category": record["category"],
                "rule": record["rule"],
                "path": case_path.relative_to(bundle).as_posix(),
                "sha256": sha256_file(case_path),
            }
        )
        coordinator_rows.append(
            {
                "order": order,
                "blind_id": blind_id,
                "alert_id": record["alert_id"],
                "project": record["project"],
                "commit": record["commit"],
                "tree_oid": record["tree_oid"],
            }
        )
        frozen_sample.append({"blind_id": blind_id, **record})

    empty_labels = [
        {"order": item["order"], "blind_id": item["blind_id"], "label": "", "rationale": ""}
        for item in case_entries
    ]
    for role in ("author_1", "author_2"):
        write_csv_rows(bundle / f"labels_{role}.csv", AUTHOR_COLUMNS, empty_labels)
    write_csv_rows(OUTPUT_DIR / "third_author_adjudication.csv", ADJUDICATION_COLUMNS, [])

    guide = """# EXP-RQ1-PRECISION Annotation Guide

Judge whether the HomeCheck finding is a true instance of the named rule in the
frozen source snapshot. Inspect the complete source file and the bundled rule
evidence. Do not consult the other author or any repair outcome.

- `Correct`: the reported location and surrounding repository evidence satisfy
  the rule's defect condition.
- `Suspicious`: the available static evidence is insufficient to decide, or the
  result depends on unavailable generated/runtime behavior.
- `Incorrect`: the report does not satisfy the rule, targets the wrong construct,
  or is contradicted by the shown source evidence.

Only `Correct` counts as a strict true positive. `Suspicious` and `Incorrect`
both count as not correct in strict precision. A rationale is mandatory for
`Suspicious` and `Incorrect`. Each author completes only their own label file.
The third author receives only disagreements after both author files are frozen.
"""
    (bundle / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    write_jsonl(OUTPUT_DIR / "frozen_sample.jsonl", frozen_sample)
    write_json(OUTPUT_DIR / "allocation.json", allocation)
    write_json(
        OUTPUT_DIR / "protocol.json",
        {
            "schema_version": 1,
            "experiment": "EXP-RQ1-PRECISION",
            "package_id": OUTPUT_DIR.name,
            "status": "frozen_ready_for_independent_annotation",
            "frozen_on": "2026-08-07",
            "population_path": str(POPULATION_PATH),
            "population_sha256": sha256_file(POPULATION_PATH),
            "population_size": len(population),
            "sample_size": len(sampled),
            "performance_sample_size": PERFORMANCE_TARGET,
            "security_sample_size": sum(SECURITY_TARGETS.values()),
            "allocation": allocation,
            "deterministic_selection": "SHA-256 rank of seed and alert_id",
            "blind_order_seed": ORDER_SEED,
            "labels": list(LABELS),
            "strict_correct_labels": ["Correct"],
            "annotation": "two independent authors; third author adjudicates disagreements only",
            "analysis": {
                "estimator": "design-weighted stratified proportion",
                "variance": "without-replacement stratified variance with finite-population correction",
                "confidence_interval": "normal 95% interval clamped to [0,1]",
                "reported": [
                    "per_rule_precision",
                    "performance_precision",
                    "security_precision",
                    "overall_precision",
                    "performance_macro_precision",
                    "security_macro_precision",
                    "raw_agreement",
                    "cohen_kappa_when_estimable",
                    "adjudication_count",
                ],
            },
            "claim_scope": "HomeCheck-defined findings in the frozen 35-project population; not natural-project recall or confirmed vulnerabilities",
            "ethics": {
                "data": "public source artifacts and author-generated expert labels; no personal participant data in the package",
                "institutional_determination": "authors must confirm local requirements before involving annotators outside the author team",
            },
        },
    )
    write_json(
        bundle / "package_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-RQ1-PRECISION",
            "package_id": OUTPUT_DIR.name,
            "case_count": len(case_entries),
            "labels": list(LABELS),
            "strict_correct_labels": ["Correct"],
            "case_files": case_entries,
        },
    )
    write_json(
        OUTPUT_DIR / "coordinator_manifest.json",
        {
            "schema_version": 1,
            "experiment": "EXP-RQ1-PRECISION",
            "package_id": OUTPUT_DIR.name,
            "population_sha256": sha256_file(POPULATION_PATH),
            "protocol_sha256": sha256_file(OUTPUT_DIR / "protocol.json"),
            "allocation_sha256": sha256_file(OUTPUT_DIR / "allocation.json"),
            "frozen_sample_sha256": sha256_file(OUTPUT_DIR / "frozen_sample.jsonl"),
            "judge_bundle": "judge_bundle/",
            "judge_bundle_excludes_project_identity": True,
            "coordinator_map": coordinator_rows,
        },
    )
    (OUTPUT_DIR / "STUDY_PROTOCOL.md").write_text(
        """# EXP-RQ1-PRECISION Study Protocol

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-07
- Verification Status: FROZEN
- Version Label: exp_rq1_precision_v1

## Study Overview

- Research question: What is the strict precision of HomeCheck-defined
  performance and security findings in the frozen 35-project population?
- Design: stratified independent expert annotation of static-analysis findings.
- Unit: one frozen HomeCheck finding and its complete source file.
- Labels: Correct, Suspicious, Incorrect; only Correct is a strict true positive.

## Sampling

- Performance: 170 of 9,401 findings, stratified by all 38 observed rules.
- Security: 61 of 483 findings, with 30 no-commented-code, 30 no-cycle, and the
  sole no-unsafe-hash finding; project stratification is retained within rule.
- Selection: frozen SHA-256 ordering without replacement.

## Annotation

- Two authors label every case independently.
- Neither author can access the other author's labels during annotation.
- A third author receives and resolves disagreements only after both files are
  complete and frozen.
- Suspicious and Incorrect require a written rationale.

## Analysis

- Use design-weighted stratified precision with finite-population correction.
- Report per-rule, category, overall, and category-macro estimates, 95% intervals,
  raw agreement, Cohen's kappa when estimable, and adjudication counts.
- Scope is precision in the frozen population, not natural-project recall or
  confirmed vulnerability prevalence.

## Ethics And Data Handling

- The package contains public source artifacts and author-generated labels, not
  personal participant data.
- The authors must confirm local institutional requirements before involving
  annotators outside the author team.
- Annotation files remain local and access-controlled by role.
""",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(
        """# Frozen EXP-RQ1-PRECISION Package

This package contains the immutable 170-performance and 61-security sample,
allocation and weighting contract, blinded cases, two independent author label
files, and a disagreement-only adjudication file.

Do not edit `protocol.json`, `allocation.json`, `frozen_sample.jsonl`, case files,
source files, or rule evidence after `audit.json` passes. Only the role-specific
label CSVs and `third_author_adjudication.csv` are mutable during annotation.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "frozen_ready_for_independent_annotation",
                "output": str(OUTPUT_DIR),
                "sample_size": len(sampled),
                "performance": PERFORMANCE_TARGET,
                "security": sum(SECURITY_TARGETS.values()),
                "protocol_sha256": sha256_file(OUTPUT_DIR / "protocol.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
