#!/usr/bin/env python3
"""Freeze the revised 35-project RQ1 finding population and strata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SCAN_DIR = SCRIPT_DIR / "scan_runs" / "candidate_scan_35_gitcode_01"
SCAN_MANIFEST = SCAN_DIR / "scan_manifest.json"
OUTPUT_DIR = WORKSPACE_ROOT / "paper" / "rebuttal" / "rq1_population"
TARGET_PREFIXES = ("@performance/", "@security/")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(rule: str) -> str:
    if rule.startswith("@performance/"):
        return "performance"
    if rule.startswith("@security/"):
        return "security"
    return "excluded_non_target"


def stable_id(project: str, finding: dict[str, Any]) -> str:
    fields = [
        project,
        str(finding["relative_path"]),
        str(finding["line"]),
        str(finding["column"]),
        str(finding["rule"]),
        str(finding["message"]),
    ]
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def main() -> None:
    scan = json.loads(SCAN_MANIFEST.read_text(encoding="utf-8"))
    projects = scan["projects"]
    if scan.get("scanned_project_count") != 35 or len(projects) != 35:
        raise SystemExit("RQ1 population requires exactly 35 successfully scanned projects")
    failures = [item["name"] for item in projects if not item["status"].startswith("scanned")]
    if failures:
        raise SystemExit(f"Scan failures remain: {failures}")

    records: list[dict[str, Any]] = []
    excluded = Counter()
    rule_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    project_counts: dict[str, Counter[str]] = defaultdict(Counter)
    project_summaries: list[dict[str, Any]] = []

    for project in sorted(projects, key=lambda item: item["name"]):
        findings_path = Path(project["findings_path"])
        findings = json.loads(findings_path.read_text(encoding="utf-8"))
        target_count = 0
        for finding in findings:
            rule = str(finding["rule"])
            finding_category = category(rule)
            if not rule.startswith(TARGET_PREFIXES):
                excluded[rule] += 1
                continue
            record = {
                "alert_id": stable_id(project["name"], finding),
                "project": project["name"],
                "category": finding_category,
                "commit": project["commit"],
                "tree_oid": project["tree_oid"],
                **finding,
            }
            records.append(record)
            target_count += 1
            rule_counts[rule] += 1
            category_counts[finding_category] += 1
            project_counts[project["name"]][finding_category] += 1
        project_summaries.append(
            {
                "name": project["name"],
                "repo_url": project["repo_url"],
                "commit": project["commit"],
                "tree_oid": project["tree_oid"],
                "subpath": project["subpath"],
                "nonblank_source_loc": project["nonblank_source_loc"],
                "raw_findings": len(findings),
                "target_findings": target_count,
                "performance_findings": project_counts[project["name"]]["performance"],
                "security_findings": project_counts[project["name"]]["security"],
            }
        )

    ids = [record["alert_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("Stable alert IDs are not unique")

    records.sort(
        key=lambda item: (
            item["project"],
            item["relative_path"],
            item["line"],
            item["column"],
            item["rule"],
            item["message"],
        )
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    population_path = OUTPUT_DIR / "population_35.jsonl"
    population_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )

    security_rules = {
        rule: count for rule, count in sorted(rule_counts.items()) if rule.startswith("@security/")
    }
    summary = {
        "schema_version": 1,
        "experiment": "EXP-RQ1-PRECISION",
        "status": "population_frozen_samples_not_yet_drawn",
        "scope": "35 fresh revised project snapshots; not the deleted historical server population",
        "scan_manifest": str(SCAN_MANIFEST),
        "scan_manifest_sha256": sha256_file(SCAN_MANIFEST),
        "source_manifest": scan["source_manifest"],
        "source_manifest_sha256": scan["source_manifest_sha256"],
        "scanner": scan["codelinter"],
        "config": scan["config"],
        "config_sha256": scan["config_sha256"],
        "scan_started_at": scan["started_at"],
        "scan_completed_at": scan["completed_at"],
        "project_count": 35,
        "raw_finding_count": sum(item["raw_findings"] for item in project_summaries),
        "target_finding_count": len(records),
        "excluded_finding_count": sum(excluded.values()),
        "excluded_rule_counts": dict(sorted(excluded.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "category_rule_counts": {
            name: sum(1 for rule in rule_counts if category(rule) == name)
            for name in ("performance", "security")
        },
        "rule_counts": dict(sorted(rule_counts.items())),
        "security_sampling_contract": {
            "method": "rule-balanced project-aware sampling without replacement",
            "population_size": category_counts["security"],
            "sample_size": 61,
            "observed_rules": security_rules,
            "allocation": {
                "@security/no-commented-code": 30,
                "@security/no-cycle": 30,
                "@security/no-unsafe-hash": 1,
            },
            "within_rule_allocation": "minimum project coverage where feasible, followed by proportional allocation of the remainder; frozen seed required before drawing",
            "annotation": "two independent authors; third-author adjudication for disagreements",
            "reported_metrics": [
                "per_rule_precision",
                "micro_precision",
                "macro_precision_over_observed_rules",
                "inter_rater_agreement",
                "adjudication_count",
            ],
            "weighting": "use frozen inclusion probabilities and the observed rule populations for per-rule, micro, macro, and overall estimates",
            "claim_scope": "HomeCheck-defined security alerts in the fixed 35-project population; not confirmed vulnerabilities or evidence for unobserved rules",
            "feasible": security_rules.get("@security/no-commented-code", 0) >= 30
            and security_rules.get("@security/no-cycle", 0) >= 30
            and security_rules.get("@security/no-unsafe-hash", 0) == 1,
        },
        "performance_sampling_contract": {
            "sample_size": 170,
            "method": "rule-aware minimum allocation followed by proportional allocation of the remainder",
            "feasible": category_counts["performance"] >= 170,
        },
        "projects": project_summaries,
        "population_path": str(population_path),
        "population_sha256": sha256_file(population_path),
    }
    summary_path = OUTPUT_DIR / "population_35_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{summary_path}: projects=35 target={len(records)} "
        f"performance={category_counts['performance']} security={category_counts['security']} "
        f"sha256={sha256_file(summary_path)}"
    )


if __name__ == "__main__":
    main()
