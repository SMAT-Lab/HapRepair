#!/usr/bin/env python3
"""Reconcile the paper-facing v14 runs and generate deterministic E1 tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
DEFAULT_INPUTS = HERE / "formal_inputs_hapskill_35_v4.json"
DEFAULT_PROTOCOL = HERE / "protocol-hapskill-35-v14-namespace-export-safe.json"
DEFAULT_PRIMARY_RUN = (
    WORKSPACE_ROOT
    / "baseline_data/exp_hapskill/runs/exp_hapskill_35_luna_v14_namespace_export_safe_01"
)
DEFAULT_RETRY_RUN = (
    WORKSPACE_ROOT / "baseline_data/exp_hapskill/runs/"
    "exp_hapskill_35_luna_v14_namespace_export_safe_flutter_embedding_retry_01"
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "paper/rebuttal/e1_v14"
EXCLUDED_TREE_DIRS = {
    ".exp_agent",
    ".git",
    ".hvigor",
    "build",
    "node_modules",
    "oh_modules",
}
METRIC_FIELDS = (
    "initial_alerts",
    "final_alerts",
    "eliminated_alerts",
    "remaining_alerts",
    "introduced_alerts",
    "net_reduction",
)
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(project: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_TREE_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            result[path.relative_to(project).as_posix()] = sha256_file(path)
    return result


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode())
    return digest.hexdigest()


def per_rule_counts(findings: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item["rule"]) for item in findings).items()))


def alert_identity(finding: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(finding.get("relative_path", "")),
        str(finding.get("rule", "")),
        " ".join(str(finding.get("message", "")).split()),
    )


def category(rule: str) -> str:
    if rule.startswith("@security/"):
        return "security"
    if rule.startswith("@performance/"):
        return "performance"
    return "other"


def alert_volume(initial: int) -> str:
    if initial < 100:
        return "lt_100"
    if initial < 500:
        return "100_to_499"
    return "ge_500"


def metric_invariants(metrics: dict[str, int]) -> list[str]:
    problems: list[str] = []
    if (
        metrics["eliminated_alerts"] + metrics["remaining_alerts"]
        != metrics["initial_alerts"]
    ):
        problems.append("initial != eliminated + remaining")
    if (
        metrics["remaining_alerts"] + metrics["introduced_alerts"]
        != metrics["final_alerts"]
    ):
        problems.append("final != remaining + introduced")
    if (
        metrics["eliminated_alerts"] - metrics["introduced_alerts"]
        != metrics["net_reduction"]
    ):
        problems.append("net != eliminated - introduced")
    return problems


def trace_commands(trace_dir: Path) -> list[str]:
    commands: list[str] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in {
                "command_execution",
                "shell_command",
            }:
                continue
            command = item.get("command") or item.get("cmd")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            if isinstance(command, str):
                commands.append(command)
    return commands


def guide_access(condition_dir: Path) -> dict[str, Any]:
    plan = read_json(condition_dir / "skill_state/plans/initial.json")
    expected = {
        item["rule"]: Path(item["guide"]["guide_path"]).name
        for item in plan["rules"]
        if item.get("guide", {}).get("covered")
    }
    command_text = "\n".join(trace_commands(condition_dir / "traces"))
    accessed = sorted(rule for rule, name in expected.items() if name in command_text)
    missing = sorted(set(expected) - set(accessed))
    return {
        "required_rules": sorted(expected),
        "accessed_rules": accessed,
        "missing_rules": missing,
        "required_rule_count": len(expected),
        "accessed_rule_count": len(accessed),
        "complete": not missing,
        "criterion": (
            "A rule counts as accessed only when the exact initial-plan guide filename "
            "appears in a completed shell-command trace. Reading only the generated "
            "plan, completion report, or semantic specification does not count."
        ),
    }


def source_for_project(
    project: str, primary_run: Path, retry_run: Path
) -> tuple[Path, str]:
    if project == "flutter_embedding":
        return retry_run / "hapskill" / project, "audited_retry"
    return primary_run / "hapskill" / project, "primary_scheduler"


def resolved_initial_evidence(
    manifest: dict[str, Any], condition_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    scanner_path = condition_dir / "skill_state/homecheck/state.json"
    scanner = read_json(scanner_path)
    scanner_initial = scanner["initial_scan"]
    initial_findings_path = (
        condition_dir / "skill_state/homecheck/scans/initial/findings.json"
    )
    initial_findings = read_json(initial_findings_path)
    manifest_initial = manifest.get("initial_scan")
    if not isinstance(manifest_initial, dict):
        warnings.append(
            "run_manifest omitted initial_scan after resume; recovered from immutable "
            "HomeCheck state"
        )
    initial_hash = (
        manifest.get("initial_target_findings_sha256")
        or scanner_initial["findings_sha256"]
    )
    initial_tree_hash = manifest.get("input_tree_sha256")
    if initial_tree_hash is None:
        warnings.append(
            "run_manifest omitted input_tree_sha256 after resume; verified the pinned "
            "source tree independently"
        )
    return (
        {
            "finding_count": len(initial_findings),
            "findings_sha256": initial_hash,
            "scanner_findings_sha256": scanner_initial["findings_sha256"],
            "per_rule": per_rule_counts(initial_findings),
            "input_tree_sha256": initial_tree_hash,
            "findings_path": str(initial_findings_path),
            "scanner_state_path": str(scanner_path),
        },
        warnings,
    )


def final_evidence(condition_dir: Path) -> dict[str, Any]:
    base = condition_dir / "skill_state/homecheck/scans/final"
    findings_path = base / "findings.json"
    deltas_path = base / "alert_deltas.json"
    findings = read_json(findings_path)
    deltas = read_json(deltas_path)
    return {
        "finding_count": len(findings),
        "per_rule": per_rule_counts(findings),
        "findings": findings,
        "deltas": deltas,
        "findings_path": str(findings_path),
        "findings_sha256": sha256_file(findings_path),
        "deltas_path": str(deltas_path),
        "deltas_sha256": sha256_file(deltas_path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {field: sum(int(row[field]) for row in rows) for field in METRIC_FIELDS}
    observed_initial_rules = [
        value["eliminated_alerts"] / value["initial_alerts"]
        for value in aggregate_rules(rows)
        if value["initial_alerts"]
    ]
    return {
        "project_count": len(rows),
        **metrics,
        "gross_elimination_rate": (
            metrics["eliminated_alerts"] / metrics["initial_alerts"]
        ),
        "net_reduction_rate": metrics["net_reduction"] / metrics["initial_alerts"],
        "projects_with_zero_final_alerts": sum(
            row["final_alerts"] == 0 for row in rows
        ),
        "projects_with_introduced_alerts": sum(
            row["introduced_alerts"] > 0 for row in rows
        ),
        "validation_scans": sum(int(row["validation_scans"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "cached_input_tokens": sum(int(row["cached_input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "wall_clock_seconds_sum": sum(float(row["wall_clock_seconds"]) for row in rows),
        "traceable_api_cost": None,
        "cost_per_eliminated_alert": None,
        "cost_limitation": "Provider billing was not exposed in the Codex traces.",
        "per_rule_micro_elimination_rate": (
            metrics["eliminated_alerts"] / metrics["initial_alerts"]
        ),
        "per_rule_macro_elimination_rate": mean(observed_initial_rules),
    }


def aggregate_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    projects: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for delta in row["per_rule_alert_deltas"]:
            rule = delta["rule"]
            eliminated = int(delta["eliminated_alerts"])
            remaining = int(delta["remaining_alerts"])
            introduced = int(delta["introduced_alerts"])
            totals[rule].update(
                {
                    "initial_alerts": eliminated + remaining,
                    "eliminated_alerts": eliminated,
                    "remaining_alerts": remaining,
                    "introduced_alerts": introduced,
                    "final_alerts": remaining + introduced,
                    "net_reduction": eliminated - introduced,
                }
            )
            projects[rule].add(row["project"])
    result: list[dict[str, Any]] = []
    for rule in sorted(totals):
        item = totals[rule]
        initial = item["initial_alerts"]
        result.append(
            {
                "rule": rule,
                "category": category(rule),
                "project_count": len(projects[rule]),
                **{field: item[field] for field in METRIC_FIELDS},
                "gross_elimination_rate": (
                    item["eliminated_alerts"] / initial if initial else None
                ),
                "net_reduction_rate": item["net_reduction"] / initial
                if initial
                else None,
            }
        )
    return result


def aggregate_stratum(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return [
        {"stratum": value, **aggregate_rows(groups[value])} for value in sorted(groups)
    ]


def reconciliation_check(
    check_id: str, passed: bool, evidence: Any, severity: str = "error"
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": passed,
        "severity": severity,
        "evidence": evidence,
    }


def make_project_row(
    *,
    frozen: dict[str, Any],
    condition_dir: Path,
    evidence_source: str,
    expected_protocol_hash: str,
    expected_input_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = condition_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    initial, warnings = resolved_initial_evidence(manifest, condition_dir)
    final = final_evidence(condition_dir)
    metrics = {field: int(manifest["alert_metrics"][field]) for field in METRIC_FIELDS}
    source_tree_hash = sha256_tree(tree_manifest(Path(frozen["source_path"])))
    accesses = guide_access(condition_dir)
    checks = [
        reconciliation_check(
            "project_identity",
            manifest.get("project") == frozen["name"],
            {"manifest": manifest.get("project"), "frozen": frozen["name"]},
        ),
        reconciliation_check(
            "condition_status",
            manifest.get("status") == "completed",
            manifest.get("status"),
        ),
        reconciliation_check(
            "protocol_hash",
            manifest.get("protocol_sha256") == expected_protocol_hash,
            manifest.get("protocol_sha256"),
        ),
        reconciliation_check(
            "input_manifest_hash",
            manifest.get("project_manifest_sha256") == expected_input_hash,
            manifest.get("project_manifest_sha256"),
        ),
        reconciliation_check(
            "commit_identity",
            manifest.get("commit") == frozen.get("commit"),
            {"manifest": manifest.get("commit"), "frozen": frozen.get("commit")},
        ),
        reconciliation_check(
            "tree_oid_identity",
            manifest.get("tree_oid") == frozen.get("tree_oid"),
            {"manifest": manifest.get("tree_oid"), "frozen": frozen.get("tree_oid")},
        ),
        reconciliation_check(
            "pinned_source_tree_hash",
            source_tree_hash == frozen["frozen_input_tree_sha256"],
            {
                "observed": source_tree_hash,
                "frozen": frozen["frozen_input_tree_sha256"],
            },
        ),
        reconciliation_check(
            "run_input_tree_hash",
            manifest.get("input_tree_sha256")
            in {
                None,
                frozen["frozen_input_tree_sha256"],
            },
            {
                "manifest": manifest.get("input_tree_sha256"),
                "frozen": frozen["frozen_input_tree_sha256"],
                "null_reason": warnings or None,
            },
        ),
        reconciliation_check(
            "initial_finding_hash",
            initial["findings_sha256"] == frozen["frozen_target_findings_sha256"]
            and initial["scanner_findings_sha256"]
            == frozen["frozen_target_findings_sha256"],
            {
                "resolved": initial["findings_sha256"],
                "scanner": initial["scanner_findings_sha256"],
                "frozen": frozen["frozen_target_findings_sha256"],
            },
        ),
        reconciliation_check(
            "initial_finding_count",
            initial["finding_count"]
            == frozen["frozen_target_finding_count"]
            == metrics["initial_alerts"],
            {
                "observed": initial["finding_count"],
                "frozen": frozen["frozen_target_finding_count"],
                "metrics": metrics["initial_alerts"],
            },
        ),
        reconciliation_check(
            "final_finding_count",
            final["finding_count"] == metrics["final_alerts"],
            {"observed": final["finding_count"], "metrics": metrics["final_alerts"]},
        ),
        reconciliation_check(
            "metric_invariants",
            not metric_invariants(metrics),
            metric_invariants(metrics),
        ),
        reconciliation_check(
            "restricted_commands",
            not manifest.get("restricted_accesses"),
            manifest.get("restricted_accesses") or [],
        ),
        reconciliation_check(
            "semantic_spec_protocol",
            manifest.get("spec_protocol_violation") is False,
            manifest.get("spec_protocol_violation"),
        ),
        reconciliation_check(
            "paper_facing",
            manifest.get("paper_facing") is True,
            manifest.get("paper_facing"),
        ),
        reconciliation_check(
            "guide_access_complete",
            accesses["complete"],
            {
                "accessed": accesses["accessed_rule_count"],
                "required": accesses["required_rule_count"],
                "missing_rules": accesses["missing_rules"],
            },
            severity="warning",
        ),
    ]
    nonblank_loc = int(frozen["nonblank_source_loc"])
    row = {
        "project": frozen["name"],
        "evidence_source": evidence_source,
        "development_exposed": frozen["name"] == "wifi_testapp",
        "selection_reason": frozen["selection_reason"],
        "source_provider": frozen["source_provider"],
        "namespace": frozen["namespace"],
        "commit": frozen["commit"],
        "tree_oid": frozen["tree_oid"],
        "nonblank_source_loc": nonblank_loc,
        "initial_alert_density_per_nonblank_kloc": metrics["initial_alerts"]
        / (nonblank_loc / 1000),
        "alert_volume_stratum": alert_volume(metrics["initial_alerts"]),
        **metrics,
        "gross_elimination_rate": metrics["eliminated_alerts"]
        / metrics["initial_alerts"],
        "net_reduction_rate": metrics["net_reduction"] / metrics["initial_alerts"],
        "validation_scans": int(manifest["validation_scan_count"]),
        "maximum_validation_scans": int(manifest["maximum_validation_scans"]),
        "best_valid_round": manifest.get("best_valid_round"),
        "last_valid_round": manifest.get("last_valid_round"),
        "final_selection_status": manifest["final_candidate_selection"]["status"],
        "build_status": manifest["build_status"],
        "test_status": manifest["test_status"],
        **{field: int(manifest.get(field) or 0) for field in TOKEN_FIELDS},
        "api_cost": manifest.get("api_cost"),
        "wall_clock_seconds": float(manifest["wall_clock_seconds"]),
        "guide_rules_required": accesses["required_rule_count"],
        "guide_rules_trace_accessed": accesses["accessed_rule_count"],
        "guide_access_complete": accesses["complete"],
        "initial_per_rule": initial["per_rule"],
        "final_per_rule": final["per_rule"],
        "per_rule_alert_deltas": manifest["per_rule_alert_deltas"],
        "observed_rule_interactions": manifest["observed_rule_interactions"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "warnings": warnings,
    }
    audit = {
        "project": frozen["name"],
        "evidence_source": evidence_source,
        "status": "passed"
        if all(item["passed"] for item in checks if item["severity"] == "error")
        else "failed",
        "checks": checks,
        "warnings": warnings,
        "guide_access": accesses,
        "evidence": {
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "initial_findings": initial["findings_path"],
            "final_findings": final["findings_path"],
            "final_findings_sha256": final["findings_sha256"],
            "final_alert_deltas": final["deltas_path"],
            "final_alert_deltas_sha256": final["deltas_sha256"],
        },
    }
    return row, audit, {"findings": final["findings"], "deltas": final["deltas"]}


def csv_project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    omitted = {
        "initial_per_rule",
        "final_per_rule",
        "per_rule_alert_deltas",
        "observed_rule_interactions",
        "warnings",
    }
    return [
        {key: value for key, value in row.items() if key not in omitted} for row in rows
    ]


def interaction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for interaction in row["observed_rule_interactions"]:
            result.append(
                {
                    "project": row["project"],
                    "round": interaction["round"],
                    "provisional": interaction["provisional"],
                    "eliminated_rules": interaction["eliminated_rules"],
                    "introduced_rules": interaction["introduced_rules"],
                    "sibling_edge_count": len(interaction["sibling_edges"]),
                    "has_unresolved_sibling_exchange": interaction[
                        "has_unresolved_sibling_exchange"
                    ],
                    "interpretation": interaction["interpretation"],
                }
            )
    return result


def residual_rows(
    rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        project_evidence = evidence[row["project"]]
        introduced = Counter(
            alert_identity(item) for item in project_evidence["deltas"]["introduced"]
        )
        remaining = Counter(
            alert_identity(item) for item in project_evidence["deltas"]["remaining"]
        )
        for finding in project_evidence["findings"]:
            identity = alert_identity(finding)
            if introduced[identity]:
                classification = "introduced"
                introduced[identity] -= 1
            elif remaining[identity]:
                classification = "remaining"
                remaining[identity] -= 1
            else:
                raise ValueError(
                    f"Final finding lacks delta identity for {row['project']}: {identity}"
                )
            result.append(
                {
                    "project": row["project"],
                    "classification": classification,
                    "relative_path": finding["relative_path"],
                    "line": finding.get("line"),
                    "column": finding.get("column"),
                    "severity": finding.get("severity"),
                    "rule": finding["rule"],
                    "message": finding["message"],
                    "claim_boundary": (
                        "HomeCheck-defined residual alert; not an adjudication of semantic "
                        "correctness or vulnerability status."
                    ),
                }
            )
        if sum(int(value) for value in introduced.values()) or sum(
            int(value) for value in remaining.values()
        ):
            raise ValueError(f"Unmatched final alert deltas for {row['project']}")
    return result


def guide_summary(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]]
) -> dict[str, Any]:
    required = sum(row["guide_rules_required"] for row in rows)
    accessed = sum(row["guide_rules_trace_accessed"] for row in rows)
    missing_by_rule: Counter[str] = Counter()
    for audit in audits:
        missing_by_rule.update(audit["guide_access"]["missing_rules"])
    return {
        "project_count": len(rows),
        "projects_with_complete_initial_rule_guide_access": sum(
            row["guide_access_complete"] for row in rows
        ),
        "initial_project_rule_requirements": required,
        "trace_accessed_initial_project_rules": accessed,
        "trace_access_rate": accessed / required,
        "missing_access_by_rule": dict(sorted(missing_by_rule.items())),
        "criterion": audits[0]["guide_access"]["criterion"],
        "interpretation": (
            "This is trace-supported access evidence only. It does not prove that a guide "
            "caused a repair, and missing access must not be described as guide use."
        ),
    }


def summary_markdown(aggregate: dict[str, Any], guide: dict[str, Any]) -> str:
    full = aggregate["full_35"]
    sensitivity = aggregate["sensitivity_excluding_wifi_testapp"]
    return f"""# E1 v14 Reconciled Summary

The reconciled dataset contains 34 valid manifests from the original v14 scheduler
and the independently audited `flutter_embedding` retry. The original invalid
`flutter_embedding` condition remains preserved and excluded.

## Main descriptive result

- Projects: {full["project_count"]}
- Alerts: {full["initial_alerts"]} initial, {full["final_alerts"]} final,
  {full["eliminated_alerts"]} eliminated, {full["remaining_alerts"]} remaining,
  {full["introduced_alerts"]} introduced, and {full["net_reduction"]} net reduced
- Gross detected-alert elimination: {full["gross_elimination_rate"]:.4%}
- Net detected-alert reduction: {full["net_reduction_rate"]:.4%}
- Zero-final-alert projects: {full["projects_with_zero_final_alerts"]}
- Validation scans: {full["validation_scans"]}
- Total trace tokens: {full["total_tokens"]}
- API cost: unavailable because provider billing was not exposed

## Sensitivity result

Excluding development-exposed `wifi_testapp`: {sensitivity["project_count"]} projects,
{sensitivity["initial_alerts"]} initial alerts, {sensitivity["final_alerts"]} final
alerts, and {sensitivity["net_reduction_rate"]:.4%} net detected-alert reduction.

## Static-guide access boundary

Exact guide filenames appeared in completed command traces for
{guide["trace_accessed_initial_project_rules"]} of
{guide["initial_project_rule_requirements"]} initial project-rule requirements
({guide["trace_access_rate"]:.4%}). Only
{guide["projects_with_complete_initial_rule_guide_access"]} projects had complete
trace-supported initial-rule guide access. This result must be reported as observed;
opening semantic specifications or reading the generated plan is not counted as
opening a bundled static guide.

All alert results are HomeCheck-defined detected-alert outcomes. They do not by
themselves establish semantic correctness or confirmed vulnerability repair.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs_path = args.inputs.resolve()
    protocol_path = args.protocol.resolve()
    primary_run = args.primary_run.resolve()
    retry_run = args.retry_run.resolve()
    output = args.output.resolve()
    inputs = read_json(inputs_path)
    projects = inputs["projects"]
    frozen_by_name = {item["name"]: item for item in projects}
    expected_names = set(frozen_by_name)
    if len(projects) != 35 or len(expected_names) != 35:
        raise ValueError("The frozen input manifest must contain 35 unique projects")
    protocol_hash = sha256_file(protocol_path)
    input_hash = sha256_file(inputs_path)
    scheduler_path = primary_run / "formal_scheduler/manifest.json"
    scheduler = read_json(scheduler_path)
    scheduler_tasks = {item["project"]: item for item in scheduler["tasks"]}
    if set(scheduler_tasks) != expected_names:
        raise ValueError("Scheduler project membership differs from frozen inputs")
    if scheduler.get("status") != "completed_with_protocol_violations":
        raise ValueError("Unexpected original scheduler status")
    completed = sorted(
        name for name, item in scheduler_tasks.items() if item["status"] == "completed"
    )
    violations = sorted(
        name
        for name, item in scheduler_tasks.items()
        if item["status"] == "protocol_violation"
    )
    if len(completed) != 34 or violations != ["flutter_embedding"]:
        raise ValueError(
            "Expected 34 scheduler completions and flutter protocol violation"
        )

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    final_evidence_by_project: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        condition_dir, source = source_for_project(name, primary_run, retry_run)
        row, audit, final_evidence_item = make_project_row(
            frozen=frozen_by_name[name],
            condition_dir=condition_dir,
            evidence_source=source,
            expected_protocol_hash=protocol_hash,
            expected_input_hash=input_hash,
        )
        rows.append(row)
        audits.append(audit)
        final_evidence_by_project[name] = final_evidence_item

    failed_projects = [item["project"] for item in audits if item["status"] != "passed"]
    aggregate = {
        "full_35": aggregate_rows(rows),
        "sensitivity_excluding_wifi_testapp": aggregate_rows(
            [row for row in rows if row["project"] != "wifi_testapp"]
        ),
        "claim_boundary": (
            "Detected-alert reduction under HomeCheck; not semantic correctness, "
            "real-world recall, or confirmed vulnerability repair."
        ),
    }
    rules = aggregate_rules(rows)
    residuals = residual_rows(rows, final_evidence_by_project)
    interactions = interaction_rows(rows)
    guide = guide_summary(rows, audits)
    strata = {
        "definitions": {
            "selection_reason": "Frozen outcome-independent selection reason.",
            "alert_volume_stratum": "Initial alerts: <100, 100-499, or >=500.",
            "build_status": "Evaluator build availability/outcome recorded by the run.",
        },
        "by_selection_reason": aggregate_stratum(rows, "selection_reason"),
        "by_alert_volume": aggregate_stratum(rows, "alert_volume_stratum"),
        "by_build_status": aggregate_stratum(rows, "build_status"),
    }
    audit = {
        "schema_version": 1,
        "status": "passed" if not failed_projects else "failed",
        "experiment": "EXP-HAPSKILL-35-v14-E1-reconciliation",
        "join_policy": (
            "Use the 34 completed primary-scheduler conditions and replace only the "
            "preserved invalid flutter_embedding row with its independently audited retry."
        ),
        "inputs": {
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "project_manifest": str(inputs_path),
            "project_manifest_sha256": input_hash,
            "primary_scheduler": str(scheduler_path),
            "primary_scheduler_sha256": sha256_file(scheduler_path),
            "retry_run": str(retry_run),
        },
        "scheduler_evidence": {
            "status": scheduler["status"],
            "completed_project_count": len(completed),
            "protocol_violation_projects": violations,
        },
        "included_project_count": len(rows),
        "failed_projects": failed_projects,
        "projects": audits,
    }

    write_json(output / "reconciliation_audit.json", audit)
    write_json(output / "aggregate.json", aggregate)
    write_json(output / "projects.json", rows)
    write_csv(output / "projects.csv", csv_project_rows(rows))
    write_json(output / "rules.json", rules)
    write_csv(output / "rules.csv", rules)
    write_json(output / "residual_findings.json", residuals)
    write_csv(output / "residual_findings.csv", residuals)
    write_json(output / "interactions.json", interactions)
    write_csv(output / "interactions.csv", interactions)
    write_json(output / "project_strata.json", strata)
    write_json(output / "guide_access.json", guide)
    (output / "summary.md").write_text(
        summary_markdown(aggregate, guide), encoding="utf-8"
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "artifact_count": len(output_hashes),
        "artifacts": output_hashes,
    }
    write_json(output / "manifest.json", manifest)
    if failed_projects:
        raise RuntimeError(
            "E1 reconciliation failed for: " + ", ".join(failed_projects)
        )
    return {"audit": audit, "aggregate": aggregate, "guide_access": guide}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--primary-run", type=Path, default=DEFAULT_PRIMARY_RUN)
    parser.add_argument("--retry-run", type=Path, default=DEFAULT_RETRY_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
