#!/usr/bin/env python3
"""Build the audited formal_03 versus v14 Skill paired analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "paper/rebuttal/agent_ref_v14_pairing"
E1_ROOT = WORKSPACE_ROOT / "paper/rebuttal/e1_v14"
REFERENCE_ROOT = (
    WORKSPACE_ROOT
    / "baseline_data/exp_agent_ref_10/runs/exp_agent_ref_10_luna_v4_formal_03"
)
INPUTS = HERE / "formal_inputs_agent_ref_10_v4.json"
REFERENCE_AUDIT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_results_audit_20260807.json"
)
REFERENCE_SUMMARY = (
    WORKSPACE_ROOT
    / "paper/rebuttal/results/exp_agent_ref_10_v4_formal_03_evaluation_summary_20260807.json"
)
SCHEDULER = REFERENCE_ROOT / "formal_scheduler/manifest.json"
PACKAGE_ID = "AN-AGENT-REF-V14-PAIRING-20260807"
METRIC_FIELDS = (
    "initial_alerts",
    "final_alerts",
    "eliminated_alerts",
    "remaining_alerts",
    "introduced_alerts",
    "net_reduction",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric_invariants(metrics: dict[str, int]) -> list[str]:
    problems: list[str] = []
    if metrics["eliminated_alerts"] + metrics["remaining_alerts"] != metrics[
        "initial_alerts"
    ]:
        problems.append("initial != eliminated + remaining")
    if metrics["remaining_alerts"] + metrics["introduced_alerts"] != metrics[
        "final_alerts"
    ]:
        problems.append("final != remaining + introduced")
    if metrics["eliminated_alerts"] - metrics["introduced_alerts"] != metrics[
        "net_reduction"
    ]:
        problems.append("net != eliminated - introduced")
    return problems


def aggregate_metrics(rows: Iterable[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = list(rows)
    metrics = {
        field: sum(int(row[condition][field]) for row in selected)
        for field in METRIC_FIELDS
    }
    return {
        "project_count": len(selected),
        **metrics,
        "gross_elimination_rate": rate(
            metrics["eliminated_alerts"], metrics["initial_alerts"]
        ),
        "net_reduction_rate": rate(metrics["net_reduction"], metrics["initial_alerts"]),
        "projects_with_zero_final_alerts": sum(
            row[condition]["final_alerts"] == 0 for row in selected
        ),
        "projects_with_introduced_alerts": sum(
            row[condition]["introduced_alerts"] > 0 for row in selected
        ),
        "validation_scans": sum(
            int(row[condition]["validation_scans"]) for row in selected
        ),
        "total_tokens": sum(int(row[condition]["total_tokens"]) for row in selected),
        "wall_clock_seconds_sum": sum(
            float(row[condition]["wall_clock_seconds"]) for row in selected
        ),
    }


def paired_aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    skill = aggregate_metrics(selected, "skill")
    reference = aggregate_metrics(selected, "reference")
    return {
        "denominator": "Only the seven projects with terminal completed outputs in both conditions.",
        "skill": skill,
        "reference": reference,
        "reference_minus_skill": {
            field: reference[field] - skill[field]
            for field in (
                "final_alerts",
                "eliminated_alerts",
                "remaining_alerts",
                "introduced_alerts",
                "net_reduction",
                "validation_scans",
                "total_tokens",
                "wall_clock_seconds_sum",
            )
        },
    }


def condition_metrics(
    metrics: dict[str, Any], manifest: dict[str, Any], validation_key: str
) -> dict[str, Any]:
    result = {field: int(metrics[field]) for field in METRIC_FIELDS}
    return {
        **result,
        "gross_elimination_rate": rate(
            result["eliminated_alerts"], result["initial_alerts"]
        ),
        "net_reduction_rate": rate(
            result["net_reduction"], result["initial_alerts"]
        ),
        "validation_scans": int(manifest[validation_key]),
        "input_tokens": int(manifest["input_tokens"]),
        "cached_input_tokens": int(manifest["cached_input_tokens"]),
        "output_tokens": int(manifest["output_tokens"]),
        "total_tokens": int(manifest["total_tokens"]),
        "api_cost": manifest.get("api_cost_if_traceable", manifest.get("api_cost")),
        "wall_clock_seconds": float(manifest["wall_clock_seconds"]),
    }


def per_rule_map(deltas: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        item["rule"]: {
            "eliminated_alerts": int(item["eliminated_alerts"]),
            "remaining_alerts": int(item["remaining_alerts"]),
            "introduced_alerts": int(item["introduced_alerts"]),
        }
        for item in deltas
    }


def load_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(INPUTS)
    return payload, {item["name"]: item for item in payload["projects"]}


def source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_project_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    e1_rows = {row["project"]: row for row in read_json(E1_ROOT / "projects.json")}
    e1_audit_rows = {
        row["project"]: row
        for row in read_json(E1_ROOT / "reconciliation_audit.json")["projects"]
    }
    _, input_rows = load_inputs()
    scheduler = read_json(SCHEDULER)
    scheduler_tasks = {task["project"]: task for task in scheduler["tasks"]}
    failure_audit = next(
        check
        for check in read_json(REFERENCE_AUDIT)["checks"]
        if check["name"] == "three_genuine_bounded_guard_failures"
    )
    failure_rows = {row["project"]: row for row in failure_audit["projects"]}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    problems: list[str] = []

    for ordinal, input_row in enumerate(read_json(INPUTS)["projects"], start=1):
        project = input_row["name"]
        e1 = e1_rows.get(project)
        task = scheduler_tasks.get(project)
        if e1 is None or task is None:
            problems.append(f"{project}: missing E1 row or scheduler task")
            continue
        skill_path = Path(e1["run_manifest_path"])
        reference_path = REFERENCE_ROOT / "vanilla" / project / "run_manifest.json"
        skill_manifest = read_json(skill_path)
        reference_manifest = read_json(reference_path)
        frozen_tree = input_rows[project]["frozen_input_tree_sha256"]
        frozen_findings = input_rows[project]["frozen_target_findings_sha256"]
        e1_checks = {
            check["id"]: check for check in e1_audit_rows[project]["checks"]
        }
        audited_skill_tree = e1_checks["pinned_source_tree_hash"]["evidence"][
            "observed"
        ]
        audited_skill_findings = e1_checks["initial_finding_hash"]["evidence"][
            "resolved"
        ]
        skill_tree = skill_manifest.get("input_tree_sha256") or audited_skill_tree
        skill_findings = (
            skill_manifest.get("initial_target_findings_sha256")
            or audited_skill_findings
        )
        audited_failure = failure_rows.get(project)
        expected_reference_manifest_hash = task.get("runner_manifest_sha256") or (
            audited_failure.get("manifest_sha256") if audited_failure else None
        )
        hash_checks = {
            "skill_manifest_matches_e1_hash": sha256_file(skill_path)
            == e1["run_manifest_sha256"],
            "reference_manifest_matches_terminal_audit_hash": (
                expected_reference_manifest_hash == sha256_file(reference_path)
            ),
            "skill_status_is_completed": skill_manifest.get("status") == "completed",
            "reference_status_matches_scheduler": reference_manifest.get("status")
            == task["status"],
            "skill_frozen_tree_matches": skill_tree == frozen_tree,
            "reference_frozen_tree_matches": reference_manifest.get(
                "frozen_source_tree_sha256"
            )
            == frozen_tree,
            "skill_frozen_findings_match": skill_findings == frozen_findings,
            "reference_frozen_findings_match": reference_manifest.get(
                "frozen_findings_sha256"
            )
            == frozen_findings,
            "reference_byte_identical_input": reference_manifest.get(
                "byte_identical_input"
            )
            is True,
            "reference_finding_identity_matches": reference_manifest.get(
                "finding_identity_matches"
            )
            is True,
        }
        failed_hashes = [name for name, passed in hash_checks.items() if not passed]
        if failed_hashes:
            problems.append(f"{project}: failed hash checks: {', '.join(failed_hashes)}")

        common = {
            "ordinal": ordinal,
            "project": project,
            "development_exposed": bool(e1["development_exposed"]),
            "selection_reason": e1["selection_reason"],
            "alert_volume_stratum": e1["alert_volume_stratum"],
            "frozen_input": {
                "tree_sha256": frozen_tree,
                "findings_sha256": frozen_findings,
                "finding_count": int(input_row["frozen_target_finding_count"]),
            },
            "skill_frozen_input_evidence": {
                "tree_sha256": skill_tree,
                "tree_source": (
                    "run_manifest"
                    if skill_manifest.get("input_tree_sha256")
                    else "v14_reconciliation_audit"
                ),
                "findings_sha256": skill_findings,
                "findings_source": (
                    "run_manifest"
                    if skill_manifest.get("initial_target_findings_sha256")
                    else "v14_reconciliation_audit"
                ),
            },
            "hash_checks": hash_checks,
            "skill_manifest": source_record(skill_path),
            "reference_manifest": source_record(reference_path),
        }
        if task["status"] == "completed":
            skill_metrics = condition_metrics(
                skill_manifest["alert_metrics"], skill_manifest, "validation_scan_count"
            )
            reference_metrics = condition_metrics(
                reference_manifest["alert_metrics"],
                reference_manifest,
                "validation_scan_count",
            )
            if skill_metrics["initial_alerts"] != reference_metrics["initial_alerts"]:
                problems.append(f"{project}: paired initial alert counts differ")
            for label, metrics in (
                ("skill", skill_metrics),
                ("reference", reference_metrics),
            ):
                invariant_problems = metric_invariants(metrics)
                if invariant_problems:
                    problems.append(
                        f"{project}/{label}: {', '.join(invariant_problems)}"
                    )
            skill_rules = skill_manifest.get("per_rule_alert_deltas", [])
            reference_rules = reference_manifest.get("per_rule_alert_deltas", [])
            rows.append(
                {
                    **common,
                    "pair_status": "paired_completed",
                    "comparability": "direct_alert_metric_comparison",
                    "skill": skill_metrics,
                    "reference": reference_metrics,
                    "differences_reference_minus_skill": {
                        field: reference_metrics[field] - skill_metrics[field]
                        for field in METRIC_FIELDS[1:]
                    },
                    "skill_per_rule": per_rule_map(skill_rules),
                    "reference_per_rule": per_rule_map(reference_rules),
                }
            )
        else:
            audited = audited_failure
            if audited is None:
                problems.append(f"{project}: scheduler failure lacks failure audit")
                continue
            skill_metrics = condition_metrics(
                skill_manifest["alert_metrics"], skill_manifest, "validation_scan_count"
            )
            failure = {
                **common,
                "pair_status": "reference_bounded_guard_failure",
                "comparability": "not_numerically_paired",
                "reference_status": "failed",
                "reference_failure": audited["failure"],
                "failed_round": audited["failed_round"],
                "attempt_count": audited["attempt_count"],
                "accepted_rounds_before_failure": audited[
                    "accepted_rounds_before_failure"
                ],
                "all_attempts_coverage_complete": audited[
                    "all_attempts_coverage_complete"
                ],
                "all_attempts_guard_failed": audited["all_attempts_guard_failed"],
                "reference_wall_clock_seconds_until_failure": float(
                    reference_manifest["wall_clock_seconds"]
                ),
                "reference_effectiveness_metrics": None,
                "skill_context_only": skill_metrics,
                "denominator_policy": (
                    "The reference failure remains a failure. Skill metrics are context "
                    "only and no zero or synthetic reference metric is imputed."
                ),
            }
            rows.append(failure)
            failures.append(failure)
    return rows, failures, problems


def build_rules(completed: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "projects": set(),
            "initial_alerts": 0,
            "skill_eliminated_alerts": 0,
            "skill_remaining_alerts": 0,
            "skill_introduced_alerts": 0,
            "reference_eliminated_alerts": 0,
            "reference_remaining_alerts": 0,
            "reference_introduced_alerts": 0,
        }
    )
    for row in completed:
        rules = set(row["skill_per_rule"]) | set(row["reference_per_rule"])
        for rule in rules:
            skill = row["skill_per_rule"].get(
                rule,
                {"eliminated_alerts": 0, "remaining_alerts": 0, "introduced_alerts": 0},
            )
            reference = row["reference_per_rule"].get(
                rule,
                {"eliminated_alerts": 0, "remaining_alerts": 0, "introduced_alerts": 0},
            )
            skill_initial = skill["eliminated_alerts"] + skill["remaining_alerts"]
            reference_initial = (
                reference["eliminated_alerts"] + reference["remaining_alerts"]
            )
            if skill_initial != reference_initial:
                raise RuntimeError(
                    f"{row['project']}/{rule}: per-rule initial support differs"
                )
            target = aggregate[rule]
            if skill_initial:
                target["projects"].add(row["project"])
                target["initial_alerts"] += skill_initial
            for condition, values in (("skill", skill), ("reference", reference)):
                for field in ("eliminated_alerts", "remaining_alerts", "introduced_alerts"):
                    target[f"{condition}_{field}"] += values[field]

    common: list[dict[str, Any]] = []
    introduced_only: list[dict[str, Any]] = []
    for rule in sorted(aggregate):
        item = aggregate[rule]
        initial = item["initial_alerts"]
        row = {
            "rule": rule,
            "category": (
                "security"
                if rule.startswith("@security/")
                else "performance"
                if rule.startswith("@performance/")
                else "other"
            ),
            "project_count": len(item["projects"]),
            "initial_alerts": initial,
            "skill_eliminated_alerts": item["skill_eliminated_alerts"],
            "skill_remaining_alerts": item["skill_remaining_alerts"],
            "skill_introduced_alerts": item["skill_introduced_alerts"],
            "skill_gross_elimination_rate": rate(
                item["skill_eliminated_alerts"], initial
            ),
            "reference_eliminated_alerts": item["reference_eliminated_alerts"],
            "reference_remaining_alerts": item["reference_remaining_alerts"],
            "reference_introduced_alerts": item["reference_introduced_alerts"],
            "reference_gross_elimination_rate": rate(
                item["reference_eliminated_alerts"], initial
            ),
        }
        if initial:
            row["support"] = "common_initial_support"
            common.append(row)
        else:
            row["support"] = "introduced_only"
            introduced_only.append(row)
    return {
        "denominator": (
            "Per-rule outcomes use only the seven completed paired projects and exact "
            "common initial rule support. Introduced-only rules are separate."
        ),
        "common_support": common,
        "introduced_only": introduced_only,
    }


def build_strata(completed: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[row[key]].append(row)
    return [
        {"stratum": value, **paired_aggregate(grouped[value])}
        for value in sorted(grouped)
    ]


def build_efficiency(completed: list[dict[str, Any]]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in ("skill", "reference"):
        tokens = [int(row[condition]["total_tokens"]) for row in completed]
        walls = [float(row[condition]["wall_clock_seconds"]) for row in completed]
        eliminated = sum(int(row[condition]["eliminated_alerts"]) for row in completed)
        conditions[condition] = {
            "project_count": len(completed),
            "total_tokens": sum(tokens),
            "median_project_tokens": median(tokens),
            "tokens_per_eliminated_alert": rate(sum(tokens), eliminated),
            "wall_clock_seconds_sum": sum(walls),
            "median_project_wall_clock_seconds": median(walls),
            "billed_cost": None,
            "cost_per_eliminated_alert": None,
            "cost_limitation": "Provider billing was not exposed in the Codex traces.",
        }
    permission = next(
        row for row in completed if row["project"] == "applications_permission_manager"
    )
    return {
        "denominator": "Seven completed paired projects; failed reference conditions excluded.",
        "conditions": conditions,
        "reference_minus_skill": {
            "total_tokens": conditions["reference"]["total_tokens"]
            - conditions["skill"]["total_tokens"],
            "wall_clock_seconds_sum": conditions["reference"]["wall_clock_seconds_sum"]
            - conditions["skill"]["wall_clock_seconds_sum"],
        },
        "wall_clock_comparability": {
            "verdict": "descriptive_only",
            "reason": (
                "The reference applications_permission_manager wall clock includes a "
                "host-reboot interruption, so aggregate wall time is not a clean runtime "
                "comparison."
            ),
            "affected_project": permission["project"],
            "recorded_reference_wall_clock_seconds": permission["reference"][
                "wall_clock_seconds"
            ],
        },
        "token_comparability": (
            "Trace-derived token totals are available for both completed conditions, but "
            "token accounting reflects each condition's actual interaction trajectory."
        ),
    }


def write_projects_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "project",
        "pair_status",
        "development_exposed",
        "selection_reason",
        "alert_volume_stratum",
        "skill_initial_alerts",
        "skill_final_alerts",
        "skill_eliminated_alerts",
        "skill_introduced_alerts",
        "skill_total_tokens",
        "skill_wall_clock_seconds",
        "reference_initial_alerts",
        "reference_final_alerts",
        "reference_eliminated_alerts",
        "reference_introduced_alerts",
        "reference_total_tokens",
        "reference_wall_clock_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            skill = row.get("skill", row.get("skill_context_only", {}))
            reference = row.get("reference", {})
            writer.writerow(
                {
                    "project": row["project"],
                    "pair_status": row["pair_status"],
                    "development_exposed": row["development_exposed"],
                    "selection_reason": row["selection_reason"],
                    "alert_volume_stratum": row["alert_volume_stratum"],
                    **{f"skill_{key}": skill.get(key) for key in METRIC_FIELDS[:3]},
                    "skill_introduced_alerts": skill.get("introduced_alerts"),
                    "skill_total_tokens": skill.get("total_tokens"),
                    "skill_wall_clock_seconds": skill.get("wall_clock_seconds"),
                    **{
                        f"reference_{key}": reference.get(key)
                        for key in METRIC_FIELDS[:3]
                    },
                    "reference_introduced_alerts": reference.get("introduced_alerts"),
                    "reference_total_tokens": reference.get("total_tokens"),
                    "reference_wall_clock_seconds": reference.get("wall_clock_seconds"),
                }
            )


def write_rules_csv(path: Path, rules: dict[str, Any]) -> None:
    rows = rules["common_support"] + rules["introduced_only"]
    fields = list(rows[0]) if rows else ["rule"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_markdown(
    paired: dict[str, Any], sensitivity: dict[str, Any], failures: list[dict[str, Any]], efficiency: dict[str, Any]
) -> str:
    skill = paired["skill"]
    reference = paired["reference"]
    no_wifi = sensitivity["excluding_wifi_testapp"]
    return f"""# formal_03 vs v14 Skill paired analysis

## Manuscript takeaway

On the seven projects completed by both conditions, v14 Skill reduced {skill['initial_alerts']:,} detected alerts to {skill['final_alerts']:,}, while the contemporary reference agent reduced the same {reference['initial_alerts']:,} alerts to {reference['final_alerts']:,}. The reference introduced {reference['introduced_alerts']:,} alerts and the Skill introduced {skill['introduced_alerts']:,}. This is a bounded contextual comparison, not evidence of semantic correctness or universal superiority.

Excluding the development-exposed `wifi_testapp`, the paired denominator is {no_wifi['skill']['project_count']} projects and {no_wifi['skill']['initial_alerts']:,} initial alerts; final counts are {no_wifi['skill']['final_alerts']:,} for Skill and {no_wifi['reference']['final_alerts']:,} for the reference.

## Failure boundary

The reference failed on {len(failures)} of the ten frozen projects: {', '.join(row['project'] for row in failures)}. These bounded guard failures are retained as failures. They are excluded from numerical paired aggregates, and no zero or synthetic effectiveness values are assigned. Skill outcomes on those projects are context only.

## Efficiency boundary

Across the seven completed pairs, trace-derived token totals were {efficiency['conditions']['skill']['total_tokens']:,} for Skill and {efficiency['conditions']['reference']['total_tokens']:,} for the reference. Provider billing was unavailable for both. Aggregate wall time is descriptive only because the reference `applications_permission_manager` duration includes a host-reboot interruption.

## Comparability

All ten rows match the same frozen source-tree and initial-finding hashes. Direct alert-metric comparison is limited to the seven completed pairs. HomeCheck alert elimination measures disappearance of detected alerts, not semantic correctness, recall, or confirmed vulnerability repair. Both conditions are single runs and remain subject to nondeterminism.

## Route decision

The comparison, sensitivity, per-rule, failure, project-stratum, and efficiency slices are durable and audited. Route to manuscript/rebuttal writing with narrowed claims; do not rerun `formal_03` or impute the three failures.
"""


def build_manifest(output: Path, generated: list[Path]) -> dict[str, Any]:
    sources = [
        INPUTS,
        REFERENCE_AUDIT,
        REFERENCE_SUMMARY,
        SCHEDULER,
        E1_ROOT / "projects.json",
        E1_ROOT / "rules.json",
        E1_ROOT / "aggregate.json",
        E1_ROOT / "project_strata.json",
        Path(__file__).resolve(),
    ]
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "status": "complete",
        "created_date": "2026-08-07",
        "parent_object": {
            "experiment": "EXP-AGENT-REF-10",
            "run_id": "exp_agent_ref_10_luna_v4_formal_03",
            "result_audit": source_record(REFERENCE_AUDIT),
        },
        "comparison_target": {
            "experiment": "EXP-HAPSKILL-35",
            "run_id": "exp_hapskill_35_luna_v14_namespace_export_safe_01",
            "reconciliation": source_record(E1_ROOT / "reconciliation_audit.json"),
        },
        "research_question": (
            "How does the completed contemporary reference-agent subset compare with "
            "v14 Skill under the same frozen inputs, and what failure, sensitivity, "
            "per-rule, project-stratum, and efficiency boundaries must qualify it?"
        ),
        "stop_condition": (
            "Stop after the seven completed pairs, wifi sensitivity, three failures, "
            "per-rule common support, project strata, and trace efficiency are audited."
        ),
        "comparability": {
            "fixed": [
                "ten-project frozen selection",
                "source-tree hashes",
                "initial-finding identities and hashes",
                "HomeCheck alert metric contract",
                "maximum five validation scans",
            ],
            "changed": "Agent condition: v14 HapRepair Skill versus contemporary reference agent.",
            "direct_numeric_scope": "seven completed paired projects",
            "failed_condition_policy": "No imputation; failures remain visible and outside paired aggregates.",
        },
        "execution_envelope": {
            "route": "CPU-only deterministic post-processing of existing artifacts",
            "new_model_runs": 0,
            "condition_reruns": 0,
            "resource_class": "negligible relative to experiment execution",
        },
        "writing_mapping": {
            "paper_role": "main_text_and_appendix",
            "section_id": "evaluation-contemporary-reference",
            "item_id": "AN-AGENT-REF-V14-PAIRING",
            "claim_links": ["AE-C2", "R1-C10", "R3-C2", "R3-C5", "R3-C6", "R3-C7"],
            "analysis_role": "contemporary baseline, robustness, failure boundary, and efficiency",
            "reviewer_question": "Does HapRepair have a fair contemporary comparator with transparent failures and cost evidence?",
            "target_display": "Main comparison table plus appendix per-rule and failure tables",
            "failure_interpretation": "Three reference failures narrow the numerical denominator and must not be converted to zeros.",
        },
        "slices": [
            {"slice_id": "paired-completed", "class": "claim-carrying", "status": "completed"},
            {"slice_id": "wifi-sensitivity", "class": "claim-carrying", "status": "completed"},
            {"slice_id": "bounded-failures", "class": "claim-carrying", "status": "completed"},
            {"slice_id": "per-rule-common-support", "class": "supporting", "status": "completed"},
            {"slice_id": "project-strata", "class": "supporting", "status": "completed"},
            {"slice_id": "efficiency", "class": "supporting", "status": "completed"},
        ],
        "sources": [source_record(path) for path in sources],
        "outputs": [source_record(path) for path in generated],
        "next_route": "write",
    }


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reference_audit = read_json(REFERENCE_AUDIT)
    if reference_audit.get("status") != "closed" or reference_audit.get("passed") is not True:
        raise RuntimeError("formal_03 result audit is not closed and passing")
    e1_audit = read_json(E1_ROOT / "reconciliation_audit.json")
    e1_audit_passed = e1_audit.get("status") == "passed"
    if not e1_audit_passed:
        raise RuntimeError("v14 E1 reconciliation audit is not closed and passing")

    project_rows, failures, problems = build_project_rows()
    completed = [row for row in project_rows if row["pair_status"] == "paired_completed"]
    if len(completed) != 7 or len(failures) != 3:
        problems.append(
            f"terminal partition mismatch: {len(completed)} completed, {len(failures)} failed"
        )
    paired = paired_aggregate(completed)
    expected = reference_audit["completed_condition_aggregate"]
    if any(paired["reference"][key] != expected[key] for key in METRIC_FIELDS):
        problems.append("reference paired aggregate does not match formal_03 audit")

    sensitivity = {
        "full_completed_pair": paired,
        "excluding_wifi_testapp": paired_aggregate(
            [row for row in completed if row["project"] != "wifi_testapp"]
        ),
        "project_strata": {
            "by_selection_reason": build_strata(completed, "selection_reason"),
            "by_alert_volume": build_strata(completed, "alert_volume_stratum"),
        },
        "development_exposure_note": (
            "wifi_testapp was exposed during harness development; exclusion is a "
            "pre-specified sensitivity slice, not a replacement primary denominator."
        ),
    }
    rules = build_rules(completed)
    efficiency = build_efficiency(completed)
    failures_payload = {
        "failure_count": len(failures),
        "denominator": "Three of ten frozen reference conditions failed.",
        "failure_policy": "No effectiveness metrics are imputed for failed conditions.",
        "failures": failures,
    }

    projects_path = output / "projects.json"
    projects_csv = output / "projects.csv"
    rules_path = output / "rules.json"
    rules_csv = output / "rules.csv"
    failures_path = output / "failures.json"
    sensitivity_path = output / "sensitivity.json"
    efficiency_path = output / "efficiency.json"
    summary_path = output / "summary.md"
    write_json(projects_path, project_rows)
    write_projects_csv(projects_csv, project_rows)
    write_json(rules_path, rules)
    write_rules_csv(rules_csv, rules)
    write_json(failures_path, failures_payload)
    write_json(sensitivity_path, sensitivity)
    write_json(efficiency_path, efficiency)
    summary_path.write_text(
        summary_markdown(paired, sensitivity, failures, efficiency), encoding="utf-8"
    )

    generated = [
        projects_path,
        projects_csv,
        rules_path,
        rules_csv,
        failures_path,
        sensitivity_path,
        efficiency_path,
        summary_path,
    ]
    manifest_path = output / "manifest.json"
    write_json(manifest_path, build_manifest(output, generated))
    checks = [
        {
            "name": "source_audits_closed",
            "passed": reference_audit.get("passed") is True and e1_audit_passed,
        },
        {"name": "exact_terminal_partition_7_completed_3_failed", "passed": len(completed) == 7 and len(failures) == 3},
        {"name": "all_frozen_input_hash_checks_pass", "passed": all(all(row["hash_checks"].values()) for row in project_rows)},
        {"name": "no_failed_condition_imputation", "passed": all(row["reference_effectiveness_metrics"] is None for row in failures)},
        {"name": "reference_aggregate_matches_parent_audit", "passed": "reference paired aggregate does not match formal_03 audit" not in problems},
        {"name": "wifi_sensitivity_has_six_pairs", "passed": sensitivity["excluding_wifi_testapp"]["skill"]["project_count"] == 6},
        {"name": "per_rule_common_initial_support", "passed": all(row["initial_alerts"] > 0 for row in rules["common_support"])},
        {"name": "provider_cost_unavailable_is_explicit", "passed": all(efficiency["conditions"][condition]["billed_cost"] is None for condition in ("skill", "reference"))},
        {"name": "all_generated_files_exist", "passed": all(path.is_file() for path in generated)},
        {"name": "analysis_problems_empty", "passed": not problems, "problems": problems},
    ]
    audit = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "status": "closed" if all(check["passed"] for check in checks) else "failed",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "manifest": source_record(manifest_path),
        "output_hashes": [source_record(path) for path in generated],
        "claim_boundary": (
            "Direct numerical claims cover seven completed pairs. Three reference "
            "failures remain failures. Alert elimination is not semantic correctness."
        ),
        "next_route": "write" if all(check["passed"] for check in checks) else "blocker",
    }
    audit_path = output / "audit.json"
    write_json(audit_path, audit)
    if not audit["passed"]:
        raise RuntimeError(f"paired analysis audit failed: {problems}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = run(args.output.resolve())
    audit_path = args.output.resolve() / "audit.json"
    print(f"{audit_path}: PASS sha256={sha256_file(audit_path)}")
    print(f"next_route={audit['next_route']}")


if __name__ == "__main__":
    main()
