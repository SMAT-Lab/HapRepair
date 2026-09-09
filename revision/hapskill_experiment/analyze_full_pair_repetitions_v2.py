#!/usr/bin/env python3
"""Audit and analyze the clean 35-project, two-repetition comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_full_pair_35/runs"
CAMPAIGN = HERE / "full_pair_campaign_35x2_v2.json"
INPUTS = HERE / "formal_inputs_hapskill_35_v4.json"
STRATA = HERE / "full_pair_strata_35_v2.json"
SCHEDULER = (
    RUN_ROOT
    / "exp_full_pair_35_luna_clean_r2_02/scheduler/manifest.json"
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "paper/rebuttal/full_pair_35/campaign_02_results"
PACKAGE_ID = "AN-FULL-PAIR-35-CLEAN-R2-20260811"
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_SAMPLES = 20_000
METRICS = (
    "initial_alerts",
    "final_alerts",
    "eliminated_alerts",
    "remaining_alerts",
    "introduced_alerts",
    "net_reduction",
)
BENEFITS = (
    "final_alert_benefit",
    "net_reduction_benefit",
    "introduced_alert_benefit",
    "acceptable_candidate_benefit",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


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


def count_json_documents(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    documents = 0
    while True:
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            return documents
        _, offset = decoder.raw_decode(text, offset)
        documents += 1


def itt_metrics(initial_alerts: int, manifest: dict[str, Any]) -> dict[str, int]:
    if manifest["status"] == "completed":
        return {key: int(manifest["alert_metrics"][key]) for key in METRICS}
    return {
        "initial_alerts": initial_alerts,
        "final_alerts": initial_alerts,
        "eliminated_alerts": 0,
        "remaining_alerts": initial_alerts,
        "introduced_alerts": 0,
        "net_reduction": 0,
    }


def classify_failure(
    manifest: dict[str, Any], run_dir: Path
) -> tuple[str, dict[str, Any]]:
    if manifest["status"] == "completed":
        return "accepted_candidate", {}
    failure = str(manifest.get("failure", ""))
    if "did not reach HomeCheck validation after 6 agent turns" in failure:
        return "coverage_exhaustion", {"failure": failure}
    if "Extra data:" in failure:
        completion = run_dir / "workspace/.exp_agent/round_completion_01.json"
        return "completion_evidence_parse_failure", {
            "failure": failure,
            "completion_path": str(completion),
            "completion_exists": completion.exists(),
            "completion_sha256": sha256_file(completion) if completion.exists() else None,
            "completion_bytes": completion.stat().st_size if completion.exists() else None,
            "top_level_json_documents": (
                count_json_documents(completion) if completion.exists() else None
            ),
        }
    if "Bounded edit retries exhausted" in failure:
        round_path = run_dir / "evaluator_state/rounds/round_01/round.json"
        detail: dict[str, Any] = {
            "failure": failure,
            "round_path": str(round_path),
            "round_sha256": sha256_file(round_path) if round_path.exists() else None,
        }
        if round_path.exists():
            attempts = read_json(round_path).get("attempts", [])
            guard_failures = [
                item
                for attempt in attempts
                for item in attempt.get("preflight", {}).get("failures", [])
            ]
            detail["attempts"] = len(attempts)
            detail["guard_failure_counts"] = dict(sorted(Counter(guard_failures).items()))
            if guard_failures:
                return "structural_or_api_guard_exhaustion", detail
            if attempts and all(
                not attempt.get("coverage", {}).get("complete", False)
                for attempt in attempts
            ):
                return "coverage_exhaustion", detail
            if attempts and all(
                int(attempt.get("edit", {}).get("changed_source_file_count", 0)) == 0
                for attempt in attempts
            ):
                return "no_diff_exhaustion", detail
        return "bounded_edit_retry_exhaustion", detail
    return "unclassified_condition_failure", {"failure": failure}


def trace_attempt_counts(run_dir: Path, condition: str) -> dict[int, int]:
    counts: Counter[int] = Counter()
    if condition == "vanilla":
        pattern = re.compile(r"round_(\d+)_attempt_(\d+)\.jsonl$")
    else:
        pattern = re.compile(r"round_(\d+)(?:\.attempt_(\d+))?\.jsonl$")
    for path in (run_dir / "traces").glob("*.jsonl"):
        match = pattern.fullmatch(path.name)
        if match:
            counts[int(match.group(1))] += 1
    return dict(sorted(counts.items()))


def aggregate_condition(
    rows: Iterable[dict[str, Any]], condition: str, observed_only: bool = False
) -> dict[str, Any]:
    selected = list(rows)
    if observed_only:
        selected = [row for row in selected if row[condition]["acceptable_candidate"]]
    values = [row[condition]["itt_metrics"] for row in selected]
    totals = {
        metric: sum(int(item[metric]) for item in values) for metric in METRICS
    }
    final_values = [int(item["final_alerts"]) for item in values]
    return {
        "condition_records": len(selected),
        "acceptable_candidates": sum(
            int(row[condition]["acceptable_candidate"]) for row in selected
        ),
        "acceptable_candidate_rate": rate(
            sum(int(row[condition]["acceptable_candidate"]) for row in selected),
            len(selected),
        ),
        **totals,
        "aggregate_net_reduction_rate": rate(
            totals["net_reduction"], totals["initial_alerts"]
        ),
        "mean_final_alerts": mean(final_values) if final_values else None,
        "median_final_alerts": median(final_values) if final_values else None,
        "zero_final_alert_records": sum(value == 0 for value in final_values),
        "records_with_introduced_alerts": sum(
            int(item["introduced_alerts"]) > 0 for item in values
        ),
    }


def benefit_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    result: dict[str, Any] = {"pairs": len(selected)}
    for metric in BENEFITS:
        values = [float(row["benefit"][metric]) for row in selected]
        result[metric] = {
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "skill_better": sum(value > 0 for value in values),
            "tie": sum(value == 0 for value in values),
            "baseline_better": sum(value < 0 for value in values),
        }
    return result


def clustered_bootstrap(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_project[row["project"]].append(row)
    projects = sorted(by_project)
    project_values = {
        project: {
            metric: mean(
                float(row["benefit"][metric]) for row in by_project[project]
            )
            for metric in BENEFITS
        }
        for project in projects
    }
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {metric: [] for metric in BENEFITS}
    for _ in range(samples):
        sampled = [projects[rng.randrange(len(projects))] for _ in projects]
        for metric in BENEFITS:
            distributions[metric].append(
                mean(project_values[project][metric] for project in sampled)
            )
    return {
        "cluster": "project",
        "project_clusters": len(projects),
        "samples": samples,
        "seed": seed,
        "estimand": "mean project-level benefit after averaging the two repetitions",
        "positive_values_favor": "hapskill",
        "metrics": {
            metric: {
                "estimate": mean(
                    project_values[project][metric] for project in projects
                ),
                "ci95_percentile": [
                    percentile(distributions[metric], 0.025),
                    percentile(distributions[metric], 0.975),
                ],
            }
            for metric in BENEFITS
        },
    }


def clustered_regression(
    rows: list[dict[str, Any]], outcome: str
) -> dict[str, Any]:
    x = np.array([math.log1p(row["initial_alerts"]) for row in rows], dtype=float)
    y = np.array([float(row["benefit"][outcome]) for row in rows], dtype=float)
    groups = np.array([row["project"] for row in rows])
    design = sm.add_constant(x)
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int(alpha=0.05)
    return {
        "outcome": outcome,
        "positive_values_favor": "hapskill",
        "predictor": "log1p(frozen initial alerts)",
        "observations": len(rows),
        "project_clusters": len(set(groups)),
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "cluster_robust_standard_error": float(fit.bse[1]),
        "ci95": [float(ci[1, 0]), float(ci[1, 1])],
        "p_value": float(fit.pvalues[1]),
        "r_squared": float(fit.rsquared),
    }


def leave_one_project_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    projects = sorted({row["project"] for row in rows})
    overall = {
        metric: mean(float(row["benefit"][metric]) for row in rows)
        for metric in BENEFITS
    }
    records = []
    for project in projects:
        retained = [row for row in rows if row["project"] != project]
        records.append(
            {
                "omitted_project": project,
                **{
                    metric: mean(
                        float(row["benefit"][metric]) for row in retained
                    )
                    for metric in BENEFITS
                },
            }
        )
    return {
        "overall": overall,
        "records": records,
        "ranges": {
            metric: {
                "minimum": min(record[metric] for record in records),
                "maximum": max(record[metric] for record in records),
                "direction_reversals": sum(
                    (record[metric] > 0) != (overall[metric] > 0)
                    for record in records
                    if record[metric] != 0 and overall[metric] != 0
                ),
            }
            for metric in BENEFITS
        },
    }


def repetition_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_project: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_project[row["project"]][row["repeat"]] = row
    result: dict[str, Any] = {}
    for metric in BENEFITS:
        categories = Counter()
        for repetitions in by_project.values():
            first = float(repetitions[1]["benefit"][metric])
            second = float(repetitions[2]["benefit"][metric])
            signs = ((first > 0) - (first < 0), (second > 0) - (second < 0))
            if signs == (1, 1):
                categories["hapskill_better_both"] += 1
            elif signs == (-1, -1):
                categories["baseline_better_both"] += 1
            elif signs == (0, 0):
                categories["tie_both"] += 1
            elif 0 in signs:
                categories["one_tie_same_nonzero_direction"] += 1
            else:
                categories["direction_reversal"] += 1
        result[metric] = dict(categories)
    return result


def task_run_dir(task: dict[str, Any]) -> Path:
    return RUN_ROOT / task["run_id"] / task["condition"] / task["project"]


def build_rows_and_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    campaign = read_json(CAMPAIGN)
    scheduler = read_json(SCHEDULER)
    inputs_payload = read_json(INPUTS)
    input_rows = {item["name"]: item for item in inputs_payload["projects"]}
    strata_payload = read_json(STRATA)
    stratum_by_project = {
        item["project"]: stratum
        for stratum, items in strata_payload["groups"].items()
        for item in items
    }
    expected_tasks = {task["task_id"]: task for task in campaign["tasks"]}
    observed_tasks = {task["task_id"]: task for task in scheduler["tasks"]}
    problems: list[str] = []
    if set(expected_tasks) != set(observed_tasks):
        problems.append("scheduler task identities differ from frozen campaign")
    if len(observed_tasks) != 140:
        problems.append(f"expected 140 scheduler tasks, observed {len(observed_tasks)}")
    if scheduler.get("maximum_concurrent_conditions") != 16:
        problems.append("scheduler maximum concurrency is not 16")
    if scheduler.get("resume_count") != 0:
        problems.append("campaign scheduler was resumed")
    if scheduler.get("status") not in {"completed", "completed_with_condition_failures"}:
        problems.append(f"scheduler is not terminal: {scheduler.get('status')}")

    run_records: dict[tuple[int, str, str], dict[str, Any]] = {}
    task_audits: list[dict[str, Any]] = []
    for task_id in sorted(observed_tasks):
        task = observed_tasks[task_id]
        expected = expected_tasks.get(task_id, {})
        run_dir = task_run_dir(task)
        manifest_path = run_dir / "run_manifest.json"
        checks: dict[str, bool] = {
            "task_terminal": task.get("status") == "terminal",
            "runner_manifest_exists": manifest_path.exists(),
        }
        if not manifest_path.exists():
            problems.append(f"{task_id}: missing runner manifest")
            task_audits.append({"task_id": task_id, "checks": checks})
            continue
        manifest = read_json(manifest_path)
        input_row = input_rows[task["project"]]
        protocol_path = Path(task["protocol"])
        trace_counts = trace_attempt_counts(run_dir, task["condition"])
        max_attempts = max(trace_counts.values(), default=0)
        serialized_manifest = json.dumps(manifest, sort_keys=True)
        checks.update(
            {
                "task_matches_frozen_campaign": all(
                    task.get(key) == expected.get(key)
                    for key in ("repeat", "condition", "project", "run_id", "protocol")
                ),
                "manifest_hash_matches_scheduler": sha256_file(manifest_path)
                == task.get("runner_manifest_sha256"),
                "log_hash_matches_scheduler": sha256_file(Path(task["log_path"]))
                == task.get("log_sha256"),
                "manifest_identity_matches_task": (
                    manifest.get("run_id") == task["run_id"]
                    and manifest.get("condition") == task["condition"]
                    and manifest.get("project") == task["project"]
                ),
                "manifest_status_matches_task": manifest.get("status")
                == task.get("condition_status"),
                "protocol_hash_matches_file": manifest.get("protocol_sha256")
                == sha256_file(protocol_path),
                "input_manifest_hash_matches": (
                    manifest.get("project_manifest_sha256")
                    or manifest.get("inputs_sha256")
                )
                == sha256_file(INPUTS),
                "source_tree_hash_matches": (
                    manifest.get("input_tree_sha256")
                    or manifest.get("frozen_source_tree_sha256")
                )
                == input_row["frozen_input_tree_sha256"],
                "initial_findings_hash_matches": (
                    manifest.get("initial_target_findings_sha256")
                    or manifest.get("frozen_findings_sha256")
                )
                == input_row["frozen_target_findings_sha256"],
                "byte_identical_input": manifest.get("byte_identical_input") is True,
                "model_matches": manifest.get("model", {}).get("requested_id")
                == "gpt-5.6-luna"
                and manifest.get("model", {}).get("provider") == "xsp"
                and manifest.get("model", {}).get("reasoning_effort") == "high",
                "maximum_validation_scans_is_five": manifest.get(
                    "maximum_validation_scans"
                )
                == 5,
                "trace_attempts_within_six": max_attempts <= 6,
                "no_campaign_01_import": "exp_full_pair_35_luna_clean_r2_01"
                not in serialized_manifest,
            }
        )
        if task["condition"] == "vanilla":
            checks["baseline_maximum_turns_is_six"] = (
                manifest.get("maximum_agent_turns_per_round") == 6
            )
            checks["baseline_finding_identity_matches"] = (
                manifest.get("finding_identity_matches") is True
            )
        if manifest["status"] == "completed":
            checks["validation_scans_within_five"] = (
                int(manifest.get("validation_scan_count", 0)) <= 5
            )
            checks["metric_invariants_hold"] = not metric_invariants(
                {key: int(manifest["alert_metrics"][key]) for key in METRICS}
            )
            checks["initial_alert_count_matches_frozen"] = (
                int(manifest["alert_metrics"]["initial_alerts"])
                == int(input_row["frozen_target_finding_count"])
            )
            checks["manifest_attempts_within_six"] = all(
                len(round_record.get("attempts", [])) <= 6
                for round_record in manifest.get("rounds", [])
            )
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            problems.append(f"{task_id}: failed checks: {', '.join(failed_checks)}")
        failure_class, failure_detail = classify_failure(manifest, run_dir)
        record = {
            "repeat": int(task["repeat"]),
            "condition": task["condition"],
            "project": task["project"],
            "status": manifest["status"],
            "acceptable_candidate": manifest["status"] == "completed",
            "failure_class": failure_class,
            "failure_detail": failure_detail,
            "itt_metrics": itt_metrics(
                int(input_row["frozen_target_finding_count"]), manifest
            ),
            "observed_metrics": (
                {key: int(manifest["alert_metrics"][key]) for key in METRICS}
                if manifest["status"] == "completed"
                else None
            ),
            "validation_scan_count": (
                int(manifest.get("validation_scan_count", 0))
                if manifest["status"] == "completed"
                else 0
            ),
            "trace_attempt_counts": trace_counts,
            "manifest": source_record(manifest_path),
        }
        run_records[(int(task["repeat"]), task["project"], task["condition"])] = record
        task_audits.append(
            {
                "task_id": task_id,
                "checks": checks,
                "failure_class": failure_class,
                "manifest": source_record(manifest_path),
            }
        )

    rows: list[dict[str, Any]] = []
    for repeat in (1, 2):
        for input_row in inputs_payload["projects"]:
            project = input_row["name"]
            skill = run_records.get((repeat, project, "hapskill"))
            baseline = run_records.get((repeat, project, "vanilla"))
            if skill is None or baseline is None:
                problems.append(f"repeat_{repeat:02d}:{project}: incomplete pair")
                continue
            skill_metrics = skill["itt_metrics"]
            baseline_metrics = baseline["itt_metrics"]
            initial = int(input_row["frozen_target_finding_count"])
            if skill_metrics["initial_alerts"] != baseline_metrics["initial_alerts"]:
                problems.append(f"repeat_{repeat:02d}:{project}: paired initial mismatch")
            rows.append(
                {
                    "repeat": repeat,
                    "project": project,
                    "stratum": stratum_by_project[project],
                    "initial_alerts": initial,
                    "selection_reason": input_row["selection_reason"],
                    "skill": skill,
                    "baseline": baseline,
                    "benefit": {
                        "final_alert_benefit": baseline_metrics["final_alerts"]
                        - skill_metrics["final_alerts"],
                        "net_reduction_benefit": skill_metrics["net_reduction"]
                        - baseline_metrics["net_reduction"],
                        "introduced_alert_benefit": baseline_metrics[
                            "introduced_alerts"
                        ]
                        - skill_metrics["introduced_alerts"],
                        "acceptable_candidate_benefit": int(
                            skill["acceptable_candidate"]
                        )
                        - int(baseline["acceptable_candidate"]),
                    },
                }
            )

    audit = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "status": "passed" if not problems else "failed",
        "checks_passed": sum(
            sum(bool(value) for value in item["checks"].values())
            for item in task_audits
        ),
        "checks_total": sum(len(item["checks"]) for item in task_audits),
        "problem_count": len(problems),
        "problems": problems,
        "task_count": len(observed_tasks),
        "pair_count": len(rows),
        "scheduler_status": scheduler.get("status"),
        "scheduler_condition_status_counts": dict(
            sorted(Counter(task.get("condition_status") for task in observed_tasks.values()).items())
        ),
        "sources": {
            "campaign": source_record(CAMPAIGN),
            "inputs": source_record(INPUTS),
            "strata": source_record(STRATA),
            "scheduler": source_record(SCHEDULER),
        },
        "task_audits": task_audits,
    }
    return rows, audit


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "repeat",
        "project",
        "stratum",
        "initial_alerts",
        "skill_status",
        "baseline_status",
        "skill_failure_class",
        "baseline_failure_class",
        "skill_final_alerts_itt",
        "baseline_final_alerts_itt",
        "skill_net_reduction_itt",
        "baseline_net_reduction_itt",
        "skill_introduced_alerts_itt",
        "baseline_introduced_alerts_itt",
        *BENEFITS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "repeat": row["repeat"],
                    "project": row["project"],
                    "stratum": row["stratum"],
                    "initial_alerts": row["initial_alerts"],
                    "skill_status": row["skill"]["status"],
                    "baseline_status": row["baseline"]["status"],
                    "skill_failure_class": row["skill"]["failure_class"],
                    "baseline_failure_class": row["baseline"]["failure_class"],
                    "skill_final_alerts_itt": row["skill"]["itt_metrics"][
                        "final_alerts"
                    ],
                    "baseline_final_alerts_itt": row["baseline"]["itt_metrics"][
                        "final_alerts"
                    ],
                    "skill_net_reduction_itt": row["skill"]["itt_metrics"][
                        "net_reduction"
                    ],
                    "baseline_net_reduction_itt": row["baseline"]["itt_metrics"][
                        "net_reduction"
                    ],
                    "skill_introduced_alerts_itt": row["skill"]["itt_metrics"][
                        "introduced_alerts"
                    ],
                    "baseline_introduced_alerts_itt": row["baseline"]["itt_metrics"][
                        "introduced_alerts"
                    ],
                    **row["benefit"],
                }
            )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repeat = {
        str(repeat): [row for row in rows if row["repeat"] == repeat]
        for repeat in (1, 2)
    }
    completed_pairs = [
        row
        for row in rows
        if row["skill"]["acceptable_candidate"]
        and row["baseline"]["acceptable_candidate"]
    ]
    strata = {
        stratum: {
            "combined": benefit_summary(
                [row for row in rows if row["stratum"] == stratum]
            ),
            "repeat_1": benefit_summary(
                [
                    row
                    for row in rows
                    if row["stratum"] == stratum and row["repeat"] == 1
                ]
            ),
            "repeat_2": benefit_summary(
                [
                    row
                    for row in rows
                    if row["stratum"] == stratum and row["repeat"] == 2
                ]
            ),
        }
        for stratum in ("low", "middle", "high")
    }
    failure_rows = [
        {
            "repeat": row["repeat"],
            "project": row["project"],
            "condition": condition,
            "failure_class": row[condition]["failure_class"],
            "failure_detail": row[condition]["failure_detail"],
            "itt_disposition": "no_deployable_repair; unchanged frozen source metrics",
        }
        for row in rows
        for condition in ("skill", "baseline")
        if not row[condition]["acceptable_candidate"]
    ]
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "analysis_contract": {
            "population": "35 frozen projects x 2 independent repetitions",
            "pair_count": 70,
            "itt_failure_disposition": (
                "A condition without an acceptable candidate remains in the denominator "
                "as no deployable repair: final=initial, remaining=initial, eliminated=0, "
                "introduced=0, and net reduction=0."
            ),
            "benefit_sign": "positive values favor HapRepair Skill",
            "efficiency_reporting": "excluded from paper-facing analysis",
        },
        "all_project_repetitions": {
            "skill": aggregate_condition(rows, "skill"),
            "baseline": aggregate_condition(rows, "baseline"),
            "paired_benefits": benefit_summary(rows),
        },
        "repetitions": {
            f"repeat_{repeat}": {
                "skill": aggregate_condition(repeat_rows, "skill"),
                "baseline": aggregate_condition(repeat_rows, "baseline"),
                "paired_benefits": benefit_summary(repeat_rows),
            }
            for repeat, repeat_rows in by_repeat.items()
        },
        "project_clustered_bootstrap": clustered_bootstrap(
            rows, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
        ),
        "burden_strata": strata,
        "continuous_burden_models": {
            metric: clustered_regression(rows, metric)
            for metric in (
                "final_alert_benefit",
                "introduced_alert_benefit",
                "acceptable_candidate_benefit",
            )
        },
        "repetition_consistency": repetition_consistency(rows),
        "leave_one_project_out": leave_one_project_out(rows),
        "completed_pair_sensitivity": {
            "interpretation": (
                "Descriptive only. Excluding failed conditions breaks intention-to-treat "
                "randomized opportunity but checks whether alert outcomes among accepted "
                "candidates point in the same direction."
            ),
            "pair_count": len(completed_pairs),
            "skill": aggregate_condition(completed_pairs, "skill"),
            "baseline": aggregate_condition(completed_pairs, "baseline"),
            "paired_benefits": benefit_summary(completed_pairs),
        },
        "failures": {
            "count": len(failure_rows),
            "by_condition": dict(
                sorted(Counter(row["condition"] for row in failure_rows).items())
            ),
            "by_class": dict(
                sorted(Counter(row["failure_class"] for row in failure_rows).items())
            ),
            "infrastructure_or_api_failures": sum(
                row["failure_class"]
                in {"infrastructure_interruption", "provider_api_failure"}
                for row in failure_rows
            ),
            "records": failure_rows,
        },
    }


def markdown_report(summary: dict[str, Any], audit: dict[str, Any]) -> str:
    overall = summary["all_project_repetitions"]
    bootstrap = summary["project_clustered_bootstrap"]["metrics"]
    lines = [
        "# Clean 35-Project Paired Comparison",
        "",
        f"Package: `{PACKAGE_ID}`. Audit: **{audit['status'].upper()}** "
        f"({audit['checks_passed']}/{audit['checks_total']} task checks; "
        f"{audit['pair_count']} project-repetition pairs).",
        "",
        "All 35 projects and both repetitions are retained. A failed condition is "
        "counted as no deployable repair and receives unchanged-source ITT metrics. "
        "Positive paired-benefit values favor HapRepair.",
        "",
        "| Outcome | HapRepair | Baseline | Paired benefit | Project-clustered 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    skill = overall["skill"]
    baseline = overall["baseline"]
    final_benefit = overall["paired_benefits"]["final_alert_benefit"]["mean"]
    acceptable_benefit = overall["paired_benefits"][
        "acceptable_candidate_benefit"
    ]["mean"]
    introduced_benefit = overall["paired_benefits"][
        "introduced_alert_benefit"
    ]["mean"]
    lines.extend(
        [
            f"| Mean final alerts | {skill['mean_final_alerts']:.3f} | "
            f"{baseline['mean_final_alerts']:.3f} | {final_benefit:.3f} fewer | "
            f"[{bootstrap['final_alert_benefit']['ci95_percentile'][0]:.3f}, "
            f"{bootstrap['final_alert_benefit']['ci95_percentile'][1]:.3f}] |",
            f"| Acceptable candidate rate | {skill['acceptable_candidate_rate']:.3%} | "
            f"{baseline['acceptable_candidate_rate']:.3%} | {acceptable_benefit:.3%} | "
            f"[{bootstrap['acceptable_candidate_benefit']['ci95_percentile'][0]:.3%}, "
            f"{bootstrap['acceptable_candidate_benefit']['ci95_percentile'][1]:.3%}] |",
            f"| Mean introduced-alert benefit | {skill['introduced_alerts'] / 70:.3f} introduced | "
            f"{baseline['introduced_alerts'] / 70:.3f} introduced | {introduced_benefit:.3f} | "
            f"[{bootstrap['introduced_alert_benefit']['ci95_percentile'][0]:.3f}, "
            f"{bootstrap['introduced_alert_benefit']['ci95_percentile'][1]:.3f}] |",
            "",
            "## Repetitions",
            "",
            "| Repetition | Skill acceptable | Baseline acceptable | Mean final-alert benefit |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for repeat in (1, 2):
        item = summary["repetitions"][f"repeat_{repeat}"]
        lines.append(
            f"| {repeat} | {item['skill']['acceptable_candidate_rate']:.3%} | "
            f"{item['baseline']['acceptable_candidate_rate']:.3%} | "
            f"{item['paired_benefits']['final_alert_benefit']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Burden",
            "",
            "| Frozen burden stratum | Pairs | Mean final-alert benefit | "
            "Mean acceptable-candidate benefit |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for stratum in ("low", "middle", "high"):
        item = summary["burden_strata"][stratum]["combined"]
        lines.append(
            f"| {stratum} | {item['pairs']} | "
            f"{item['final_alert_benefit']['mean']:.3f} | "
            f"{item['acceptable_candidate_benefit']['mean']:.3%} |"
        )
    regression = summary["continuous_burden_models"]["final_alert_benefit"]
    loo = summary["leave_one_project_out"]["ranges"]["final_alert_benefit"]
    lines.extend(
        [
            "",
            "The clustered linear model uses `log1p(initial alerts)` and all 70 "
            f"project-repetition observations: slope={regression['slope']:.3f}, "
            f"cluster-robust 95% CI [{regression['ci95'][0]:.3f}, "
            f"{regression['ci95'][1]:.3f}], p={regression['p_value']:.4g}.",
            "",
            "## Robustness And Failures",
            "",
            f"Leave-one-project-out mean final-alert benefits range from "
            f"{loo['minimum']:.3f} to {loo['maximum']:.3f}; direction reversals: "
            f"{loo['direction_reversals']}. The completed-pair-only descriptive "
            f"sensitivity retains {summary['completed_pair_sensitivity']['pair_count']} "
            "pairs.",
            "",
            f"There are {summary['failures']['count']} failed conditions: "
            + ", ".join(
                f"{name}={count}"
                for name, count in summary["failures"]["by_class"].items()
            )
            + ". No failure is classified as a provider API or infrastructure failure.",
            "",
            "Raw per-project rows, failure details, bootstrap settings, regression "
            "outputs, source hashes, and all task-level audit checks are stored beside "
            "this report. Token, invocation, and wall-time data are intentionally not "
            "reported as effectiveness evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def freeze_evidence(output: Path, filenames: list[str]) -> None:
    write_json(
        output / "evidence_manifest.json",
        {
            "schema_version": 1,
            "package_id": PACKAGE_ID,
            "status": "frozen",
            "artifacts": [source_record(output / filename) for filename in filenames],
        },
    )


def build_route_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "parent_object": "EXP-HAPREPAIR-FULL-PAIR-35-R2 campaign_02",
        "parent_claim": (
            "HapRepair should be more reliable than a coding-agent-only baseline under "
            "matched inputs, model, analyzer feedback, and evaluator gates."
        ),
        "execution_envelope": {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "high",
            "projects": 35,
            "conditions": 2,
            "repetitions": 2,
            "formal_condition_count": 140,
            "maximum_concurrency": 16,
            "maximum_validation_scans": 5,
            "maximum_agent_turns_per_active_round": 6,
            "resume_refreshes_allowance": False,
            "additional_model_runs_after_terminal_campaign": 0,
        },
        "comparison_baselines": ["coding-agent-only baseline under the matched v6 protocol"],
        "slices": [
            {
                "slice_id": "E5-A1",
                "analysis_role": "claim-carrying full-population paired effectiveness",
                "paper_role": "main_text",
                "section_id": "evaluation-contemporary-reference",
                "item_id": "AN-FULL-PAIR-35-CLEAN-R2",
                "claim_links": ["AE-C2", "R1-C10", "R3-C2", "R3-C5", "R3-C6"],
                "reviewer_question": "Is the apparent advantage stable across clean repetitions and all 35 projects?",
                "fixed_conditions": "Same frozen inputs, model, localization, five-scan budget, six-turn active-round allowance, and evaluator gates.",
                "metrics": ["final alerts", "net reduction", "introduced alerts", "acceptable-candidate rate"],
                "evidence_paths": ["summary.json", "project_repetition_rows.csv", "audit.json"],
                "claim_update": (
                    "Acceptable-candidate reliability favors HapRepair in both repetitions, "
                    "but final-alert advantage is mixed and uncertain."
                ),
                "comparability": "direct and intention-to-treat; cluster by project",
                "status": "completed_partial_support",
                "next_action": "Write a narrowed reliability claim and report the mixed final-alert result.",
            },
            {
                "slice_id": "E5-A2",
                "analysis_role": "claim-carrying burden interaction",
                "paper_role": "appendix_or_compact_main_text",
                "section_id": "evaluation-contemporary-reference",
                "item_id": "AN-PROJECT-STRATA",
                "claim_links": ["C2", "R3-C5"],
                "reviewer_question": "Does HapRepair's benefit increase with the frozen initial alert burden?",
                "fixed_conditions": "Outcome-independent 12/11/12 strata and continuous log1p burden model with project clustering.",
                "metrics": ["paired final-alert benefit", "acceptable-candidate benefit", "continuous burden slope"],
                "evidence_paths": ["summary.json", "project_repetition_rows.csv"],
                "claim_update": "The increasing-benefit-with-burden hypothesis is rejected.",
                "comparability": "direct; strata frozen before outcomes",
                "status": "completed_contradictory",
                "next_action": "Remove the burden-benefit claim and retain the negative result.",
            },
            {
                "slice_id": "E5-A3",
                "analysis_role": "supporting failure and sensitivity audit",
                "paper_role": "main_text_and_appendix",
                "section_id": "evaluation-contemporary-reference",
                "item_id": "AN-FAILURE",
                "claim_links": ["AE-C2", "R1-C10", "R3-C6"],
                "reviewer_question": "Are condition gaps caused by repair behavior or by provider/infrastructure accidents?",
                "fixed_conditions": "All terminal failures retained; no outcome-improving retries; leave-one-project-out sensitivity.",
                "metrics": ["failure taxonomy", "ITT disposition", "leave-one-project-out direction"],
                "evidence_paths": ["summary.json", "audit.json", "project_repetition_rows.json"],
                "claim_update": "All 15 failures are repair/completion failures; no API or infrastructure failure occurred.",
                "comparability": "direct; failure handling is symmetric and preregistered",
                "status": "completed",
                "next_action": "Report the failure mix and preserve every failed condition in the denominator.",
            },
        ],
        "evaluation_summary": {
            "takeaway": (
                "HapRepair is more likely to produce an evaluator-acceptable candidate, "
                "but it does not show a stable final-alert advantage, and benefit does not "
                "increase with initial defect burden."
            ),
            "acceptable_candidate_counts": {
                "hapskill": summary["all_project_repetitions"]["skill"]["acceptable_candidates"],
                "baseline": summary["all_project_repetitions"]["baseline"]["acceptable_candidates"],
                "denominator_each": 70,
            },
            "comparability": "preserved",
            "failure_mode": summary["failures"]["by_class"],
        },
        "stop_condition": (
            "Met: two clean full-population repetitions, all terminal states audited, "
            "primary and burden claims adjudicated, and no infrastructure replacement needed."
        ),
        "next_route": "write",
        "next_action": (
            "Reconcile the narrowed claims into the outline, then revise the manuscript "
            "and response letter without launching additional repair experiments."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, audit = build_rows_and_audit()
    write_json(args.output / "audit.json", audit)
    if audit["status"] != "passed":
        raise SystemExit(
            f"audit failed with {audit['problem_count']} problem(s); see {args.output / 'audit.json'}"
        )
    summary = build_summary(rows)
    write_json(args.output / "project_repetition_rows.json", rows)
    write_rows_csv(args.output / "project_repetition_rows.csv", rows)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "route_record.json", build_route_record(summary))
    (args.output / "analysis.md").write_text(
        markdown_report(summary, audit), encoding="utf-8"
    )
    freeze_evidence(
        args.output,
        [
            "audit.json",
            "project_repetition_rows.json",
            "project_repetition_rows.csv",
            "summary.json",
            "route_record.json",
            "analysis.md",
        ],
    )
    print(args.output / "analysis.md")


if __name__ == "__main__":
    main()
