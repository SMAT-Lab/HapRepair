#!/usr/bin/env python3
"""Audit and reconcile the completed EXP-AGENT-REF-10 formal_03 scheduler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import resume_agent_ref_interrupted_v4 as resume


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
RUN_ROOT = resume.DEFAULT_RUN_ROOT / resume.RUN_ID
SCHEDULER = RUN_ROOT / "formal_scheduler/manifest.json"
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_results_audit_20260807.json"
)
COMPLETED = (
    "TextComponentTest",
    "ace_ets_module_swiper_api11",
    "applications_photos",
    "wifi_testapp",
    "HealthyPotAssistant",
    "asn1_ber",
    "applications_permission_manager",
)
FAILED = ("acts_validator", "bluetoothtest", "ohos_cordova")


def add(checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **evidence})


def condition_path(project: str) -> Path:
    return RUN_ROOT / "vanilla" / project / "run_manifest.json"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def trace_usage(manifest: dict[str, Any]) -> Counter[str]:
    usage: Counter[str] = Counter()
    for round_item in manifest["rounds"]:
        for attempt in round_item["attempts"]:
            usage.update((attempt.get("turn") or {}).get("usage") or {})
    return usage


def main() -> None:
    checks: list[dict[str, Any]] = []
    scheduler = resume.read_json(SCHEDULER)
    statuses = Counter(item["status"] for item in scheduler["tasks"])
    observed_completed = tuple(
        item["project"] for item in scheduler["tasks"] if item["status"] == "completed"
    )
    observed_failed = tuple(
        item["project"] for item in scheduler["tasks"] if item["status"] == "failed"
    )
    add(
        checks,
        "scheduler_terminal_status_and_exact_partition",
        scheduler.get("status") == "completed_with_protocol_failures"
        and statuses == {"completed": 7, "failed": 3}
        and observed_completed == COMPLETED
        and observed_failed == FAILED,
        scheduler=str(SCHEDULER),
        scheduler_sha256=resume.sha256_file(SCHEDULER),
        status_counts=dict(statuses),
        completed=list(observed_completed),
        failed=list(observed_failed),
    )

    completed_rows: list[dict[str, Any]] = []
    completed_failures: list[dict[str, Any]] = []
    task_by_project = {item["project"]: item for item in scheduler["tasks"]}
    for project in COMPLETED:
        path = condition_path(project)
        manifest = resume.read_json(path)
        metrics = manifest.get("alert_metrics") or {}
        metric_identity = (
            metrics.get("eliminated_alerts", 0) + metrics.get("remaining_alerts", 0)
            == metrics.get("initial_alerts")
            and metrics.get("remaining_alerts", 0) + metrics.get("introduced_alerts", 0)
            == metrics.get("final_alerts")
            and metrics.get("net_reduction")
            == metrics.get("initial_alerts", 0) - metrics.get("final_alerts", 0)
        )
        scanner = resume.read_json(path.parent / "evaluator_state/homecheck/state.json")
        task = task_by_project[project]
        valid = (
            manifest.get("status") == "completed"
            and metric_identity
            and not manifest.get("restricted_accesses")
            and 0 <= int(manifest.get("validation_scan_count", -1)) <= 5
            and scanner.get("validation_scans_consumed")
            == manifest.get("validation_scan_count")
            and scanner.get("final_scan", {}).get("target_finding_count")
            == metrics.get("final_alerts")
            and task.get("runner_manifest_sha256") == resume.sha256_file(path)
        )
        if not valid:
            completed_failures.append({"project": project, "manifest": str(path)})
        completed_rows.append(
            {
                "project": project,
                **metrics,
                "validation_scan_count": manifest.get("validation_scan_count"),
                "input_tokens": manifest.get("input_tokens"),
                "cached_input_tokens": manifest.get("cached_input_tokens"),
                "output_tokens": manifest.get("output_tokens"),
                "total_tokens": manifest.get("total_tokens"),
                "wall_clock_seconds": manifest.get("wall_clock_seconds"),
                "manifest": str(path),
                "manifest_sha256": resume.sha256_file(path),
            }
        )
    add(
        checks,
        "seven_completed_condition_manifests_reconcile",
        not completed_failures,
        failures=completed_failures,
        projects=completed_rows,
    )

    failed_rows: list[dict[str, Any]] = []
    failed_problems: list[str] = []
    for project in FAILED:
        path = condition_path(project)
        manifest = resume.read_json(path)
        round_number = 2 if project == "ohos_cordova" else 1
        round_path = (
            path.parent
            / "evaluator_state/rounds"
            / f"round_{round_number:02d}/round.json"
        )
        round_record = resume.read_json(round_path)
        attempts = round_record.get("attempts") or []
        all_coverage = all(
            item.get("coverage", {}).get("complete") is True for item in attempts
        )
        guard_failure = all(
            bool((item.get("preflight") or {}).get("failures")) for item in attempts
        )
        valid = (
            manifest.get("status") == "failed"
            and manifest.get("failure")
            == f"RuntimeError: Bounded edit retries exhausted in round {round_number}"
            and round_record.get("status") == "bounded_edit_retry_exhausted"
            and len(attempts) == resume.MAX_AGENT_TURNS_PER_ROUND
            and all_coverage
            and guard_failure
        )
        if not valid:
            failed_problems.append(project)
        failed_rows.append(
            {
                "project": project,
                "failure": manifest.get("failure"),
                "failed_round": round_number,
                "attempt_count": len(attempts),
                "all_attempts_coverage_complete": all_coverage,
                "all_attempts_guard_failed": guard_failure,
                "accepted_rounds_before_failure": len(manifest.get("rounds") or []),
                "manifest": str(path),
                "manifest_sha256": resume.sha256_file(path),
            }
        )
    add(
        checks,
        "three_genuine_bounded_guard_failures",
        not failed_problems,
        failures=failed_problems,
        projects=failed_rows,
    )

    permission_path = condition_path(resume.PROJECT)
    permission = resume.read_json(permission_path)
    recovery = (
        permission_path.parent
        / "evaluator_state/interruption_recovery"
        / resume.RECOVERY_ID
    )
    reconstruction = resume.read_json(recovery / "reconstruction.json")
    incident_paths = (
        resume.INCIDENT,
        WORKSPACE_ROOT
        / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_01_python_incident_20260807.json",
        WORKSPACE_ROOT
        / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_02_reconstruction_incident_20260807.json",
        resume.RESUME_AUTHORIZATION,
    )
    incident_statuses = [
        resume.read_json(path).get("passed") for path in incident_paths
    ]
    add(
        checks,
        "continuation_authorizations_and_incidents_closed",
        all(incident_statuses)
        and permission.get("reboot_continuation", {}).get("same_condition") is True
        and permission.get("reboot_continuation", {}).get(
            "fresh_stochastic_condition_started"
        )
        is False,
        artifacts=[
            {"path": str(path), "sha256": resume.sha256_file(path)}
            for path in incident_paths
        ],
    )

    expected_attempt_patch = resume.read_json(resume.INCIDENT)["interrupted_condition"][
        "attempt_03_patch_sha256"
    ]
    expected_partial_patch = resume.read_json(resume.INCIDENT)["interrupted_condition"][
        "partial_attempt_04_patch_sha256"
    ]
    add(
        checks,
        "partial_state_preserved_and_attempt_03_exactly_reconstructed",
        reconstruction.get("byte_exact_patch_reconstruction") is True
        and reconstruction.get("reconstructed_attempt_03", {}).get("patch_sha256")
        == expected_attempt_patch
        and reconstruction.get("partial_attempt_04_patch", {}).get("patch_sha256")
        == expected_partial_patch
        and reconstruction.get("normalized_completed_patch", {}).get(
            "repaired_no_newline_boundaries"
        )
        == 6,
        reconstruction=str(recovery / "reconstruction.json"),
        reconstruction_sha256=resume.sha256_file(recovery / "reconstruction.json"),
    )

    round_one, round_two = permission["rounds"]
    round_one_threads = {item["thread_id"] for item in round_one["attempts"]}
    round_two_threads = {item["thread_id"] for item in round_two["attempts"]}
    add(
        checks,
        "active_round_thread_scope_preserved",
        round_one_threads == {"019fd7bc-4ec9-7da2-bed1-cb77dfd382cd"}
        and len(round_two_threads) == 1
        and not round_one_threads.intersection(round_two_threads)
        and [len(round_one["attempts"]), len(round_two["attempts"])] == [5, 1],
        round_01_threads=sorted(round_one_threads),
        round_02_threads=sorted(round_two_threads),
        attempt_counts=[len(round_one["attempts"]), len(round_two["attempts"])],
    )

    usage = trace_usage(permission)
    usage_fields = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
    )
    add(
        checks,
        "trace_usage_and_coverage_reconcile",
        all(int(permission[key]) == int(usage[key]) for key in usage_fields)
        and all(
            attempt.get("coverage", {}).get("complete") is True
            for round_item in permission["rounds"]
            for attempt in round_item["attempts"]
        ),
        manifest_usage={key: permission[key] for key in usage_fields},
        trace_usage={key: usage[key] for key in usage_fields},
    )

    scanner = resume.read_json(
        permission_path.parent / "evaluator_state/homecheck/state.json"
    )
    scan_counts = [scanner["initial_scan"]["target_finding_count"]] + [
        item["target_finding_count"] for item in scanner["validation_scans"]
    ]
    add(
        checks,
        "permission_manager_scan_sequence_and_finalization",
        scan_counts == [80, 1, 0]
        and scanner["final_scan"]["target_finding_count"] == 0
        and scanner["final_scan"]["budget_consumed"] is False
        and permission["alert_metrics"]
        == {
            "initial_alerts": 80,
            "final_alerts": 0,
            "eliminated_alerts": 80,
            "remaining_alerts": 0,
            "introduced_alerts": 0,
            "net_reduction": 80,
        }
        and permission["best_valid_round"] == 2,
        scan_counts=scan_counts,
        final_scan_count=scanner["final_scan"]["target_finding_count"],
        final_candidate_selection=permission["final_candidate_selection"],
    )

    aggregate_fields = (
        "initial_alerts",
        "final_alerts",
        "eliminated_alerts",
        "remaining_alerts",
        "introduced_alerts",
        "net_reduction",
    )
    aggregate = {
        key: sum(int(row[key]) for row in completed_rows) for key in aggregate_fields
    }
    sensitivity_rows = [
        row for row in completed_rows if row["project"] != "wifi_testapp"
    ]
    sensitivity = {
        key: sum(int(row[key]) for row in sensitivity_rows) for key in aggregate_fields
    }
    add(
        checks,
        "completed_condition_aggregates",
        aggregate
        == {
            "initial_alerts": 2752,
            "final_alerts": 3,
            "eliminated_alerts": 2751,
            "remaining_alerts": 1,
            "introduced_alerts": 2,
            "net_reduction": 2749,
        }
        and sensitivity
        == {
            "initial_alerts": 1678,
            "final_alerts": 3,
            "eliminated_alerts": 1677,
            "remaining_alerts": 1,
            "introduced_alerts": 2,
            "net_reduction": 1675,
        },
        seven_completed_conditions=aggregate,
        six_completed_conditions_excluding_wifi_testapp=sensitivity,
        denominator_note=(
            "These are completed-condition aggregates. The three bounded guard "
            "failures are reported separately and are not assigned synthetic metrics."
        ),
    )

    tests = run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "test_agent_ref_v4.py",
            "test_resume_agent_ref_interrupted_v4.py",
        ]
    )
    lint = run(
        [
            "ruff",
            "check",
            "resume_agent_ref_interrupted_v4.py",
            "test_resume_agent_ref_interrupted_v4.py",
            Path(__file__).name,
        ]
    )
    add(
        checks,
        "post_run_tests_compile_and_lint",
        tests.returncode == 0
        and "Ran 13 tests" in tests.stderr
        and lint.returncode == 0,
        test_returncode=tests.returncode,
        test_output=(tests.stdout + tests.stderr)[-5000:],
        lint_returncode=lint.returncode,
        lint_output=(lint.stdout + lint.stderr)[-3000:],
    )

    add(
        checks,
        "formal_04_absent_and_no_automatic_failure_reruns",
        not (resume.DEFAULT_RUN_ROOT / "exp_agent_ref_10_luna_v4_formal_04").exists()
        and all(
            len(list((condition_path(project).parent / "traces").glob("*.jsonl"))) > 0
            for project in FAILED
        ),
        formal_04=str(resume.DEFAULT_RUN_ROOT / "exp_agent_ref_10_luna_v4_formal_04"),
        failed_projects=list(FAILED),
    )

    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": resume.RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if passed else "open_failed_audit",
        "passed": passed,
        "scheduler_status": scheduler.get("status"),
        "completed_condition_count": len(COMPLETED),
        "bounded_guard_failure_count": len(FAILED),
        "completed_condition_aggregate": aggregate,
        "completed_sensitivity_excluding_wifi_testapp": sensitivity,
        "checks": checks,
        "claim_boundary": (
            "formal_03 produced seven completed reference-agent conditions and three "
            "bounded guard failures. Aggregates cover only completed conditions; no "
            "effectiveness value is imputed for failed conditions. Alert elimination "
            "is not semantic correctness."
        ),
        "next_action": (
            "Use this reconciled package for paired reporting against existing Skill "
            "results; do not rerun formal_03 conditions."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={resume.sha256_file(OUTPUT)}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
