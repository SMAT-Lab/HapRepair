#!/usr/bin/env python3
"""Audit the final-v14 run without rewriting its frozen runner outputs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
RUNS_DIR = ORACLE_DIR / "generation_runs"
RUNNER = SCRIPT_DIR / "run_generation_v14_static_skill.py"
EXPECTED_CASE_COUNT = 63
METADATA_FILES = {
    "HAPREPAIR_TASK.json",
    "HAPREPAIR_COMPLETION.json",
    "completion.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def repair_snapshot(root: Path) -> dict[str, bytes]:
    """Capture all candidate files while excluding runner/model metadata."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in METADATA_FILES
        and ".git" not in path.relative_to(root).parts
    }


def unified_diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    chunks: list[str] = []
    for relative in sorted(set(before) | set(after)):
        old = (
            before.get(relative, b"").decode("utf-8", errors="replace").splitlines(True)
        )
        new = (
            after.get(relative, b"").decode("utf-8", errors="replace").splitlines(True)
        )
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def max_concurrency(events: list[dict[str, Any]]) -> int:
    points: list[tuple[datetime, int]] = []
    for event in events:
        if event["event"] == "request_started":
            points.append((datetime.fromisoformat(event["timestamp"]), 1))
        elif event["event"] == "request_finished":
            points.append((datetime.fromisoformat(event["timestamp"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def result_acceptance_consistent(result: dict[str, Any]) -> bool:
    accepted = all(
        (
            result["exit_code"] == 0,
            result["event_count"] > 0,
            result["source_diff_present"],
            not result["deleted_input_paths"],
            not result["missing_required_paths"],
            result["task_unchanged"],
            not result["restricted_commands"],
        )
    )
    return (result["status"] == "accepted") == accepted


def capture_recovery(
    *, run_dir: Path, item: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any] | None:
    if result["status"] != "rejected":
        return None
    case_dir = run_dir / item["case_dir"]
    before = repair_snapshot(case_dir / "baseline")
    after = repair_snapshot(case_dir / "workspace")
    missing_inputs = sorted(set(before) - set(after))
    patch = unified_diff(before, after)
    recoverable = all(
        (
            result["exit_code"] == 0,
            result["event_count"] > 0,
            result["task_unchanged"],
            not result["restricted_commands"],
            not missing_inputs,
            bool(patch),
            set(result["rejection_reasons"])
            <= {"no_source_diff", "required_source_missing"},
        )
    )
    if not recoverable:
        return None
    changed = sorted(
        relative
        for relative in set(before) | set(after)
        if before.get(relative) != after.get(relative)
    )
    return {
        "case_id": item["case_id"],
        "blind_id": item["blind_id"],
        "original_status": result["status"],
        "original_rejection_reasons": result["rejection_reasons"],
        "original_result_sha256": sha256_file(case_dir / "result.json"),
        "trace_sha256": sha256_file(case_dir / "trace.jsonl"),
        "capture_bug": (
            "The frozen runner snapshots only .ets/.ts files, although this case's "
            "input and repair use .json5/.json resource files."
        ),
        "changed_files": [
            {
                "path": relative,
                "change": "added" if relative not in before else "modified",
                "sha256": sha256_bytes(after[relative]),
                "size_bytes": len(after[relative]),
            }
            for relative in changed
        ],
        "missing_original_inputs_under_full_capture": missing_inputs,
        "recovered_patch": patch,
        "recovered_patch_sha256": sha256_bytes(patch.encode("utf-8")),
        "decision_status": "pending_author_confirmation",
    }


def write_recovery_evidence(
    *, audit_dir: Path, run_dir: Path, item: dict[str, Any], recovery: dict[str, Any]
) -> None:
    destination = audit_dir / "capture_recovery_evidence" / item["blind_id"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidate.patch").write_text(
        recovery["recovered_patch"], encoding="utf-8"
    )
    workspace = run_dir / item["case_dir"] / "workspace"
    for record in recovery["changed_files"]:
        target = destination / "candidate_files" / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / record["path"], target)
    manifest = {
        key: value for key, value in recovery.items() if key != "recovered_patch"
    }
    manifest["candidate_patch"] = "candidate.patch"
    manifest["candidate_patch_sha256"] = sha256_file(destination / "candidate.patch")
    write_json(destination / "recovery_manifest.json", manifest)


def audit(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    protocol = read_json(run_dir / "protocol_snapshot.json")
    manifest = read_json(run_dir / "input_manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    verification = read_json(run_dir / "verification.json")
    events = [
        json.loads(line)
        for line in (run_dir / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event["event"] == "request_started"]
    finishes = [event for event in events if event["event"] == "request_finished"]
    results: list[dict[str, Any]] = []
    recoveries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    artifact_checks: list[bool] = []
    for item in manifest:
        case_dir = run_dir / item["case_dir"]
        result = read_json(case_dir / "result.json")
        results.append(result)
        artifact_checks.extend(
            (
                sha256_file(case_dir / "trace.jsonl") == result["trace_sha256"],
                sha256_file(case_dir / "stderr.log") == result["stderr_sha256"],
                sha256_file(case_dir / "candidate.patch")
                == result["source_diff_sha256"],
                read_json(case_dir / "attempt.json")["case_id"] == item["case_id"],
            )
        )
        recovery = capture_recovery(run_dir=run_dir, item=item, result=result)
        if recovery is not None:
            recoveries.append((item, recovery))

    start_counts = Counter(event["case_id"] for event in starts)
    finish_counts = Counter(event["case_id"] for event in finishes)
    result_by_case = {result["case_id"]: result for result in results}
    status_counts = Counter(result["status"] for result in results)
    false_verifier_checks = sorted(
        name for name, passed in verification["checks"].items() if not passed
    )
    expected_false_checks = [
        "all_63_candidates_accepted",
        "all_source_diffs_present",
        "no_restricted_commands",
    ]
    tokens_complete = all(
        all(
            isinstance(result["usage"].get(key), int) and result["usage"][key] >= 0
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "total_tokens",
            )
        )
        and result["usage"]["total_tokens"]
        == result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
        for result in results
    )
    checks = {
        "manifest_has_63_cases": len(manifest) == EXPECTED_CASE_COUNT,
        "request_started_exactly_once_per_case": len(starts) == EXPECTED_CASE_COUNT
        and set(start_counts) == {item["case_id"] for item in manifest}
        and all(count == 1 for count in start_counts.values()),
        "request_finished_exactly_once_per_case": len(finishes) == EXPECTED_CASE_COUNT
        and set(finish_counts) == {item["case_id"] for item in manifest}
        and all(count == 1 for count in finish_counts.values()),
        "all_63_terminal_results_present": len(results) == EXPECTED_CASE_COUNT,
        "result_identity_and_status_match_events": all(
            result_by_case[event["case_id"]]["blind_id"] == event["blind_id"]
            and result_by_case[event["case_id"]]["status"] == event["status"]
            for event in finishes
        ),
        "frozen_acceptance_logic_reproduced": all(
            result_acceptance_consistent(result) for result in results
        ),
        "all_case_artifact_hashes_match": all(artifact_checks),
        "requested_model_matches_protocol_for_all_cases": all(
            result["requested_model"] == protocol["model"]["requested_id"]
            for result in results
        ),
        "usage_complete_and_arithmetically_consistent": tokens_complete,
        "timing_complete_and_positive": all(
            result["wall_seconds"] > 0 for result in results
        ),
        "no_systemic_failure": all(
            not result["systemic_failure"] for result in results
        ),
        "metrics_terminal_counts_match_results": metrics["status"] == "complete"
        and metrics["attempted_case_count"] == EXPECTED_CASE_COUNT
        and metrics["accepted_candidate_count"] == status_counts["accepted"]
        and metrics["rejected_case_count"] == status_counts["rejected"],
        "strict_verifier_failures_are_fully_explained": false_verifier_checks
        == expected_false_checks,
        "parallel_amendment_respected": max_concurrency(events) <= 4
        and not (run_dir / "parallel_execution" / "SYSTEMIC_STOP").exists(),
        "capture_recovery_is_unique": len(recoveries) == 1,
    }
    recovery_records = [recovery for _, recovery in recoveries]
    report = {
        "schema_version": 1,
        "analysis_id": "E3-05-final-v14-artifact-audit-v1",
        "parent_run_id": run_id,
        "parent_question": protocol["research_question"],
        "analysis_question": (
            "Are all 63 single-invocation outputs complete, hash-consistent, and "
            "classifiable without adding a model call?"
        ),
        "inspection_only": True,
        "additional_model_calls": 0,
        "fixed_conditions": [
            "Frozen benchmark, model, provider, prompt, Skill, and per-case invocation budget",
            "Original result.json, metrics.json, verification.json, traces, and workspaces remain unchanged",
            "Rejected cases are visible and receive no retry",
        ],
        "execution_envelope": {
            "resource_class": "local deterministic filesystem audit",
            "model_or_gpu_required": False,
            "binding_constraint": "Do not mutate frozen outputs or decide the recovery counting policy without author confirmation.",
        },
        "observed": {
            "request_started_count": len(starts),
            "request_finished_count": len(finishes),
            "raw_accepted_candidate_count": status_counts["accepted"],
            "raw_rejected_case_count": status_counts["rejected"],
            "max_observed_concurrency": max_concurrency(events),
            "raw_rejected_cases": [
                {
                    "case_id": result["case_id"],
                    "blind_id": result["blind_id"],
                    "rejection_reasons": result["rejection_reasons"],
                    "restricted_commands": result["restricted_commands"],
                }
                for result in results
                if result["status"] == "rejected"
            ],
            "capture_recoverable_count": len(recovery_records),
            "capture_recoveries": [
                {
                    key: value
                    for key, value in recovery.items()
                    if key != "recovered_patch"
                }
                for recovery in recovery_records
            ],
            "counting_views": {
                "frozen_runner_raw": {
                    "candidate_count": status_counts["accepted"],
                    "generation_failure_count": status_counts["rejected"],
                },
                "capture_aware_if_author_approved": {
                    "candidate_count": status_counts["accepted"] + len(recoveries),
                    "generation_failure_count": status_counts["rejected"]
                    - len(recoveries),
                },
            },
            "strict_verifier_all_passed": verification["all_passed"],
            "strict_verifier_false_checks": false_verifier_checks,
        },
        "checks": checks,
        "artifact_audit_passed": all(checks.values()),
        "claim_update": (
            "The 63-call generation is terminal and reconstructable. One resource-file "
            "candidate was misclassified by the frozen .ets/.ts-only capture layer; its "
            "counting treatment remains pending author confirmation. Semantic correctness "
            "remains unmeasured until blind annotation."
        ),
        "comparability": (
            "Case-level inputs and generation conditions remain comparable. The two "
            "reported counting views differ only in whether the verified capture-layer "
            "recovery is admitted."
        ),
        "next_action": (
            "Obtain the author's counting decision, freeze one package policy, then build "
            "the new blind annotation package without rerunning any case."
        ),
    }
    audit_dir = run_dir / "artifact_audit_v1"
    for item, recovery in recoveries:
        write_recovery_evidence(
            audit_dir=audit_dir, run_dir=run_dir, item=item, recovery=recovery
        )
    write_json(audit_dir / "audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = audit(args.run_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["artifact_audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
