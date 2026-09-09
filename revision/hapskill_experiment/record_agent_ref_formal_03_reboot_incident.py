#!/usr/bin/env python3
"""Record the host-reboot interruption of one formal_03 condition."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from resume_agent_ref_interrupted_v4 import (
    AUTHORIZATION,
    INCIDENT,
    PROJECT,
    RUN_ID,
    inspect_interrupted_state,
)
from run_agent_baseline import sha256_file


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUN = WORKSPACE_ROOT / "baseline_data/exp_agent_ref_10/runs" / RUN_ID
SERVICE = "exp-agent-ref-10-v4-formal-03.service"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def main() -> None:
    condition = RUN / "vanilla" / PROJECT
    scheduler_path = RUN / "formal_scheduler/manifest.json"
    condition_manifest_path = condition / "run_manifest.json"
    scheduler = read_json(scheduler_path)
    manifest = read_json(condition_manifest_path)
    state = inspect_interrupted_state(condition)
    trace_paths = sorted((condition / "traces").glob("round_01_attempt_*.jsonl"))
    thread_ids = []
    for path in trace_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"thread.started", "session.started"}:
                value = (
                    event.get("thread_id") or event.get("session_id") or event.get("id")
                )
                if isinstance(value, str):
                    thread_ids.append(value)
                break
    task = next(item for item in scheduler["tasks"] if item["project"] == PROJECT)
    current_boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    )
    current_boot_started = command_output(["uptime", "-s"])
    prior_service_journal = command_output(
        [
            "journalctl",
            "--user",
            "-b",
            "-1",
            "-u",
            SERVICE,
            "--no-pager",
            "-o",
            "short-iso",
        ]
    )
    authorization = read_json(AUTHORIZATION)
    interrupted = {
        "project": PROJECT,
        "condition_manifest": str(condition_manifest_path),
        "condition_status": manifest.get("status"),
        "scheduler_task_status": task.get("status"),
        "rounds_recorded": len(manifest.get("rounds") or []),
        "completed_attempt_count": len(trace_paths),
        "completed_trace_paths": [str(path) for path in trace_paths],
        "completed_trace_sha256": [sha256_file(path) for path in trace_paths],
        "thread_id": thread_ids[0] if len(set(thread_ids)) == 1 else None,
        "same_thread_all_completed_attempts": len(set(thread_ids)) == 1,
        "attempt_03_patch_sha256": state["historical_patches"][-1]["sha256"],
        "partial_attempt_04_patch_sha256": state["current_partial_candidate"][
            "patch_sha256"
        ],
        "partial_attempt_04_changed_source_file_count": state[
            "current_partial_candidate"
        ]["changed_source_file_count"],
        "attempt_04_trace_present": (
            condition / "traces/round_01_attempt_04.jsonl"
        ).exists(),
        "completion_report_present": state["completion_report_present"],
        "validation_scans_consumed": state["validation_scans_consumed"],
        "round_snapshot": str(condition / "evaluator_state/rounds/round_01/snapshot"),
    }
    passed = (
        scheduler.get("run_id") == RUN_ID
        and manifest.get("run_id") == RUN_ID
        and manifest.get("project") == PROJECT
        and manifest.get("status") == "preparing"
        and task.get("status") == "active"
        and len(manifest.get("rounds") or []) == 0
        and len(trace_paths) == 3
        and interrupted["same_thread_all_completed_attempts"]
        and interrupted["thread_id"] == "019fd7bc-4ec9-7da2-bed1-cb77dfd382cd"
        and interrupted["attempt_03_patch_sha256"]
        == "248e03e038a619af54b5a8836ede046ac3afe1e08ac80d890fe92b35cc9b8892"
        and interrupted["partial_attempt_04_patch_sha256"]
        == "df9b049e5e01e592ff2a01715157c0c8e85f155ad11364a06660d3886a599557"
        and not interrupted["attempt_04_trace_present"]
        and not interrupted["completion_report_present"]
        and interrupted["validation_scans_consumed"] == 0
        and "Started exp-agent-ref-10-v4-formal-03.service" in prior_service_journal
        and current_boot_started == "2026-08-07 01:07:07"
        and authorization.get("authorized_run_id") == RUN_ID
        and authorization.get("passed") is True
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": ("closed_reboot_only_continuation_authorized" if passed else "open"),
        "passed": passed,
        "classification": "host reboot during an active model turn",
        "failure_layer": "environment",
        "scheduler_manifest": str(scheduler_path),
        "scheduler_manifest_status": scheduler.get("status"),
        "service": SERVICE,
        "prior_boot_service_journal": prior_service_journal,
        "current_boot_id": current_boot_id,
        "current_boot_started_local": current_boot_started,
        "interrupted_condition": interrupted,
        "authorization": {
            "path": str(AUTHORIZATION),
            "sha256": sha256_file(AUTHORIZATION),
            "authorized_run_id": authorization.get("authorized_run_id"),
        },
        "continuation_policy": {
            "fresh_stochastic_condition": False,
            "same_run_id": RUN_ID,
            "same_project": PROJECT,
            "same_active_round": 1,
            "same_thread_id": interrupted["thread_id"],
            "next_attempt": 4,
            "maximum_attempt": 6,
            "archive_partial_attempt_04_before_restoration": True,
            "restore_completed_attempt_03_exactly": True,
            "continue_existing_validation_budget": True,
        },
        "claim_boundary": (
            "This is an environment-only continuation inside the existing formal_03 "
            "condition. It does not authorize a new condition, a new seed, a new run "
            "ID, a protocol change, or reruns of the three completed guard failures."
        ),
    }
    INCIDENT.parent.mkdir(parents=True, exist_ok=True)
    INCIDENT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{INCIDENT}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
