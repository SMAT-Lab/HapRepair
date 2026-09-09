#!/usr/bin/env python3
"""Preserve the failed standard-patch reconstruction before any model turn."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import resume_agent_ref_interrupted_v4 as resume


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_02_reconstruction_incident_20260807.json"
)
SERVICE_LOG = (
    resume.DEFAULT_RUN_ROOT
    / resume.RUN_ID
    / "formal_scheduler/resume_applications_permission_manager.log"
)


def main() -> None:
    run_dir = resume.DEFAULT_RUN_ROOT / resume.RUN_ID / "vanilla" / resume.PROJECT
    recovery = run_dir / "evaluator_state/interruption_recovery" / resume.RECOVERY_ID
    manifest = resume.read_json(run_dir / "run_manifest.json")
    scanner = resume.read_json(run_dir / "evaluator_state/homecheck/state.json")
    log = SERVICE_LOG.read_text(encoding="utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as temporary:
        current = resume.guards.source_diff(
            run_dir / "evaluator_state/rounds/round_01/snapshot/files",
            run_dir / "workspace",
            Path(temporary) / "current.patch",
        )
    partial_patch = recovery / "partial_attempt_04.patch"
    passed = (
        manifest.get("status") == "preparing"
        and manifest.get("rounds") == []
        and scanner.get("validation_scans_consumed") == 0
        and current["patch_sha256"]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        and partial_patch.is_file()
        and resume.sha256_file(partial_patch)
        == "df9b049e5e01e592ff2a01715157c0c8e85f155ad11364a06660d3886a599557"
        and (recovery / "partial_attempt_04/manifest.json").is_file()
        and (recovery / "partial_attempt_04_control/plan.json").is_file()
        and not (run_dir / "traces/round_01_attempt_04.jsonl").exists()
        and "malformed patch at line 35" in log
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": resume.RUN_ID,
        "project": resume.PROJECT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed_excluded_pre_model_reconstruction_failure"
        if passed
        else "open",
        "passed": passed,
        "classification": (
            "difflib patch with a missing-EOF-newline boundary was rejected by "
            "the standard patch utility"
        ),
        "failure_layer": "implementation",
        "service_log": str(SERVICE_LOG),
        "service_log_sha256": resume.sha256_file(SERVICE_LOG),
        "condition_unchanged": {
            "status": manifest.get("status"),
            "rounds": len(manifest.get("rounds") or []),
            "validation_scans_consumed": scanner.get("validation_scans_consumed"),
            "workspace_vs_round_snapshot_patch_sha256": current["patch_sha256"],
            "attempt_04_trace_created": (
                run_dir / "traces/round_01_attempt_04.jsonl"
            ).exists(),
        },
        "preserved_partial_attempt_04": {
            "recovery_directory": str(recovery),
            "patch": str(partial_patch),
            "patch_sha256": resume.sha256_file(partial_patch),
            "snapshot_manifest": str(recovery / "partial_attempt_04/manifest.json"),
            "control_archive": str(recovery / "partial_attempt_04_control"),
        },
        "remediation": (
            "Normalize difflib's concatenated no-newline record using the exact "
            "round snapshot, test this boundary deterministically, restore the round "
            "snapshot again, and require the regenerated candidate patch to retain "
            "the original attempt-3 SHA-256."
        ),
        "claim_boundary": (
            "The partial attempt-4 state was archived before restoration. No model "
            "turn, validation scan, or condition-manifest mutation occurred."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
