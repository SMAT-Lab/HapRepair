#!/usr/bin/env python3
"""Preserve the pre-mutation resume launch that used the wrong host Python."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import resume_agent_ref_interrupted_v4 as resume


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OLD_AUDIT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_resume_g2_20260807.json"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_01_python_incident_20260807.json"
)
SERVICE_LOG = (
    resume.DEFAULT_RUN_ROOT
    / resume.RUN_ID
    / "formal_scheduler/resume_applications_permission_manager.log"
)


def version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    return (result.stdout + result.stderr).strip()


def main() -> None:
    run_dir = resume.DEFAULT_RUN_ROOT / resume.RUN_ID / "vanilla" / resume.PROJECT
    manifest = resume.read_json(run_dir / "run_manifest.json")
    scanner = resume.read_json(run_dir / "evaluator_state/homecheck/state.json")
    old_audit = resume.read_json(OLD_AUDIT)
    log = SERVICE_LOG.read_text(encoding="utf-8", errors="replace")
    system_python = version("/usr/bin/python3")
    frozen_python = version("/data/zhihao/miniconda3/bin/python3.13")
    passed = (
        old_audit.get("passed") is True
        and old_audit.get("resume_authorized") is True
        and manifest.get("status") == "preparing"
        and manifest.get("rounds") == []
        and scanner.get("validation_scans_consumed") == 0
        and not (run_dir / "evaluator_state/interruption_recovery").exists()
        and not (run_dir / "traces/round_01_attempt_04.jsonl").exists()
        and "Container tool versions differ" in log
        and system_python == "Python 3.12.3"
        and frozen_python == "Python 3.13.11"
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": resume.RUN_ID,
        "project": resume.PROJECT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed_excluded_pre_mutation_launcher_failure" if passed else "open",
        "passed": passed,
        "classification": "resume service selected Python 3.12 instead of frozen Python 3.13",
        "failure_layer": "environment",
        "old_authorization": {
            "path": str(OLD_AUDIT),
            "sha256": resume.sha256_file(OLD_AUDIT),
        },
        "service_log": str(SERVICE_LOG),
        "service_log_sha256": resume.sha256_file(SERVICE_LOG),
        "observed_system_python": system_python,
        "required_frozen_python": frozen_python,
        "condition_unchanged": {
            "status": manifest.get("status"),
            "rounds": len(manifest.get("rounds") or []),
            "validation_scans_consumed": scanner.get("validation_scans_consumed"),
            "recovery_archive_created": (
                run_dir / "evaluator_state/interruption_recovery"
            ).exists(),
            "attempt_04_trace_created": (
                run_dir / "traces/round_01_attempt_04.jsonl"
            ).exists(),
        },
        "remediation": (
            "Pin the oneshot resume service to /data/zhihao/miniconda3/bin/python3.13, "
            "extend the dedicated audit to verify that executable, and issue a new "
            "authorization artifact without changing the experiment condition."
        ),
        "claim_boundary": (
            "No workspace restoration, model turn, HomeCheck validation, or condition "
            "manifest mutation occurred. This launcher failure is excluded."
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
