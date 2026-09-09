#!/usr/bin/env python3
"""Authorize the reboot-only continuation of one formal_03 condition."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import resume_agent_ref_interrupted_v4 as resume


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
OUTPUT = resume.RESUME_AUTHORIZATION
SERVICE = (
    Path.home() / ".config/systemd/user/exp-agent-ref-10-v4-formal-03-resume.service"
)
TEST = HERE / "test_resume_agent_ref_interrupted_v4.py"
RECORDER = HERE / "record_agent_ref_formal_03_reboot_incident.py"
LAUNCH_RECORDER = HERE / "record_agent_ref_formal_03_resume_launch_incident.py"
LAUNCH_INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_01_python_incident_20260807.json"
)
RECONSTRUCTION_RECORDER = (
    HERE / "record_agent_ref_formal_03_resume_reconstruction_incident.py"
)
RECONSTRUCTION_INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_resume_launch_02_reconstruction_incident_20260807.json"
)
FROZEN_PYTHON = Path("/data/zhihao/miniconda3/bin/python3.13")


def add(checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **evidence})


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    checks: list[dict[str, Any]] = []
    run_dir = resume.DEFAULT_RUN_ROOT / resume.RUN_ID / "vanilla" / resume.PROJECT
    manifest_path = run_dir / "run_manifest.json"
    scheduler_path = (
        resume.DEFAULT_RUN_ROOT / resume.RUN_ID / "formal_scheduler/manifest.json"
    )
    incident = resume.read_json(resume.INCIDENT)
    manifest = resume.read_json(manifest_path)
    scheduler = resume.read_json(scheduler_path)
    task = next(
        item for item in scheduler["tasks"] if item["project"] == resume.PROJECT
    )
    state = resume.inspect_interrupted_state(run_dir)

    add(
        checks,
        "closed_host_reboot_incident",
        incident.get("status") == "closed_reboot_only_continuation_authorized"
        and incident.get("passed") is True
        and incident.get("classification") == "host reboot during an active model turn",
        incident=str(resume.INCIDENT),
        incident_sha256=resume.sha256_file(resume.INCIDENT),
    )

    launch_incident = resume.read_json(LAUNCH_INCIDENT)
    add(
        checks,
        "excluded_pre_mutation_python_launcher_failure",
        launch_incident.get("status") == "closed_excluded_pre_mutation_launcher_failure"
        and launch_incident.get("passed") is True
        and launch_incident.get("condition_unchanged", {}).get(
            "validation_scans_consumed"
        )
        == 0
        and launch_incident.get("condition_unchanged", {}).get(
            "attempt_04_trace_created"
        )
        is False,
        incident=str(LAUNCH_INCIDENT),
        incident_sha256=resume.sha256_file(LAUNCH_INCIDENT),
    )

    reconstruction_incident = resume.read_json(RECONSTRUCTION_INCIDENT)
    add(
        checks,
        "excluded_pre_model_patch_reconstruction_failure",
        reconstruction_incident.get("status")
        == "closed_excluded_pre_model_reconstruction_failure"
        and reconstruction_incident.get("passed") is True
        and reconstruction_incident.get("condition_unchanged", {}).get(
            "validation_scans_consumed"
        )
        == 0
        and reconstruction_incident.get("condition_unchanged", {}).get(
            "attempt_04_trace_created"
        )
        is False,
        incident=str(RECONSTRUCTION_INCIDENT),
        incident_sha256=resume.sha256_file(RECONSTRUCTION_INCIDENT),
    )

    try:
        frozen = resume.validate_frozen_authorization(
            resume.DEFAULT_PROTOCOL, resume.DEFAULT_INPUTS, run_dir
        )
        frozen_error = None
    except Exception as error:  # pragma: no cover - evidence collection
        frozen = None
        frozen_error = f"{type(error).__name__}: {error}"
    add(
        checks,
        "formal_03_g2_and_frozen_harness_unchanged",
        frozen is not None,
        authorization=frozen,
        error=frozen_error,
    )

    partial = incident["interrupted_condition"]
    add(
        checks,
        "single_interrupted_condition_identity",
        manifest.get("run_id") == resume.RUN_ID
        and manifest.get("project") == resume.PROJECT
        and manifest.get("status") == "preparing"
        and manifest.get("rounds") == []
        and task.get("status") == "active"
        and partial["project"] == resume.PROJECT,
        condition_manifest=str(manifest_path),
        scheduler_manifest=str(scheduler_path),
        condition_status=manifest.get("status"),
        task_status=task.get("status"),
    )

    observed_patch = state["historical_patches"][-1]["sha256"]
    observed_partial = state["current_partial_candidate"]["patch_sha256"]
    archived_partial_path = (
        run_dir
        / "evaluator_state/interruption_recovery"
        / resume.RECOVERY_ID
        / "partial_attempt_04.patch"
    )
    archived_partial = (
        resume.sha256_file(archived_partial_path)
        if archived_partial_path.is_file()
        else None
    )
    add(
        checks,
        "interrupted_attempt_state_is_byte_identical",
        observed_patch == partial["attempt_03_patch_sha256"]
        and archived_partial == partial["partial_attempt_04_patch_sha256"]
        and observed_partial
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        and state["validation_scans_consumed"] == 0
        and state["completion_report_present"] is False
        and not (run_dir / "traces/round_01_attempt_04.jsonl").exists(),
        completed_attempt_03_patch_sha256=observed_patch,
        archived_partial_attempt_04_patch_sha256=archived_partial,
        current_workspace_patch_sha256=observed_partial,
        validation_scans_consumed=state["validation_scans_consumed"],
    )

    try:
        thread, commands, usage, recovered = resume.historical_trace_state(
            run_dir / "traces"
        )
        trace_error = None
    except Exception as error:  # pragma: no cover - evidence collection
        thread, commands, usage, recovered = None, [], {}, []
        trace_error = f"{type(error).__name__}: {error}"
    restricted = [
        item for item in commands if resume.is_restricted_command(item["command"])
    ]
    add(
        checks,
        "three_completed_turns_same_active_round_thread",
        thread == partial["thread_id"]
        and len(recovered) == 3
        and not restricted
        and int(usage.get("input_tokens", 0)) > 0
        and int(usage.get("output_tokens", 0)) > 0,
        thread_id=thread,
        recovered_attempts=len(recovered),
        usage=dict(usage),
        restricted_commands=restricted,
        error=trace_error,
    )

    source_text = Path(resume.__file__).read_text(encoding="utf-8")
    archive_index = source_text.index(
        "guards.snapshot_workspace(", source_text.index("def archive_and_restore")
    )
    restore_index = source_text.index(
        "guards.restore_snapshot(", source_text.index("def archive_and_restore")
    )
    add(
        checks,
        "partial_archive_before_deterministic_reconstruction",
        archive_index < restore_index
        and "normalize_difflib_patch(" in source_text
        and "patch_workspace(workspace, Path(normalized_patch" in source_text
        and "byte_exact_patch_reconstruction" in source_text
        and "partial_attempt_04_patch_sha256" in source_text,
        resume_script=str(Path(resume.__file__)),
    )

    add(
        checks,
        "same_condition_thread_and_budget_continuation",
        "start_attempt = 4" in source_text
        and "range(start_attempt, MAX_AGENT_TURNS_PER_ROUND + 1)" in source_text
        and "active_thread = thread_id" in source_text
        and "validation_scans_consumed" in source_text
        and '"fresh_stochastic_condition_started": False' in source_text
        and "copy_project" not in source_text,
        maximum_attempt=resume.MAX_AGENT_TURNS_PER_ROUND,
        maximum_validation_scans=resume.read_json(resume.DEFAULT_PROTOCOL)[
            "common_contract"
        ]["maximum_post_edit_validation_scans"],
    )

    with tempfile.TemporaryDirectory() as temporary:
        normalized = resume.normalize_difflib_patch(
            run_dir / "evaluator_state/rounds/round_01/snapshot/files",
            run_dir / "evaluator_state/rounds/round_01/attempt_03.patch",
            Path(temporary) / "normalized_attempt_03.patch",
        )
        patch_check = subprocess.run(
            [
                "patch",
                "--dry-run",
                "--batch",
                "--forward",
                "-p1",
                "-i",
                normalized["path"],
            ],
            cwd=run_dir / "workspace",
            capture_output=True,
            text=True,
            check=False,
        )
    add(
        checks,
        "live_no_newline_patch_normalization",
        normalized["repaired_no_newline_boundaries"] == 6
        and patch_check.returncode == 0,
        normalized_patch=normalized,
        patch_dry_run_returncode=patch_check.returncode,
        patch_dry_run_output=(patch_check.stdout + patch_check.stderr)[-4000:],
    )

    tests = run([sys.executable, "-m", "unittest", "-v", TEST.name])
    add(
        checks,
        "focused_reconstruction_and_resume_tests",
        tests.returncode == 0 and "Ran 5 tests" in tests.stderr,
        returncode=tests.returncode,
        stdout=tests.stdout[-2000:],
        stderr=tests.stderr[-4000:],
    )

    lint = run(
        [
            "ruff",
            "check",
            Path(resume.__file__).name,
            TEST.name,
            RECORDER.name,
            LAUNCH_RECORDER.name,
            RECONSTRUCTION_RECORDER.name,
            Path(__file__).name,
        ]
    )
    compile_result = run(
        [
            sys.executable,
            "-m",
            "py_compile",
            Path(resume.__file__).name,
            TEST.name,
            RECORDER.name,
            LAUNCH_RECORDER.name,
            RECONSTRUCTION_RECORDER.name,
            Path(__file__).name,
        ]
    )
    add(
        checks,
        "python_compile_and_ruff",
        lint.returncode == 0 and compile_result.returncode == 0,
        lint_returncode=lint.returncode,
        lint_output=(lint.stdout + lint.stderr)[-3000:],
        compile_returncode=compile_result.returncode,
        compile_output=(compile_result.stdout + compile_result.stderr)[-3000:],
    )

    audit_state = run([sys.executable, Path(resume.__file__).name, "--audit-state"])
    try:
        audit_payload = json.loads(audit_state.stdout)
    except json.JSONDecodeError:
        audit_payload = {}
    add(
        checks,
        "read_only_live_resume_preflight",
        audit_state.returncode == 0
        and audit_payload.get("status") == "audited"
        and audit_payload.get("project") == resume.PROJECT
        and audit_payload.get("current_partial_candidate", {}).get("patch_sha256")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        and resume.sha256_file(
            run_dir
            / "evaluator_state/interruption_recovery"
            / resume.RECOVERY_ID
            / "partial_attempt_04.patch"
        )
        == partial["partial_attempt_04_patch_sha256"],
        returncode=audit_state.returncode,
        payload=audit_payload,
        stderr=audit_state.stderr[-2000:],
    )

    service_text = SERVICE.read_text(encoding="utf-8") if SERVICE.is_file() else ""
    python_version = run([str(FROZEN_PYTHON), "--version"])
    add(
        checks,
        "dedicated_oneshot_systemd_service",
        "Type=oneshot" in service_text
        and str(Path(resume.__file__).resolve()) in service_text
        and str(resume.RESUME_AUTHORIZATION) in service_text
        and f"ExecStart={FROZEN_PYTHON} " in service_text
        and (python_version.stdout + python_version.stderr).strip() == "Python 3.13.11"
        and "Restart=" not in service_text
        and "run_formal_agent_ref_v4.py" not in service_text,
        service=str(SERVICE),
        service_sha256=resume.sha256_file(SERVICE) if SERVICE.is_file() else None,
        frozen_python=str(FROZEN_PYTHON),
        frozen_python_version=(python_version.stdout + python_version.stderr).strip(),
    )

    add(
        checks,
        "forbidden_formal_04_and_failed_condition_reruns_absent",
        not (resume.DEFAULT_RUN_ROOT / "exp_agent_ref_10_luna_v4_formal_04").exists()
        and resume.PROJECT in source_text
        and "acts_validator" not in source_text
        and "bluetoothtest" not in source_text
        and "ohos_cordova" not in source_text,
        forbidden_formal_04=str(
            resume.DEFAULT_RUN_ROOT / "exp_agent_ref_10_luna_v4_formal_04"
        ),
    )

    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "gate": "formal_03 reboot-only condition continuation authorization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if passed else "open_failed_audit",
        "passed": passed,
        "resume_authorized": passed,
        "authorized_run_id": resume.RUN_ID,
        "authorized_project": resume.PROJECT,
        "authorized_next_attempt": 4,
        "resume_script": str(Path(resume.__file__).resolve()),
        "resume_script_sha256": resume.sha256_file(Path(resume.__file__).resolve()),
        "incident": str(resume.INCIDENT),
        "incident_sha256": resume.sha256_file(resume.INCIDENT),
        "checks": checks,
        "claim_boundary": (
            "Authorization is limited to deterministic restoration followed by the "
            "same active-round thread and remaining attempt/scan budget for "
            "applications_permission_manager inside formal_03."
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
