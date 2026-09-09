#!/usr/bin/env python3
"""Resume the one formal_03 condition interrupted by the 2026-08-07 reboot."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
SKILL_SCRIPTS = HAPREPAIR_ROOT / "skills/haprepair-openharmony-repair-v14-dev/scripts"
for import_dir in (BASELINE_DIR, SKILL_SCRIPTS):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import homecheck  # type: ignore  # noqa: E402
import repair_session as guards  # type: ignore  # noqa: E402
from agent_ref_session import (  # noqa: E402
    audit_completion,
    candidate_score,
    completion_schema,
    describe_interactions,
    is_restricted_command,
    make_reference_plan,
    read_json,
    write_json,
)
from build_gate import prepare_build_gate, run_build_gate  # type: ignore  # noqa: E402
from run_agent_baseline import (  # type: ignore  # noqa: E402
    compute_alert_metrics,
    extract_commands,
    extract_thread_id,
    extract_usage,
    parse_codex_events,
    provider_endpoint_fingerprint,
    sha256_file,
    source_diff,
    utc_now,
)
from run_agent_ref_v4 import (  # noqa: E402
    DEFAULT_IMAGE,
    MAX_AGENT_TURNS_PER_ROUND,
    configure_host_runtime,
    findings_from_scan,
    load_contract,
    per_rule_deltas,
    prompt_for_attempt,
    run_codex_turn,
    validate_runtime,
)


RUN_ID = "exp_agent_ref_10_luna_v4_formal_03"
PROJECT = "applications_permission_manager"
DEFAULT_PROTOCOL = HERE / "protocol-agent-ref-10-v4-formal-03.json"
DEFAULT_INPUTS = HERE / "formal_inputs_agent_ref_10_v4.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_agent_ref_10/runs"
AUTHORIZATION = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_retry_g2_20260806.json"
)
INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_03_reboot_incident_20260807.json"
)
RESUME_AUTHORIZATION = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_resume_retry_03_g2_20260807.json"
)
RECOVERY_ID = "formal_03_host_reboot_20260807"
EXPECTED_COMPLETED_ATTEMPTS = 3


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_trace(path: Path) -> tuple[str | None, list[dict[str, Any]], Counter[str]]:
    events = parse_codex_events(path.read_text(encoding="utf-8", errors="replace"))
    usage: Counter[str] = Counter()
    usage.update(extract_usage(events))
    return extract_thread_id(events), extract_commands(events), usage


def historical_trace_state(
    traces: Path,
) -> tuple[str, list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    thread_ids: list[str] = []
    all_commands: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for attempt in range(1, EXPECTED_COMPLETED_ATTEMPTS + 1):
        path = traces / f"round_01_attempt_{attempt:02d}.jsonl"
        patch_path = (
            traces.parent
            / "evaluator_state/rounds/round_01"
            / f"attempt_{attempt:02d}.patch"
        )
        preflight_path = (
            traces.parent
            / "evaluator_state/rounds/round_01"
            / f"preflight_{attempt:02d}"
        )
        if not path.is_file():
            raise RuntimeError(f"Missing completed historical trace: {path}")
        if not patch_path.is_file() or not preflight_path.is_dir():
            raise RuntimeError(
                f"Historical attempt {attempt} lacks its patch or preflight evidence"
            )
        raw_trace = path.read_text(encoding="utf-8", errors="replace")
        thread_id, commands, turn_usage = parse_trace(path)
        if not thread_id:
            raise RuntimeError(f"Historical trace has no thread identity: {path}")
        if not any(
            event.get("type") == "turn.completed"
            for event in parse_codex_events(raw_trace)
        ):
            raise RuntimeError(f"Historical trace is not a completed turn: {path}")
        thread_ids.append(thread_id)
        all_commands.extend(commands)
        usage.update(turn_usage)
        records.append(
            {
                "attempt": attempt,
                "thread_id": thread_id,
                "recovered_after_reboot": True,
                "turn": {
                    "thread_id": thread_id,
                    "usage": dict(turn_usage),
                    "commands": commands,
                    "trace_path": str(path),
                    "trace_sha256": sha256_file(path),
                    "stderr_path": str(
                        traces / f"round_01_attempt_{attempt:02d}.stderr.log"
                    ),
                },
                "coverage": {
                    "complete": True,
                    "recovery_evidence": (
                        "The frozen runner created preflight artifacts only after "
                        "the completion and source-diff gates passed."
                    ),
                },
                "edit": {
                    "patch_path": str(patch_path),
                    "patch_sha256": sha256_file(patch_path),
                },
                "preflight": {
                    "status": "failed_retained_candidate",
                    "recovery_evidence": str(preflight_path),
                },
            }
        )
    if len(set(thread_ids)) != 1:
        raise RuntimeError(f"Historical attempts used different threads: {thread_ids}")
    extra = sorted(traces.glob("round_01_attempt_04.jsonl"))
    if extra:
        raise RuntimeError(
            f"Attempt 4 already has a trace and needs a new audit: {extra}"
        )
    return thread_ids[0], all_commands, usage, records


def validate_frozen_authorization(
    protocol_path: Path, inputs_path: Path, run_dir: Path
) -> dict[str, Any]:
    authorization = read_json(AUTHORIZATION)
    if (
        authorization.get("status") != "closed"
        or authorization.get("passed") is not True
        or authorization.get("formal_execution_authorized") is not True
        or authorization.get("authorized_run_id") != RUN_ID
    ):
        raise RuntimeError("formal_03 G2 authorization is not closed and passing")
    artifacts = {
        "agent_ref_session.py": HERE / "agent_ref_session.py",
        "run_agent_ref_v4.py": HERE / "run_agent_ref_v4.py",
        "run_formal_agent_ref_v4.py": HERE / "run_formal_agent_ref_v4.py",
        "test_agent_ref_v4.py": HERE / "test_agent_ref_v4.py",
        "audit_agent_ref_v4.py": HERE / "audit_agent_ref_v4.py",
        "audit_agent_ref_v4_formal_03_retry.py": HERE
        / "audit_agent_ref_v4_formal_03_retry.py",
    }
    expected = authorization.get("runner_artifacts") or {}
    drift = {
        name: {"expected": expected.get(name), "observed": sha256_file(path)}
        for name, path in artifacts.items()
        if expected.get(name) != sha256_file(path)
    }
    contract = next(
        item
        for item in authorization["checks"]
        if item.get("name") == "frozen_contract_and_exact_order"
    )
    if contract.get("protocol_sha256") != sha256_file(protocol_path):
        drift["protocol"] = {
            "expected": contract.get("protocol_sha256"),
            "observed": sha256_file(protocol_path),
        }
    if contract.get("inputs_sha256") != sha256_file(inputs_path):
        drift["inputs"] = {
            "expected": contract.get("inputs_sha256"),
            "observed": sha256_file(inputs_path),
        }
    manifest = read_json(run_dir / "run_manifest.json")
    if (
        manifest.get("run_id") != RUN_ID
        or manifest.get("project") != PROJECT
        or manifest.get("protocol_sha256") != sha256_file(protocol_path)
        or manifest.get("inputs_sha256") != sha256_file(inputs_path)
    ):
        drift["condition_identity"] = {"observed": manifest}
    if drift:
        raise RuntimeError(f"Frozen formal_03 artifacts drifted: {drift}")
    return {
        "path": str(AUTHORIZATION),
        "sha256": sha256_file(AUTHORIZATION),
        "verified_runner_artifacts": expected,
    }


def validate_resume_authorization() -> dict[str, Any]:
    if not RESUME_AUTHORIZATION.is_file():
        raise RuntimeError(f"Resume authorization is missing: {RESUME_AUTHORIZATION}")
    authorization = read_json(RESUME_AUTHORIZATION)
    observed_script = sha256_file(Path(__file__).resolve())
    if (
        authorization.get("status") != "closed"
        or authorization.get("passed") is not True
        or authorization.get("resume_authorized") is not True
        or authorization.get("authorized_run_id") != RUN_ID
        or authorization.get("authorized_project") != PROJECT
        or authorization.get("resume_script_sha256") != observed_script
        or any(not item.get("passed") for item in authorization.get("checks", []))
    ):
        raise RuntimeError("Dedicated formal_03 resume authorization is not passing")
    return {
        "path": str(RESUME_AUTHORIZATION),
        "sha256": sha256_file(RESUME_AUTHORIZATION),
        "resume_script_sha256": observed_script,
    }


def snapshot_descriptor(path: Path) -> dict[str, Any]:
    return {
        "files_dir": str(path / "files"),
        "manifest_path": str(path / "manifest.json"),
        "file_count": len(read_json(path / "manifest.json")),
    }


HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def normalize_difflib_patch(
    original_root: Path, patch_path: Path, output_path: Path
) -> dict[str, Any]:
    """Add standard EOF markers where difflib concatenated patch records."""
    pending = patch_path.read_text(encoding="utf-8").splitlines(keepends=True)
    normalized: list[str] = []
    original_lines: list[str] | None = None
    old_index = 0
    old_remaining = 0
    new_remaining = 0
    index = 0
    repaired_boundaries = 0
    while index < len(pending):
        line = pending[index]
        if old_remaining == 0 and new_remaining == 0:
            if line.startswith("--- a/"):
                relative = line[len("--- a/") :].rstrip("\r\n")
                original_lines = (
                    (original_root / relative)
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines(keepends=True)
                )
                normalized.append(line)
                index += 1
                if index >= len(pending) or not pending[index].startswith("+++ b/"):
                    raise RuntimeError(f"Malformed file header in {patch_path}")
                normalized.append(pending[index])
                index += 1
                continue
            match = HUNK_HEADER.match(line)
            if match:
                if original_lines is None:
                    raise RuntimeError(f"Hunk precedes a file header in {patch_path}")
                old_index = int(match.group(1)) - 1
                old_remaining = int(match.group(2) or "1")
                new_remaining = int(match.group(4) or "1")
                normalized.append(line)
                index += 1
                continue
            normalized.append(line)
            index += 1
            continue

        prefix = line[:1]
        if prefix not in {" ", "-", "+"}:
            raise RuntimeError(f"Unexpected patch record in {patch_path}: {line!r}")
        if prefix in {" ", "-"}:
            if original_lines is None or old_index >= len(original_lines):
                raise RuntimeError(f"Patch exceeds original file in {patch_path}")
            expected = original_lines[old_index]
            expected_record = prefix + expected
            if expected.endswith(("\n", "\r")):
                if line != expected_record:
                    raise RuntimeError(
                        f"Patch context differs from its frozen snapshot: {line!r}"
                    )
                normalized.append(line)
            else:
                marker = prefix + expected
                if not line.startswith(marker):
                    raise RuntimeError(
                        f"EOF patch context differs from its frozen snapshot: {line!r}"
                    )
                normalized.append(marker + "\n")
                normalized.append("\\ No newline at end of file\n")
                remainder = line[len(marker) :]
                if remainder:
                    pending.insert(index + 1, remainder)
                    repaired_boundaries += 1
            old_index += 1
            old_remaining -= 1
            if prefix == " ":
                new_remaining -= 1
        else:
            normalized.append(line)
            new_remaining -= 1
        if old_remaining < 0 or new_remaining < 0:
            raise RuntimeError(f"Patch hunk counts underflow in {patch_path}")
        index += 1
    if old_remaining or new_remaining:
        raise RuntimeError(f"Patch ended inside a hunk: {patch_path}")
    output_path.write_text("".join(normalized), encoding="utf-8")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "repaired_no_newline_boundaries": repaired_boundaries,
    }


def patch_workspace(workspace: Path, patch_path: Path) -> None:
    result = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch_path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Cannot reconstruct completed candidate from {patch_path}: "
            f"{result.stdout}\n{result.stderr}"
        )


def inspect_interrupted_state(run_dir: Path) -> dict[str, Any]:
    workspace = run_dir / "workspace"
    round_dir = run_dir / "evaluator_state/rounds/round_01"
    scanner = read_json(run_dir / "evaluator_state/homecheck/state.json")
    with tempfile.TemporaryDirectory() as temporary:
        current = guards.source_diff(
            round_dir / "snapshot/files",
            workspace,
            Path(temporary) / "partial_attempt_04.patch",
        )
    patches = [
        {
            "attempt": attempt,
            "path": str(round_dir / f"attempt_{attempt:02d}.patch"),
            "sha256": sha256_file(round_dir / f"attempt_{attempt:02d}.patch"),
        }
        for attempt in range(1, EXPECTED_COMPLETED_ATTEMPTS + 1)
    ]
    return {
        "completion_report_present": (
            workspace / ".exp_agent/round_01/completion.json"
        ).exists(),
        "validation_scans_consumed": scanner["validation_scans_consumed"],
        "historical_patches": patches,
        "current_partial_candidate": current,
    }


def archive_and_restore(run_dir: Path, incident: dict[str, Any]) -> dict[str, Any]:
    workspace = run_dir / "workspace"
    round_dir = run_dir / "evaluator_state/rounds/round_01"
    recovery = run_dir / "evaluator_state/interruption_recovery" / RECOVERY_ID
    control_archive = recovery / "partial_attempt_04_control"
    expected_partial = incident["interrupted_condition"][
        "partial_attempt_04_patch_sha256"
    ]
    if recovery.exists():
        partial = snapshot_descriptor(recovery / "partial_attempt_04")
        partial_patch_path = recovery / "partial_attempt_04.patch"
        if (
            not partial_patch_path.is_file()
            or sha256_file(partial_patch_path) != expected_partial
            or not control_archive.is_dir()
        ):
            raise RuntimeError(
                "Existing partial-attempt archive is incomplete or drifted"
            )
        partial_patch = {
            "patch_path": str(partial_patch_path),
            "patch_sha256": sha256_file(partial_patch_path),
            "reused_after_failed_standard_patch_reconstruction": True,
        }
    else:
        recovery.mkdir(parents=True)
        partial = guards.snapshot_workspace(workspace, recovery / "partial_attempt_04")
        partial_patch = guards.source_diff(
            round_dir / "snapshot/files",
            workspace,
            recovery / "partial_attempt_04.patch",
        )
        shutil.copytree(workspace / ".exp_agent/round_01", control_archive)
        if partial_patch["patch_sha256"] != expected_partial:
            raise RuntimeError(
                "Interrupted workspace changed after incident audit: "
                f"{partial_patch['patch_sha256']} != {expected_partial}"
            )

    round_snapshot = snapshot_descriptor(round_dir / "snapshot")
    guards.restore_snapshot(workspace, round_snapshot)
    completed_patch = round_dir / "attempt_03.patch"
    normalized_patch = normalize_difflib_patch(
        round_dir / "snapshot/files",
        completed_patch,
        recovery / "normalized_attempt_03.patch",
    )
    patch_workspace(workspace, Path(normalized_patch["path"]))
    reconstructed = guards.source_diff(
        round_dir / "snapshot/files",
        workspace,
        recovery / "reconstructed_attempt_03.patch",
    )
    expected_completed = incident["interrupted_condition"]["attempt_03_patch_sha256"]
    if reconstructed["patch_sha256"] != expected_completed:
        raise RuntimeError(
            "Reconstructed candidate differs from completed attempt 3: "
            f"{reconstructed['patch_sha256']} != {expected_completed}"
        )
    result = {
        "recovery_id": RECOVERY_ID,
        "archived_at": utc_now(),
        "partial_attempt_04_snapshot": partial,
        "partial_attempt_04_patch": partial_patch,
        "partial_attempt_04_control_archive": str(control_archive),
        "restored_from_round_snapshot": round_snapshot,
        "applied_completed_patch": str(completed_patch),
        "normalized_completed_patch": normalized_patch,
        "reconstructed_attempt_03": reconstructed,
        "byte_exact_patch_reconstruction": True,
    }
    write_json(recovery / "reconstruction.json", result)
    return result


def recovered_preflight(
    run_dir: Path,
    workspace: Path,
    round_snapshot: dict[str, Any],
    public_baseline: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    destination = (
        run_dir
        / "evaluator_state/interruption_recovery"
        / RECOVERY_ID
        / "reconstructed_attempt_03_preflight"
    )
    destination.mkdir()
    structure = guards.structural_guard(
        Path(round_snapshot["files_dir"]), workspace, destination
    )
    observed_api = (
        guards.public_api_inventory(workspace)
        if structure["status"] == "passed"
        else {"api": {}}
    )
    public = (
        guards.compare_public_api(public_baseline, observed_api)
        if structure["status"] == "passed"
        else {"status": "skipped"}
    )
    namespace = guards.namespace_export_guard(
        Path(round_snapshot["files_dir"]), workspace
    )
    failures: list[str] = []
    if structure["status"] != "passed":
        failures.append("structural/declaration guard failed")
    if public["status"] != "passed":
        failures.append("public API guard failed")
    if namespace["status"] != "passed":
        failures.extend(namespace["failures"])
    if not failures:
        raise RuntimeError(
            "Completed attempt 3 now passes preflight; interruption state is inconsistent"
        )
    preflight = {
        "structure": structure,
        "public_api": public,
        "namespace_export": namespace,
        "failures": failures,
        "recomputed_after_reboot": True,
    }
    write_json(destination / "preflight.json", preflight)
    return preflight, "Repair the retained candidate in place: " + "; ".join(failures)


def complete_scheduler(run_root: Path, manifest: dict[str, Any]) -> None:
    scheduler_path = run_root / RUN_ID / "formal_scheduler/manifest.json"
    scheduler = read_json(scheduler_path)
    task = next(item for item in scheduler["tasks"] if item["project"] == PROJECT)
    task.update(
        {
            "status": "completed" if manifest["status"] == "completed" else "failed",
            "completed_at": utc_now(),
            "runner_manifest": str(
                run_root / RUN_ID / "vanilla" / PROJECT / "run_manifest.json"
            ),
            "runner_manifest_sha256": sha256_file(
                run_root / RUN_ID / "vanilla" / PROJECT / "run_manifest.json"
            ),
            "continuation": RECOVERY_ID,
        }
    )
    statuses = Counter(item["status"] for item in scheduler["tasks"])
    scheduler.setdefault("continuation_history", []).append(
        {
            "recovery_id": RECOVERY_ID,
            "completed_at": utc_now(),
            "project": PROJECT,
            "condition_status": manifest["status"],
            "status_counts": dict(statuses),
        }
    )
    scheduler["status"] = "completed_with_protocol_failures"
    scheduler["completed_at"] = utc_now()
    scheduler["updated_at"] = utc_now()
    scheduler["final_status_counts"] = dict(statuses)
    write_json(scheduler_path, scheduler)


def run_condition(
    protocol_path: Path,
    inputs_path: Path,
    run_root: Path,
    image: str,
) -> Path:
    protocol, project = load_contract(protocol_path, inputs_path, PROJECT)
    if protocol["formal_run"]["run_id"] != RUN_ID:
        raise RuntimeError("Resume protocol has a different formal run ID")
    run_dir = run_root / RUN_ID / "vanilla" / PROJECT
    manifest_path = run_dir / "run_manifest.json"
    if not INCIDENT.is_file():
        raise RuntimeError(f"Reboot incident artifact is missing: {INCIDENT}")
    incident = read_json(INCIDENT)
    if incident.get("status") != "closed_reboot_only_continuation_authorized":
        raise RuntimeError("Reboot incident audit does not authorize continuation")
    authorization = validate_frozen_authorization(protocol_path, inputs_path, run_dir)
    resume_authorization = validate_resume_authorization()
    validate_runtime(protocol, image)
    configure_host_runtime(protocol)

    with exclusive_lock(run_dir / "evaluator_state/interruption_recovery/resume.lock"):
        manifest = read_json(manifest_path)
        if manifest.get("status") != "preparing" or manifest.get("rounds"):
            raise RuntimeError(
                f"Expected the untouched interrupted manifest, observed {manifest.get('status')}"
            )
        traces = run_dir / "traces"
        thread_id, all_commands, usage, recovered_attempts = historical_trace_state(
            traces
        )
        restricted = [
            item for item in all_commands if is_restricted_command(item["command"])
        ]
        if restricted:
            raise RuntimeError(f"Historical trace has restricted access: {restricted}")
        recovery = archive_and_restore(run_dir, incident)
        workspace = run_dir / "workspace"
        evaluator_state = run_dir / "evaluator_state"
        scanner_dir = evaluator_state / "homecheck"
        scanner_state = read_json(scanner_dir / "state.json")
        if scanner_state["validation_scans_consumed"] != 0:
            raise RuntimeError(
                "Interrupted condition already consumed a validation scan"
            )
        initial_findings = findings_from_scan(scanner_state["initial_scan"])
        current_findings = findings_from_scan(scanner_state["current_scan"])
        baseline_snapshot = snapshot_descriptor(evaluator_state / "snapshots/initial")
        round_snapshot = snapshot_descriptor(
            evaluator_state / "rounds/round_01/snapshot"
        )
        public_baseline = read_json(evaluator_state / "public_api_baseline.json")
        preflight, feedback = recovered_preflight(
            run_dir, workspace, round_snapshot, public_baseline
        )
        recovered_attempts[-1]["preflight"] = preflight
        recovered_attempts[-1]["edit"].update(recovery["reconstructed_attempt_03"])

        availability = project["validation_availability"]["build"] == "available"
        build_setup = (
            prepare_build_gate(PROJECT, workspace, run_dir / "build_setup")
            if availability
            else {"available": False, "test_available": False}
        )
        maximum = int(protocol["common_contract"]["maximum_post_edit_validation_scans"])
        source = Path(project["source_path"]).resolve()
        codex_home = run_dir / "codex_home"
        build_home = run_dir / "build_home"
        last_valid = baseline_snapshot
        last_valid_round = 0
        best_valid = baseline_snapshot
        best_valid_round = 0
        best_score = [len(initial_findings), 0, 0, 0]
        counters = {
            "coverage_retry_count": 0,
            "no_diff_retry_count": 0,
            "repair_required_round_count": 0,
        }
        interactions: list[dict[str, Any]] = []
        resume_started = time.monotonic()
        manifest.setdefault("continuation_history", []).append(
            {
                "recovery_id": RECOVERY_ID,
                "resumed_at": utc_now(),
                "classification": "host_reboot_only",
                "incident": str(INCIDENT),
                "incident_sha256": sha256_file(INCIDENT),
                "authorization": authorization,
                "resume_authorization": resume_authorization,
                "reconstruction": recovery,
                "thread_id": thread_id,
                "next_attempt": 4,
            }
        )
        manifest["status"] = "resuming_after_host_reboot"
        manifest["active_round_recovery"] = {
            "round": 1,
            "next_attempt": 4,
            "thread_id": thread_id,
            "feedback": feedback,
        }
        write_json(manifest_path, manifest)

        try:
            for round_number in range(1, maximum + 1):
                if not current_findings:
                    break
                round_dir = evaluator_state / "rounds" / f"round_{round_number:02d}"
                if round_number == 1:
                    plan = read_json(workspace / ".exp_agent/round_01/plan.json")
                    active_snapshot = round_snapshot
                    attempts = recovered_attempts
                    start_attempt = 4
                    active_thread = thread_id
                    round_feedback = feedback
                    round_repair_required = True
                else:
                    round_dir.mkdir(parents=True)
                    active_snapshot = guards.snapshot_workspace(
                        workspace, round_dir / "snapshot"
                    )
                    plan = make_reference_plan(current_findings)
                    control = workspace / ".exp_agent" / f"round_{round_number:02d}"
                    control.mkdir(parents=True, exist_ok=True)
                    write_json(control / "plan.json", plan)
                    write_json(control / "completion-schema.json", completion_schema())
                    attempts = []
                    start_attempt = 1
                    active_thread = None
                    round_feedback = "Complete the frozen plan."
                    round_repair_required = False
                control = workspace / ".exp_agent" / f"round_{round_number:02d}"
                plan_path = control / "plan.json"
                completion_path = control / "completion.json"
                schema_path = control / "completion-schema.json"
                passed = False
                round_no_diff = 0

                for attempt in range(start_attempt, MAX_AGENT_TURNS_PER_ROUND + 1):
                    manifest["active_round_recovery"] = {
                        "round": round_number,
                        "active_attempt": attempt,
                        "thread_id": active_thread,
                        "candidate_retained": True,
                    }
                    write_json(manifest_path, manifest)
                    completion_path.unlink(missing_ok=True)
                    turn = run_codex_turn(
                        workspace=workspace,
                        codex_home=codex_home,
                        build_home=build_home,
                        prompt=prompt_for_attempt(
                            project,
                            round_number,
                            attempt,
                            workspace,
                            plan_path,
                            completion_path,
                            schema_path,
                            round_feedback,
                            availability,
                        ),
                        trace_path=traces
                        / f"round_{round_number:02d}_attempt_{attempt:02d}.jsonl",
                        stderr_path=traces
                        / f"round_{round_number:02d}_attempt_{attempt:02d}.stderr.log",
                        model=protocol["model"],
                        thread_id=active_thread,
                        image=image,
                    )
                    active_thread = turn["thread_id"]
                    all_commands.extend(turn["commands"])
                    usage.update(turn["usage"])
                    restricted = [
                        item
                        for item in turn["commands"]
                        if is_restricted_command(item["command"])
                    ]
                    if restricted:
                        raise RuntimeError(f"Restricted artifact access: {restricted}")
                    report = (
                        read_json(completion_path) if completion_path.is_file() else {}
                    )
                    coverage = audit_completion(plan, report)
                    edit = guards.source_diff(
                        Path(active_snapshot["files_dir"]),
                        workspace,
                        round_dir / f"attempt_{attempt:02d}.patch",
                    )
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt,
                        "thread_id": active_thread,
                        "turn": turn,
                        "coverage": coverage,
                        "edit": edit,
                        "continued_after_host_reboot": True,
                    }
                    if not coverage["complete"]:
                        counters["coverage_retry_count"] += 1
                        round_feedback = coverage["feedback"]
                        attempts.append(attempt_record)
                        continue
                    if edit["changed_source_file_count"] == 0:
                        round_no_diff += 1
                        counters["no_diff_retry_count"] += 1
                        round_feedback = (
                            "No ArkTS/TypeScript source diff exists; make the required "
                            "source repairs."
                        )
                        attempts.append(attempt_record)
                        if round_no_diff > int(
                            protocol["scheduling"]["maximum_no_diff_retries_per_round"]
                        ):
                            raise RuntimeError("Bounded no-diff retries exhausted")
                        continue
                    preflight_dir = round_dir / f"preflight_{attempt:02d}"
                    preflight_dir.mkdir()
                    structure = guards.structural_guard(
                        Path(last_valid["files_dir"]), workspace, preflight_dir
                    )
                    observed_api = (
                        guards.public_api_inventory(workspace)
                        if structure["status"] == "passed"
                        else {"api": {}}
                    )
                    public = (
                        guards.compare_public_api(public_baseline, observed_api)
                        if structure["status"] == "passed"
                        else {"status": "skipped"}
                    )
                    namespace = guards.namespace_export_guard(
                        Path(active_snapshot["files_dir"]), workspace
                    )
                    failures: list[str] = []
                    if structure["status"] != "passed":
                        failures.append("structural/declaration guard failed")
                    if public["status"] != "passed":
                        failures.append("public API guard failed")
                    if namespace["status"] != "passed":
                        failures.extend(namespace["failures"])
                    attempt_record["preflight"] = {
                        "structure": structure,
                        "public_api": public,
                        "namespace_export": namespace,
                        "failures": failures,
                    }
                    if failures:
                        round_repair_required = True
                        round_feedback = (
                            "Repair the retained candidate in place: "
                            + "; ".join(failures)
                        )
                        attempts.append(attempt_record)
                        continue
                    build = run_build_gate(
                        PROJECT,
                        workspace,
                        round_dir / "build",
                        build_setup,
                        label=f"attempt_{attempt:02d}",
                    )
                    attempt_record["build_gate"] = build
                    if build["status"] == "failed":
                        round_repair_required = True
                        round_feedback = (
                            "Evaluator build failed. Repair the retained candidate in "
                            "place using the recorded build log."
                        )
                        attempts.append(attempt_record)
                        continue
                    passed = True
                    attempts.append(attempt_record)
                    break

                if not passed:
                    counters["repair_required_round_count"] += 1
                    write_json(
                        round_dir / "round.json",
                        {
                            "round": round_number,
                            "status": "bounded_edit_retry_exhausted",
                            "candidate_retained": True,
                            "attempts": attempts,
                            "continuation": RECOVERY_ID,
                        },
                    )
                    raise RuntimeError(
                        f"Bounded edit retries exhausted in round {round_number}"
                    )
                if round_repair_required:
                    counters["repair_required_round_count"] += 1

                validation = homecheck.scan_session(
                    argparse.Namespace(state_dir=scanner_dir, kind="validation")
                )
                next_findings = findings_from_scan(validation)
                round_metrics, round_deltas = compute_alert_metrics(
                    current_findings, next_findings
                )
                total_metrics, _ = compute_alert_metrics(
                    initial_findings, next_findings
                )
                total_diff = source_diff(
                    source, workspace, round_dir / "candidate_total.patch"
                )
                score = candidate_score(
                    total_metrics,
                    total_diff["changed_source_file_count"],
                    round_number,
                )
                accepted = guards.snapshot_workspace(
                    workspace,
                    evaluator_state / "snapshots" / f"accepted_{round_number:02d}",
                )
                last_valid, last_valid_round = accepted, round_number
                best_updated = score < best_score
                if best_updated:
                    best_valid, best_valid_round, best_score = (
                        accepted,
                        round_number,
                        score,
                    )
                interaction = describe_interactions(round_number, round_deltas)
                if interaction:
                    interactions.append(interaction)
                record = {
                    "round": round_number,
                    "status": "accepted",
                    "thread_id": active_thread,
                    "attempts": attempts,
                    "validation_scan": validation,
                    "round_metrics": round_metrics,
                    "total_metrics": total_metrics,
                    "candidate_score": score,
                    "best_valid_updated": best_updated,
                    "interaction": interaction,
                    "continuation": RECOVERY_ID,
                }
                manifest["rounds"].append(record)
                manifest.pop("active_round_recovery", None)
                write_json(round_dir / "round.json", record)
                write_json(manifest_path, manifest)
                current_findings = next_findings

            latest_is_best = last_valid_round == best_valid_round
            if not latest_is_best:
                guards.snapshot_workspace(
                    workspace,
                    evaluator_state / "snapshots/abandoned_final_candidate",
                )
                guards.restore_snapshot(workspace, best_valid)
                selection = {
                    "status": "restored_best_valid_candidate",
                    "latest_valid_round": last_valid_round,
                    "selected_round": best_valid_round,
                    "selected_score": best_score,
                }
            else:
                selection = {
                    "status": "latest_candidate_is_best_valid",
                    "latest_valid_round": last_valid_round,
                    "selected_round": best_valid_round,
                    "selected_score": best_score,
                }
            final_scan = homecheck.scan_session(
                argparse.Namespace(state_dir=scanner_dir, kind="final")
            )
            final_findings = findings_from_scan(final_scan)
            final_metrics, final_deltas = compute_alert_metrics(
                initial_findings, final_findings
            )
            scanner_state = read_json(scanner_dir / "state.json")
            restricted = [
                item for item in all_commands if is_restricted_command(item["command"])
            ]
            final_diff = source_diff(
                source, workspace, run_dir / "source_changes.patch"
            )
            started_at = datetime.fromisoformat(manifest["started_at"])
            manifest.update(
                {
                    "status": "protocol_violation" if restricted else "completed",
                    "completed_at": utc_now(),
                    "wall_clock_seconds": (
                        datetime.now(started_at.tzinfo) - started_at
                    ).total_seconds(),
                    "resume_wall_clock_seconds": time.monotonic() - resume_started,
                    "wall_clock_note": (
                        "wall_clock_seconds includes the host-reboot interruption; "
                        "resume_wall_clock_seconds measures the audited continuation."
                    ),
                    "alert_metrics": final_metrics,
                    "per_rule_alert_deltas": per_rule_deltas(final_deltas),
                    "validation_scan_count": scanner_state["validation_scans_consumed"],
                    **counters,
                    "best_valid_round": best_valid_round,
                    "best_valid_score": best_score,
                    "final_candidate_selection": selection,
                    "observed_rule_interactions": interactions,
                    "build_status": (
                        manifest["rounds"][-1]["attempts"][-1]["build_gate"]["status"]
                        if manifest["rounds"]
                        else "not_run"
                    ),
                    "test_status": "not_available",
                    "input_tokens": usage["input_tokens"],
                    "cached_input_tokens": usage["cached_input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "api_cost_if_traceable": None,
                    "restricted_accesses": restricted,
                    "source_diff": final_diff,
                    "isolation": {
                        "haprepair_repository_mounted": False,
                        "skill_state_mounted": False,
                        "corpus_mounted": False,
                        "other_runs_mounted": False,
                    },
                    "provider_endpoint_fingerprint": provider_endpoint_fingerprint(
                        codex_home, protocol["model"]["provider"]
                    ),
                    "reboot_continuation": {
                        "recovery_id": RECOVERY_ID,
                        "incident": str(INCIDENT),
                        "incident_sha256": sha256_file(INCIDENT),
                        "same_condition": True,
                        "same_thread_round_01": thread_id,
                        "fresh_stochastic_condition_started": False,
                    },
                }
            )
            manifest.pop("active_round_recovery", None)
            write_json(manifest_path, manifest)
            complete_scheduler(run_root, manifest)
            if restricted:
                raise RuntimeError("Restricted access detected after trace audit")
        except BaseException as error:
            manifest.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "failure": f"{type(error).__name__}: {error}",
                    "resume_wall_clock_seconds": time.monotonic() - resume_started,
                }
            )
            write_json(manifest_path, manifest)
            complete_scheduler(run_root, manifest)
            raise
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--audit-state", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    inputs_path = args.project_manifest.resolve()
    run_root = args.run_root.resolve()
    run_dir = run_root / RUN_ID / "vanilla" / PROJECT
    if args.audit_state:
        validate_frozen_authorization(protocol_path, inputs_path, run_dir)
        payload = inspect_interrupted_state(run_dir)
        payload.update({"run_id": RUN_ID, "project": PROJECT, "status": "audited"})
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(run_condition(protocol_path, inputs_path, run_root, args.container_image))


if __name__ == "__main__":
    main()
