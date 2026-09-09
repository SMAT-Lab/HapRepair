#!/usr/bin/env python3
"""Resume an interrupted v14 condition without recreating its workspace.

This continuation runner is deliberately separate from the frozen v14 runner.
It reuses the existing workspace, candidate-session state, Codex home, and
active-round thread.  A failed provider turn is therefore recoverable without
silently starting the project from its pinned source again.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_condition_v14 import (
    CONTAINER_STATE,
    CONTAINER_RUNTIME_LIB_DIR,
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    DEFAULT_RUN_ROOT,
    HOST_CODEX_HOME,
    audit_spec_access,
    classify_command,
    command_status,
    common_mounts,
    container_to_host,
    current_findings,
    extract_commands,
    extract_thread_id,
    extract_usage,
    is_restricted_command,
    load_project,
    parse_codex_events,
    per_rule_counts,
    per_rule_deltas,
    prompt_for_attempt,
    read_json,
    remove_empty_auth_mount_placeholder,
    run_skill,
    sha256_file,
    source_diff,
    validate_method_artifacts,
    validate_runtime,
    write_json,
    utc_now,
)


RESUME_CODEX_TIMEOUT_SECONDS = float(
    os.environ.get("HAPREPAIR_RESUME_CODEX_TIMEOUT_SECONDS", "1800")
)


def run_resumed_codex_turn(
    *,
    workspace: Path,
    state_dir: Path,
    codex_home: Path,
    codelinter_result: Path,
    prompt: str,
    trace_path: Path,
    stderr_path: Path,
    model: dict[str, Any],
    thread_id: str | None,
    image: str,
) -> dict[str, Any]:
    """Run a resumed turn with a bounded provider wait and process cleanup.

    The frozen v14 runner intentionally has no provider timeout.  Resumption
    needs one because a disconnected provider can otherwise hold the whole
    scheduler forever.  The workspace and thread remain durable when this
    subprocess is terminated, so the next resume continues the same turn.
    """
    common = [
        "--json",
        "--model",
        model["requested_id"],
        "-c",
        f'model_provider="{model["provider"]}"',
        "-c",
        f'model_reasoning_effort="{model["reasoning_effort"]}"',
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    codex_command = (
        ["codex", "exec", *common, "--cd", "/workspace", "-"]
        if thread_id is None
        else ["codex", "exec", "resume", *common, thread_id, "-"]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "bridge",
        "--workdir",
        "/workspace",
        "--env",
        "CODEX_HOME=/codex-home",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HAPREPAIR_AGENT_MODE=1",
        "--env",
        "HOME=/run-state/build-home",
        "--env",
        "HVIGOR_USER_HOME=/run-state/build-home",
        "--env",
        "npm_config_userconfig=/run-state/build-home/.npmrc",
        "--env",
        "NPM_CONFIG_USERCONFIG=/run-state/build-home/.npmrc",
        "--env",
        f"LD_LIBRARY_PATH={CONTAINER_RUNTIME_LIB_DIR}",
        *common_mounts(workspace, state_dir, codex_home, codelinter_result),
        "--volume",
        f"{HOST_CODEX_HOME / 'auth.json'}:/codex-home/auth.json:ro",
        image,
        *codex_command,
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workspace,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            input=prompt, timeout=RESUME_CODEX_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        remove_empty_auth_mount_placeholder(codex_home)
        trace_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise TimeoutError(
            f"Resumed Codex turn exceeded {RESUME_CODEX_TIMEOUT_SECONDS:g}s; "
            f"workspace and thread retained for another resume"
        ) from error
    remove_empty_auth_mount_placeholder(codex_home)
    trace_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    events = parse_codex_events(stdout)
    if process.returncode != 0:
        raise RuntimeError(f"Codex turn exited {process.returncode}; see {stderr_path}")
    return {
        "elapsed_seconds": time.monotonic() - started,
        "thread_id": extract_thread_id(events) or thread_id,
        "usage": extract_usage(events),
        "commands": extract_commands(events),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "stderr_path": str(stderr_path),
    }


def trace_files(traces: Path, round_number: int) -> list[Path]:
    return sorted(traces.glob(f"round_{round_number:02d}*.jsonl"))


def remaining_attempt_range(*, existing_attempts: int, maximum_attempts: int) -> range:
    """Return the unspent active-round allowance without refreshing it."""
    if existing_attempts < 0 or maximum_attempts < 1:
        raise ValueError("Invalid active-round attempt budget")
    next_attempt = existing_attempts + 1
    return range(next_attempt, maximum_attempts + 1)


def trace_state(
    paths: list[Path],
) -> tuple[str | None, list[dict[str, Any]], Counter[str]]:
    """Recover the last thread, command evidence, and token usage from traces."""
    thread_id: str | None = None
    commands: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    for path in paths:
        events = parse_codex_events(path.read_text(encoding="utf-8", errors="replace"))
        thread_id = extract_thread_id(events) or thread_id
        commands.extend(extract_commands(events))
        for key, value in extract_usage(events).items():
            usage[key] += value
    return thread_id, commands, usage


def manifest_commands(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for round_item in manifest.get("rounds", []):
        for attempt in round_item.get("attempts", []):
            commands.extend((attempt.get("turn") or {}).get("commands") or [])
    return commands


def resume_condition(
    *,
    run_id: str,
    project_name: str,
    protocol_path: Path,
    project_manifest_path: Path,
    run_root: Path,
    image: str,
) -> Path:
    protocol = read_json(protocol_path)
    validate_method_artifacts(protocol)
    validate_runtime(protocol, image)
    project = load_project(project_name, project_manifest_path)

    run_dir = run_root.resolve() / run_id / "hapskill" / project_name
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Cannot resume missing condition manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {"failed", "resuming", "preparing"}:
        raise RuntimeError(
            f"Only interrupted conditions are resumable; {project_name} is "
            f"{manifest.get('status')}"
        )
    if manifest.get("run_id") != run_id or manifest.get("project") != project_name:
        raise RuntimeError("Existing condition identity does not match resume request")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Existing condition was created with a different protocol")

    workspace = run_dir / "workspace"
    state_dir = run_dir / "skill_state"
    traces = run_dir / "traces"
    codex_home = run_dir / "codex_home"
    codelinter_result = run_dir / "codelinter_result"
    session_path = state_dir / "session.json"
    if not all(
        path.exists()
        for path in (
            workspace,
            state_dir,
            traces,
            codex_home,
            codelinter_result,
            session_path,
        )
    ):
        raise RuntimeError(
            f"Condition state is incomplete and cannot be resumed: {run_dir}"
        )

    session = read_json(session_path)
    if not isinstance(session.get("active_round"), dict):
        raise RuntimeError("Failed condition has no active round to resume")

    build_setup = read_json(run_dir / "build_gate" / "setup.json")
    maximum_scans = int(protocol["common_contract"]["maximum_validation_scans"])
    maximum_attempts = int(protocol["scheduling"]["maximum_agent_turns_per_round"])
    control = workspace / ".exp_agent"
    manifest.setdefault("resume_history", []).append(
        {"resumed_at": utc_now(), "previous_failure": manifest.get("failure")}
    )
    manifest.pop("failure", None)
    manifest["status"] = "resuming"
    manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    write_json(manifest_path, manifest)

    total_usage: Counter[str] = Counter()
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        total_usage[key] = int(manifest.get(key, 0) or 0)
    all_commands = manifest_commands(manifest)
    experiment_started = time.monotonic()

    try:
        while True:
            session = read_json(session_path)
            active = session.get("active_round")
            consumed = int(
                read_json(state_dir / "homecheck" / "state.json").get(
                    "validation_scans_consumed", 0
                )
            )
            if active is None:
                findings = current_findings(
                    workspace=workspace, state_dir=state_dir, codex_home=codex_home
                )
                if not findings or consumed >= maximum_scans:
                    break
                begin = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=image,
                    arguments=["begin-round", "--state-dir", str(CONTAINER_STATE)],
                )
                active = read_json(session_path)["active_round"]
                if int(begin["round"]) != int(active["round"]):
                    raise RuntimeError(
                        "Resumed active-round number differs from controller state"
                    )
                # The original runner materializes these evaluator-owned inputs
                # immediately after begin-round. Preserve that contract when a
                # resumed condition advances to a new round.
                round_number = int(active["round"])
                write_json(
                    control / f"findings_round_{round_number:02d}.json",
                    current_findings(
                        workspace=workspace, state_dir=state_dir, codex_home=codex_home
                    ),
                )
                write_json(
                    control / f"round_plan_{round_number:02d}.json",
                    begin["plan"],
                )

            round_number = int(active["round"])
            findings_path = control / f"findings_round_{round_number:02d}.json"
            plan_path = control / f"round_plan_{round_number:02d}.json"
            completion_path = control / f"round_completion_{round_number:02d}.json"
            if not findings_path.is_file() or not plan_path.is_file():
                # A controller can have created the round just before a process
                # interruption. Reconstruct the two small evaluator files from
                # its durable active-round record instead of rescanning or
                # recreating the project.
                input_scan = active.get("input_scan") or {}
                input_path = input_scan.get("findings_path")
                active_plan_path = active.get("plan_path")
                if not findings_path.is_file() and input_path:
                    write_json(
                        findings_path,
                        read_json(
                            container_to_host(
                                input_path,
                                workspace=workspace,
                                state_dir=state_dir,
                                codex_home=codex_home,
                            )
                        ),
                    )
                if not plan_path.is_file() and active_plan_path:
                    write_json(
                        plan_path,
                        read_json(
                            container_to_host(
                                active_plan_path,
                                workspace=workspace,
                                state_dir=state_dir,
                                codex_home=codex_home,
                            )
                        ),
                    )
            if not findings_path.is_file() or not plan_path.is_file():
                raise RuntimeError(
                    f"Missing persisted active-round inputs for round {round_number}"
                )
            findings = read_json(findings_path)
            plan = read_json(plan_path)
            localization_sha = sha256_file(findings_path)
            plan_sha = sha256_file(plan_path)

            old_trace_paths = trace_files(traces, round_number)
            thread_id, round_commands, trace_usage = trace_state(old_trace_paths)
            total_usage.update(trace_usage)
            all_commands.extend(round_commands)
            attempts: list[dict[str, Any]] = []
            attempt_range = remaining_attempt_range(
                existing_attempts=len(old_trace_paths),
                maximum_attempts=maximum_attempts,
            )
            feedback = "The previous Codex turn was interrupted by a provider/network failure. Continue this same active round from the current workspace and report."
            gate_recovery = False
            validation: dict[str, Any] | None = None
            use_persisted_completion = completion_path.is_file()

            for attempt in attempt_range:
                prompt: str | None = None
                turn: dict[str, Any] | None = None
                if not use_persisted_completion:
                    prompt = prompt_for_attempt(
                        project=project,
                        round_number=round_number,
                        maximum_scans=maximum_scans,
                        attempt=attempt,
                        findings_path=findings_path,
                        plan_path=plan_path,
                        completion_path=completion_path,
                        feedback=feedback,
                        gate_recovery=gate_recovery,
                        build_available=bool(build_setup.get("available")),
                    )
                    suffix = "" if attempt == 1 else f".attempt_{attempt:02d}"
                    prompt_path = (
                        traces / f"round_{round_number:02d}{suffix}.prompt.txt"
                    )
                    prompt_path.write_text(prompt + "\n", encoding="utf-8")
                    turn = run_resumed_codex_turn(
                        workspace=workspace,
                        state_dir=state_dir,
                        codex_home=codex_home,
                        codelinter_result=codelinter_result,
                        prompt=prompt,
                        trace_path=traces / f"round_{round_number:02d}{suffix}.jsonl",
                        stderr_path=traces
                        / f"round_{round_number:02d}{suffix}.stderr.log",
                        model=protocol["model"],
                        thread_id=thread_id,
                        image=image,
                    )
                    thread_id = turn["thread_id"]
                    round_commands.extend(turn["commands"])
                    all_commands.extend(turn["commands"])
                    for key, value in turn["usage"].items():
                        total_usage[key] += value
                    if (
                        sha256_file(findings_path) != localization_sha
                        or sha256_file(plan_path) != plan_sha
                    ):
                        raise RuntimeError(
                            "Agent modified persisted active-round inputs"
                        )

                try:
                    completion = run_skill(
                        workspace=workspace,
                        state_dir=state_dir,
                        codex_home=codex_home,
                        codelinter_result=codelinter_result,
                        image=image,
                        arguments=[
                            "record-completion",
                            "--state-dir",
                            str(CONTAINER_STATE),
                            "--report",
                            f"/workspace/.exp_agent/{completion_path.name}",
                        ],
                    )
                except Exception as error:
                    feedback = f"The persisted completion report could not be recorded: {error}. Rewrite it completely and continue."
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "status": "completion_recovery_required",
                            "feedback": feedback,
                        }
                    )
                    use_persisted_completion = False
                    continue

                spec_access = audit_spec_access(round_commands, plan)
                if not completion["complete"] or not spec_access["complete"]:
                    problems = list(completion.get("problems") or [])
                    if not spec_access["complete"]:
                        problems.append(
                            "trace lacks exact normative spec access for: "
                            + ", ".join(spec_access["missing_rules"])
                        )
                    feedback = "Coverage evidence is incomplete: " + "; ".join(problems)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "completion": completion,
                            "spec_access": spec_access,
                            "status": "coverage_required",
                            "feedback": feedback,
                        }
                    )
                    use_persisted_completion = False
                    continue

                preflight = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=image,
                    arguments=["preflight-round", "--state-dir", str(CONTAINER_STATE)],
                )
                if preflight["status"] != "preflight_passed":
                    feedback = preflight["feedback"]
                    gate_recovery = True
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "completion": completion,
                            "spec_access": spec_access,
                            "preflight": preflight,
                            "status": preflight["status"],
                            "feedback": feedback,
                        }
                    )
                    use_persisted_completion = False
                    continue

                validation = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=image,
                    arguments=["validate-round", "--state-dir", str(CONTAINER_STATE)],
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "turn": turn,
                        "completion": completion,
                        "spec_access": spec_access,
                        "preflight": preflight,
                        "validation": validation,
                        "status": validation["status"],
                    }
                )
                session = read_json(session_path)
                if (
                    validation["status"] == "accepted"
                    or session.get("active_round") is None
                ):
                    break
                feedback = validation["feedback"]
                gate_recovery = True
                use_persisted_completion = False

            if validation is None:
                raise RuntimeError(
                    f"Round {round_number} did not reach validation after resume"
                )
            # A resume invocation owns one bounded batch.  If validation still
            # rejects the retained candidate (for example, because the same
            # evaluator build infrastructure failure persists), return control
            # to the scheduler instead of starting another unbounded batch in
            # this process.  The durable workspace, report, traces, and thread
            # remain available to a later explicit resume.
            if (
                validation["status"] != "accepted"
                and read_json(session_path).get("active_round") is not None
            ):
                raise RuntimeError(
                    f"Round {round_number} did not reach accepted validation "
                    f"after {maximum_attempts} resumed agent turns"
                )
            if (
                sha256_file(findings_path) != localization_sha
                or sha256_file(plan_path) != plan_sha
            ):
                raise RuntimeError("Active-round inputs changed during resume")
            manifest["rounds"].append(
                {
                    "round": round_number,
                    "input_alerts": len(findings),
                    "input_per_rule": per_rule_counts(findings),
                    "localization_path": str(findings_path),
                    "localization_sha256": localization_sha,
                    "plan_path": str(plan_path),
                    "plan_sha256": plan_sha,
                    "completion_path": str(completion_path),
                    "attempts": attempts,
                    "validation": validation,
                    "resumed": True,
                }
            )
            write_json(manifest_path, manifest)
            if (
                validation["status"] == "accepted"
                and validation["total_metrics"]["final_alerts"] == 0
            ):
                break

        final = run_skill(
            workspace=workspace,
            state_dir=state_dir,
            codex_home=codex_home,
            codelinter_result=codelinter_result,
            image=image,
            arguments=["finalize", "--state-dir", str(CONTAINER_STATE)],
        )
        final_scan = final["final_scan"]
        final_findings_path = container_to_host(
            final_scan["findings_path"],
            workspace=workspace,
            state_dir=state_dir,
            codex_home=codex_home,
        )
        final_deltas_path = container_to_host(
            final_scan["alert_deltas_path"],
            workspace=workspace,
            state_dir=state_dir,
            codex_home=codex_home,
        )
        final_findings = read_json(final_findings_path)
        final_deltas = read_json(final_deltas_path)
        session = read_json(session_path)
        scanner = read_json(state_dir / "homecheck" / "state.json")
        restricted = [
            item
            for item in all_commands
            if is_restricted_command(str(item.get("command", "")))
        ]
        spec_access = [
            attempt.get("spec_access")
            for item in manifest["rounds"]
            for attempt in item.get("attempts", [])
            if attempt.get("spec_access")
        ]
        spec_violation = any(not item["complete"] for item in spec_access[-1:])
        agent_builds = [
            item
            for item in all_commands
            if classify_command(str(item.get("command", ""))) == "build"
        ]
        agent_tests = [
            item
            for item in all_commands
            if classify_command(str(item.get("command", ""))) == "test"
        ]
        source = Path(manifest["source"]).resolve()
        diff = source_diff(source, workspace, run_dir / "source_changes.patch")
        last_valid_round = int(session.get("last_valid_round", 0))
        rounds = session.get("rounds") or []
        build_status = "not_available"
        if build_setup.get("available"):
            build_status = (
                rounds[last_valid_round - 1]["build_gate"]["status"]
                if last_valid_round and len(rounds) >= last_valid_round
                else "not_run"
            )
        manifest.update(
            {
                "status": "protocol_violation"
                if restricted or spec_violation
                else "completed",
                "completed_at": utc_now(),
                "wall_clock_seconds": time.monotonic() - experiment_started,
                "finalization": final,
                "alert_metrics": final_scan["metrics"],
                "final_per_rule": per_rule_counts(final_findings),
                "per_rule_alert_deltas": per_rule_deltas(final_deltas),
                "validation_scan_count": scanner.get("validation_scans_consumed", 0),
                "last_valid_round": last_valid_round,
                "best_valid_round": session.get("best_valid_round"),
                "best_valid_score": session.get("best_valid_score"),
                "observed_rule_interactions": session.get("rule_interactions", []),
                "final_candidate_selection": session.get("final_candidate_selection"),
                "build_status": build_status,
                "test_status": command_status(all_commands, "test"),
                "agent_build_count": len(agent_builds),
                "agent_test_count": len(agent_tests),
                "build_test_commands": agent_builds + agent_tests,
                "input_tokens": total_usage.get("input_tokens", 0),
                "cached_input_tokens": total_usage.get("cached_input_tokens", 0),
                "output_tokens": total_usage.get("output_tokens", 0),
                "total_tokens": total_usage.get("total_tokens", 0),
                "api_cost": None,
                "api_cost_note": "Provider billing was not exposed in the Codex trace.",
                "restricted_accesses": restricted,
                "spec_access_audits": spec_access,
                "spec_protocol_violation": spec_violation,
                "source_diff": diff,
            }
        )
        write_json(manifest_path, manifest)
        print(
            f"[resumed] hapskill {project_name}: {final_scan['metrics']['eliminated_alerts']} eliminated, {final_scan['metrics']['introduced_alerts']} introduced",
            flush=True,
        )
        return manifest_path
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["failed_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        run_dir = args.run_root.resolve() / args.run_id / "hapskill" / args.project
        manifest = read_json(run_dir / "run_manifest.json")
        session = read_json(run_dir / "skill_state" / "session.json")
        active = session.get("active_round") or {}
        round_number = active.get("round")
        paths = (
            trace_files(run_dir / "traces", int(round_number)) if round_number else []
        )
        thread_id, commands, usage = trace_state(paths)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "project": args.project,
                    "status": manifest.get("status"),
                    "active_round": round_number,
                    "validation_scans_consumed": read_json(
                        run_dir / "skill_state" / "homecheck" / "state.json"
                    ).get("validation_scans_consumed"),
                    "persisted_completion": bool(
                        round_number
                        and (
                            run_dir
                            / "workspace"
                            / ".exp_agent"
                            / f"round_completion_{int(round_number):02d}.json"
                        ).is_file()
                    ),
                    "trace_count": len(paths),
                    "thread_id": thread_id,
                    "command_count": len(commands),
                    "trace_usage": usage,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    path = resume_condition(
        run_id=args.run_id,
        project_name=args.project,
        protocol_path=args.protocol.resolve(),
        project_manifest_path=args.project_manifest.resolve(),
        run_root=args.run_root.resolve(),
        image=args.container_image,
    )
    print(path)


if __name__ == "__main__":
    main()
