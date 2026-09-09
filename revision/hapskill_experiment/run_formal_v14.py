#!/usr/bin/env python3
"""Run the frozen HapRepair Skill condition across all 35 projects."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
DEFAULT_PROTOCOL = HERE / "protocol-hapskill-35-v14-namespace-export-safe.json"
DEFAULT_INPUTS = HERE / "formal_inputs_hapskill_35_v4.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "runs"
RUNNER = HERE / "run_condition_v14.py"
RESUME_RUNNER = HERE / "resume_condition_v14.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(protocol_path: Path, input_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(protocol_path)
    inputs = read_json(input_path)
    if protocol.get("status") != "frozen":
        raise ValueError("Formal v14 protocol must be frozen")
    if protocol.get("paper_facing") is not True:
        raise ValueError("Formal v14 protocol must be paper-facing")
    if protocol["formal_run"].get("condition") != "hapskill":
        raise ValueError("Formal v14 scheduler only accepts the HapRepair Skill condition")
    if protocol["dataset"]["formal_inputs_sha256"] != sha256_file(input_path):
        raise ValueError("Formal input manifest hash does not match the protocol")
    projects = inputs.get("projects")
    if not isinstance(projects, list) or len(projects) != 35:
        raise ValueError("Formal v14 input manifest must contain exactly 35 projects")
    if protocol["dataset"].get("project_count") != len(projects):
        raise ValueError("Protocol and input project counts differ")
    names = [item.get("name") for item in projects]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every frozen project must have a name")
    if len(set(names)) != len(names):
        raise ValueError("Project names must be unique")
    tasks = [
        {"ordinal": ordinal, "project": name, "condition": "hapskill", "status": "pending"}
        for ordinal, name in enumerate(sorted(names), start=1)
    ]
    return protocol, inputs, tasks


def condition_dir(run_root: Path, run_id: str, project: str) -> Path:
    return run_root / run_id / "hapskill" / project


def condition_manifest(run_root: Path, run_id: str, project: str) -> Path:
    return condition_dir(run_root, run_id, project) / "run_manifest.json"


def run_one(
    *,
    task: dict[str, Any],
    run_id: str,
    run_root: Path,
    protocol: Path,
    inputs: Path,
    log_dir: Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    project = task["project"]
    manifest_path = condition_manifest(run_root, run_id, project)
    if manifest_path.is_file():
        observed = read_json(manifest_path)
        if observed.get("status") == "completed":
            return {
                **task,
                "status": "already_completed",
                "runner_manifest": str(manifest_path),
                "runner_manifest_sha256": sha256_file(manifest_path),
            }
    if manifest_path.parent.exists():
        if not resume_existing:
            raise RuntimeError(f"Incomplete condition requires audit: {manifest_path.parent}")
        observed = read_json(manifest_path) if manifest_path.is_file() else {}
        if observed.get("status") not in {"failed", "resuming", "preparing"}:
            raise RuntimeError(
                f"Only interrupted conditions can be resumed: {manifest_path.parent}"
            )

    log_path = log_dir / f"{task['ordinal']:02d}_{project}.log"
    runner = RESUME_RUNNER if resume_existing else RUNNER
    command = [
        sys.executable,
        str(runner),
        "--run-id",
        run_id,
        "--project",
        project,
        "--protocol",
        str(protocol),
        "--project-manifest",
        str(inputs),
        "--run-root",
        str(run_root),
    ]
    started = time.monotonic()
    with log_path.open("ab" if resume_existing else "wb") as log:
        if resume_existing:
            log.write(f"\n[resume] {utc_now()} continuing existing workspace and Codex thread\n".encode())
        result = subprocess.run(
            command,
            cwd=HERE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"Condition failed for {project}; see {log_path}")
    observed = read_json(manifest_path)
    if observed.get("status") != "completed":
        raise RuntimeError(
            f"Condition ended as {observed.get('status')} for {project}; see {log_path}"
        )
    return {
        **task,
        "status": "completed",
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "command": command,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "runner_manifest": str(manifest_path),
        "runner_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    input_path = args.project_manifest.resolve()
    run_root = args.run_root.resolve()
    protocol, _inputs, tasks = load_contract(protocol_path, input_path)
    run_id = args.run_id or protocol["formal_run"]["run_id"]
    if run_id != protocol["formal_run"]["run_id"]:
        raise SystemExit("Run ID override differs from the frozen protocol")
    maximum = args.max_concurrent or int(protocol["execution"]["maximum_concurrent_conditions"])
    if maximum != int(protocol["execution"]["maximum_concurrent_conditions"]):
        raise SystemExit("Concurrency override differs from the frozen protocol")
    summary = {
        "experiment": protocol["experiment"],
        "run_id": run_id,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "project_manifest": str(input_path),
        "project_manifest_sha256": sha256_file(input_path),
        "project_count": len(tasks),
        "condition": "hapskill",
        "maximum_concurrent_conditions": maximum,
        "tasks": tasks,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    scheduler_dir = run_root / run_id / "formal_scheduler"
    scheduler_path = scheduler_dir / "manifest.json"
    log_dir = scheduler_dir / "logs"
    if scheduler_path.is_file():
        if not args.resume:
            raise SystemExit("Scheduler already exists; use --resume after auditing it")
        state = read_json(scheduler_path)
        if state.get("protocol_sha256") != summary["protocol_sha256"]:
            raise SystemExit("Protocol drifted after scheduling started")
        if state.get("project_manifest_sha256") != summary["project_manifest_sha256"]:
            raise SystemExit("Input manifest drifted after scheduling started")
        previous_status = state.get("status")
        previous_failure = state.get("failure")
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["last_resumed_at"] = utc_now()
        state["status"] = "running"
        state.setdefault("resume_history", []).append(
            {
                "resumed_at": state["last_resumed_at"],
                "previous_status": previous_status,
                "previous_failure": previous_failure,
            }
        )
    else:
        if args.resume:
            raise SystemExit("No scheduler exists to resume")
        log_dir.mkdir(parents=True, exist_ok=False)
        state = {**summary, "status": "running", "created_at": utc_now(), "resume_count": 0, "tasks": tasks}
    scheduler_dir.mkdir(parents=True, exist_ok=True)

    pending: list[dict[str, Any]] = []
    for task in state["tasks"]:
        manifest_path = condition_manifest(run_root, run_id, task["project"])
        observed = read_json(manifest_path) if manifest_path.is_file() else {}
        condition_status = observed.get("status")
        if condition_status == "completed":
            task["status"] = "completed"
            task["runner_manifest"] = str(manifest_path)
        elif args.resume and condition_status == "protocol_violation":
            # A protocol-violating condition is finalized evidence, not an
            # interrupted workspace. Preserve it and continue the remaining
            # conditions without rerunning or mutating its project state.
            task["status"] = "protocol_violation"
            task["runner_manifest"] = str(manifest_path)
            task["runner_manifest_sha256"] = sha256_file(manifest_path)
            task["audit_note"] = (
                "finalized condition retained as invalid evidence; excluded from "
                "successful-condition metrics"
            )
        elif manifest_path.parent.exists():
            existing = observed
            if args.resume and existing.get("status") in {"failed", "resuming", "preparing"}:
                task["status"] = "resume_pending"
                pending.append(task)
            else:
                task["status"] = "incomplete_requires_audit"
                state["status"] = "blocked_incomplete_condition"
                state["updated_at"] = utc_now()
                write_json(scheduler_path, state)
                raise SystemExit(f"Incomplete condition requires audit: {manifest_path.parent}")
        else:
            task["status"] = "pending"
            pending.append(task)
    write_json(scheduler_path, state)

    next_index = 0
    stopped = False
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        while next_index < len(pending) and len(active) < maximum:
            task = pending[next_index]
            next_index += 1
            active[executor.submit(run_one, task=task, run_id=run_id, run_root=run_root, protocol=protocol_path, inputs=input_path, log_dir=log_dir, resume_existing=args.resume and task.get("status") == "resume_pending")] = task
            print(f"[scheduler] started {task['project']}", flush=True)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    record = future.result()
                except Exception as error:
                    task["status"] = "failed"
                    task["failure"] = f"{type(error).__name__}: {error}"
                    state["status"] = "stopped_condition_failure"
                    state["failure"] = task["failure"]
                    stopped = True
                    print(f"[scheduler] FAILED {task['project']}: {error}", flush=True)
                else:
                    task.update(record)
                    print(f"[scheduler] completed {task['project']}", flush=True)
                state["updated_at"] = utc_now()
                write_json(scheduler_path, state)
            if not stopped:
                while next_index < len(pending) and len(active) < maximum:
                    task = pending[next_index]
                    next_index += 1
                    active[executor.submit(run_one, task=task, run_id=run_id, run_root=run_root, protocol=protocol_path, inputs=input_path, log_dir=log_dir, resume_existing=args.resume and task.get("status") == "resume_pending")] = task
                    print(f"[scheduler] started {task['project']}", flush=True)

    if stopped:
        state["completed_at"] = utc_now()
        write_json(scheduler_path, state)
        raise SystemExit(1)

    protocol_violations = [
        task["project"]
        for task in state["tasks"]
        if task.get("status") == "protocol_violation"
    ]
    if protocol_violations:
        state["status"] = "completed_with_protocol_violations"
        state["protocol_violation_projects"] = protocol_violations
    else:
        state["status"] = "completed"
    state.pop("failure", None)
    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    write_json(scheduler_path, state)
    print(scheduler_path)


if __name__ == "__main__":
    main()
