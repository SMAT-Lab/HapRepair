#!/usr/bin/env python3
"""Concurrent, resumable scheduler for the independent v4 formal experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "runs"
RUNNER = HERE / "run_condition.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def parse_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    values = protocol["formal_run"]["condition_order"]
    tasks = []
    for ordinal, value in enumerate(values, start=1):
        project, condition = value.rsplit(":", 1)
        if condition not in {"hapskill", "vanilla"}:
            raise ValueError(f"Unknown condition in task {value!r}")
        tasks.append(
            {
                "ordinal": ordinal,
                "project": project,
                "condition": condition,
                "status": "pending",
            }
        )
    return tasks


def validate_contract(
    protocol_path: Path, input_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(protocol_path)
    inputs = read_json(input_path)
    tasks = parse_tasks(protocol)
    if protocol.get("status") != "frozen":
        raise ValueError("Formal protocol is not frozen")
    if protocol["formal_run"]["run_id"] == "exp_hapskill_10_luna_formal_04":
        raise ValueError("The withdrawn formal_04 run is forbidden")
    if protocol["dataset"]["formal_inputs_sha256"] != sha256_file(input_path):
        raise ValueError("Formal input manifest hash does not match the protocol")
    projects = {item["name"] for item in inputs["projects"]}
    task_projects = [item["project"] for item in tasks]
    if len(task_projects) != len(set(task_projects)):
        raise ValueError("Each project must appear once in an independent formal run")
    if set(task_projects) != projects:
        raise ValueError("Protocol task membership differs from the frozen inputs")
    expected = protocol["formal_run"]["condition"]
    if any(item["condition"] != expected for item in tasks):
        raise ValueError("Formal task condition differs from the protocol condition")
    return protocol, inputs, tasks


def runner_manifest(run_root: Path, run_id: str, task: dict[str, Any]) -> Path:
    return (
        run_root
        / run_id
        / task["condition"]
        / task["project"]
        / "run_manifest.json"
    )


def run_one(
    *,
    task: dict[str, Any],
    run_id: str,
    run_root: Path,
    protocol_path: Path,
    input_path: Path,
    device: str,
    log_dir: Path,
) -> dict[str, Any]:
    manifest_path = runner_manifest(run_root, run_id, task)
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
        raise RuntimeError(
            f"Incomplete condition requires audit before resume: {manifest_path.parent}"
        )

    log_path = log_dir / (
        f"{task['ordinal']:02d}_{task['project']}_{task['condition']}.log"
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--project",
        task["project"],
        "--condition",
        task["condition"],
        "--protocol",
        str(protocol_path),
        "--project-manifest",
        str(input_path),
        "--run-root",
        str(run_root),
        "--device",
        device,
    ]
    started = time.monotonic()
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            cwd=HERE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"Condition failed; see {log_path}")
    observed = read_json(manifest_path)
    if observed.get("status") != "completed":
        raise RuntimeError(
            f"Condition ended as {observed.get('status')}; see {manifest_path}"
        )
    return {
        **task,
        "status": "completed",
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "device": device,
        "command": command,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "runner_manifest": str(manifest_path),
        "runner_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--devices")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    input_path = args.project_manifest.resolve()
    run_root = args.run_root.resolve()
    protocol, inputs, tasks = validate_contract(protocol_path, input_path)
    run_id = args.run_id or protocol["formal_run"]["run_id"]
    if run_id != protocol["formal_run"]["run_id"]:
        raise SystemExit("Run ID override differs from the frozen protocol")

    scheduling = protocol["execution"]
    max_concurrent = args.max_concurrent or int(
        scheduling["maximum_concurrent_conditions"]
    )
    devices = (args.devices or ",".join(scheduling["retrieval_devices"])).split(",")
    devices = [item.strip() for item in devices if item.strip()]
    if max_concurrent != int(scheduling["maximum_concurrent_conditions"]):
        raise SystemExit("Concurrency override differs from the frozen protocol")
    if devices != scheduling["retrieval_devices"]:
        raise SystemExit("Device override differs from the frozen protocol")

    dry_summary = {
        "experiment": protocol["experiment"],
        "run_id": run_id,
        "protocol_sha256": sha256_file(protocol_path),
        "project_manifest_sha256": sha256_file(input_path),
        "project_count": inputs["project_count"],
        "task_count": len(tasks),
        "condition": protocol["formal_run"]["condition"],
        "maximum_concurrent_conditions": max_concurrent,
        "retrieval_devices": devices,
        "tasks": tasks,
    }
    if args.dry_run:
        print(json.dumps(dry_summary, ensure_ascii=False, indent=2))
        return

    scheduler_dir = run_root / run_id / "formal_scheduler"
    scheduler_path = scheduler_dir / "manifest.json"
    log_dir = scheduler_dir / "logs"
    if scheduler_path.is_file():
        if not args.resume:
            raise SystemExit("Scheduler already exists; pass --resume after auditing it")
        state = read_json(scheduler_path)
        if state["protocol_sha256"] != sha256_file(protocol_path):
            raise SystemExit("Protocol drifted after scheduling started")
        if state["project_manifest_sha256"] != sha256_file(input_path):
            raise SystemExit("Input manifest drifted after scheduling started")
        expected = [(item["project"], item["condition"]) for item in tasks]
        observed = [(item["project"], item["condition"]) for item in state["tasks"]]
        if expected != observed:
            raise SystemExit("Task membership/order drifted after scheduling started")
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["last_resumed_at"] = utc_now()
    else:
        if args.resume:
            raise SystemExit("No scheduler exists to resume")
        log_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            **dry_summary,
            "status": "running",
            "created_at": utc_now(),
            "resume_count": 0,
            "protocol": str(protocol_path),
            "project_manifest": str(input_path),
            "tasks": tasks,
            "abort_policy": scheduling["abort_policy"],
            "resume_policy": scheduling["resume_policy"],
        }
        write_json(scheduler_path, state)

    pending = []
    for task in state["tasks"]:
        manifest_path = runner_manifest(run_root, run_id, task)
        if manifest_path.is_file() and read_json(manifest_path).get("status") == "completed":
            task["status"] = "completed"
            task["runner_manifest"] = str(manifest_path)
        elif manifest_path.parent.exists():
            task["status"] = "incomplete_requires_audit"
            state["status"] = "blocked_incomplete_condition"
            state["updated_at"] = utc_now()
            write_json(scheduler_path, state)
            raise SystemExit(f"Incomplete condition requires audit: {manifest_path.parent}")
        else:
            task["status"] = "pending"
            pending.append(task)
    if args.max_tasks is not None:
        pending = pending[: args.max_tasks]

    state["status"] = "running"
    state["updated_at"] = utc_now()
    write_json(scheduler_path, state)
    next_index = 0
    stopped = False
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        while (next_index < len(pending) and not stopped) or active:
            while (
                not stopped
                and next_index < len(pending)
                and len(active) < max_concurrent
            ):
                task = pending[next_index]
                device = devices[(task["ordinal"] - 1) % len(devices)]
                task["status"] = "running"
                task["started_at"] = utc_now()
                future = executor.submit(
                    run_one,
                    task=task,
                    run_id=run_id,
                    run_root=run_root,
                    protocol_path=protocol_path,
                    input_path=input_path,
                    device=device,
                    log_dir=log_dir,
                )
                active[future] = task
                next_index += 1
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    task.update(future.result())
                except Exception as error:
                    task["status"] = "failed"
                    task["failure"] = f"{type(error).__name__}: {error}"
                    stopped = True
                    state["status"] = "stopped_condition_failure"
                state["updated_at"] = utc_now()
                write_json(scheduler_path, state)

    completed = sum(item["status"] == "completed" for item in state["tasks"])
    if stopped:
        state["status"] = "stopped_condition_failure"
    elif completed == len(state["tasks"]):
        state["status"] = "completed"
        state["completed_at"] = utc_now()
    else:
        state["status"] = "paused_after_max_tasks"
    state["updated_at"] = utc_now()
    write_json(scheduler_path, state)
    print(scheduler_path)
    if stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
