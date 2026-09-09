#!/usr/bin/env python3
"""Run the frozen 35-project Skill/baseline campaign in one global pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
DEFAULT_CAMPAIGN = HERE / "full_pair_campaign_35x2_v1.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_full_pair_35/runs"
RUNNERS = {
    "hapskill": HERE / "run_condition_v14.py",
    "vanilla": HERE / "run_agent_ref_v4.py",
}
TERMINAL_CONDITION_STATUSES = {"completed", "protocol_violation", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_dir(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / task["run_id"] / task["condition"] / task["project"]


def condition_manifest(run_root: Path, task: dict[str, Any]) -> Path:
    return condition_dir(run_root, task) / "run_manifest.json"


def validate_campaign(
    campaign_path: Path, *, require_g2: bool
) -> tuple[dict[str, Any], Path]:
    campaign = read_json(campaign_path)
    if campaign.get("status") != "prepared":
        raise ValueError("Campaign must be prepared and immutable")
    if campaign.get("project_count") != 35 or campaign.get("task_count") != 140:
        raise ValueError("Campaign must contain 140 tasks over 35 projects")
    if campaign.get("maximum_concurrent_conditions") != 16:
        raise ValueError("Campaign concurrency must be exactly 16")
    inputs = Path(campaign["input_manifest"]).resolve()
    if sha256_file(inputs) != campaign["input_manifest_sha256"]:
        raise ValueError("Shared input manifest drifted")
    for item in campaign["protocols"].values():
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Condition protocol drifted: {path}")
    task_ids = [task["task_id"] for task in campaign["tasks"]]
    ordinals = [task["ordinal"] for task in campaign["tasks"]]
    if len(set(task_ids)) != 140 or ordinals != list(range(1, 141)):
        raise ValueError("Campaign task identities or order are invalid")
    if require_g2:
        gate_path = Path(campaign["g2_authorization"]).resolve()
        if not gate_path.is_file():
            raise ValueError(f"G2 authorization is missing: {gate_path}")
        gate = read_json(gate_path)
        if (
            gate.get("status") != "closed"
            or gate.get("passed") is not True
            or gate.get("formal_execution_authorized") is not True
            or gate.get("campaign_sha256") != sha256_file(campaign_path)
            or any(not item.get("passed") for item in gate.get("checks", []))
        ):
            raise ValueError("G2 authorization is not closed and passing")
    return campaign, inputs


def classify_existing(run_root: Path, task: dict[str, Any]) -> str:
    directory = condition_dir(run_root, task)
    manifest_path = directory / "run_manifest.json"
    if not directory.exists():
        return "untouched"
    if manifest_path.is_file():
        status = read_json(manifest_path).get("status")
        if status in TERMINAL_CONDITION_STATUSES:
            return str(status)
    return "incomplete_requires_audit"


def runner_command(*, task: dict[str, Any], inputs: Path, run_root: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNERS[task["condition"]]),
        "--run-id",
        task["run_id"],
        "--project",
        task["project"],
        "--protocol",
        task["protocol"],
        "--project-manifest",
        str(inputs),
        "--run-root",
        str(run_root),
    ]


def run_one(
    *,
    task: dict[str, Any],
    inputs: Path,
    run_root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    log_path = (
        log_dir / f"{task['ordinal']:03d}_{task['task_id'].replace(':', '__')}.log"
    )
    command = runner_command(task=task, inputs=inputs, run_root=run_root)
    started = time.monotonic()
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            cwd=HERE,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    manifest_path = condition_manifest(run_root, task)
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Runner exited {result.returncode} without a condition manifest; see {log_path}"
        )
    manifest = read_json(manifest_path)
    condition_status = manifest.get("status")
    if condition_status not in TERMINAL_CONDITION_STATUSES:
        raise RuntimeError(
            f"Runner left non-terminal status {condition_status!r}; see {manifest_path}"
        )
    return {
        **task,
        "status": "terminal",
        "condition_status": condition_status,
        "runner_exit_code": result.returncode,
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "runner_manifest": str(manifest_path),
        "runner_manifest_sha256": sha256_file(manifest_path),
        "condition_failure": manifest.get("failure"),
    }


def task_summary(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                task.get("condition_status", task["status"]) for task in tasks
            ).items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--resume-scheduler", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    campaign_path = args.campaign.resolve()
    run_root = args.run_root.resolve()
    campaign, inputs = validate_campaign(campaign_path, require_g2=not args.dry_run)
    maximum = args.max_concurrent or campaign["maximum_concurrent_conditions"]
    if maximum != campaign["maximum_concurrent_conditions"]:
        raise SystemExit("Concurrency override differs from the frozen campaign")
    commands = [
        {**task, "command": runner_command(task=task, inputs=inputs, run_root=run_root)}
        for task in campaign["tasks"]
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "campaign_id": campaign["campaign_id"],
                    "campaign_sha256": sha256_file(campaign_path),
                    "task_count": len(commands),
                    "maximum_concurrent_conditions": maximum,
                    "tasks": commands,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    scheduler_dir = run_root / campaign["campaign_id"] / "scheduler"
    scheduler_path = scheduler_dir / "manifest.json"
    log_dir = scheduler_dir / "logs"
    if scheduler_path.exists():
        if not args.resume_scheduler:
            raise SystemExit(
                "Scheduler already exists; use --resume-scheduler after audit"
            )
        state = read_json(scheduler_path)
        if state.get("campaign_sha256") != sha256_file(campaign_path):
            raise SystemExit("Campaign drifted after scheduler creation")
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["last_resumed_at"] = utc_now()
    else:
        if args.resume_scheduler:
            raise SystemExit("No scheduler exists to resume")
        log_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            "experiment": campaign["experiment"],
            "campaign_id": campaign["campaign_id"],
            "campaign": str(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "input_manifest": str(inputs),
            "input_manifest_sha256": sha256_file(inputs),
            "maximum_concurrent_conditions": maximum,
            "status": "running",
            "created_at": utc_now(),
            "resume_count": 0,
            "tasks": copy_tasks(campaign["tasks"]),
        }

    pending: list[dict[str, Any]] = []
    blockers: list[str] = []
    for task in state["tasks"]:
        existing = classify_existing(run_root, task)
        if existing in TERMINAL_CONDITION_STATUSES:
            manifest_path = condition_manifest(run_root, task)
            manifest = read_json(manifest_path)
            task.update(
                {
                    "status": "terminal",
                    "condition_status": existing,
                    "runner_manifest": str(manifest_path),
                    "runner_manifest_sha256": sha256_file(manifest_path),
                    "condition_failure": manifest.get("failure"),
                }
            )
        elif existing == "untouched":
            task["status"] = "pending"
            pending.append(task)
        else:
            task["status"] = existing
            blockers.append(task["task_id"])
    if blockers:
        state["status"] = "blocked_incomplete_conditions"
        state["blocked_tasks"] = blockers
        state["updated_at"] = utc_now()
        write_json(scheduler_path, state)
        raise SystemExit(f"Incomplete conditions require audit: {blockers}")
    write_json(scheduler_path, state)

    next_index = 0
    harness_failure = False
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        while next_index < len(pending) and len(active) < maximum:
            task = pending[next_index]
            next_index += 1
            task["status"] = "active"
            active[
                executor.submit(
                    run_one,
                    task=task,
                    inputs=inputs,
                    run_root=run_root,
                    log_dir=log_dir,
                )
            ] = task
            print(f"[scheduler] started {task['task_id']}", flush=True)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    task.update(future.result())
                except Exception as error:
                    task["status"] = "incomplete_requires_audit"
                    task["scheduler_failure"] = f"{type(error).__name__}: {error}"
                    harness_failure = True
                    print(
                        f"[scheduler] INCOMPLETE {task['task_id']}: {error}", flush=True
                    )
                else:
                    print(
                        f"[scheduler] terminal {task['task_id']}: {task['condition_status']}",
                        flush=True,
                    )
                state["status_counts"] = task_summary(state["tasks"])
                state["updated_at"] = utc_now()
                write_json(scheduler_path, state)
            if not harness_failure:
                while next_index < len(pending) and len(active) < maximum:
                    task = pending[next_index]
                    next_index += 1
                    task["status"] = "active"
                    active[
                        executor.submit(
                            run_one,
                            task=task,
                            inputs=inputs,
                            run_root=run_root,
                            log_dir=log_dir,
                        )
                    ] = task
                    print(f"[scheduler] started {task['task_id']}", flush=True)

    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    state["status_counts"] = task_summary(state["tasks"])
    if harness_failure:
        state["status"] = "blocked_incomplete_condition"
    elif any(task.get("condition_status") != "completed" for task in state["tasks"]):
        state["status"] = "completed_with_condition_failures"
    else:
        state["status"] = "completed"
    write_json(scheduler_path, state)
    print(scheduler_path)
    if harness_failure:
        raise SystemExit(1)


def copy_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(task) for task in tasks]


if __name__ == "__main__":
    main()
