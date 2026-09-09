#!/usr/bin/env python3
"""Execute the frozen EXP-AGENT-10 formal task order durably and resumably."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_agent_baseline import sha256_file, utc_now
from validation_gate import write_json


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"
DEFAULT_RUN_ROOT = HERE.parents[1].parent / "baseline_data" / "exp_agent_10" / "runs"
RUNNERS = {
    "coding_agent": HERE / "run_agent_baseline.py",
    "haprepair": HERE / "run_haprepair_condition.py",
}


def parse_tasks(protocol: dict[str, Any]) -> list[dict[str, str]]:
    tasks = []
    for ordinal, value in enumerate(protocol["formal_run"]["task_order"], start=1):
        project, condition = value.split(":", 1)
        if condition not in RUNNERS:
            raise ValueError(f"Unknown formal condition: {condition}")
        tasks.append(
            {
                "ordinal": ordinal,
                "project": project,
                "condition": condition,
                "status": "pending",
            }
        )
    return tasks


def load_runner_manifest(
    run_root: Path, run_id: str, condition: str, project: str
) -> dict[str, Any] | None:
    path = run_root / run_id / condition / project / "run_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_id = protocol["formal_run"]["run_id"]
    run_root = args.run_root.resolve()
    orchestration_dir = run_root / run_id / "orchestration"
    manifest_path = orchestration_dir / "manifest.json"
    logs_dir = orchestration_dir / "logs"
    protocol_sha256 = sha256_file(protocol_path)
    frozen_tasks = parse_tasks(protocol)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != protocol_sha256:
            raise SystemExit("Formal protocol changed after orchestration started")
        expected = [
            (item["project"], item["condition"]) for item in frozen_tasks
        ]
        observed = [
            (item["project"], item["condition"]) for item in manifest["tasks"]
        ]
        if observed != expected:
            raise SystemExit("Formal task order changed after orchestration started")
    else:
        orchestration_dir.mkdir(parents=True, exist_ok=False)
        logs_dir.mkdir()
        manifest = {
            "schema_version": 1,
            "experiment": "EXP-AGENT-10",
            "run_id": run_id,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "python": sys.version,
            "tasks": frozen_tasks,
        }
        write_json(manifest_path, manifest)

    completed_this_invocation = 0
    for task in manifest["tasks"]:
        existing = load_runner_manifest(
            run_root, run_id, task["condition"], task["project"]
        )
        if existing and existing.get("status") == "completed":
            task["status"] = "completed"
            task["runner_manifest"] = str(
                run_root
                / run_id
                / task["condition"]
                / task["project"]
                / "run_manifest.json"
            )
            continue
        if task.get("status") == "completed" and not existing:
            raise SystemExit(f"Completed task lost its runner manifest: {task}")
        run_dir = run_root / run_id / task["condition"] / task["project"]
        if run_dir.exists():
            task["status"] = "incomplete_requires_audit"
            manifest["status"] = "stopped_incomplete_task"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(f"Incomplete formal task requires audit: {run_dir}")
        if args.max_tasks is not None and completed_this_invocation >= args.max_tasks:
            manifest["status"] = "paused_after_max_tasks"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            print(f"[paused] {completed_this_invocation} task(s) completed")
            return

        runner = RUNNERS[task["condition"]]
        command = [
            sys.executable,
            str(runner),
            "--run-id",
            run_id,
            "--project",
            task["project"],
            "--protocol",
            str(protocol_path),
            "--run-root",
            str(run_root),
        ]
        if task["condition"] == "haprepair":
            command.extend(["--device", args.device])
        log_path = logs_dir / (
            f"{task['ordinal']:02d}_{task['project']}_{task['condition']}.log"
        )
        task.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "command": command,
                "log_path": str(log_path),
            }
        )
        manifest["status"] = "running"
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(
            f"[formal {task['ordinal']:02d}/{len(manifest['tasks'])}] "
            f"{task['project']} {task['condition']}",
            flush=True,
        )
        started = time.monotonic()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("wb") as log:
            result = subprocess.run(
                command,
                cwd=HERE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        task["completed_at"] = utc_now()
        task["wall_seconds"] = time.monotonic() - started
        task["exit_code"] = result.returncode
        task["log_sha256"] = sha256_file(log_path)
        runner_manifest = load_runner_manifest(
            run_root, run_id, task["condition"], task["project"]
        )
        if result.returncode != 0 or not runner_manifest:
            task["status"] = "failed"
            manifest["status"] = "stopped_task_failure"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(f"Formal task failed; see {log_path}")
        task["runner_manifest"] = str(
            run_root
            / run_id
            / task["condition"]
            / task["project"]
            / "run_manifest.json"
        )
        task["runner_manifest_sha256"] = sha256_file(Path(task["runner_manifest"]))
        if runner_manifest.get("status") != "completed":
            task["status"] = "failed"
            manifest["status"] = "stopped_task_failure"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(
                f"Formal task ended with runner status {runner_manifest.get('status')}; "
                f"see {task['runner_manifest']}"
            )
        task["status"] = "completed"
        completed_this_invocation += 1
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)

    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"[completed] {len(manifest['tasks'])} formal tasks")


if __name__ == "__main__":
    main()
