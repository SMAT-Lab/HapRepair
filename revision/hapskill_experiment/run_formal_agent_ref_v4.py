#!/usr/bin/env python3
"""Schedule the frozen ten-project EXP-AGENT-REF-10 v4 run."""

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
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_agent_baseline import sha256_file, utc_now  # type: ignore  # noqa: E402
from run_agent_ref_v4 import validate_authorization  # noqa: E402


DEFAULT_PROTOCOL = HERE / "protocol-agent-ref-10-v4.json"
DEFAULT_INPUTS = HERE / "formal_inputs_agent_ref_10_v4.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_agent_ref_10/runs"
RUNNER = HERE / "run_agent_ref_v4.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_contract(
    protocol_path: Path, input_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(protocol_path)
    inputs = read_json(input_path)
    if protocol.get("status") != "frozen" or inputs.get("status") != "frozen":
        raise ValueError("Protocol and formal inputs must be frozen")
    if protocol["formal_run"]["condition"] != "vanilla":
        raise ValueError("Reference scheduler accepts only the vanilla condition")
    if protocol["dataset"]["formal_inputs_sha256"] != sha256_file(input_path):
        raise ValueError("Formal input hash differs from the protocol")
    order = protocol["formal_run"]["condition_order"]
    expected = [f"{item['name']}:vanilla" for item in inputs["projects"]]
    expected_count = int(protocol["dataset"]["project_count"])
    if order != expected or len(order) != expected_count:
        raise ValueError("Exact project membership/order contract failed")
    tasks = [
        {
            "ordinal": index,
            "project": item["name"],
            "condition": "vanilla",
            "status": "pending",
        }
        for index, item in enumerate(inputs["projects"], 1)
    ]
    return protocol, tasks


def condition_dir(root: Path, run_id: str, project: str) -> Path:
    return root / run_id / "vanilla" / project


def condition_manifest(root: Path, run_id: str, project: str) -> Path:
    return condition_dir(root, run_id, project) / "run_manifest.json"


def classify_existing(root: Path, run_id: str, task: dict[str, Any]) -> str:
    directory = condition_dir(root, run_id, task["project"])
    manifest_path = directory / "run_manifest.json"
    if not directory.exists():
        return "untouched"
    if (
        manifest_path.is_file()
        and read_json(manifest_path).get("status") == "completed"
    ):
        return "completed"
    return "incomplete_requires_audit"


def run_one(
    task: dict[str, Any],
    run_id: str,
    root: Path,
    protocol: Path,
    inputs: Path,
    logs: Path,
) -> dict[str, Any]:
    log_path = logs / f"{task['ordinal']:02d}_{task['project']}.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--project",
        task["project"],
        "--protocol",
        str(protocol),
        "--project-manifest",
        str(inputs),
        "--run-root",
        str(root),
    ]
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
    manifest_path = condition_manifest(root, run_id, task["project"])
    if (
        result.returncode
        or not manifest_path.is_file()
        or read_json(manifest_path).get("status") != "completed"
    ):
        raise RuntimeError(f"Condition failed for {task['project']}; see {log_path}")
    return {
        **task,
        "status": "completed",
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
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
    inputs_path = args.project_manifest.resolve()
    root = args.run_root.resolve()
    protocol, tasks = load_contract(protocol_path, inputs_path)
    run_id = args.run_id or protocol["formal_run"]["run_id"]
    maximum = args.max_concurrent or int(
        protocol["execution"]["maximum_concurrent_conditions"]
    )
    allowed_run_ids = protocol["formal_run"].get("run_ids") or [
        protocol["formal_run"].get("run_id")
    ]
    if run_id not in allowed_run_ids:
        raise SystemExit("Run ID differs from the frozen protocol")
    if maximum != int(protocol["execution"]["maximum_concurrent_conditions"]):
        raise SystemExit("Concurrency differs from the frozen protocol")
    summary = {
        "experiment": protocol["experiment"],
        "run_id": run_id,
        "condition": "vanilla",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "project_manifest": str(inputs_path),
        "project_manifest_sha256": sha256_file(inputs_path),
        "project_count": len(tasks),
        "maximum_concurrent_conditions": maximum,
        "tasks": tasks,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    authorization = validate_authorization(protocol_path, inputs_path, run_id)
    summary["static_g2_authorization"] = authorization

    scheduler_dir = root / run_id / "formal_scheduler"
    manifest_path = scheduler_dir / "manifest.json"
    logs = scheduler_dir / "logs"
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("Scheduler already exists; resume requires audit")
        state = read_json(manifest_path)
        if (
            state["protocol_sha256"] != summary["protocol_sha256"]
            or state["project_manifest_sha256"] != summary["project_manifest_sha256"]
        ):
            raise SystemExit("Frozen scheduler inputs drifted")
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["last_resumed_at"] = utc_now()
    else:
        if args.resume:
            raise SystemExit("No scheduler exists to resume")
        logs.mkdir(parents=True, exist_ok=False)
        state = {
            **summary,
            "status": "running",
            "created_at": utc_now(),
            "resume_count": 0,
        }
    pending = []
    for task in state["tasks"]:
        existing = classify_existing(root, run_id, task)
        if existing == "completed":
            task["status"] = "completed"
            task["runner_manifest"] = str(
                condition_manifest(root, run_id, task["project"])
            )
        elif existing == "untouched":
            task["status"] = "pending"
            pending.append(task)
        else:
            task["status"] = existing
            state["status"] = "blocked_incomplete_condition"
            state["updated_at"] = utc_now()
            write_json(manifest_path, state)
            raise SystemExit(
                f"Incomplete condition requires audit: {condition_dir(root, run_id, task['project'])}"
            )
    write_json(manifest_path, state)

    next_index = 0
    stopped = False
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        while next_index < len(pending) and len(active) < maximum:
            task = pending[next_index]
            next_index += 1
            task["status"] = "active"
            active[
                executor.submit(
                    run_one, task, run_id, root, protocol_path, inputs_path, logs
                )
            ] = task
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    task.update(future.result())
                except Exception as error:
                    task["status"] = "failed"
                    task["failure"] = f"{type(error).__name__}: {error}"
                    state["status"] = "stopped_condition_failure"
                    state["failure"] = task["failure"]
                    stopped = True
                state["updated_at"] = utc_now()
                write_json(manifest_path, state)
            if not stopped:
                while next_index < len(pending) and len(active) < maximum:
                    task = pending[next_index]
                    next_index += 1
                    task["status"] = "active"
                    active[
                        executor.submit(
                            run_one,
                            task,
                            run_id,
                            root,
                            protocol_path,
                            inputs_path,
                            logs,
                        )
                    ] = task
    state["completed_at"] = utc_now()
    state["status"] = "stopped_condition_failure" if stopped else "completed"
    write_json(manifest_path, state)
    if stopped:
        raise SystemExit(1)
    print(manifest_path)


if __name__ == "__main__":
    main()
