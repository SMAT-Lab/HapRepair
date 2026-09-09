#!/usr/bin/env python3
"""Run the remaining EXP-HAPSKILL-10 project pairs concurrently."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_paired as paired


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol-v3.json"
DEFAULT_INPUTS = HERE / "formal_inputs.json"
DEFAULT_RUN_ROOT = paired.DEFAULT_RUN_ROOT
RUNNER = HERE / "run_condition.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(
    *,
    task: dict[str, Any],
    run_id: str,
    run_root: Path,
    protocol: Path,
    project_manifest: Path,
    device: str,
    log_dir: Path,
) -> dict[str, Any]:
    project = task["project"]
    condition = task["condition"]
    manifest_path = paired.condition_manifest(run_root, run_id, condition, project)
    if manifest_path.is_file():
        observed = read_json(manifest_path)
        if observed.get("status") == "completed":
            return {
                **task,
                "status": "already_completed",
                "runner_manifest": str(manifest_path),
                "runner_manifest_sha256": paired.sha256_file(manifest_path),
            }
    if manifest_path.parent.exists():
        raise RuntimeError(
            f"incomplete condition requires audit: {manifest_path.parent}"
        )

    log_path = log_dir / f"{task['ordinal']:02d}_{project}_{condition}.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--project",
        project,
        "--condition",
        condition,
        "--protocol",
        str(protocol),
        "--project-manifest",
        str(project_manifest),
        "--run-root",
        str(run_root),
        "--device",
        device,
    ]
    started_at = paired.utc_now()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            cwd=HERE,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"condition failed; see {log_path}")
    observed = read_json(manifest_path)
    if observed.get("status") != "completed":
        raise RuntimeError(
            f"condition ended as {observed.get('status')}; see {manifest_path}"
        )
    return {
        **task,
        "status": "completed",
        "started_at": started_at,
        "completed_at": paired.utc_now(),
        "wall_seconds": time.monotonic() - started,
        "command": command,
        "log_path": str(log_path),
        "log_sha256": paired.sha256_file(log_path),
        "runner_manifest": str(manifest_path),
        "runner_manifest_sha256": paired.sha256_file(manifest_path),
    }


def reconcile_orchestration(
    *, run_root: Path, run_id: str, projects: list[str]
) -> None:
    path = run_root / run_id / "orchestration" / "manifest.json"
    manifest = read_json(path)
    for project in projects:
        summary_path = run_root / run_id / "paired" / project / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing paired summary: {summary_path}")
        manifest["paired_summaries"][project] = str(summary_path)
    manifest["scheduling_amendment"] = str(
        run_root / run_id / "parallel_scheduler" / "manifest.json"
    )
    paired.write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-ordinal", type=int, default=7)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--max-concurrent-projects", type=int, default=1)
    parser.add_argument("--scheduler-label", default="parallel_scheduler")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    args = parser.parse_args()

    if args.max_concurrent_projects < 1:
        raise SystemExit("--max-concurrent-projects must be at least 1")

    protocol_path = args.protocol.resolve()
    project_manifest = args.project_manifest.resolve()
    run_root = args.run_root.resolve()
    amendment = args.amendment.resolve()
    protocol = read_json(protocol_path)
    all_tasks = paired.tasks(protocol, pilot=False)
    selected = [task for task in all_tasks if task["ordinal"] >= args.start_ordinal]
    projects = list(dict.fromkeys(task["project"] for task in selected))
    pairs = {
        project: [task for task in selected if task["project"] == project]
        for project in projects
    }
    if any(len(tasks) != 2 for tasks in pairs.values()):
        raise RuntimeError("start ordinal must select complete project pairs")
    if not amendment.is_file():
        raise FileNotFoundError(f"missing scheduling amendment: {amendment}")
    if args.dry_run:
        print(json.dumps({"projects": projects, "pairs": pairs}, indent=2))
        return
    if args.reconcile_only:
        reconcile_orchestration(
            run_root=run_root, run_id=args.run_id, projects=projects
        )
        return

    scheduler_dir = run_root / args.run_id / args.scheduler_label
    scheduler_dir.mkdir(parents=True, exist_ok=False)
    log_dir = scheduler_dir / "logs"
    log_dir.mkdir()
    scheduler_manifest = scheduler_dir / "manifest.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-HAPSKILL-10",
        "run_id": args.run_id,
        "status": "running",
        "created_at": paired.utc_now(),
        "scheduling_mode": (
            "project_ordered_pair_concurrent"
            if args.max_concurrent_projects == 1
            else "multi_project_pair_concurrent"
        ),
        "maximum_concurrent_projects": args.max_concurrent_projects,
        "maximum_concurrent_conditions": args.max_concurrent_projects * 2,
        "start_ordinal": args.start_ordinal,
        "protocol": str(protocol_path),
        "protocol_sha256": paired.sha256_file(protocol_path),
        "project_manifest": str(project_manifest),
        "project_manifest_sha256": paired.sha256_file(project_manifest),
        "amendment": str(amendment),
        "amendment_sha256": paired.sha256_file(amendment),
        "projects": [],
    }
    paired.write_json(scheduler_manifest, state)

    project_records: dict[str, dict[str, Any]] = {}
    for project in projects:
        project_record: dict[str, Any] = {
            "project": project,
            "status": "running",
            "started_at": paired.utc_now(),
            "conditions": [],
        }
        project_records[project] = project_record
        state["projects"].append(project_record)
    paired.write_json(scheduler_manifest, state)

    with ThreadPoolExecutor(max_workers=args.max_concurrent_projects * 2) as executor:
        futures = {
            executor.submit(
                run_one,
                task=task,
                run_id=args.run_id,
                run_root=run_root,
                protocol=protocol_path,
                project_manifest=project_manifest,
                device=args.device,
                log_dir=log_dir,
            ): task
            for project in projects
            for task in pairs[project]
        }
        try:
            for future in as_completed(futures):
                task = futures[future]
                project_records[task["project"]]["conditions"].append(future.result())
                paired.write_json(scheduler_manifest, state)
        except Exception:
            task = futures[future]
            project_records[task["project"]]["status"] = "failed"
            state["status"] = "stopped_condition_failure"
            state["updated_at"] = paired.utc_now()
            paired.write_json(scheduler_manifest, state)
            raise

    for project in projects:
        project_record = project_records[project]
        project_record["conditions"].sort(key=lambda item: item["ordinal"])
        project_record["status"] = "completed"
        project_record["completed_at"] = paired.utc_now()
        summary = paired.paired_summary(run_root, args.run_id, project)
        summary_path = run_root / args.run_id / "paired" / project / "summary.json"
        paired.write_json(summary_path, summary)
        project_record["paired_summary"] = str(summary_path)
        project_record["paired_summary_sha256"] = paired.sha256_file(summary_path)
        paired.write_json(scheduler_manifest, state)

    state["status"] = "completed"
    state["completed_at"] = paired.utc_now()
    paired.write_json(scheduler_manifest, state)
    print(scheduler_manifest)


if __name__ == "__main__":
    main()
