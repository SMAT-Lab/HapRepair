#!/usr/bin/env python3
"""Rerun the three failed HapRepair Skill conditions without touching baseline."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from run_full_pair_repetitions_v2 import (
    DEFAULT_CAMPAIGN,
    DEFAULT_RUN_ROOT,
    run_one,
    task_summary,
    utc_now,
    write_json,
)


DEFAULT_REPLACEMENT_ID = "exp_full_pair_35_luna_skill_failed3_replacement_03"
DEFAULT_RUN_IDS = (
    "exp_hapskill_35_luna_v16_failed3_replacement_repeat_01",
    "exp_hapskill_35_luna_v16_failed3_replacement_repeat_02",
)
REPLACEMENT_TASKS = {
    (1, "asn1_ber"),
    (2, "asn1_ber"),
    (2, "bluetoothtest"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def replacement_tasks(
    campaign: dict[str, Any], run_ids: tuple[str, str]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for source in campaign["tasks"]:
        identity = (int(source["repeat"]), str(source["project"]))
        if source["condition"] != "hapskill" or identity not in REPLACEMENT_TASKS:
            continue
        task = dict(source)
        task["run_id"] = run_ids[int(task["repeat"]) - 1]
        task["status"] = "pending"
        tasks.append(task)
    if len(tasks) != 3:
        raise ValueError(f"Expected three failed Skill tasks, found {len(tasks)}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--replacement-id", default=DEFAULT_REPLACEMENT_ID)
    parser.add_argument("--repeat-1-run-id", default=DEFAULT_RUN_IDS[0])
    parser.add_argument("--repeat-2-run-id", default=DEFAULT_RUN_IDS[1])
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    campaign_path = args.campaign.resolve()
    run_root = args.run_root.resolve()
    campaign = read_json(campaign_path)
    if campaign.get("project_count") != 35:
        raise SystemExit("The frozen campaign must contain 35 projects")
    if not 1 <= args.max_concurrent <= 16:
        raise SystemExit("Replacement concurrency must be between 1 and 16")
    inputs = Path(campaign["input_manifest"]).resolve()
    tasks = replacement_tasks(
        campaign, (args.repeat_1_run_id, args.repeat_2_run_id)
    )
    if args.dry_run:
        print(json.dumps(tasks, ensure_ascii=False, indent=2))
        return

    scheduler_dir = run_root / args.replacement_id / "scheduler"
    scheduler_path = scheduler_dir / "manifest.json"
    log_dir = scheduler_dir / "logs"
    if scheduler_dir.exists():
        raise SystemExit(f"Replacement scheduler already exists: {scheduler_dir}")
    log_dir.mkdir(parents=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "experiment": campaign["experiment"],
        "replacement_id": args.replacement_id,
        "source_campaign": str(campaign_path),
        "input_manifest": str(inputs),
        "maximum_concurrent_conditions": args.max_concurrent,
        "replacement_policy": (
            "One post-hoc recovery attempt for each of the three failed Skill "
            "conditions. Prior records are preserved and baseline remains read-only."
        ),
        "status": "running",
        "created_at": utc_now(),
        "tasks": tasks,
    }
    write_json(scheduler_path, state)

    next_index = 0
    harness_failure = False
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        while next_index < len(tasks) and len(active) < args.max_concurrent:
            task = tasks[next_index]
            next_index += 1
            task["status"] = "active"
            active[executor.submit(
                run_one,
                task=task,
                inputs=inputs,
                run_root=run_root,
                log_dir=log_dir,
            )] = task
            print(f"[replacement] started {task['task_id']}", flush=True)

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    task.update(future.result())
                except Exception as error:
                    task["status"] = "incomplete"
                    task["scheduler_failure"] = f"{type(error).__name__}: {error}"
                    harness_failure = True
                    print(
                        f"[replacement] INCOMPLETE {task['task_id']}: {error}",
                        flush=True,
                    )
                else:
                    print(
                        f"[replacement] terminal {task['task_id']}: "
                        f"{task['condition_status']}",
                        flush=True,
                    )
                state["status_counts"] = task_summary(tasks)
                state["updated_at"] = utc_now()
                write_json(scheduler_path, state)

            if not harness_failure:
                while next_index < len(tasks) and len(active) < args.max_concurrent:
                    task = tasks[next_index]
                    next_index += 1
                    task["status"] = "active"
                    active[executor.submit(
                        run_one,
                        task=task,
                        inputs=inputs,
                        run_root=run_root,
                        log_dir=log_dir,
                    )] = task
                    print(f"[replacement] started {task['task_id']}", flush=True)

    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    state["status_counts"] = task_summary(tasks)
    if harness_failure:
        state["status"] = "incomplete"
    elif any(task.get("condition_status") != "completed" for task in tasks):
        state["status"] = "completed_with_condition_failures"
    else:
        state["status"] = "completed"
    write_json(scheduler_path, state)
    print(scheduler_path)
    if harness_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
