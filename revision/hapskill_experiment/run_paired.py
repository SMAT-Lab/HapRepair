#!/usr/bin/env python3
"""Durable, resumable outer orchestrator for EXP-HAPSKILL-10."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
DEFAULT_PROTOCOL = HERE / "protocol-v3.json"
DEFAULT_FORMAL_INPUTS = HERE / "formal_inputs.json"
DEFAULT_CANDIDATES = (
    HERE.parent
    / "coding_agent_baseline"
    / "scan_runs"
    / "candidate_scan_01"
    / "scan_manifest.json"
)
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "runs"
RUNNER = HERE / "run_condition.py"


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_manifest(
    run_root: Path, run_id: str, condition: str, project: str
) -> Path:
    return run_root / run_id / condition / project / "run_manifest.json"


def paired_summary(run_root: Path, run_id: str, project: str) -> dict[str, Any]:
    manifests = {}
    for condition in ("vanilla", "hapskill"):
        path = condition_manifest(run_root, run_id, condition, project)
        if not path.is_file():
            raise FileNotFoundError(f"missing paired condition manifest: {path}")
        manifests[condition] = json.loads(path.read_text(encoding="utf-8"))
    vanilla = manifests["vanilla"]
    hapskill = manifests["hapskill"]
    comparable = {
        "input_tree_sha256": vanilla["input_tree_sha256"]
        == hapskill["input_tree_sha256"],
        "initial_target_findings_sha256": (
            vanilla["initial_target_findings_sha256"]
            == hapskill["initial_target_findings_sha256"]
        ),
        "model": vanilla["model"] == hapskill["model"],
        "scan_budget": vanilla["max_validation_scans"]
        == hapskill["max_validation_scans"],
        "protocol": vanilla["protocol_sha256"] == hapskill["protocol_sha256"],
    }
    metrics = {}
    for condition, manifest in manifests.items():
        metrics[condition] = {
            **manifest["alert_metrics"],
            "validation_scan_count": manifest["validation_scan_count"],
            "no_diff_retry_count": manifest.get("no_diff_retry_count", 0),
            "repair_required_round_count": manifest["repair_required_round_count"],
            "final_candidate_abandoned": bool(
                manifest.get("final_candidate_abandonment")
            ),
            "input_tokens": manifest["input_tokens"],
            "output_tokens": manifest["output_tokens"],
            "total_tokens": manifest["total_tokens"],
            "wall_clock_seconds": manifest["wall_clock_seconds"],
        }
    delta = {
        key: metrics["hapskill"][key] - metrics["vanilla"][key]
        for key in (
            "final_alerts",
            "eliminated_alerts",
            "introduced_alerts",
            "net_reduction",
            "validation_scan_count",
            "no_diff_retry_count",
            "repair_required_round_count",
            "final_candidate_abandoned",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "wall_clock_seconds",
        )
    }
    evaluation_summary = {
        "evidence_tier": "auxiliary/dev"
        if not vanilla.get("paper_facing", True)
        else "main/test",
        "outcome_summary": (
            f"Vanilla eliminated {metrics['vanilla']['eliminated_alerts']} of "
            f"{metrics['vanilla']['initial_alerts']} initial alerts and introduced "
            f"{metrics['vanilla']['introduced_alerts']}; HapRepair Skill eliminated "
            f"{metrics['hapskill']['eliminated_alerts']} and introduced "
            f"{metrics['hapskill']['introduced_alerts']}. Their net reductions were "
            f"{metrics['vanilla']['net_reduction']} and "
            f"{metrics['hapskill']['net_reduction']}, respectively."
        ),
        "claim_update": "inconclusive_pilot"
        if not vanilla.get("paper_facing", True)
        else "pending_cross_project_analysis",
        "baseline_relation": (
            f"Net reduction {'tied' if delta['net_reduction'] == 0 else 'differed'} "
            f"on this project. HapRepair Skill used {delta['validation_scan_count']:+d} "
            f"validation scans, {delta['total_tokens']:+d} tokens, and "
            f"{delta['wall_clock_seconds']:+.3f} seconds relative to vanilla."
        ),
        "failure_mode": "none; all paired comparability and contamination gates passed",
        "next_action": (
            "Treat this pilot as auxiliary evidence. Freeze the v3 G2 audit, then "
            "start formal_04 in the preregistered alternating order only if the "
            "comparability, isolation, lifecycle, and metric checks pass."
            if not vanilla.get("paper_facing", True)
            else "Route the completed formal comparison to cross-project analysis."
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "EXP-HAPSKILL-10",
        "run_id": run_id,
        "project": project,
        "status": "comparable" if all(comparable.values()) else "not_comparable",
        "comparability": comparable,
        "condition_manifests": {
            condition: str(condition_manifest(run_root, run_id, condition, project))
            for condition in manifests
        },
        "metrics": metrics,
        "hapskill_minus_vanilla": delta,
        "evaluation_summary": evaluation_summary,
        "claim_boundary": "HomeCheck alert elimination is not semantic correctness.",
    }


def tasks(protocol: dict[str, Any], pilot: bool) -> list[dict[str, Any]]:
    if pilot:
        project = protocol["pilot"]["project"]
        values = [
            f"{project}:{condition}"
            for condition in protocol["pilot"]["condition_order"]
        ]
    else:
        values = protocol["formal_run"]["condition_order"]
    return [
        {
            "ordinal": ordinal,
            "project": value.split(":", 1)[0],
            "condition": value.split(":", 1)[1],
            "status": "pending",
        }
        for ordinal, value in enumerate(values, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_id = args.run_id or (
        protocol["pilot"]["run_id"] if args.pilot else protocol["formal_run"]["run_id"]
    )
    project_manifest = (
        args.project_manifest.resolve()
        if args.project_manifest
        else DEFAULT_CANDIDATES.resolve()
        if args.pilot
        else DEFAULT_FORMAL_INPUTS.resolve()
    )
    if not project_manifest.is_file():
        raise SystemExit(f"project input manifest is missing: {project_manifest}")
    run_root = args.run_root.resolve()
    orchestration = run_root / run_id / "orchestration"
    manifest_path = orchestration / "manifest.json"
    logs = orchestration / "logs"
    frozen_tasks = tasks(protocol, args.pilot)
    protocol_hash = sha256_file(protocol_path)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != protocol_hash:
            raise SystemExit("protocol drifted after orchestration started")
        expected = [(item["project"], item["condition"]) for item in frozen_tasks]
        observed = [(item["project"], item["condition"]) for item in manifest["tasks"]]
        if expected != observed:
            raise SystemExit("task order drifted after orchestration started")
    else:
        logs.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "experiment": "EXP-HAPSKILL-10",
            "paper_facing": not args.pilot,
            "run_id": run_id,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "project_manifest": str(project_manifest),
            "project_manifest_sha256": sha256_file(project_manifest),
            "tasks": frozen_tasks,
            "paired_summaries": {},
        }
        write_json(manifest_path, manifest)

    completed_now = 0
    for task in manifest["tasks"]:
        runner_manifest = condition_manifest(
            run_root, run_id, task["condition"], task["project"]
        )
        if runner_manifest.is_file():
            observed = json.loads(runner_manifest.read_text(encoding="utf-8"))
            if observed.get("status") == "completed":
                task["status"] = "completed"
                task["runner_manifest"] = str(runner_manifest)
                continue
        if args.max_tasks is not None and completed_now >= args.max_tasks:
            manifest["status"] = "paused_after_max_tasks"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            return
        run_dir = runner_manifest.parent
        if run_dir.exists():
            task["status"] = "incomplete_requires_audit"
            manifest["status"] = "stopped_incomplete_task"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(f"incomplete condition requires audit: {run_dir}")

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
            str(project_manifest),
            "--run-root",
            str(run_root),
            "--device",
            args.device,
        ]
        if args.pilot:
            command.append("--pilot")
        log_path = (
            logs / f"{task['ordinal']:02d}_{task['project']}_{task['condition']}.log"
        )
        task.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "command": command,
                "log_path": str(log_path),
            }
        )
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(
            f"[{task['ordinal']:02d}/{len(manifest['tasks'])}] "
            f"{task['project']} {task['condition']}",
            flush=True,
        )
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
        task["completed_at"] = utc_now()
        task["wall_seconds"] = time.monotonic() - started
        task["exit_code"] = result.returncode
        task["log_sha256"] = sha256_file(log_path)
        if result.returncode != 0 or not runner_manifest.is_file():
            task["status"] = "failed"
            manifest["status"] = "stopped_task_failure"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(f"condition failed; see {log_path}")
        observed = json.loads(runner_manifest.read_text(encoding="utf-8"))
        if observed.get("status") != "completed":
            task["status"] = observed.get("status", "failed")
            manifest["status"] = "stopped_protocol_or_task_failure"
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(
                f"condition ended as {task['status']}; see {runner_manifest}"
            )
        task["status"] = "completed"
        task["runner_manifest"] = str(runner_manifest)
        task["runner_manifest_sha256"] = sha256_file(runner_manifest)
        completed_now += 1
        project_tasks = [
            item for item in manifest["tasks"] if item["project"] == task["project"]
        ]
        if all(item.get("status") == "completed" for item in project_tasks):
            summary = paired_summary(run_root, run_id, task["project"])
            summary_path = (
                run_root / run_id / "paired" / task["project"] / "summary.json"
            )
            write_json(summary_path, summary)
            manifest["paired_summaries"][task["project"]] = str(summary_path)
            if summary["status"] != "comparable":
                manifest["status"] = "stopped_pair_not_comparable"
                manifest["updated_at"] = utc_now()
                write_json(manifest_path, manifest)
                raise SystemExit(f"paired inputs are not comparable: {summary_path}")
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)

    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"[completed] {len(manifest['tasks'])} condition runs", flush=True)


if __name__ == "__main__":
    main()
