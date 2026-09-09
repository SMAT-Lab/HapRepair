#!/usr/bin/env python3
"""Preserve the pre-model DevEco Node launch incident from formal_01."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUN = (
    WORKSPACE_ROOT
    / "baseline_data/exp_agent_ref_10/runs/exp_agent_ref_10_luna_v4_formal_01"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_01_node_path_incident_20260806.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    scheduler_path = RUN / "formal_scheduler/manifest.json"
    scheduler = read_json(scheduler_path)
    failures = []
    for manifest_path in sorted((RUN / "vanilla").glob("*/run_manifest.json")):
        manifest = read_json(manifest_path)
        failures.append(
            {
                "project": manifest["project"],
                "status": manifest["status"],
                "failure": manifest.get("failure"),
                "round_count": len(manifest.get("rounds") or []),
                "manifest": str(manifest_path),
            }
        )
    trace_count = len(list((RUN / "vanilla").glob("*/traces/*.jsonl")))
    scan_count = len(
        list((RUN / "vanilla").glob("*/evaluator_state/homecheck/scans/*"))
    )
    pending = [
        item["project"] for item in scheduler["tasks"] if item["status"] == "pending"
    ]
    passed = (
        scheduler["status"] == "stopped_condition_failure"
        and len(failures) == 8
        and all(item["status"] == "failed" for item in failures)
        and all(
            "No such file or directory: 'node'" in item["failure"] for item in failures
        )
        and all(item["round_count"] == 0 for item in failures)
        and trace_count == 0
        and scan_count == 0
        and len(pending) == 2
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": "exp_agent_ref_10_luna_v4_formal_01",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed_excluded_infrastructure_failure" if passed else "open",
        "passed": passed,
        "classification": "pre-model evaluator environment failure",
        "cause": "The systemd service PATH omitted the frozen DevEco Node required by evaluator declaration/public-API extraction.",
        "scheduler_manifest": str(scheduler_path),
        "failed_condition_count": len(failures),
        "pending_condition_count": len(pending),
        "pending_projects": pending,
        "model_turn_count": trace_count,
        "homecheck_scan_count": scan_count,
        "condition_failures": failures,
        "evidence_policy": "Preserve formal_01 unchanged and exclude it from effectiveness, token, cost, and timing results.",
        "retry_authorized_in_principle": passed,
        "required_retry": "Use a new audited run ID with the DevEco Node directory prepended by the runner itself.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
