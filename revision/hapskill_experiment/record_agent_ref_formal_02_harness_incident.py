#!/usr/bin/env python3
"""Preserve and exclude formal_02 after its prompt/report contract failure."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUN = (
    WORKSPACE_ROOT
    / "baseline_data/exp_agent_ref_10/runs/exp_agent_ref_10_luna_v4_formal_02"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_02_harness_incident_20260806.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    scheduler_path = RUN / "formal_scheduler/manifest.json"
    scheduler = read_json(scheduler_path)
    statuses: dict[str, int] = {}
    for task in scheduler["tasks"]:
        statuses[task["status"]] = statuses.get(task["status"], 0) + 1
    failed = []
    for task in scheduler["tasks"]:
        manifest_path = RUN / "vanilla" / task["project"] / "run_manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") == "failed":
            failed.append(
                {
                    "project": task["project"],
                    "failure": manifest.get("failure"),
                    "round_count": len(manifest.get("rounds") or []),
                    "manifest": str(manifest_path),
                }
            )

    acts_completion = read_json(
        RUN / "vanilla/acts_validator/workspace/.exp_agent/round_01/completion.json"
    )
    acts_repository_evidence = sum(
        isinstance(item.get("repository_evidence"), list)
        and bool(item["repository_evidence"])
        for item in acts_completion["entity_repairs"]
    )
    wrong_bluetooth_report = (
        RUN / "vanilla/bluetoothtest/workspace/round_01/completion.json"
    )
    intended_bluetooth_report = (
        RUN / "vanilla/bluetoothtest/workspace/.exp_agent/round_01/completion.json"
    )
    passed = (
        scheduler["status"] == "stopped_condition_failure"
        and statuses == {"active": 1, "completed": 3, "failed": 6}
        and len(failed) == 6
        and all("Bounded edit retries exhausted" in item["failure"] for item in failed)
        and acts_repository_evidence == 404
        and wrong_bluetooth_report.is_file()
        and not intended_bluetooth_report.exists()
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": "exp_agent_ref_10_luna_v4_formal_02",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed_excluded_harness_failure" if passed else "open",
        "passed": passed,
        "classification": "evaluator prompt/report contract implementation failure",
        "scheduler_manifest": str(scheduler_path),
        "scheduler_status_counts_at_stop": statuses,
        "failed_conditions": failed,
        "root_causes": [
            {
                "id": "wrong_agent_visible_control_path",
                "expected": "/workspace/.exp_agent/round_01/completion.json",
                "prompted": "/workspace/round_01/completion.json",
                "evidence": str(wrong_bluetooth_report),
            },
            {
                "id": "ambiguous_evidence_field_contract",
                "auditor_required": "non-empty JSON array named evidence",
                "prompt_worded": "non-empty repository evidence",
                "acts_validator_repository_evidence_records": acts_repository_evidence,
            },
        ],
        "evidence_policy": "Exclude all formal_02 conditions, including completed rows, because every agent received the faulty prompt contract.",
        "retry_authorized_in_principle": passed,
        "required_retry": "Fix exact container paths, publish an explicit completion JSON Schema, add end-to-end regression tests, audit, and use a new run ID.",
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
