#!/usr/bin/env python3
"""Authorize the path/schema-corrected EXP-AGENT-REF-10 formal_03 retry."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import audit_agent_ref_v4 as base
import run_agent_ref_v4 as runner


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
PRIOR_PROTOCOL = HERE / "protocol-agent-ref-10-v4-formal-02.json"
RETRY_PROTOCOL = HERE / "protocol-agent-ref-10-v4-formal-03.json"
INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_02_harness_incident_20260806.json"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_retry_g2_20260806.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base.PROTOCOL = RETRY_PROTOCOL
    base.OUTPUT = OUTPUT
    base.main()
    result = read_json(OUTPUT)
    prior = read_json(PRIOR_PROTOCOL)
    retry = read_json(RETRY_PROTOCOL)
    normalized = copy.deepcopy(retry)
    normalized["formal_run"] = copy.deepcopy(prior["formal_run"])
    incident = read_json(INCIDENT)

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        control = workspace / ".exp_agent/round_01"
        control.mkdir(parents=True)
        plan = control / "plan.json"
        completion = control / "completion.json"
        schema = control / "completion-schema.json"
        prompt = runner.prompt_for_attempt(
            {"name": "smoke", "commit": "frozen"},
            1,
            1,
            workspace,
            plan,
            completion,
            schema,
            "smoke",
            False,
        )
    exact_paths = (
        all(
            value in prompt
            for value in (
                "/workspace/.exp_agent/round_01/plan.json",
                "/workspace/.exp_agent/round_01/completion.json",
                "/workspace/.exp_agent/round_01/completion-schema.json",
            )
        )
        and "/workspace/round_01/" not in prompt
    )
    schema_payload = runner.completion_schema()
    repair_schema = schema_payload["properties"]["entity_repairs"]["items"]
    exact_schema = (
        "evidence" in repair_schema["required"]
        and repair_schema["properties"]["evidence"]
        == {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        }
        and "repository_evidence" not in repair_schema["properties"]
        and 'exact key "evidence"' in prompt
    )
    result["checks"].extend(
        [
            {
                "name": "scientific_contract_unchanged_from_formal_02",
                "passed": normalized == prior,
                "prior_protocol": str(PRIOR_PROTOCOL),
                "retry_protocol": str(RETRY_PROTOCOL),
                "allowed_delta": [
                    "new formal run ID",
                    "formal_02 harness-incident supersession provenance",
                    "formal_03 authorization artifact",
                ],
            },
            {
                "name": "formal_02_harness_incident_closed",
                "passed": incident.get("status") == "closed_excluded_harness_failure"
                and incident.get("passed") is True
                and incident.get("scheduler_status_counts_at_stop")
                == {"active": 1, "completed": 3, "failed": 6},
                "incident": str(INCIDENT),
                "classification": incident.get("classification"),
            },
            {
                "name": "agent_visible_control_path_round_trip",
                "passed": exact_paths,
                "prompt_excerpt": [
                    line
                    for line in prompt.splitlines()
                    if line.startswith(
                        ("Plan:", "Completion report:", "Exact completion")
                    )
                ],
            },
            {
                "name": "exact_completion_evidence_schema",
                "passed": exact_schema,
                "evidence_schema": repair_schema["properties"].get("evidence"),
            },
        ]
    )
    passed = all(item["passed"] for item in result["checks"])
    result["gate"] = "formal_03 path/schema-only retry authorization"
    result["status"] = "closed" if passed else "open_failed_audit"
    result["passed"] = passed
    result["formal_execution_authorized"] = passed
    result["authorized_run_id"] = retry["formal_run"]["run_id"]
    result["runner_artifacts"][Path(__file__).name] = base.sha256_file(Path(__file__))
    result["claim_boundary"] = (
        "formal_02 is wholly excluded. formal_03 changes only evaluator prompt paths, "
        "the explicit completion-report schema, regression coverage, and retry provenance."
    )
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={base.sha256_file(OUTPUT)}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
