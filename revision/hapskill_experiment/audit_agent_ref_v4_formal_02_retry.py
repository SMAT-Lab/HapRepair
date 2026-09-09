#!/usr/bin/env python3
"""Authorize only the environment-corrected EXP-AGENT-REF-10 formal_02 retry."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import audit_agent_ref_v4 as base
import run_agent_ref_v4 as runner


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
BASE_PROTOCOL = HERE / "protocol-agent-ref-10-v4.json"
RETRY_PROTOCOL = HERE / "protocol-agent-ref-10-v4-formal-02.json"
INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_formal_01_node_path_incident_20260806.json"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_02_retry_g2_20260806.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base.PROTOCOL = RETRY_PROTOCOL
    base.OUTPUT = OUTPUT
    base.main()
    result = read_json(OUTPUT)
    original = read_json(BASE_PROTOCOL)
    retry = read_json(RETRY_PROTOCOL)
    normalized = copy.deepcopy(retry)
    normalized["runtime"].pop("node")
    normalized["runtime"].pop("node_bin")
    normalized["formal_run"]["run_id"] = original["formal_run"]["run_id"]
    normalized["formal_run"].pop("supersedes_run_id")
    normalized["formal_run"].pop("superseded_reason")
    normalized["formal_run"].pop("authorization_artifact")
    incident = read_json(INCIDENT)
    try:
        host_runtime = runner.configure_host_runtime(retry)
        smoke_project = Path(base.read_json(base.INPUTS)["projects"][6]["source_path"])
        public_api = runner.guards.public_api_inventory(smoke_project)
        node_smoke = bool(public_api.get("api") is not None)
        node_smoke_error = None
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        host_runtime = {}
        node_smoke = False
        node_smoke_error = f"{type(error).__name__}: {error}"
    result["checks"].extend(
        [
            {
                "name": "scientific_contract_unchanged",
                "passed": normalized == original,
                "base_protocol": str(BASE_PROTOCOL),
                "retry_protocol": str(RETRY_PROTOCOL),
                "allowed_delta": [
                    "new formal run ID and supersession provenance",
                    "frozen DevEco Node version and binary directory",
                    "retry-specific static G2 authorization artifact",
                ],
            },
            {
                "name": "formal_01_incident_closed_before_model_execution",
                "passed": incident.get("status")
                == "closed_excluded_infrastructure_failure"
                and incident.get("passed") is True
                and incident.get("model_turn_count") == 0
                and incident.get("homecheck_scan_count") == 0
                and incident.get("failed_condition_count") == 8
                and incident.get("pending_condition_count") == 2,
                "incident": str(INCIDENT),
                "classification": incident.get("classification"),
            },
            {
                "name": "frozen_node_path_public_api_smoke",
                "passed": node_smoke
                and host_runtime.get("path_policy") == "prepended_frozen_deveco_node",
                "host_runtime": host_runtime,
                "project": str(smoke_project),
                "error": node_smoke_error,
            },
        ]
    )
    passed = all(item["passed"] for item in result["checks"])
    result["gate"] = "formal_02 environment-only retry authorization"
    result["status"] = "closed" if passed else "open_failed_audit"
    result["passed"] = passed
    result["formal_execution_authorized"] = passed
    result["authorized_run_id"] = retry["formal_run"]["run_id"]
    result["runner_artifacts"][Path(__file__).name] = base.sha256_file(Path(__file__))
    result["claim_boundary"] = (
        "formal_01 contains no model or HomeCheck evidence. formal_02 changes only "
        "the evaluator Node PATH and retry provenance; scientific inputs and method remain fixed."
    )
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={base.sha256_file(OUTPUT)}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
