#!/usr/bin/env python3
"""Authorize the environment-only formal_02 retry after the formal_01 incident."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_formal as formal


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASE_PROTOCOL = HERE / "protocol-hapskill-35-v4.json"
RETRY_PROTOCOL = HERE / "protocol-hapskill-35-v4-formal-02.json"
INPUTS = HERE / "formal_inputs_hapskill_35_v4.json"
INCIDENT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_hapskill_35_formal_01_node_path_incident_20260804.json"
)
DRY_RUN = (
    WORKSPACE_ROOT
    / "baseline_data/exp_hapskill/smoke/g2_v4_hapskill_formal_02_node_dry_01"
    / "hapskill/CanvasTest/run_manifest.json"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_hapskill_v4_formal_02_retry_g2_20260804.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )


def add(checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    checks.append({"name": name, "passed": passed, **evidence})


def main() -> None:
    checks: list[dict[str, Any]] = []
    base = read_json(BASE_PROTOCOL)
    retry, inputs, tasks = formal.validate_contract(RETRY_PROTOCOL, INPUTS)
    normalized = copy.deepcopy(retry)
    normalized["runtime"].pop("node")
    normalized["runtime"].pop("node_bin")
    normalized["execution"].pop("host_path_policy")
    normalized["formal_run"]["run_id"] = base["formal_run"]["run_id"]
    normalized["formal_run"].pop("supersedes_run_id")
    normalized["formal_run"].pop("superseded_reason")
    add(
        checks,
        "scientific_contract_unchanged",
        normalized == base,
        base_protocol=str(BASE_PROTOCOL),
        base_protocol_sha256=sha256_file(BASE_PROTOCOL),
        retry_protocol=str(RETRY_PROTOCOL),
        retry_protocol_sha256=sha256_file(RETRY_PROTOCOL),
        allowed_delta=[
            "formal run ID and incident provenance",
            "frozen DevEco Node path/version",
            "host PATH injection policy for HomeCheck subprocesses",
        ],
    )
    add(
        checks,
        "retry_membership_and_input",
        len(tasks) == inputs["project_count"] == 35
        and {item["condition"] for item in tasks} == {"hapskill"}
        and retry["dataset"]["formal_inputs_sha256"] == sha256_file(INPUTS)
        and retry["formal_run"]["supersedes_run_id"]
        == "exp_hapskill_35_luna_v4_formal_01",
        input_sha256=sha256_file(INPUTS),
        run_id=retry["formal_run"]["run_id"],
    )

    incident = read_json(INCIDENT)
    add(
        checks,
        "formal_01_incident_closed",
        incident["status"] == "closed_excluded_infrastructure_failure"
        and incident["model_turn_count"] == 0
        and incident["failed_condition_count"] == 8
        and incident["pending_condition_count"] == 27
        and incident["retry_authorized_in_principle"] is True,
        incident=str(INCIDENT),
        incident_sha256=sha256_file(INCIDENT),
        classification=incident["classification"],
    )

    dry = read_json(DRY_RUN)
    add(
        checks,
        "minimal_path_dry_run",
        dry["status"] == "dry_run_verified"
        and dry["project"] == "CanvasTest"
        and dry["agent_runtime"]["node"] == retry["runtime"]["node"]
        and dry["agent_runtime"]["node_bin"] == retry["runtime"]["node_bin"]
        and dry["byte_identical_input"]
        and dry["initial_target_findings_sha256"] == dry["frozen_initial_sha256"],
        manifest=str(DRY_RUN),
        manifest_sha256=sha256_file(DRY_RUN),
        agent_runtime=dry["agent_runtime"],
    )

    node = run([retry["runtime"]["node_bin"] + "/node", "--version"], HERE)
    add(
        checks,
        "frozen_node_runtime",
        node.returncode == 0 and node.stdout.strip() == retry["runtime"]["node"],
        command=node.args,
        returncode=node.returncode,
        version=node.stdout.strip(),
    )
    retry_dir = (
        WORKSPACE_ROOT
        / "baseline_data/exp_hapskill/runs/exp_hapskill_35_luna_v4_formal_02"
    )
    add(
        checks,
        "retry_run_not_started",
        not retry_dir.exists(),
        path=str(retry_dir),
    )

    jobs = [
        (
            "harness",
            ["python3", "-m", "unittest", "-v", "test_harness.py"],
            HERE,
            17,
        ),
        (
            "skill",
            ["python3", "-m", "unittest", "-v", "test_haprepair_skill.py"],
            HAPREPAIR_ROOT / "skills/haprepair-openharmony-repair/scripts",
            22,
        ),
        (
            "baseline_gates",
            ["python3", "-m", "unittest", "-v", "test_baseline_runner.py"],
            HAPREPAIR_ROOT / "revision/coding_agent_baseline",
            22,
        ),
    ]
    for name, command, cwd, expected in jobs:
        result = run(command, cwd)
        output = result.stdout + result.stderr
        add(
            checks,
            f"tests:{name}",
            result.returncode == 0 and f"Ran {expected} tests" in output,
            command=command,
            cwd=str(cwd),
            expected_test_count=expected,
            returncode=result.returncode,
            output=output,
        )
    ruff = run(
        [
            "ruff",
            "check",
            "run_condition.py",
            "run_formal.py",
            "test_harness.py",
            "audit_v4_formal_02_retry.py",
        ],
        HERE,
    )
    add(
        checks,
        "ruff",
        ruff.returncode == 0,
        command=ruff.args,
        returncode=ruff.returncode,
        output=ruff.stdout + ruff.stderr,
    )

    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "experiment": "EXP-HAPSKILL-35",
        "gate": "formal_02 environment-only retry authorization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if passed else "open_failed_audit",
        "passed": passed,
        "formal_execution_authorized": passed,
        "authorized_run_id": "exp_hapskill_35_luna_v4_formal_02",
        "checks": checks,
        "claim_boundary": (
            "This authorization repairs evaluator launch infrastructure only; "
            "formal_01 is excluded and contributes no model or effectiveness evidence."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={sha256_file(OUTPUT)}")
    if not passed:
        for check in checks:
            if not check["passed"]:
                print(f"FAILED: {check['name']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
