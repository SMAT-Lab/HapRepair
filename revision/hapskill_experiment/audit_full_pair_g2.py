#!/usr/bin/env python3
"""Close the static G2 gate for the clean full-population paired campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HERE.parents[2]
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
for directory in (HERE, BASELINE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import prepare_full_pair_campaign as prepare  # noqa: E402
import run_agent_ref_v4 as baseline_runner  # noqa: E402
import run_condition_v14 as skill_runner  # noqa: E402
import run_full_pair_repetitions as scheduler  # noqa: E402


DEFAULT_OUTPUT = prepare.G2
RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_full_pair_35/runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def verify_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for project in inputs["projects"]:
        source = Path(project["source_path"])
        findings = Path(project["frozen_target_findings_path"])
        observed_findings = (
            prepare.sha256_file(findings) if findings.is_file() else None
        )
        observed_tree = None
        if source.is_dir():
            observed_tree = baseline_runner.sha256_tree(
                baseline_runner.tree_manifest(source)
            )
        if (
            observed_findings != project["frozen_target_findings_sha256"]
            or observed_tree != project["frozen_input_tree_sha256"]
        ):
            failures.append(
                {
                    "project": project["name"],
                    "expected_findings": project["frozen_target_findings_sha256"],
                    "observed_findings": observed_findings,
                    "expected_tree": project["frozen_input_tree_sha256"],
                    "observed_tree": observed_tree,
                }
            )
    return {"passed": not failures, "failures": failures}


def verify_tasks(campaign: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    projects = {item["name"] for item in inputs["projects"]}
    counts = Counter((task["project"], task["condition"]) for task in campaign["tasks"])
    failures = [
        {
            "project": project,
            "condition": condition,
            "count": counts[(project, condition)],
        }
        for project in sorted(projects)
        for condition in ("hapskill", "vanilla")
        if counts[(project, condition)] != 2
    ]
    commands = [
        scheduler.runner_command(task=task, inputs=prepare.INPUTS, run_root=RUN_ROOT)
        for task in campaign["tasks"]
    ]
    return {
        "passed": not failures and len(commands) == 140,
        "failures": failures,
        "command_count": len(commands),
        "unique_task_count": len({task["task_id"] for task in campaign["tasks"]}),
    }


def verify_no_prior_outputs(campaign: dict[str, Any]) -> dict[str, Any]:
    existing = [
        str(scheduler.condition_dir(RUN_ROOT, task))
        for task in campaign["tasks"]
        if scheduler.condition_dir(RUN_ROOT, task).exists()
    ]
    scheduler_dir = RUN_ROOT / campaign["campaign_id"]
    if scheduler_dir.exists():
        existing.append(str(scheduler_dir))
    return {"passed": not existing, "existing": sorted(set(existing))}


def run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "test_agent_ref_v4.py",
        "test_full_pair_campaign.py",
        "test_analyze_agent_ref_v14_pairing.py",
    ]
    result = subprocess.run(
        command,
        cwd=HERE,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=prepare.CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    campaign_path = args.campaign.resolve()
    output = args.output.resolve()
    campaign, inputs_path = scheduler.validate_campaign(campaign_path, require_g2=False)
    inputs = prepare.read_json(inputs_path)
    skill_protocol = prepare.read_json(prepare.SKILL_PROTOCOL)
    baseline_protocol = prepare.read_json(prepare.BASELINE_PROTOCOL)
    strata = prepare.read_json(prepare.STRATA)

    contract_equal = (
        skill_protocol["common_contract"] == baseline_protocol["common_contract"]
        and skill_protocol["scheduling"] == baseline_protocol["scheduling"]
        and skill_protocol["model"] == baseline_protocol["model"]
    )
    contract_values = {
        "maximum_validation_scans": skill_protocol["common_contract"].get(
            "maximum_validation_scans"
        ),
        "maximum_agent_turns_per_round": skill_protocol["scheduling"].get(
            "maximum_agent_turns_per_round"
        ),
        "maximum_concurrent_conditions": campaign.get("maximum_concurrent_conditions"),
    }
    input_audit = verify_inputs(inputs)
    task_audit = verify_tasks(campaign, inputs)
    output_audit = verify_no_prior_outputs(campaign)
    tests = run_tests()

    runtime_error = None
    runtime: dict[str, Any] = {}
    try:
        skill_runner.validate_method_artifacts(skill_protocol)
        runtime["skill"] = skill_runner.validate_runtime(
            skill_protocol, skill_protocol["runtime"]["container_image"]
        )
        runtime["baseline"] = baseline_runner.validate_runtime(
            baseline_protocol, baseline_protocol["runtime"]["container_image"]
        )
    except Exception as error:
        runtime_error = f"{type(error).__name__}: {error}"

    checks = [
        check(
            "frozen_contract_and_exact_order",
            baseline_protocol["dataset"]["formal_inputs_sha256"]
            == prepare.sha256_file(inputs_path)
            and len(baseline_protocol["formal_run"]["condition_order"]) == 35,
            protocol_sha256=prepare.sha256_file(prepare.BASELINE_PROTOCOL),
            inputs_sha256=prepare.sha256_file(inputs_path),
            project_order=[item["name"] for item in inputs["projects"]],
        ),
        check(
            "equal_model_feedback_and_turn_contract",
            contract_equal
            and contract_values
            == {
                "maximum_validation_scans": 5,
                "maximum_agent_turns_per_round": 6,
                "maximum_concurrent_conditions": 16,
            },
            observed=contract_values,
        ),
        check(
            "complete_two_by_two_task_expansion",
            task_audit["passed"],
            failures=task_audit["failures"],
            command_count=task_audit["command_count"],
            unique_task_count=task_audit["unique_task_count"],
        ),
        check(
            "frozen_initial_inputs",
            input_audit["passed"],
            failures=input_audit["failures"],
        ),
        check(
            "pre_registered_burden_strata",
            strata.get("group_sizes") == {"low": 12, "middle": 11, "high": 12}
            and strata.get("input_manifest_sha256") == prepare.sha256_file(inputs_path),
            group_sizes=strata.get("group_sizes"),
            strata_sha256=prepare.sha256_file(prepare.STRATA),
        ),
        check(
            "clean_output_roots",
            output_audit["passed"],
            existing=output_audit["existing"],
        ),
        check(
            "unit_tests",
            tests["passed"],
            command=tests["command"],
            exit_code=tests["exit_code"],
            stdout=tests["stdout"],
            stderr=tests["stderr"],
        ),
        check(
            "runtime_and_method_identity",
            runtime_error is None,
            runtime=runtime,
            error=runtime_error,
        ),
        check(
            "no_refresh_resume_implementation",
            "remaining_attempt_range" in prepare.sha256_file.__module__
            or (
                "return range(next_attempt, maximum_attempts + 1)"
                in (HERE / "resume_condition_v14.py").read_text(encoding="utf-8")
                and "attempt_limit = next_attempt + maximum_attempts - 1"
                not in (HERE / "resume_condition_v14.py").read_text(encoding="utf-8")
            ),
            resume_script=str(HERE / "resume_condition_v14.py"),
        ),
    ]
    passed = all(item["passed"] for item in checks)
    artifact_paths = {
        name: str(path.resolve())
        for name, path in {
            "agent_ref_session.py": HERE / "agent_ref_session.py",
            "run_agent_ref_v4.py": HERE / "run_agent_ref_v4.py",
            "run_condition_v14.py": HERE / "run_condition_v14.py",
            "resume_condition_v14.py": HERE / "resume_condition_v14.py",
            "run_full_pair_repetitions.py": HERE / "run_full_pair_repetitions.py",
            "prepare_full_pair_campaign.py": HERE / "prepare_full_pair_campaign.py",
            "audit_full_pair_g2.py": Path(__file__).resolve(),
            "test_full_pair_campaign.py": HERE / "test_full_pair_campaign.py",
        }.items()
    }
    runner_artifacts = {
        name: prepare.sha256_file(Path(path)) for name, path in artifact_paths.items()
    }
    result = {
        "schema_version": 1,
        "experiment": campaign["experiment"],
        "gate": "clean full-population paired campaign static G2",
        "created_at": utc_now(),
        "status": "closed" if passed else "failed",
        "passed": passed,
        "formal_execution_authorized": passed,
        "campaign": str(campaign_path),
        "campaign_sha256": prepare.sha256_file(campaign_path),
        "authorized_run_ids": prepare.RUN_IDS["vanilla"],
        "all_campaign_run_ids": prepare.RUN_IDS,
        "runner_artifact_paths": artifact_paths,
        "runner_artifacts": runner_artifacts,
        "checks": checks,
    }
    write_json(output, result)
    print(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
