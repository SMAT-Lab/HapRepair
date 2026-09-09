#!/usr/bin/env python3
"""Audit and close the v4 formal-input/protocol G2 without running a model."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_formal as formal
import freeze_v4_inputs as freeze


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
OUTPUT = (
    WORKSPACE_ROOT
    / "paper"
    / "rebuttal"
    / "gates"
    / "exp_hapskill_v4_g2_formal_freeze_20260804.json"
)
PROTOCOL_35 = HERE / "protocol-hapskill-35-v4.json"
PROTOCOL_10 = HERE / "protocol-agent-ref-10-v4.json"
INPUTS_35 = HERE / "formal_inputs_hapskill_35_v4.json"
INPUTS_10 = HERE / "formal_inputs_agent_ref_10_v4.json"
DRY_35 = (
    WORKSPACE_ROOT
    / "baseline_data/exp_hapskill/smoke/g2_v4_hapskill_condition_dry_04"
    / "hapskill/applications_permission_manager/run_manifest.json"
)
DRY_10 = (
    WORKSPACE_ROOT
    / "baseline_data/exp_hapskill/smoke/g2_v4_vanilla_condition_dry_04"
    / "vanilla/applications_permission_manager/run_manifest.json"
)
RESUME_SMOKE = (
    WORKSPACE_ROOT
    / "baseline_data/exp_hapskill/smoke/g2_scheduler_resume_05"
    / "exp_hapskill_35_luna_v4_formal_01/formal_scheduler/manifest.json"
)
EXPECTED_IMAGE = (
    "sha256:9cc027cd2f9404ccd58c61e367fe712895188a926c07d464ea0936e893b9a89a"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    protocol_35, inputs_35, tasks_35 = formal.validate_contract(
        PROTOCOL_35, INPUTS_35
    )
    protocol_10, inputs_10, tasks_10 = formal.validate_contract(
        PROTOCOL_10, INPUTS_10
    )

    add(
        checks,
        "frozen_population_35",
        inputs_35["status"] == "frozen"
        and inputs_35["project_count"] == len(inputs_35["projects"]) == 35
        and inputs_35["raw_finding_count"] == 9934
        and inputs_35["target_finding_count"] == 9884
        and inputs_35["excluded_non_target_finding_count"] == 50,
        manifest=str(INPUTS_35),
        manifest_sha256=freeze.sha256_file(INPUTS_35),
        raw_findings=inputs_35["raw_finding_count"],
        target_findings=inputs_35["target_finding_count"],
        excluded=inputs_35["excluded_non_target_finding_count"],
    )
    categories: Counter[str] = Counter()
    for project in inputs_35["projects"]:
        categories.update(project["target_category_counts"])
    add(
        checks,
        "target_category_totals",
        categories == {"performance": 9401, "security": 483},
        categories=dict(categories),
    )

    source_failures = []
    finding_failures = []
    git_failures = []
    for project in inputs_35["projects"]:
        current_tree = freeze.sha256_tree(
            freeze.tree_manifest(Path(project["source_path"]).resolve())
        )
        if current_tree != project["frozen_input_tree_sha256"]:
            source_failures.append(project["name"])
        finding_path = Path(project["frozen_target_findings_path"])
        if (
            not finding_path.is_file()
            or freeze.sha256_file(finding_path)
            != project["frozen_target_findings_sha256"]
            or len(read_json(finding_path)) != project["frozen_target_finding_count"]
        ):
            finding_failures.append(project["name"])
        try:
            observed_git = freeze.git_identity(project)
        except (RuntimeError, subprocess.SubprocessError, ValueError) as error:
            git_failures.append({"project": project["name"], "error": str(error)})
        else:
            if observed_git != project["verified_git_identity"]:
                git_failures.append(
                    {"project": project["name"], "error": "identity record drift"}
                )
    add(
        checks,
        "all_35_source_tree_hashes",
        not source_failures,
        failures=source_failures,
    )
    add(
        checks,
        "all_35_finding_artifacts",
        not finding_failures,
        failures=finding_failures,
    )
    add(checks, "all_35_git_identities", not git_failures, failures=git_failures)

    selected = read_json(
        HAPREPAIR_ROOT / "revision/coding_agent_baseline/selected_projects.json"
    )
    selected_names = [item["name"] for item in selected["projects"]]
    reference_names = [item["name"] for item in inputs_10["projects"]]
    by_name = {item["name"]: item for item in inputs_35["projects"]}
    add(
        checks,
        "unchanged_reference_subset",
        inputs_10["project_count"] == 10
        and inputs_10["target_finding_count"] == 4393
        and reference_names == selected_names
        and "wifi_testapp" in reference_names
        and all(item == by_name[item["name"]] for item in inputs_10["projects"]),
        manifest=str(INPUTS_10),
        manifest_sha256=freeze.sha256_file(INPUTS_10),
        names=reference_names,
        target_findings=inputs_10["target_finding_count"],
    )

    common_protocol = all(
        protocol["model"]
        == {
            **protocol["model"],
            "requested_id": "gpt-5.6-luna",
            "provider": "xsp",
            "reasoning_effort": "high",
            "runs_per_project_condition": 1,
        }
        and protocol["common_contract"]["maximum_post_edit_validation_scans"] == 5
        and protocol["scheduling"]["thread_scope"] == "active_round"
        and protocol["execution"]["maximum_concurrent_conditions"] == 8
        and protocol["execution"]["retrieval_devices"]
        == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        and protocol["execution"]["formal_04_forbidden"]
        and all(
            key in protocol["execution"]
            for key in ("retry_policy", "abort_policy", "resume_policy")
        )
        and protocol["runtime"]["container_image_id"] == EXPECTED_IMAGE
        for protocol in (protocol_35, protocol_10)
    )
    add(
        checks,
        "shared_formal_contract",
        common_protocol,
        protocol_35_sha256=freeze.sha256_file(PROTOCOL_35),
        protocol_10_sha256=freeze.sha256_file(PROTOCOL_10),
        model=protocol_35["model"],
        execution=protocol_35["execution"],
    )
    add(
        checks,
        "independent_condition_membership",
        len(tasks_35) == 35
        and {item["condition"] for item in tasks_35} == {"hapskill"}
        and len(tasks_10) == 10
        and {item["condition"] for item in tasks_10} == {"vanilla"},
        hapskill_task_count=len(tasks_35),
        reference_task_count=len(tasks_10),
    )
    add(
        checks,
        "retrieval_and_isolation_contract",
        protocol_35["retrieval"]["require_success_for_covered_rules"] is False
        and protocol_35["retrieval"]["role"] == "optional same-rule repair evidence"
        and protocol_10["retrieval"]["exposed_to_agent"] is False
        and protocol_35["retrieval"]["skill_protocol_sha256"]
        == protocol_10["retrieval"]["skill_protocol_sha256"]
        == freeze.sha256_file(
            HAPREPAIR_ROOT
            / "skills/haprepair-openharmony-repair/references/protocol-v4.json"
        ),
        hapskill_retrieval=protocol_35["retrieval"],
        reference_isolation=protocol_10["retrieval"]["isolation"],
    )

    dry_35 = read_json(DRY_35)
    dry_10 = read_json(DRY_10)
    add(
        checks,
        "condition_dry_runs",
        dry_35["status"] == dry_10["status"] == "dry_run_verified"
        and dry_35["byte_identical_input"]
        and dry_10["byte_identical_input"]
        and dry_35["initial_target_findings_sha256"]
        == dry_35["frozen_initial_sha256"]
        and dry_10["initial_target_findings_sha256"]
        == dry_10["frozen_initial_sha256"]
        and dry_35["input_tree_sha256"] == dry_10["input_tree_sha256"]
        and dry_35["isolation"]["broker_exposed"] is True
        and dry_10["isolation"]["broker_exposed"] is False
        and not dry_10["isolation"]["haprepair_repository_mounted"]
        and not dry_10["isolation"]["corpus_mounted"],
        hapskill_manifest=str(DRY_35),
        hapskill_manifest_sha256=freeze.sha256_file(DRY_35),
        vanilla_manifest=str(DRY_10),
        vanilla_manifest_sha256=freeze.sha256_file(DRY_10),
    )
    resume = read_json(RESUME_SMOKE)
    add(
        checks,
        "scheduler_abort_resume_dry_run",
        resume["status"] == "paused_after_max_tasks"
        and resume["task_count"] == 35
        and sum(item["status"] == "pending" for item in resume["tasks"]) == 35
        and resume["maximum_concurrent_conditions"] == 8
        and resume["resume_count"] == 1
        and bool(resume.get("last_resumed_at")),
        manifest=str(RESUME_SMOKE),
        manifest_sha256=freeze.sha256_file(RESUME_SMOKE),
        status=resume["status"],
        resume_count=resume["resume_count"],
    )

    pilot = read_json(
        WORKSPACE_ROOT
        / "paper/rebuttal/gates/exp_hapskill_v4_development_pilot_wifi_02_audit_20260804.json"
    )
    thread = read_json(
        WORKSPACE_ROOT
        / "paper/rebuttal/gates/exp_hapskill_v4_thread_scope_smoke_wifi_01_audit_20260804.json"
    )
    decision = read_json(
        WORKSPACE_ROOT
        / "paper/rebuttal/gates/exp_agent_ref_10_keep_wifi_decision_20260804.json"
    )
    add(
        checks,
        "development_gates_and_wifi_decision",
        pilot["decision"] == "close_g1_v4_core_and_keep_g2_open"
        and pilot["result"]["all_round_coverage_audits_passed"]
        and thread["decision"] == "accept_active_round_thread_scope_for_g2_protocol_design"
        and thread["thread_audit"]["distinct_threads"]
        and decision["decision"] == "retain_wifi_testapp_in_original_reference_subset"
        and decision["reference_subset_policy"]["wifi_testapp_included"],
        pilot_result=pilot["result"],
        thread_scope=thread["protocol"]["thread_scope"],
        reference_subset_policy=decision["reference_subset_policy"],
    )

    image = run(
        ["docker", "image", "inspect", "hybrid-gym-codex:0.146.0", "--format", "{{.Id}}"],
        WORKSPACE_ROOT,
    )
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ],
        WORKSPACE_ROOT,
    )
    gpu_lines = [line for line in gpu.stdout.splitlines() if line.strip()]
    add(
        checks,
        "runtime_and_gpu_capacity",
        image.returncode == 0
        and image.stdout.strip() == EXPECTED_IMAGE
        and gpu.returncode == 0
        and len(gpu_lines) == 4
        and all("A100" in line for line in gpu_lines),
        image_id=image.stdout.strip(),
        gpus=gpu_lines,
    )

    formal_dirs = [
        WORKSPACE_ROOT
        / "baseline_data/exp_hapskill/runs/exp_hapskill_35_luna_v4_formal_01",
        WORKSPACE_ROOT
        / "baseline_data/exp_hapskill/runs/exp_agent_ref_10_luna_v4_formal_01",
        WORKSPACE_ROOT
        / "baseline_data/exp_hapskill/runs/exp_hapskill_10_luna_formal_04",
    ]
    add(
        checks,
        "formal_runs_not_started",
        not any(path.exists() for path in formal_dirs),
        paths={str(path): path.exists() for path in formal_dirs},
    )

    test_jobs = [
        (
            "harness",
            ["python3", "-m", "unittest", "-v", "test_harness.py"],
            HERE,
            16,
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
    for name, command, cwd, expected in test_jobs:
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
            "freeze_v4_inputs.py",
            "run_formal.py",
            "run_condition.py",
            "test_harness.py",
            "audit_v4_g2.py",
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
        "experiment": ["EXP-HAPSKILL-35", "EXP-AGENT-REF-10"],
        "gate": "G2 formal input and protocol freeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if passed else "open_failed_audit",
        "passed": passed,
        "formal_execution_authorized": passed,
        "authorization_scope": (
            "Only the two frozen v4 run IDs and protocols; formal_04 remains forbidden."
        ),
        "checks": checks,
        "launch_order": [
            "EXP-HAPSKILL-35 first with controlled concurrency",
            "EXP-AGENT-REF-10 after the 35-project Skill run completes",
        ],
        "claim_boundary": (
            "This gate authorizes execution, not any effectiveness, semantic-correctness, "
            "RAG-causality, or superiority claim."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={freeze.sha256_file(OUTPUT)}")
    if not passed:
        for check in checks:
            if not check["passed"]:
                print(f"FAILED: {check['name']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
