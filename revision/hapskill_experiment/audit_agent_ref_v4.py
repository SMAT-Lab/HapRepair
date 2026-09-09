#!/usr/bin/env python3
"""Run the static G2 audit for EXP-AGENT-REF-10 without launching a model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_agent_baseline import sha256_file, tree_manifest  # type: ignore  # noqa: E402
from run_agent_ref_v4 import sha256_tree, validate_runtime  # noqa: E402
import run_formal_agent_ref_v4 as formal  # noqa: E402


PROTOCOL = HERE / "protocol-agent-ref-10-v4.json"
INPUTS = HERE / "formal_inputs_agent_ref_10_v4.json"
RUNNER = HERE / "run_agent_ref_v4.py"
SESSION = HERE / "agent_ref_session.py"
SCHEDULER = HERE / "run_formal_agent_ref_v4.py"
TEST = HERE / "test_agent_ref_v4.py"
E1_PROJECTS = WORKSPACE_ROOT / "paper/rebuttal/e1_v14/projects.json"
OUTPUT = (
    WORKSPACE_ROOT / "paper/rebuttal/gates/exp_agent_ref_10_v4_static_g2_20260806.json"
)
FORMAL_ROOT = WORKSPACE_ROOT / "baseline_data/exp_agent_ref_10/runs"
SDK_MANIFESTS = {
    "openharmony_sdk_manifest_sha256": WORKSPACE_ROOT
    / "baseline_data/openharmony_sdk/install_manifest.json",
    "sdk_build_components_sha256": WORKSPACE_ROOT
    / "baseline_data/openharmony_sdk/install_manifest_build_components.json",
    "sdk_native_api20_sha256": WORKSPACE_ROOT
    / "baseline_data/openharmony_sdk/install_manifest_native_api20.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )


def add(checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **evidence})


def main() -> None:
    checks: list[dict[str, Any]] = []
    protocol, tasks = formal.load_contract(PROTOCOL, INPUTS)
    inputs = read_json(INPUTS)
    expected_names = [
        item.split(":", 1)[0] for item in protocol["formal_run"]["condition_order"]
    ]
    add(
        checks,
        "frozen_contract_and_exact_order",
        protocol["status"] == inputs["status"] == "frozen"
        and len(tasks) == 10
        and [item["project"] for item in tasks] == expected_names
        and protocol["dataset"]["formal_inputs_sha256"] == sha256_file(INPUTS),
        protocol_sha256=sha256_file(PROTOCOL),
        inputs_sha256=sha256_file(INPUTS),
        order=expected_names,
    )
    add(
        checks,
        "model_budget_and_thread_scope",
        protocol["model"]["requested_id"] == "gpt-5.6-luna"
        and protocol["model"]["provider"] == "xsp"
        and protocol["model"]["reasoning_effort"] == "high"
        and protocol["common_contract"]["maximum_post_edit_validation_scans"] == 5
        and protocol["scheduling"]["thread_scope"] == "active_round"
        and protocol["execution"]["maximum_concurrent_conditions"] == 8,
        model=protocol["model"],
        common_contract=protocol["common_contract"],
        scheduling=protocol["scheduling"],
    )

    runtime_hashes = {
        key: sha256_file(path) if path.is_file() else None
        for key, path in SDK_MANIFESTS.items()
    }
    config = (
        HAPREPAIR_ROOT
        / "skills/haprepair-openharmony-repair-v14-dev/assets/homecheck-config.json5"
    )
    overlay = (
        HAPREPAIR_ROOT
        / "skills/haprepair-openharmony-repair-v14-dev/assets/homecheck-overlay.json"
    )
    runtime_hashes["codelinter_config_sha256"] = sha256_file(config)
    runtime_hashes["codelinter_overlay_sha256"] = sha256_file(overlay)
    add(
        checks,
        "frozen_scanner_and_sdk_artifacts",
        all(runtime_hashes[key] == protocol["runtime"][key] for key in runtime_hashes),
        observed=runtime_hashes,
        expected={key: protocol["runtime"][key] for key in runtime_hashes},
    )
    try:
        runtime_identity = validate_runtime(
            protocol, protocol["runtime"]["container_image"]
        )
    except (OSError, RuntimeError, KeyError) as error:
        runtime_identity = {"error": f"{type(error).__name__}: {error}"}
    add(
        checks,
        "runtime_identity",
        runtime_identity.get("image_id") == protocol["runtime"]["container_image_id"]
        and protocol["runtime"]["codex_cli"]
        in runtime_identity.get("versions", {}).get("container_codex", "")
        and protocol["runtime"]["python"]
        in runtime_identity.get("versions", {}).get("host_python", ""),
        observed=runtime_identity,
    )

    source_failures = []
    finding_failures = []
    for project in inputs["projects"]:
        observed_tree = sha256_tree(tree_manifest(Path(project["source_path"])))
        if observed_tree != project["frozen_input_tree_sha256"]:
            source_failures.append(project["name"])
        finding_path = Path(project["frozen_target_findings_path"])
        if (
            not finding_path.is_file()
            or sha256_file(finding_path) != project["frozen_target_findings_sha256"]
            or len(read_json(finding_path)) != project["frozen_target_finding_count"]
        ):
            finding_failures.append(project["name"])
    add(
        checks,
        "all_ten_source_and_finding_hashes",
        not source_failures and not finding_failures,
        source_failures=source_failures,
        finding_failures=finding_failures,
    )
    e1 = {item["project"]: item for item in read_json(E1_PROJECTS)}
    join_failures = [
        item["name"]
        for item in inputs["projects"]
        if item["name"] not in e1
        or e1[item["name"]]["initial_alerts"] != item["frozen_target_finding_count"]
        or e1[item["name"]]["commit"] != item["commit"]
        or e1[item["name"]]["tree_oid"] != item["tree_oid"]
    ]
    add(
        checks,
        "e1_matching_project_join",
        not join_failures,
        e1_artifact=str(E1_PROJECTS),
        failures=join_failures,
    )

    runner_text = RUNNER.read_text(encoding="utf-8")
    session_text = SESSION.read_text(encoding="utf-8")
    scheduler_text = SCHEDULER.read_text(encoding="utf-8")
    add(
        checks,
        "agent_isolation_and_mount_allowlist",
        "SKILL_SCRIPTS" in runner_text
        and 'f"{SKILL_SCRIPTS}' not in runner_text
        and 'f"{HAPREPAIR_ROOT}' not in runner_text
        and 'skill_state_mounted": False' in runner_text
        and 'other_runs_mounted": False' in runner_text
        and "RULES_ROOT" in runner_text
        and "SDK_ROOT" in runner_text,
        runner=str(RUNNER),
    )
    add(
        checks,
        "coverage_and_sanitized_plan_gate",
        "assert_plan_is_sanitized" in session_text
        and "workspace_container_path" in runner_text
        and "completion_schema" in runner_text
        and "expected != observed" in session_text
        and 'coverage["complete"]' in runner_text
        and "consulted_specs" not in session_text,
        session=str(SESSION),
    )
    add(
        checks,
        "active_round_threads_and_bounded_retries",
        "thread_id: str | None = None" in runner_text
        and 'thread_id = turn["thread_id"]' in runner_text
        and "maximum_no_diff_retries_per_round" in runner_text,
        runner=str(RUNNER),
    )
    add(
        checks,
        "candidate_retention_and_best_valid_restoration",
        "Repair the retained candidate in place" in runner_text
        and "guards.restore_snapshot(workspace, best_valid)" in runner_text
        and "candidate_score" in runner_text
        and "score < best_score" in runner_text,
        runner=str(RUNNER),
    )
    add(
        checks,
        "scanner_budget_and_evaluator_final_scan",
        'kind="initial"' in runner_text
        and 'kind="validation"' in runner_text
        and 'kind="final"' in runner_text
        and runner_text.index("guards.restore_snapshot(workspace, best_valid)")
        < runner_text.index('kind="final"'),
        runner=str(RUNNER),
    )
    add(
        checks,
        "scheduler_abort_and_resume_semantics",
        "if not stopped" in scheduler_text
        and "incomplete_requires_audit" in scheduler_text
        and "wait(active" in scheduler_text
        and "maximum_concurrent_conditions" in scheduler_text,
        scheduler=str(SCHEDULER),
    )
    forbidden_dir = FORMAL_ROOT / "exp_agent_ref_10_luna_v4_formal_04"
    authorized_dir = FORMAL_ROOT / protocol["formal_run"]["run_id"]
    add(
        checks,
        "formal_04_forbidden_and_no_run_started",
        protocol["execution"]["formal_04_forbidden"] is True
        and "formal_04" in scheduler_text
        and "validate_authorization(" in runner_text
        and "validate_authorization(" in scheduler_text
        and not forbidden_dir.exists()
        and not authorized_dir.exists(),
        forbidden_path=str(forbidden_dir),
        authorized_path=str(authorized_dir),
    )

    dry_failures = []
    for task in tasks:
        result = run(
            [
                sys.executable,
                str(RUNNER),
                "--run-id",
                protocol["formal_run"]["run_id"],
                "--project",
                task["project"],
                "--protocol",
                str(PROTOCOL),
                "--project-manifest",
                str(INPUTS),
                "--run-root",
                str(FORMAL_ROOT),
                "--dry-run",
            ]
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {}
        if (
            result.returncode
            or payload.get("status") != "dry_run_verified"
            or not payload.get("byte_identical_input")
            or payload.get("isolation", {}).get("haprepair_repository_mounted")
            is not False
        ):
            dry_failures.append(
                {
                    "project": task["project"],
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:],
                }
            )
    add(checks, "all_ten_condition_dry_runs", not dry_failures, failures=dry_failures)
    scheduler_dry = run(
        [
            sys.executable,
            str(SCHEDULER),
            "--protocol",
            str(PROTOCOL),
            "--project-manifest",
            str(INPUTS),
            "--run-root",
            str(FORMAL_ROOT),
            "--dry-run",
        ]
    )
    scheduler_payload = (
        json.loads(scheduler_dry.stdout) if scheduler_dry.returncode == 0 else {}
    )
    add(
        checks,
        "scheduler_dry_run",
        scheduler_dry.returncode == 0
        and scheduler_payload.get("project_count") == 10
        and scheduler_payload.get("maximum_concurrent_conditions") == 8
        and [item["project"] for item in scheduler_payload.get("tasks", [])]
        == expected_names,
        output=scheduler_payload if scheduler_payload else scheduler_dry.stderr,
    )
    tests = run([sys.executable, "-m", "unittest", "-v", TEST.name])
    add(
        checks,
        "focused_tests",
        tests.returncode == 0 and "Ran 8 tests" in tests.stderr + tests.stdout,
        command=tests.args,
        output=tests.stdout + tests.stderr,
    )
    compile_result = run(
        [
            sys.executable,
            "-m",
            "py_compile",
            SESSION.name,
            RUNNER.name,
            SCHEDULER.name,
            TEST.name,
            Path(__file__).name,
        ]
    )
    add(
        checks,
        "python_compile",
        compile_result.returncode == 0,
        output=compile_result.stdout + compile_result.stderr,
    )
    ruff = run(
        [
            "ruff",
            "check",
            SESSION.name,
            RUNNER.name,
            SCHEDULER.name,
            TEST.name,
            Path(__file__).name,
        ]
    )
    add(checks, "ruff", ruff.returncode == 0, output=ruff.stdout + ruff.stderr)

    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "gate": "dedicated reference runner static G2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if passed else "open_failed_audit",
        "passed": passed,
        "formal_execution_authorized": passed,
        "authorized_run_id": protocol["formal_run"]["run_id"],
        "checks": checks,
        "runner_artifacts": {
            path.name: sha256_file(path)
            for path in (SESSION, RUNNER, SCHEDULER, TEST, Path(__file__))
        },
        "claim_boundary": "Static authorization confirms runner identity, isolation, lifecycle, and frozen inputs; it contains no reference-agent effectiveness result.",
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
