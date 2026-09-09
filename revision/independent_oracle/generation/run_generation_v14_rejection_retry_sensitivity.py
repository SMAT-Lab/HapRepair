#!/usr/bin/env python3
"""Run the three-case final-v14 protocol-rejection retry sensitivity."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_generation_v14_static_skill as base


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
DEFAULT_CONTRACT = SCRIPT_DIR / "protocol_v14_rejection_retry_sensitivity.json"


def read_contract(path: Path) -> dict[str, Any]:
    contract = base.read_json(path)
    if contract.get("status") != "frozen":
        raise SystemExit("Retry-sensitivity contract is not frozen")
    parent_path = base.resolve_repo_path(contract["parent_protocol"])
    base.assert_hash(parent_path, contract["parent_protocol_sha256"], "parent protocol")
    parent = base.read_json(parent_path)
    selected = contract.get("selected_cases", [])
    if len(selected) != 3:
        raise SystemExit("Retry sensitivity must contain exactly three cases")
    case_ids = [row["case_id"] for row in selected]
    source_ids = [row["source_blind_id"] for row in selected]
    if len(case_ids) != len(set(case_ids)) or len(source_ids) != len(set(source_ids)):
        raise SystemExit("Retry-sensitivity IDs must be unique")
    if contract.get("model_calls") != 3 or contract.get("max_concurrency") != 3:
        raise SystemExit("Retry-sensitivity execution budget must remain 3/3")
    if contract["reporting_policy"].get("replace_parent_main_result") is not False:
        raise SystemExit("Retry sensitivity cannot replace the parent main result")
    contract["_parent"] = parent
    contract["_parent_path"] = parent_path
    return contract


def effective_protocol(contract: dict[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(contract["_parent"])
    protocol["experiment"] = contract["experiment"]
    protocol["run_id"] = contract["run_id"]
    protocol["tier"] = contract["tier"]
    protocol["research_question"] = (
        "Under an explicitly separate retry sensitivity, do fresh generations for "
        "the three parent protocol rejections yield protocol-compliant candidates?"
    )
    protocol["research_objective"] = (
        "Measure the sensitivity of final-v14 correctness to retrying only the three "
        "restricted-command parent rejections without reclassifying them as API failures."
    )
    protocol["sensitivity_parent"] = {
        "run_id": contract["parent_run_id"],
        "classification": contract["classification"],
        "replace_parent_main_result": False,
    }
    return protocol


def validate_parent_results(contract: dict[str, Any]) -> None:
    parent_dir = base.RUNS_DIR / contract["parent_run_id"]
    for row in contract["selected_cases"]:
        result_path = parent_dir / "cases" / row["source_blind_id"] / "result.json"
        base.assert_hash(
            result_path,
            row["parent_result_sha256"],
            f"parent result {row['source_blind_id']}",
        )
        result = base.read_json(result_path)
        if result.get("case_id") != row["case_id"]:
            raise SystemExit(f"Parent case identity mismatch: {row['source_blind_id']}")
        if result.get("rejection_reasons") != [row["parent_rejection_reason"]]:
            raise SystemExit(f"Parent rejection reason drift: {row['source_blind_id']}")
        if result.get("systemic_failure") is not False:
            raise SystemExit(
                f"Parent result was systemic, not a protocol rejection: {row['source_blind_id']}"
            )
        if not result.get("usage") or not result.get("source_diff_present"):
            raise SystemExit(
                f"Parent result lacks completed-generation evidence: {row['source_blind_id']}"
            )


def contract_context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    contract = read_contract(path)
    validate_parent_results(contract)
    protocol = effective_protocol(contract)
    paths = base.validate_frozen_inputs(protocol)
    return contract, protocol, paths


def validate(args: argparse.Namespace) -> None:
    contract_path = args.contract.resolve()
    contract, protocol, paths = contract_context(contract_path)
    manifest = base.read_json(paths["case_manifest"])
    available = {case["case_id"] for case in manifest["cases"]}
    selected = {row["case_id"] for row in contract["selected_cases"]}
    checks = {
        "contract_frozen": contract["status"] == "frozen",
        "parent_protocol_hash_matches": base.sha256_file(contract["_parent_path"])
        == contract["parent_protocol_sha256"],
        "exactly_three_parent_rejections": len(selected) == 3,
        "selected_cases_exist": selected <= available,
        "parent_results_are_completed_protocol_rejections": True,
        "same_model": protocol["model"] == contract["_parent"]["model"],
        "same_method": protocol["method"] == contract["_parent"]["method"],
        "same_benchmark": protocol["benchmark"] == contract["_parent"]["benchmark"],
        "same_leakage_gate": protocol["leakage_gate"]
        == contract["_parent"]["leakage_gate"],
        "not_classified_as_api_retry": "not an API" in contract["classification"],
        "parent_main_result_not_replaced": contract["reporting_policy"][
            "replace_parent_main_result"
        ]
        is False,
    }
    result = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "run_id": contract["run_id"],
        "contract_sha256": base.sha256_file(contract_path),
        "checks": checks,
        "all_passed": all(checks.values()),
        "model_calls": 0,
    }
    if args.output:
        base.write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(2)


def prepare(args: argparse.Namespace) -> None:
    contract_path = args.contract.resolve()
    contract, protocol, paths = contract_context(contract_path)
    if args.run_id != contract["run_id"]:
        raise SystemExit(f"Retry run ID must be {contract['run_id']}")
    run_dir = base.RUNS_DIR / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    manifest = base.read_json(paths["case_manifest"])
    cases_by_id = {case["case_id"]: case for case in manifest["cases"]}
    indexed = base.finding_index(paths["defective_report"], paths["defective_project"])
    prompt = paths["prompt_template"].read_text(encoding="utf-8")
    prepared = []
    retry_map = []
    for index, selection in enumerate(contract["selected_cases"], start=1):
        case = cases_by_id[selection["case_id"]]
        blind_id = f"RTRY-{index:03d}"
        case_dir = run_dir / "cases" / blind_id
        baseline = case_dir / "baseline"
        workspace = case_dir / "workspace"
        codex_home = case_dir / "codex_home"
        run_state = case_dir / "run_state"
        baseline.mkdir(parents=True)
        workspace.mkdir(parents=True)
        (run_state / "home").mkdir(parents=True)
        baseline_files = base.copy_case_sources(
            paths["defective_project"], case["defective_files"], baseline
        )
        workspace_files = base.copy_case_sources(
            paths["defective_project"], case["defective_files"], workspace
        )
        if baseline_files != workspace_files:
            raise SystemExit(f"Baseline/workspace copy mismatch for {case['case_id']}")
        task = base.render_task(
            case=case,
            blind_id=blind_id,
            findings=base.case_findings(case, indexed),
        )
        task["experiment"] = contract["experiment"]
        task_path = workspace / "HAPREPAIR_TASK.json"
        base.write_json(task_path, task)
        prompt_path = case_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        base.prepare_codex_home(codex_home)
        installed = base.install_skill(paths["skill"], codex_home)
        if installed["tree_sha256"] != protocol["method"]["skill_tree_sha256"]:
            raise SystemExit(f"Installed Skill drift for {case['case_id']}")
        prepared.append(
            {
                "case_id": case["case_id"],
                "blind_id": blind_id,
                "rule": case["rule"],
                "case_dir": case_dir.relative_to(run_dir).as_posix(),
                "input_files": workspace_files,
                "task_sha256": base.sha256_file(task_path),
                "prompt_sha256": base.sha256_file(prompt_path),
                "skill_tree_sha256": installed["tree_sha256"],
                "reference_project_mounted": False,
                "knowledge_base_mounted": False,
                "historical_outputs_mounted": False,
            }
        )
        retry_map.append(
            {
                "retry_blind_id": blind_id,
                "source_blind_id": selection["source_blind_id"],
                "case_id": case["case_id"],
                "rule": case["rule"],
                "parent_result_sha256": selection["parent_result_sha256"],
            }
        )

    base.write_json(run_dir / "protocol_snapshot.json", protocol)
    base.write_json(
        run_dir / "retry_contract_snapshot.json",
        {key: value for key, value in contract.items() if not key.startswith("_")},
    )
    base.write_json(run_dir / "input_manifest.json", prepared)
    base.write_json(run_dir / "source_to_retry_blind_map.json", retry_map)
    base.write_json(
        run_dir / "environment.prepare.json",
        {
            "prepared_at": base.utc_now(),
            "contract_path": str(contract_path),
            "contract_sha256": base.sha256_file(contract_path),
            "parent_protocol_sha256": contract["parent_protocol_sha256"],
            "base_runner_sha256": base.sha256_file(Path(base.__file__)),
            "retry_runner_sha256": base.sha256_file(Path(__file__)),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "git_status_short": base.git_output("status", "--short"),
            "case_count": len(prepared),
            "max_concurrency": contract["max_concurrency"],
        },
    )
    print(json.dumps({"status": "prepared", "case_count": len(prepared)}, indent=2))


def execute(args: argparse.Namespace) -> None:
    contract, protocol, _ = contract_context(args.contract.resolve())
    if args.run_id != contract["run_id"]:
        raise SystemExit(f"Retry run ID must be {contract['run_id']}")
    run_dir = base.RUNS_DIR / args.run_id
    prepared = base.read_json(run_dir / "input_manifest.json")
    workers = min(contract["max_concurrency"], len(prepared))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                base.execute_case, protocol=protocol, run_dir=run_dir, item=item
            ): item
            for item in prepared
        }
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"retry progress {len(results)}/{len(prepared)}: "
                f"{item['blind_id']} {result['status']}",
                flush=True,
            )
    metrics = base.summarize_run(run_dir, protocol)
    metrics["classification"] = contract["classification"]
    metrics["parent_run_id"] = contract["parent_run_id"]
    metrics["replace_parent_main_result"] = False
    base.write_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def verify(args: argparse.Namespace) -> None:
    contract_path = args.contract.resolve()
    contract, protocol, _ = contract_context(contract_path)
    if args.run_id != contract["run_id"]:
        raise SystemExit(f"Retry run ID must be {contract['run_id']}")
    run_dir = base.RUNS_DIR / args.run_id
    prepared = base.read_json(run_dir / "input_manifest.json")
    source_map = base.read_json(run_dir / "source_to_retry_blind_map.json")
    results = []
    for item in prepared:
        result_path = run_dir / item["case_dir"] / "result.json"
        if result_path.is_file():
            results.append(base.read_json(result_path))
    checks = {
        "contract_hash_matches_snapshot": base.sha256_file(contract_path)
        == base.sha256_file(run_dir / "retry_contract_snapshot.json"),
        "exactly_three_inputs": len(prepared) == 3,
        "exactly_three_source_mappings": len(source_map) == 3,
        "case_ids_match_contract": {row["case_id"] for row in prepared}
        == {row["case_id"] for row in contract["selected_cases"]},
        "input_hashes_match": all(
            base.sha256_file(run_dir / item["case_dir"] / "baseline" / record["path"])
            == record["sha256"]
            for item in prepared
            for record in item["input_files"]
        ),
        "task_hashes_match": all(
            base.sha256_file(
                run_dir / item["case_dir"] / "workspace" / "HAPREPAIR_TASK.json"
            )
            == item["task_sha256"]
            for item in prepared
        ),
        "results_complete": len(results) == 3,
        "all_candidates_accepted": len(results) == 3
        and all(result["status"] == "accepted" for result in results),
        "no_systemic_failures": len(results) == 3
        and all(result["systemic_failure"] is False for result in results),
        "no_restricted_commands": len(results) == 3
        and all(not result["restricted_commands"] for result in results),
        "all_source_diffs_present": len(results) == 3
        and all(result["source_diff_present"] for result in results),
        "parent_main_result_not_replaced": contract["reporting_policy"][
            "replace_parent_main_result"
        ]
        is False,
    }
    verification = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "run_id": protocol["run_id"],
        "classification": contract["classification"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "accepted_candidate_count": sum(
            result["status"] == "accepted" for result in results
        ),
        "next_action": (
            "Build a separate three-case blind annotation package."
            if all(checks.values())
            else "Preserve the retry outcomes and decide how to report the sensitivity."
        ),
    }
    base.write_json(run_dir / "verification.json", verification)
    print(json.dumps(verification, indent=2))
    if not verification["all_passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path)
    validate_parser.set_defaults(func=validate)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.set_defaults(func=prepare)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--run-id", required=True)
    execute_parser.set_defaults(func=execute)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.set_defaults(func=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
