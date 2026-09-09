#!/usr/bin/env python3
"""Prepare the clean 35-project, two-condition, two-repeat campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
INPUTS = HERE / "formal_inputs_hapskill_35_v4.json"
SKILL_TEMPLATE = HERE / "protocol-hapskill-35-v14-namespace-export-safe.json"
BASELINE_TEMPLATE = HERE / "protocol-agent-ref-10-v4-formal-03.json"
SKILL_PROTOCOL = HERE / "protocol-full-pair-skill-v16.json"
BASELINE_PROTOCOL = HERE / "protocol-full-pair-baseline-v6.json"
STRATA = HERE / "full_pair_strata_35_v2.json"
CAMPAIGN = HERE / "full_pair_campaign_35x2_v2.json"
G2 = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_full_pair_35_luna_clean_r2_g2_v2_20260810.json"
)
CAMPAIGN_ID = "exp_full_pair_35_luna_clean_r2_02"
EXPERIMENT_ID = "EXP-HAPREPAIR-FULL-PAIR-35-R2"
RUN_IDS = {
    "hapskill": [
        "exp_hapskill_35_luna_v16_clean2_repeat_01",
        "exp_hapskill_35_luna_v16_clean2_repeat_02",
    ],
    "vanilla": [
        "exp_agent_ref_35_luna_v6_clean2_repeat_01",
        "exp_agent_ref_35_luna_v6_clean2_repeat_02",
    ],
}
MAX_VALIDATION_SCANS = 5
MAX_AGENT_TURNS_PER_ROUND = 6
MAX_CONCURRENT = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_contract() -> dict[str, Any]:
    return {
        "target_rule_prefixes": [
            "@performance/",
            "@security/",
            "@hw-ets-eslint/",
        ],
        "maximum_validation_scans": MAX_VALIDATION_SCANS,
        "initial_scan_counts_against_budget": False,
        "final_scan_counts_against_budget": False,
        "final_scan_is_fed_back": False,
        "thread_scope": "active_round",
        "same_initial_localization": True,
        "alert_identity": (
            "multiset over (relative_path, rule, whitespace-normalized message)"
        ),
        "source_diff_gate": (
            "Validation requires a real ArkTS/TypeScript source diff relative to "
            "the active-round snapshot."
        ),
        "target_coverage_gate": (
            "Every localized rule/file/location in the frozen active-round plan "
            "must be concretely repaired before validation."
        ),
        "candidate_retention": (
            "Retain the active candidate after structural, public-API, namespace, "
            "configured build, or configured test failure and continue in place "
            "without consuming a HomeCheck validation scan."
        ),
        "best_valid_selection": (
            "Before the evaluator-only final scan, restore the valid candidate "
            "with the lexicographically best score over final alerts, introduced "
            "alerts, changed source files, and round."
        ),
        "build_test_policy": (
            "Use only evaluator gates frozen as available in the shared input "
            "manifest; otherwise label the project statically validated only."
        ),
        "arkts_diagnostic_policy": (
            "For configured builds, compiler ignoreWarning applies only to ArkTS "
            "checker diagnostics; later failures remain fatal."
        ),
    }


def scheduling_contract() -> dict[str, Any]:
    return {
        "maximum_no_diff_retries_per_round": 2,
        "maximum_agent_turns_per_round": MAX_AGENT_TURNS_PER_ROUND,
        "thread_scope": "active_round",
        "round_target_policy": (
            "Repair every currently localized alert location before validation."
        ),
        "complete_cluster_policy": (
            "Inspect every affected file and jointly account for same-entity or "
            "interacting-rule-family findings."
        ),
        "preflight_policy": (
            "Structural, public-API, namespace, build, and test failures are "
            "repaired in the same active round without consuming a scan."
        ),
        "homogeneous_batch_policy": (
            "Batch only after demonstrating common preconditions and invariants."
        ),
        "interaction_policy": (
            "Feed residual and introduced findings into later rounds."
        ),
        "resume_budget_policy": (
            "A resumed active round inherits only unused turns from its original "
            "allowance; every existing JSONL attempt consumes one slot."
        ),
    }


def execution_contract() -> dict[str, Any]:
    return {
        "maximum_concurrent_conditions": MAX_CONCURRENT,
        "retry_policy": (
            "No stochastic condition is rerun automatically. Terminal semantic or "
            "gate failures remain formal failures. Infrastructure interruptions are "
            "preserved for audit before any replacement decision."
        ),
        "abort_policy": (
            "Continue after auditable terminal condition failures; stop launching "
            "new work only when a condition lacks a durable terminal manifest."
        ),
        "resume_policy": (
            "Scheduler resume launches untouched tasks only. Incomplete condition "
            "directories block automatic reuse. Active-round resume cannot refresh "
            "the turn allowance."
        ),
        "old_run_import_policy": "Forbidden for all candidates, traces, and metrics.",
        "paper_facing_efficiency_metrics": [],
    }


def shared_dataset(template: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    dataset = copy.deepcopy(template["dataset"])
    dataset.update(
        {
            "formal_inputs": INPUTS.name,
            "formal_inputs_sha256": sha256_file(INPUTS),
            "formal_inputs_experiment": inputs["experiment"],
            "project_count": len(inputs["projects"]),
            "selection": (
                "Complete frozen 35-project population; no outcome-based subset."
            ),
        }
    )
    return dataset


def model_contract(template: dict[str, Any]) -> dict[str, Any]:
    model = copy.deepcopy(template["model"])
    model["runs_per_project_condition"] = 2
    model["repeat_policy"] = (
        "Two independent clean workspaces; provider seed is not exposed."
    )
    return model


def skill_protocol(inputs: dict[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(read_json(SKILL_TEMPLATE))
    protocol.update(
        {
            "experiment": EXPERIMENT_ID,
            "protocol_id": "full-pair-skill-v16-clean2-r2-20260810",
            "status": "frozen",
            "paper_facing": True,
            "created_at": utc_now(),
            "common_contract": common_contract(),
            "scheduling": scheduling_contract(),
            "execution": {
                **execution_contract(),
                "skill_delivery_policy": protocol["execution"].get(
                    "skill_delivery_policy"
                ),
                "toolchain_mount_policy": protocol["execution"].get(
                    "toolchain_mount_policy"
                ),
            },
            "dataset": shared_dataset(protocol, inputs),
            "model": model_contract(protocol),
            "formal_run": {
                "condition": "hapskill",
                "run_ids": RUN_IDS["hapskill"],
                "scope": "All 35 frozen projects, two independent clean repetitions",
            },
            "stop_condition": (
                "Stop early on zero accepted target findings; otherwise stop after "
                "five post-edit validation scans or bounded active-round exhaustion."
            ),
            "abandonment_condition": (
                "Preserve a formal failure on drift, restricted access, scanner "
                "failure, or bounded active-round exhaustion."
            ),
        }
    )
    protocol["repair_references"]["runner_sha256"] = sha256_file(
        HERE / "run_condition_v14.py"
    )
    return protocol


def baseline_protocol(inputs: dict[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(read_json(BASELINE_TEMPLATE))
    protocol.update(
        {
            "experiment": EXPERIMENT_ID,
            "protocol_id": "full-pair-baseline-v6-clean2-r2-20260810",
            "status": "frozen",
            "paper_facing": True,
            "created_at": utc_now(),
            "common_contract": common_contract(),
            "scheduling": scheduling_contract(),
            "execution": execution_contract(),
            "dataset": shared_dataset(read_json(SKILL_TEMPLATE), inputs),
            "model": model_contract(protocol),
            "formal_run": {
                "condition": "vanilla",
                "run_ids": RUN_IDS["vanilla"],
                "run_id": RUN_IDS["vanilla"][0],
                "authorization_artifact": str(G2),
                "condition_order": [
                    f"{item['name']}:vanilla" for item in inputs["projects"]
                ],
            },
            "stop_condition": (
                "Stop early on zero accepted target findings; otherwise stop after "
                "five post-edit validation scans or bounded active-round exhaustion."
            ),
            "abandonment_condition": (
                "Preserve a formal failure on drift, restricted access, scanner "
                "failure, or bounded active-round exhaustion."
            ),
        }
    )
    protocol["retrieval"] = {
        "exposed_to_agent": False,
        "skill_exposed_to_agent": False,
        "semantic_specifications_exposed": False,
        "static_references_exposed": False,
        "isolation": (
            "The coding-agent-only baseline receives frozen HomeCheck locations and "
            "the same evaluator feedback/gates but no HapRepair Skill or references."
        ),
    }
    return protocol


def strata_manifest(inputs: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(
        (
            {
                "project": item["name"],
                "initial_alerts": int(item["frozen_target_finding_count"]),
            }
            for item in inputs["projects"]
        ),
        key=lambda item: (item["initial_alerts"], item["project"]),
    )
    boundaries = (("low", 0, 12), ("middle", 12, 23), ("high", 23, 35))
    groups = {
        name: [
            {**item, "rank": rank + 1}
            for rank, item in enumerate(ordered[start:end], start=start)
        ]
        for name, start, end in boundaries
    }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "status": "frozen",
        "created_at": utc_now(),
        "input_manifest": str(INPUTS),
        "input_manifest_sha256": sha256_file(INPUTS),
        "ordering": "ascending (frozen_target_finding_count, project_name)",
        "group_sizes": {name: len(items) for name, items in groups.items()},
        "groups": groups,
    }


def task_order(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for repeat in (1, 2):
        for condition in ("hapskill", "vanilla"):
            run_id = RUN_IDS[condition][repeat - 1]
            protocol = SKILL_PROTOCOL if condition == "hapskill" else BASELINE_PROTOCOL
            for project in inputs["projects"]:
                task_id = f"repeat_{repeat:02d}:{condition}:{project['name']}"
                order_key = hashlib.sha256(
                    f"{CAMPAIGN_ID}\0{task_id}".encode("utf-8")
                ).hexdigest()
                tasks.append(
                    {
                        "task_id": task_id,
                        "order_key": order_key,
                        "repeat": repeat,
                        "condition": condition,
                        "project": project["name"],
                        "run_id": run_id,
                        "protocol": str(protocol),
                    }
                )
    tasks.sort(key=lambda item: (item["order_key"], item["task_id"]))
    for ordinal, task in enumerate(tasks, start=1):
        task["ordinal"] = ordinal
        task["status"] = "pending"
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = (SKILL_PROTOCOL, BASELINE_PROTOCOL, STRATA, CAMPAIGN)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"Refusing to overwrite prepared artifacts: {existing}")

    inputs = read_json(INPUTS)
    if inputs.get("status") != "frozen" or len(inputs.get("projects", [])) != 35:
        raise RuntimeError("Expected the frozen 35-project input manifest")
    write_json(SKILL_PROTOCOL, skill_protocol(inputs))
    write_json(BASELINE_PROTOCOL, baseline_protocol(inputs))
    write_json(STRATA, strata_manifest(inputs))
    tasks = task_order(inputs)
    campaign = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "campaign_id": CAMPAIGN_ID,
        "status": "prepared",
        "created_at": utc_now(),
        "input_manifest": str(INPUTS),
        "input_manifest_sha256": sha256_file(INPUTS),
        "strata_manifest": str(STRATA),
        "strata_manifest_sha256": sha256_file(STRATA),
        "protocols": {
            "hapskill": {
                "path": str(SKILL_PROTOCOL),
                "sha256": sha256_file(SKILL_PROTOCOL),
            },
            "vanilla": {
                "path": str(BASELINE_PROTOCOL),
                "sha256": sha256_file(BASELINE_PROTOCOL),
            },
        },
        "g2_authorization": str(G2),
        "maximum_concurrent_conditions": MAX_CONCURRENT,
        "project_count": 35,
        "repetitions": 2,
        "condition_count": 2,
        "task_count": len(tasks),
        "task_order_policy": "ascending sha256(campaign_id NUL task_id)",
        "tasks": tasks,
    }
    write_json(CAMPAIGN, campaign)
    print(CAMPAIGN)


if __name__ == "__main__":
    main()
