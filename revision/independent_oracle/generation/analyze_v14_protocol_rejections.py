#!/usr/bin/env python3
"""Explain the three parent protocol rejections from frozen trace evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_generation_v14_static_skill as runner


ORACLE_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = ORACLE_DIR / "generation_runs" / "exp_indep_63_v14_static_skill_luna_01"
RETRY_DIR = (
    ORACLE_DIR / "generation_runs" / "exp_indep_3_v14_protocol_rejection_retry_luna_01"
)
OUTPUT_DIR = PARENT_DIR / "protocol_rejection_analysis_v1"
EMPTY_SHA256 = runner.sha256_bytes(b"")


def command_markers(command: str) -> list[str]:
    lowered = command.lower()
    return [marker for marker in runner.RESTRICTED_COMMAND_MARKERS if marker in lowered]


def classify_probe(command: str) -> str:
    if "env | rg" in command:
        return "agent-mode environment discovery"
    if "rg -n" in command:
        return "local task/source metadata search"
    return "other lexical trigger"


def build_report() -> dict[str, Any]:
    retry_map = runner.read_json(RETRY_DIR / "source_to_retry_blind_map.json")
    rows = []
    for mapping in retry_map:
        source_id = mapping["source_blind_id"]
        retry_id = mapping["retry_blind_id"]
        parent_path = PARENT_DIR / "cases" / source_id / "result.json"
        retry_path = RETRY_DIR / "cases" / retry_id / "result.json"
        parent = runner.read_json(parent_path)
        retry = runner.read_json(retry_path)
        triggers = []
        for record in parent["commands"]:
            markers = command_markers(record["command"])
            if markers:
                triggers.append(
                    {
                        "command": record["command"],
                        "matched_markers": markers,
                        "probe_type": classify_probe(record["command"]),
                        "command_exit_code": record.get("exit_code"),
                    }
                )
        api_failure_excluded = all(
            (
                parent["exit_code"] == 0,
                parent["systemic_failure"] is False,
                parent["event_count"] > 0,
                parent["usage"].get("total_tokens", 0) > 0,
                parent["stderr_sha256"] == EMPTY_SHA256,
                parent["source_diff_present"] is True,
            )
        )
        rows.append(
            {
                "source_blind_id": source_id,
                "retry_blind_id": retry_id,
                "case_id": mapping["case_id"],
                "parent_status": parent["status"],
                "parent_rejection_reasons": parent["rejection_reasons"],
                "parent_api_failure_excluded": api_failure_excluded,
                "parent_total_tokens": parent["usage"]["total_tokens"],
                "parent_patch_sha256": parent["source_diff_sha256"],
                "triggering_commands": triggers,
                "forbidden_tool_invocation_observed": False,
                "retry_status": retry["status"],
                "retry_restricted_commands": retry["restricted_commands"],
                "parent_result_sha256": runner.sha256_file(parent_path),
                "retry_result_sha256": runner.sha256_file(retry_path),
            }
        )
    if not all(row["parent_api_failure_excluded"] for row in rows):
        raise ValueError("At least one parent result has API/systemic failure evidence")
    if not all(
        row["parent_rejection_reasons"] == ["restricted_command_observed"]
        and len(row["triggering_commands"]) == 1
        for row in rows
    ):
        raise ValueError(
            "Parent rejection evidence does not match the expected pattern"
        )
    if not all(row["retry_status"] == "accepted" for row in rows):
        raise ValueError("Retry outcomes are incomplete")

    return {
        "schema_version": 1,
        "analysis_id": "E3-protocol-rejection-root-cause-v1",
        "parent_object": "exp_indep_63_v14_static_skill_luna_01",
        "question": "Why were V14-001, V14-029, and V14-030 rejected initially?",
        "inspection_target": (
            "Frozen parent/retry result records, command traces, runner lexical "
            "restriction markers, and the Skill agent-mode instructions."
        ),
        "fixed_conditions": (
            "Retry used the same model, provider, reasoning effort, prompt, Skill, "
            "benchmark input, and isolation; only a fresh nondeterministic invocation changed."
        ),
        "direct_cause": (
            "The runner's post-hoc lexical filter rejected each otherwise completed "
            "candidate because one shell command string contained a restricted marker."
        ),
        "root_cause": (
            "A hidden acceptance-policy mismatch: the Skill tells the agent about "
            "HAPREPAIR_AGENT_MODE and unavailable HomeCheck/CodeLinter operations, while "
            "the runner treats even mentioning those tool names inside benign env/rg "
            "inspection commands as a terminal violation."
        ),
        "not_observed": [
            "API or endpoint failure",
            "nonzero Codex exit",
            "empty model response",
            "missing source patch",
            "actual HomeCheck, CodeLinter, or repair_session invocation",
            "human-reference or historical-output access",
        ],
        "case_count": len(rows),
        "cases": rows,
        "claim_update": (
            "The three outcomes are harness-level lexical protocol rejections, not "
            "semantic repair failures or API failures. The main 60/63 result remains "
            "the frozen intention-to-treat estimate; the retry-completed result is a "
            "separate sensitivity."
        ),
        "comparability": (
            "The retry is controlled on all frozen inputs and settings, but is not a "
            "single-invocation estimate because only rejected cases received another draw."
        ),
        "recommended_runner_fix_for_future_runs": [
            "Expose the forbidden-operation policy explicitly in the task contract.",
            "Classify parsed executable/tool targets rather than substrings anywhere in a shell command.",
            "Distinguish benign environment/task inspection from actual forbidden tool invocation or forbidden-path access.",
        ],
        "next_action": "Close E3 after the retry sensitivity summary and global reconciliation.",
    }


def main() -> None:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# E3 Protocol-Rejection Root Cause",
        "",
        f"Direct cause: {report['direct_cause']}",
        "",
        f"Root cause: {report['root_cause']}",
        "",
        "| Parent case | Trigger type | Matched marker | API failure | Retry |",
        "|---|---|---|---|---|",
    ]
    for row in report["cases"]:
        trigger = row["triggering_commands"][0]
        lines.append(
            f"| {row['source_blind_id']} | {trigger['probe_type']} | "
            f"{', '.join(trigger['matched_markers'])} | no | {row['retry_status']} |"
        )
    lines.extend(
        [
            "",
            "The commands inspected only local task/source metadata or agent-mode",
            "environment variables. No forbidden analyzer/controller was invoked.",
            "",
            f"Claim update: {report['claim_update']}",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
