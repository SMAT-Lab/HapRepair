#!/usr/bin/env python3
"""Validate and summarize one paired EXP-AGENT-10 project result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_agent_baseline import canonical_finding
from validation_gate import write_json


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def required_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = manifest["alert_metrics"]
    return {
        **metrics,
        "build_status": manifest["build_status"],
        "test_status": manifest["test_status"],
        "build_count": manifest["build_count"],
        "test_count": manifest["test_count"],
        "input_tokens": manifest["input_tokens"],
        "output_tokens": manifest["output_tokens"],
        "total_tokens": manifest["total_tokens"],
        "api_cost": manifest["api_cost"],
        "wall_clock_seconds": manifest["wall_clock_seconds"],
        "validation_scan_count": manifest["validation_scan_count"],
    }


def validation_scope(manifest: dict[str, Any]) -> str:
    build_status = manifest["build_status"]
    test_status = manifest["test_status"]
    if build_status == "passed" and test_status == "passed":
        return "build and existing tests passed"
    if build_status == "passed":
        return f"build passed; existing tests {test_status.replace('_', ' ')}"
    if build_status in {"not_available", "not_run"} and test_status in {
        "not_available",
        "not_run",
    }:
        return "statically validated only"
    return f"build {build_status.replace('_', ' ')}; tests {test_status.replace('_', ' ')}"


def per_rule(path: Path) -> list[dict[str, Any]]:
    counts = Counter(item["rule"] for item in read_json(path))
    return [{"rule": rule, "count": counts[rule]} for rule in sorted(counts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--agent-run-dir", type=Path, required=True)
    parser.add_argument("--haprepair-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    agent_dir = args.agent_run_dir.resolve()
    haprepair_dir = args.haprepair_run_dir.resolve()
    agent = read_json(agent_dir / "run_manifest.json")
    haprepair = read_json(haprepair_dir / "run_manifest.json")
    checks = {
        "both_completed": agent["status"] == haprepair["status"] == "completed",
        "same_project": agent["project"] == haprepair["project"],
        "same_commit": agent["commit"] == haprepair["commit"],
        "same_tree_oid": agent["tree_oid"] == haprepair["tree_oid"],
        "same_requested_model": (
            agent["model"]["requested_id"] == haprepair["model"]["requested_id"]
        ),
        "same_provider": agent["model"]["provider"] == haprepair["model"]["provider"],
        "same_endpoint_fingerprint": (
            agent["agent_runtime"]["model_endpoint_sha256"]
            == haprepair["endpoint_sha256"]
        ),
        "same_initial_alert_count": (
            agent["alert_metrics"]["initial_alerts"]
            == haprepair["alert_metrics"]["initial_alerts"]
        ),
        "within_scan_budget": (
            agent["validation_scan_count"] <= 5
            and haprepair["validation_scan_count"] <= 5
        ),
        "agent_prohibited_data_not_mounted": (
            agent["isolation"]["haprepair_repository_mounted"] is False
        ),
        "agent_no_restricted_access": not agent["restricted_accesses"],
    }
    agent_initial = read_json(Path(agent["initial_scan"]["findings_path"]))
    haprepair_initial = read_json(Path(haprepair["initial_scan"]["findings_path"]))
    checks["same_initial_findings"] = Counter(
        canonical_finding(item) for item in agent_initial
    ) == Counter(canonical_finding(item) for item in haprepair_initial)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Paired comparability checks failed: {failed}")

    agent_metrics = required_metrics(agent)
    haprepair_metrics = required_metrics(haprepair)
    agent_scope = validation_scope(agent)
    haprepair_scope = validation_scope(haprepair)
    evidence_tier = "auxiliary/dev pilot" if args.pilot else "main/test"
    summary = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "pair_id": args.pair_id,
        "evidence_tier": evidence_tier,
        "status": "trusted_with_caveats" if args.pilot else "paired_result_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": agent["project"],
        "commit": agent["commit"],
        "tree_oid": agent["tree_oid"],
        "model": agent["model"],
        "comparability_checks": checks,
        "conditions": {
            "coding_agent": {
                "run_manifest": str(agent_dir / "run_manifest.json"),
                "metrics": agent_metrics,
                "final_per_rule": per_rule(Path(agent["final_scan"]["findings_path"])),
                "validation_scope": agent_scope,
            },
            "haprepair": {
                "run_manifest": str(haprepair_dir / "run_manifest.json"),
                "metrics": haprepair_metrics,
                "final_per_rule": per_rule(Path(haprepair["final_scan"]["findings_path"])),
                "validation_scope": haprepair_scope,
            },
        },
        "paired_deltas_coding_agent_minus_haprepair": {
            key: agent_metrics[key] - haprepair_metrics[key]
            for key in (
                "eliminated_alerts",
                "introduced_alerts",
                "net_reduction",
                "total_tokens",
                "wall_clock_seconds",
                "validation_scan_count",
            )
        },
        "evaluation_summary": {
            "result": (
                f"On this project, the coding agent eliminated "
                f"{agent_metrics['eliminated_alerts']}/{agent_metrics['initial_alerts']} "
                f"alerts and HapRepair eliminated "
                f"{haprepair_metrics['eliminated_alerts']}/{haprepair_metrics['initial_alerts']}."
            ),
            "comparability": "All frozen project, model, endpoint, localization, and scan-budget checks passed.",
            "claim_update": (
                "The paired runner is executable; one pilot cannot support a general superiority claim."
                if args.pilot
                else "This is one of ten preselected paired projects; aggregate claims require all ten pairs."
            ),
            "baseline_relation": "This validates the strong coding-agent comparator path against the matching HapRepair condition.",
            "validation": (
                f"Coding agent: {agent_scope}. HapRepair: {haprepair_scope}."
            ),
            "next_action": (
                "Freeze one formal paired run ID and execute both conditions on all ten preselected projects."
                if args.pilot
                else "Complete and verify the remaining preselected project pairs."
            ),
        },
        "claim_boundary": (
            "Alert elimination is not semantic correctness. "
            + (
                "This pilot is not a paper-facing ten-project comparison and "
                if args.pilot
                else "This single pair is not the complete paper-facing ten-project comparison and "
            )
            + "must not be combined with the historical 8,664-alert corpus."
        ),
    }
    output_dir = args.output_root.resolve() / args.pair_id
    if output_dir.exists():
        raise SystemExit(f"Paired summary already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
