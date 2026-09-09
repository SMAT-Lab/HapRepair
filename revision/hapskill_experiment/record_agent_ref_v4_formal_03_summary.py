#!/usr/bin/env python3
"""Record the final formal_03 evaluation handoff without changing run artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import resume_agent_ref_interrupted_v4 as resume


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
AUDIT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/gates/exp_agent_ref_10_v4_formal_03_results_audit_20260807.json"
)
OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/results/exp_agent_ref_10_v4_formal_03_evaluation_summary_20260807.json"
)


def main() -> None:
    audit = resume.read_json(AUDIT)
    if audit.get("status") != "closed" or audit.get("passed") is not True:
        raise RuntimeError("formal_03 results audit is not closed and passing")
    result = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-REF-10",
        "run_id": resume.RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "recorded",
        "evaluation_summary": {
            "research_question": (
                "What contextual repair effectiveness, validation behavior, and "
                "efficiency does the frozen contemporary coding agent achieve on "
                "the ten-project reference subset?"
            ),
            "outcome": (
                "Seven conditions completed and three exhausted bounded guard "
                "repair attempts. Completed conditions reduced 2,752 alerts to 3."
            ),
            "comparison": (
                "The paired join against existing EXP-HAPSKILL-35 results remains "
                "the reporting step; no superiority claim is made here."
            ),
            "evidence": {
                "completed_condition_count": 7,
                "bounded_guard_failure_count": 3,
                "completed_condition_aggregate": audit["completed_condition_aggregate"],
                "sensitivity_excluding_wifi_testapp": audit[
                    "completed_sensitivity_excluding_wifi_testapp"
                ],
            },
            "limitations": (
                "The three failed conditions have no imputed effectiveness metrics; "
                "single-run nondeterminism remains; HomeCheck elimination is not "
                "semantic correctness."
            ),
            "next_action": audit["next_action"],
        },
        "claim_update": "contextual_reference_evidence_available_with_failures",
        "baseline_relation": (
            "Comparable frozen inputs and evaluator budget; paired Skill-result join "
            "not yet reported."
        ),
        "failure_mode": "three_bounded_structural_public_api_or_namespace_guard_failures",
        "next_action": audit["next_action"],
        "results_audit": str(AUDIT),
        "results_audit_sha256": resume.sha256_file(AUDIT),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: RECORDED sha256={resume.sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
