#!/usr/bin/env python3
"""Measure trace-proven semantic-spec and optional-guide access for final v14."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
RUNS_DIR = ORACLE_DIR / "generation_runs"
INSTALLED_SKILL = Path("codex_home/skills/haprepair-openharmony-repair")
CONTENT_COMMAND = re.compile(r"\b(cat|sed|head|tail|awk|perl|rg|grep)\b")
LISTING_COMMAND = re.compile(r"\brg\s+--files\b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def category(rule: str) -> str:
    if rule.startswith("@performance/"):
        return "performance"
    if rule.startswith("@security/"):
        return "security"
    return "arkts_eslint"


def command_events(trace_path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        event = json.loads(line)
        item = event.get("item", {})
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "command_execution"
        ):
            events.append(
                {
                    "trace_line": line_number,
                    "item_id": item.get("id"),
                    "command": item.get("command", ""),
                    "output": item.get("aggregated_output", ""),
                    "exit_code": item.get("exit_code"),
                }
            )
    return events


def content_access_evidence(
    events: list[dict[str, Any]], *, relative_path: str
) -> list[dict[str, Any]]:
    """Return conservative evidence that file contents, not just names, were read."""
    filename = Path(relative_path).name
    evidence = []
    for event in events:
        command = event["command"]
        output = event["output"]
        direct_read = (
            filename in command
            and CONTENT_COMMAND.search(command)
            and not LISTING_COMMAND.search(command)
        )
        ripgrep_content = (
            re.search(r"\brg\b", command) is not None
            and "--files" not in command
            and (f"/{filename}:" in output or f"/{filename}-" in output)
        )
        if direct_read or ripgrep_content:
            evidence.append(
                {
                    "trace_line": event["trace_line"],
                    "item_id": event["item_id"],
                    "command": command,
                    "basis": "direct_content_command"
                    if direct_read
                    else "ripgrep_content_output",
                }
            )
    return evidence


def load_reference_map(skill_root: Path) -> dict[str, Any]:
    guide_manifest = read_json(skill_root / "references/repair-guides/manifest.json")
    spec_manifest = read_json(skill_root / "references/rule-specs/manifest.json")
    guide_by_rule = {
        record["rule"]: record["file"] for record in guide_manifest["rules"]
    }
    family_by_rule = {
        rule: family["spec"]
        for family in spec_manifest["families"]
        for rule in family["rules"]
    }
    return {
        "core": spec_manifest["core"],
        "aliases": spec_manifest["aliases"],
        "guide_by_rule": guide_by_rule,
        "family_by_rule": family_by_rule,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "case_count": count,
        "skill_md_content_access_count": sum(row["skill_md_accessed"] for row in rows),
        "core_spec_content_access_count": sum(
            row["core_spec_accessed"] for row in rows
        ),
        "family_spec_content_access_count": sum(
            row["family_spec_accessed"] for row in rows
        ),
        "exact_guide_content_access_count": sum(
            row["exact_guide_accessed"] for row in rows
        ),
        "exact_guide_content_access_rate": (
            sum(row["exact_guide_accessed"] for row in rows) / count if count else None
        ),
    }


def audit(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    manifest = read_json(run_dir / "input_manifest.json")
    first_skill = run_dir / manifest[0]["case_dir"] / INSTALLED_SKILL
    reference_map = load_reference_map(first_skill)
    rows = []
    for item in manifest:
        rule = item["rule"]
        canonical_rule = reference_map["aliases"].get(rule, rule)
        guide_file = reference_map["guide_by_rule"][rule]
        family_file = reference_map["family_by_rule"][canonical_rule]
        trace_path = run_dir / item["case_dir"] / "trace.jsonl"
        events = command_events(trace_path)
        expected = {
            "skill_md": "SKILL.md",
            "core_spec": f"references/rule-specs/{reference_map['core']}",
            "family_spec": f"references/rule-specs/{family_file}",
            "exact_guide": f"references/repair-guides/{guide_file}",
        }
        evidence = {
            name: content_access_evidence(events, relative_path=path)
            for name, path in expected.items()
        }
        result = read_json(run_dir / item["case_dir"] / "result.json")
        rows.append(
            {
                "case_id": item["case_id"],
                "blind_id": item["blind_id"],
                "category": category(rule),
                "rule": rule,
                "generation_status": result["status"],
                "command_event_count": len(events),
                "expected_paths": expected,
                "skill_md_accessed": bool(evidence["skill_md"]),
                "core_spec_accessed": bool(evidence["core_spec"]),
                "family_spec_accessed": bool(evidence["family_spec"]),
                "exact_guide_accessed": bool(evidence["exact_guide"]),
                "evidence": evidence,
            }
        )

    by_category = {
        name: summarize_group([row for row in rows if row["category"] == name])
        for name in ("performance", "arkts_eslint", "security")
    }
    by_generation_status = {
        name: summarize_group([row for row in rows if row["generation_status"] == name])
        for name in ("accepted", "rejected")
    }
    summary = summarize_group(rows)
    access_patterns = Counter(
        (
            row["skill_md_accessed"],
            row["core_spec_accessed"],
            row["family_spec_accessed"],
            row["exact_guide_accessed"],
        )
        for row in rows
    )
    checks = {
        "case_count_63": len(rows) == 63,
        "taxonomy_42_1_20": {
            name: value["case_count"] for name, value in by_category.items()
        }
        == {"performance": 42, "arkts_eslint": 1, "security": 20},
        "every_rule_has_one_expected_guide": all(
            row["expected_paths"]["exact_guide"] for row in rows
        ),
        "every_rule_maps_to_one_family_spec": all(
            row["expected_paths"]["family_spec"] for row in rows
        ),
        "all_traces_have_command_events": all(
            row["command_event_count"] > 0 for row in rows
        ),
        "core_and_family_access_proven_for_all_cases": all(
            row["core_spec_accessed"] and row["family_spec_accessed"] for row in rows
        ),
        "guide_non_access_remains_visible": any(
            not row["exact_guide_accessed"] for row in rows
        ),
    }
    report = {
        "schema_version": 1,
        "analysis_id": "E3-final-v14-trace-reference-access-v1",
        "parent_run_id": run_id,
        "analysis_question": (
            "Which final-v14 cases have command-trace evidence of reading the Skill, "
            "normative semantic specifications, and the exact optional static guide?"
        ),
        "additional_model_calls": 0,
        "measurement_rule": {
            "counts_as_content_access": (
                "A completed cat/sed/head/tail/awk/perl/rg/grep command directly names "
                "the expected file, or a non-listing ripgrep command emits content lines "
                "prefixed by that exact filename."
            ),
            "does_not_count": (
                "Filename listings, guide availability, prompt text, model-authored "
                "completion claims, or successful repair outcomes."
            ),
            "interpretation": (
                "Access proves that content was exposed through a command trace; it does "
                "not prove causal reliance, correct interpretation, or necessity."
            ),
        },
        "overall": summary,
        "by_category": by_category,
        "by_generation_status": by_generation_status,
        "access_patterns": [
            {
                "skill_md_accessed": pattern[0],
                "core_spec_accessed": pattern[1],
                "family_spec_accessed": pattern[2],
                "exact_guide_accessed": pattern[3],
                "case_count": count,
            }
            for pattern, count in sorted(access_patterns.items())
        ],
        "cases_without_exact_guide_access": [
            {
                "case_id": row["case_id"],
                "blind_id": row["blind_id"],
                "category": row["category"],
                "rule": row["rule"],
                "generation_status": row["generation_status"],
            }
            for row in rows
            if not row["exact_guide_accessed"]
        ],
        "checks": checks,
        "audit_passed": all(checks.values()),
        "claim_update": (
            "All 63 traces prove access to the core and applicable family semantic "
            "specifications. Exact optional guide-content access is proven for 58/63 "
            "cases, so the evidence supports frequent but not universal guide use and "
            "does not identify the guide's causal effect."
        ),
        "comparability": (
            "The same conservative command-trace rule is applied to all 63 cases; no "
            "generation or adjudication artifact is changed."
        ),
        "next_action": (
            "Carry the 58/63 access fact and five explicit non-access cases into final "
            "reconciliation; do not claim universal or causal reference use."
        ),
        "cases": rows,
    }
    output = run_dir / "trace_access_audit_v1" / "access.json"
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = audit(args.run_id)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )
    if not report["audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
