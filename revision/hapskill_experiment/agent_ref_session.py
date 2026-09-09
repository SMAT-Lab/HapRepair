#!/usr/bin/env python3
"""Method-independent planning and candidate lifecycle helpers for EXP-AGENT-REF-10."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_PLAN_MARKERS = (
    "haprepair",
    "repair-guide",
    "repair_guide",
    "semantic-spec",
    "semantic_spec",
    "skill_state",
    "independent_oracle",
    "knowledge_base",
    "retrieval",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def workspace_container_path(path: Path, workspace: Path) -> str:
    """Map a host workspace artifact to the exact agent-visible path."""
    relative = path.resolve().relative_to(workspace.resolve())
    return "/workspace/" + relative.as_posix()


def completion_schema() -> dict[str, Any]:
    """Publish the exact report shape enforced by :func:`audit_completion`."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "selected_rules",
            "entity_repairs",
            "blocked",
            "unresolved_external",
        ],
        "properties": {
            "selected_rules": {"type": "array", "items": {"type": "string"}},
            "blocked": {"type": "array", "maxItems": 0},
            "unresolved_external": {"type": "array", "maxItems": 0},
            "entity_repairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "rule",
                        "relative_path",
                        "locations",
                        "entities",
                        "evidence",
                        "transformation",
                        "status",
                    ],
                    "properties": {
                        "rule": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "locations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["line", "column"],
                                "properties": {
                                    "line": {"type": "integer"},
                                    "column": {"type": "integer"},
                                },
                            },
                        },
                        "entities": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "transformation": {"type": "string", "minLength": 1},
                        "status": {"const": "repaired"},
                    },
                },
            },
        },
    }


def make_reference_plan(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Group evaluator findings without leaking HapRepair method evidence."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rules_by_file: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        rule = str(finding["rule"])
        relative = str(finding["relative_path"])
        grouped[rule][relative].append(
            {
                "line": int(finding.get("line", 0)),
                "column": int(finding.get("column", 0)),
                "end_line": int(finding.get("end_line", finding.get("line", 0))),
                "end_column": int(finding.get("end_column", finding.get("column", 0))),
                "message": str(finding.get("message", "")),
            }
        )
        rules_by_file[relative].add(rule)

    rules = []
    for rule, files in sorted(grouped.items()):
        rules.append(
            {
                "rule": rule,
                "files": [
                    {
                        "relative_path": relative,
                        "locations": sorted(
                            locations,
                            key=lambda item: (
                                item["line"],
                                item["column"],
                                item["message"],
                            ),
                        ),
                    }
                    for relative, locations in sorted(files.items())
                ],
            }
        )
    clusters = [
        {"relative_path": relative, "rules": sorted(file_rules)}
        for relative, file_rules in sorted(rules_by_file.items())
        if len(file_rules) > 1
    ]
    plan = {
        "schema_version": 1,
        "method": "independent_coding_agent",
        "finding_count": len(findings),
        "rule_count": len(rules),
        "rules": rules,
        "same_file_interaction_clusters": clusters,
    }
    assert_plan_is_sanitized(plan)
    return plan


def assert_plan_is_sanitized(plan: dict[str, Any]) -> None:
    encoded = json.dumps(plan, ensure_ascii=False).lower()
    leaked = [marker for marker in FORBIDDEN_PLAN_MARKERS if marker in encoded]
    if leaked:
        raise ValueError(f"Reference plan leaks forbidden method evidence: {leaked}")


def _locations(items: list[dict[str, Any]]) -> Counter[tuple[int, int]]:
    return Counter(
        (int(item.get("line", -1)), int(item.get("column", -1))) for item in items
    )


def audit_completion(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Require concrete, exact rule/file/location accounting before a scan."""
    assert_plan_is_sanitized(plan)
    required_rules = {item["rule"] for item in plan["rules"]}
    selected = {str(item) for item in report.get("selected_rules", [])}
    problems: list[str] = []
    if selected != required_rules:
        problems.append(
            "selected_rules mismatch: "
            f"missing={sorted(required_rules - selected)}, "
            f"unknown={sorted(selected - required_rules)}"
        )
    if report.get("blocked") or report.get("unresolved_external"):
        problems.append("blocked or unresolved entries do not satisfy round coverage")

    repairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in report.get("entity_repairs") or []:
        if not isinstance(item, dict):
            problems.append("entity_repairs entries must be objects")
            continue
        rule = str(item.get("rule", ""))
        relative = str(item.get("relative_path", ""))
        repairs[(rule, relative)].append(item)
        if item.get("status") != "repaired":
            problems.append(f"{rule} in {relative} is not marked repaired")
        for field in ("entities", "evidence", "locations"):
            if not isinstance(item.get(field), list) or not item[field]:
                problems.append(
                    f'{rule} in {relative} requires non-empty JSON array "{field}"'
                )
        if not str(item.get("transformation", "")).strip():
            problems.append(f"{rule} in {relative} requires a transformation")

    expected_groups: set[tuple[str, str]] = set()
    for rule_record in plan["rules"]:
        rule = rule_record["rule"]
        for file_record in rule_record["files"]:
            relative = file_record["relative_path"]
            key = (rule, relative)
            expected_groups.add(key)
            group = repairs.get(key, [])
            if not group:
                problems.append(f"missing rule/file group: {rule} in {relative}")
                continue
            expected = _locations(file_record["locations"])
            observed = Counter()
            for item in group:
                observed.update(_locations(item.get("locations") or []))
            if expected != observed:
                problems.append(
                    f"location mismatch for {rule} in {relative}: "
                    f"missing={sorted((expected - observed).elements())}, "
                    f"unknown={sorted((observed - expected).elements())}"
                )
    unknown_groups = sorted(set(repairs) - expected_groups)
    if unknown_groups:
        problems.append(f"unknown rule/file groups: {unknown_groups}")
    return {
        "complete": not problems,
        "problems": problems,
        "required_rule_count": len(required_rules),
        "required_rule_file_group_count": len(expected_groups),
        "feedback": (
            "Every required rule, file, and exact evaluator location is accounted for."
            if not problems
            else "Coverage is incomplete: " + "; ".join(problems)
        ),
    }


def candidate_score(
    metrics: dict[str, int], changed_source_files: int, round_number: int
) -> list[int]:
    return [
        int(metrics["final_alerts"]),
        int(metrics["introduced_alerts"]),
        int(changed_source_files),
        int(round_number),
    ]


def describe_interactions(
    round_number: int, deltas: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """Describe same-file exchanges without hidden family/spec metadata."""
    eliminated = Counter(str(item["rule"]) for item in deltas["eliminated"])
    introduced = Counter(str(item["rule"]) for item in deltas["introduced"])
    if not eliminated or not introduced:
        return None
    edges = sorted(
        {
            (
                str(before["relative_path"]),
                str(before["rule"]),
                str(after["rule"]),
            )
            for before in deltas["eliminated"]
            for after in deltas["introduced"]
            if before["relative_path"] == after["relative_path"]
        }
    )
    return {
        "round": round_number,
        "eliminated_rules": dict(sorted(eliminated.items())),
        "introduced_rules": dict(sorted(introduced.items())),
        "same_file_exchanges": [
            {"relative_path": path, "from_rule": source, "to_rule": target}
            for path, source, target in edges
        ],
        "interpretation": "Descriptive same-file association only; no hidden rule-family metadata was used.",
    }


def is_restricted_command(command: str) -> bool:
    normalized = command.lower()
    markers = (
        "/home/zhihao/hdd/haprepair/haprepair",
        "haprepair-openharmony-repair",
        "skill_state",
        "repair-guide",
        "semantic-spec",
        "knowledge_base",
        "independent_oracle",
        "retrieval",
        "exp_hapskill",
        "exp_agent_10/runs",
        "reference_patch",
        "formal_04",
    )
    return any(marker in normalized for marker in markers)
