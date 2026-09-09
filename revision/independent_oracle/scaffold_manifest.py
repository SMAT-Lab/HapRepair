#!/usr/bin/env python3
"""Populate the independent-oracle authoring manifest with all 63 rule IDs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PAIR_MANIFEST = SCRIPT_DIR.parent / "knowledge_base" / "rule_complete_383.jsonl"
CASE_MANIFEST = SCRIPT_DIR / "cases_manifest.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def case_id(rule: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", rule.lower()).strip("_")
    return f"case_{normalized}"


def main() -> None:
    pairs = load_jsonl(PAIR_MANIFEST)
    descriptions: dict[str, str] = {}
    for pair in pairs:
        descriptions.setdefault(pair["rule"], pair["description"])

    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    existing = {case["rule"]: case for case in manifest["cases"]}
    cases: list[dict[str, Any]] = []
    for rule in sorted(descriptions):
        if rule in existing:
            case = existing[rule]
            case.setdefault("rule_description", descriptions[rule])
        else:
            case = {
                "case_id": case_id(rule),
                "rule": rule,
                "rule_description": descriptions[rule],
                "defective_files": [],
                "repaired_files": [],
                "status": "not_authored",
            }
        cases.append(case)

    manifest["cases"] = cases
    CASE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"cases: {len(cases)}")
    print(f"authored or draft: {sum(case['status'] != 'not_authored' for case in cases)}")
    print(f"not authored: {sum(case['status'] == 'not_authored' for case in cases)}")


if __name__ == "__main__":
    main()
