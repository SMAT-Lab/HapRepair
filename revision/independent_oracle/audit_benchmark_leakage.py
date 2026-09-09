#!/usr/bin/env python3
"""Screen the 63-case benchmark for exact and near duplicates in the 383-pair KB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = SCRIPT_DIR.parent / "knowledge_base" / "rule_complete_383.jsonl"
CASE_MANIFEST = SCRIPT_DIR / "cases_manifest.json"
DEFECTIVE_PROJECT = SCRIPT_DIR / "benchmark" / "defective_project"
REPAIRED_PROJECT = SCRIPT_DIR / "benchmark" / "repaired_project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assemble_case(project: Path, relative_paths: list[str]) -> str:
    return normalize_text(
        "\n\n".join(
            (project / relative_path).read_text(encoding="utf-8")
            for relative_path in relative_paths
        )
    )


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def containment(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match()
    return match.size / min(len(left), len(right))


def main() -> None:
    args = parse_args()
    for value, label in (
        (args.similarity_threshold, "similarity threshold"),
        (args.containment_threshold, "containment threshold"),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be between 0 and 1")

    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    pairs = load_jsonl(KNOWLEDGE_BASE)
    pairs_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pairs_by_rule[pair["rule"]].append(pair)

    candidates: list[dict[str, Any]] = []
    exact_pair_matches = 0
    exact_side_matches = 0
    for case in manifest["cases"]:
        defective_code = assemble_case(DEFECTIVE_PROJECT, case["defective_files"])
        repaired_code = assemble_case(REPAIRED_PROJECT, case["repaired_files"])
        for pair in pairs_by_rule[case["rule"]]:
            problem_code = normalize_text(pair["problem_code"])
            repair_code = normalize_text(pair["repair_code"])
            defective_exact = defective_code == problem_code
            repaired_exact = repaired_code == repair_code
            pair_exact = defective_exact and repaired_exact
            problem_similarity = similarity(defective_code, problem_code)
            repair_similarity = similarity(repaired_code, repair_code)
            mean_similarity = (problem_similarity + repair_similarity) / 2
            problem_containment = containment(defective_code, problem_code)
            repair_containment = containment(repaired_code, repair_code)

            exact_pair_matches += pair_exact
            exact_side_matches += defective_exact + repaired_exact
            is_candidate = (
                defective_exact
                or repaired_exact
                or problem_similarity >= args.similarity_threshold
                or repair_similarity >= args.similarity_threshold
                or problem_containment >= args.containment_threshold
                or repair_containment >= args.containment_threshold
            )
            if is_candidate:
                candidates.append(
                    {
                        "case_id": case["case_id"],
                        "rule": case["rule"],
                        "pair_id": pair["pair_id"],
                        "defective_exact": defective_exact,
                        "repaired_exact": repaired_exact,
                        "pair_exact": pair_exact,
                        "problem_similarity": round(problem_similarity, 6),
                        "repair_similarity": round(repair_similarity, 6),
                        "mean_similarity": round(mean_similarity, 6),
                        "problem_containment": round(problem_containment, 6),
                        "repair_containment": round(repair_containment, 6),
                        "decision": "manual_review_required",
                    }
                )

    candidates.sort(
        key=lambda row: (
            -max(
                row["problem_similarity"],
                row["repair_similarity"],
                row["problem_containment"],
                row["repair_containment"],
            ),
            row["case_id"],
            row["pair_id"],
        )
    )
    report = {
        "experiment": "EXP-INDEP-63",
        "benchmark_case_count": len(manifest["cases"]),
        "knowledge_base_pair_count": len(pairs),
        "knowledge_base_rule_count": len(pairs_by_rule),
        "knowledge_base_sha256": sha256_file(KNOWLEDGE_BASE),
        "case_manifest_sha256": sha256_file(CASE_MANIFEST),
        "comparison_scope": "same-rule defective/problem and reference-repair/repair code",
        "similarity_metric": "difflib.SequenceMatcher ratio with autojunk disabled",
        "containment_metric": "longest common contiguous match divided by shorter input length",
        "similarity_threshold": args.similarity_threshold,
        "containment_threshold": args.containment_threshold,
        "exact_pair_match_count": exact_pair_matches,
        "exact_side_match_count": exact_side_matches,
        "near_duplicate_candidate_count": len(candidates),
        "automatic_gate_passed": exact_side_matches == 0 and not candidates,
        "database_mutation": False,
    }
    (SCRIPT_DIR / "leakage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (SCRIPT_DIR / "leakage_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            tuple(candidates[0])
            if candidates
            else (
                "case_id",
                "rule",
                "pair_id",
                "decision",
            )
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["automatic_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
