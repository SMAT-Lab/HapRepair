#!/usr/bin/env python3
"""Audit whether each rule can support a held-out, same-rule RAG evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = SCRIPT_DIR.parent / "knowledge_base"
PAIR_MANIFEST = KNOWLEDGE_BASE_DIR / "rule_complete_383.jsonl"
NEAR_DUPLICATE_CANDIDATES = KNOWLEDGE_BASE_DIR / "near_duplicate_candidates.json"
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_components(
    pair_ids: list[str],
    edges: list[tuple[str, str, float]],
    threshold: float,
) -> list[list[str]]:
    parent = {pair_id: pair_id for pair_id in pair_ids}

    def find(pair_id: str) -> str:
        while parent[pair_id] != pair_id:
            parent[pair_id] = parent[parent[pair_id]]
            pair_id = parent[pair_id]
        return pair_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, score in edges:
        if score >= threshold:
            union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for pair_id in pair_ids:
        components[find(pair_id)].append(pair_id)
    return sorted(
        (sorted(component) for component in components.values()),
        key=lambda component: (len(component), component),
    )


def main() -> None:
    pairs = load_jsonl(PAIR_MANIFEST)
    candidates = json.loads(NEAR_DUPLICATE_CANDIDATES.read_text(encoding="utf-8"))

    pairs_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pairs_by_rule[pair["rule"]].append(pair)

    edges_by_rule: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for candidate in candidates:
        edges_by_rule[candidate["rule"]].append(
            (
                candidate["left_pair_id"],
                candidate["right_pair_id"],
                float(candidate["mean_similarity"]),
            )
        )

    rule_rows: list[dict[str, Any]] = []
    sensitivity: dict[str, dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        sensitivity[f"{threshold:.2f}"] = {
            "rules_with_at_least_two_candidate_components": 0,
            "rules_with_one_candidate_component": 0,
            "multi_pair_rules_collapsed_to_one_candidate_component": [],
        }

    for rule in sorted(pairs_by_rule):
        rule_pairs = pairs_by_rule[rule]
        pair_ids = sorted(pair["pair_id"] for pair in rule_pairs)
        component_counts: dict[str, int] = {}
        for threshold in THRESHOLDS:
            components = candidate_components(
                pair_ids, edges_by_rule.get(rule, []), threshold
            )
            threshold_key = f"{threshold:.2f}"
            component_counts[threshold_key] = len(components)
            if len(components) >= 2:
                sensitivity[threshold_key][
                    "rules_with_at_least_two_candidate_components"
                ] += 1
            else:
                sensitivity[threshold_key]["rules_with_one_candidate_component"] += 1
                if len(pair_ids) > 1:
                    sensitivity[threshold_key][
                        "multi_pair_rules_collapsed_to_one_candidate_component"
                    ].append(rule)

        pair = rule_pairs[0]
        rule_rows.append(
            {
                "rule": rule,
                "namespace": pair["namespace"],
                "category": pair["category"],
                "pair_count": len(pair_ids),
                "strict_same_rule_rag_holdout_possible_by_count": len(pair_ids) >= 2,
                "candidate_component_count_0_80": component_counts["0.80"],
                "candidate_component_count_0_85": component_counts["0.85"],
                "candidate_component_count_0_90": component_counts["0.90"],
                "candidate_component_count_0_95": component_counts["0.95"],
                "pair_ids": pair_ids,
            }
        )

    singleton_rules = [row["rule"] for row in rule_rows if row["pair_count"] == 1]
    report = {
        "experiment": "EXP-INDEP-63",
        "paper_facing_corpus": "rule_complete_383",
        "pair_count": len(pairs),
        "rule_count": len(rule_rows),
        "protocol_requirement": (
            "Hold out one pair per rule, exclude the held-out pair and confirmed "
            "near duplicates from retrieval, and retain at least one same-rule "
            "retrieval example."
        ),
        "strict_same_rule_rag_holdout": {
            "eligible_rule_count_by_pair_cardinality": sum(
                row["strict_same_rule_rag_holdout_possible_by_count"]
                for row in rule_rows
            ),
            "blocked_rule_count_by_pair_cardinality": len(singleton_rules),
            "blocked_singleton_rules": singleton_rules,
        },
        "near_duplicate_candidate_sensitivity": sensitivity,
        "interpretation": [
            "Similarity edges are review candidates, not confirmed duplicate labels.",
            "The 12 singleton rules cannot support both a held-out case and a same-rule retrieval example from the current 383-pair corpus.",
            "No final held-out/retrieval split is generated until the singleton-rule protocol is resolved and near-duplicate judgments are frozen.",
        ],
    }

    (SCRIPT_DIR / "split_eligibility.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (SCRIPT_DIR / "rule_eligibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = tuple(key for key in rule_rows[0] if key != "pair_ids")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rule_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    print(json.dumps(report["strict_same_rule_rag_holdout"], ensure_ascii=False, indent=2))
    print(json.dumps(sensitivity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
