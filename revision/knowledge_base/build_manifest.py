#!/usr/bin/env python3
"""Build auditable HapRepair knowledge-base manifests from the source workbooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

PUBLISHED_SOURCES = (
    Path("data/vul_pairs.xlsx"),
    Path("data/output.xlsx"),
    Path("data/security_pairs.xlsx"),
    Path("data/addition.xlsx"),
)
COMPLETION_SOURCES = (Path("data/missing_pairs.xlsx"),)

COLUMN_ALIASES = {
    "rule": ("Rule", "rule", "规则"),
    "description": ("Description", "description", "描述"),
    "problem_code": ("Problem Code Example", "problem_code", "问题代码样例"),
    "problem_explanation": (
        "Problem Explanation",
        "problem_explain",
        "问题解释",
    ),
    "repair_code": ("Repair Code Example", "problem_fix", "修复代码样例"),
    "diff": ("Diff", "diff", "差异"),
    "difflib": ("Difflib", "difflib"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated manifests (default: script directory)",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.80,
        help="Minimum mean code similarity for a review candidate (default: 0.80)",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def get_cell(row: pd.Series, aliases: tuple[str, ...]) -> str:
    for column in aliases:
        if column in row.index:
            return normalize_text(row[column])
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_hash(rule: str, problem_code: str, repair_code: str) -> str:
    payload = json.dumps(
        [rule, problem_code, repair_code],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_rule(rule: str) -> tuple[str, str]:
    if rule.startswith("@") and "/" in rule:
        namespace = rule[1:].split("/", 1)[0]
    else:
        namespace = "unclassified"
    category = namespace if namespace in {"performance", "security"} else "other"
    return namespace, category


def read_sources(source_paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for relative_path in source_paths:
        absolute_path = REPO_ROOT / relative_path
        workbook = pd.ExcelFile(absolute_path)
        source_rows: list[dict[str, Any]] = []
        sheet_audits: list[dict[str, Any]] = []
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(absolute_path, sheet_name=sheet_name, dtype=object)
            sheet_rule_ids: set[str] = set()
            for zero_based_index, row in frame.iterrows():
                item = {
                    field: get_cell(row, aliases)
                    for field, aliases in COLUMN_ALIASES.items()
                }
                item.update(
                    {
                        "source_file": relative_path.as_posix(),
                        "source_sheet": sheet_name,
                        "source_row": int(zero_based_index) + 2,
                    }
                )
                item["exact_hash"] = exact_hash(
                    item["rule"], item["problem_code"], item["repair_code"]
                )
                source_rows.append(item)
                if item["rule"]:
                    sheet_rule_ids.add(item["rule"])
            sheet_audits.append(
                {
                    "sheet": sheet_name,
                    "row_count": len(frame),
                    "distinct_rule_count": len(sheet_rule_ids),
                    "columns": [str(column) for column in frame.columns],
                }
            )
        rows.extend(source_rows)
        audits.append(
            {
                "source_file": relative_path.as_posix(),
                "sha256": sha256_file(absolute_path),
                "row_count": len(source_rows),
                "distinct_rule_count": len(
                    {item["rule"] for item in source_rows if item["rule"]}
                ),
                "sheets": sheet_audits,
            }
        )
    return rows, audits


def canonicalize(
    all_rows: list[dict[str, Any]], published_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["exact_hash"]].append(row)

    canonical: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for digest, occurrences in grouped.items():
        primary = occurrences[0]
        namespace, category = classify_rule(primary["rule"])
        pair = {
            "pair_id": f"pair_{digest[:16]}",
            "rule": primary["rule"],
            "namespace": namespace,
            "category": category,
            "description": primary["description"],
            "problem_code": primary["problem_code"],
            "problem_explanation": primary["problem_explanation"],
            "repair_code": primary["repair_code"],
            "diff": primary["diff"],
            "difflib": primary["difflib"],
            "source_file": primary["source_file"],
            "source_sheet": primary["source_sheet"],
            "source_row": primary["source_row"],
            "source_occurrences": [
                {
                    "source_file": item["source_file"],
                    "source_sheet": item["source_sheet"],
                    "source_row": item["source_row"],
                }
                for item in occurrences
            ],
            "exact_hash": digest,
            "duplicate_group": f"duplicate_{digest[:16]}"
            if len(occurrences) > 1
            else None,
            "included_in_published_378": digest in published_hashes,
            "included_in_rule_complete_383": True,
        }
        canonical.append(pair)
        if len(occurrences) > 1:
            duplicate_groups.append(
                {
                    "duplicate_group": pair["duplicate_group"],
                    "pair_id": pair["pair_id"],
                    "exact_hash": digest,
                    "occurrence_count": len(occurrences),
                    "occurrences": pair["source_occurrences"],
                }
            )

    canonical.sort(key=lambda item: (item["rule"], item["pair_id"]))
    duplicate_groups.sort(key=lambda item: item["duplicate_group"])
    return canonical, duplicate_groups


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def find_near_duplicates(
    pairs: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_rule[pair["rule"]].append(pair)

    candidates: list[dict[str, Any]] = []
    for rule, rule_pairs in by_rule.items():
        for left_index, left in enumerate(rule_pairs):
            for right in rule_pairs[left_index + 1 :]:
                problem_score = similarity(left["problem_code"], right["problem_code"])
                repair_score = similarity(left["repair_code"], right["repair_code"])
                mean_score = (problem_score + repair_score) / 2
                if mean_score >= threshold:
                    candidates.append(
                        {
                            "rule": rule,
                            "left_pair_id": left["pair_id"],
                            "right_pair_id": right["pair_id"],
                            "problem_code_similarity": round(problem_score, 6),
                            "repair_code_similarity": round(repair_score, 6),
                            "mean_similarity": round(mean_score, 6),
                            "decision": "manual_review_required",
                        }
                    )
    candidates.sort(
        key=lambda item: (-item["mean_similarity"], item["rule"], item["left_pair_id"])
    )
    return candidates


def count_categories(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(pair["category"] for pair in pairs)
    return dict(sorted(counts.items()))


def count_namespaces(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(pair["namespace"] for pair in pairs)
    return dict(sorted(counts.items()))


def count_rules_by(field: str, pairs: list[dict[str, Any]]) -> dict[str, int]:
    rule_values = {pair["rule"]: pair[field] for pair in pairs}
    return dict(sorted(Counter(rule_values.values()).items()))


def count_empty_fields(pairs: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "rule",
        "description",
        "problem_code",
        "problem_explanation",
        "repair_code",
        "diff",
        "difflib",
    )
    return {field: sum(not pair[field] for pair in pairs) for field in fields}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_rule_coverage(
    path: Path,
    canonical: list[dict[str, Any]],
    published: list[dict[str, Any]],
    complete: list[dict[str, Any]],
) -> None:
    published_counts = Counter(pair["rule"] for pair in published)
    complete_counts = Counter(pair["rule"] for pair in complete)
    metadata = {pair["rule"]: pair for pair in canonical}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rule",
                "namespace",
                "category",
                "published_378_pair_count",
                "rule_complete_383_pair_count",
            ),
        )
        writer.writeheader()
        for rule in sorted(metadata):
            writer.writerow(
                {
                    "rule": rule,
                    "namespace": metadata[rule]["namespace"],
                    "category": metadata[rule]["category"],
                    "published_378_pair_count": published_counts[rule],
                    "rule_complete_383_pair_count": complete_counts[rule],
                }
            )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        raise SystemExit("--near-duplicate-threshold must be between 0 and 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    published_rows, published_audits = read_sources(PUBLISHED_SOURCES)
    completion_rows, completion_audits = read_sources(COMPLETION_SOURCES)
    all_rows = published_rows + completion_rows
    published_hashes = {row["exact_hash"] for row in published_rows}
    canonical, duplicate_groups = canonicalize(all_rows, published_hashes)
    published = [pair for pair in canonical if pair["included_in_published_378"]]
    complete = [pair for pair in canonical if pair["included_in_rule_complete_383"]]
    near_duplicates = find_near_duplicates(complete, args.near_duplicate_threshold)

    pair_ids = [pair["pair_id"] for pair in canonical]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("pair_id collision detected")

    audit = {
        "deduplication_key": ["rule", "problem_code", "repair_code"],
        "normalization": "CRLF/CR converted to LF; outer whitespace stripped",
        "paper_facing_view": "rule_complete_383",
        "source_files": published_audits + completion_audits,
        "views": {
            "published_378": {
                "source_files": [path.as_posix() for path in PUBLISHED_SOURCES],
                "raw_row_count": len(published_rows),
                "exact_unique_pair_count": len(published),
                "exact_duplicate_row_count": len(published_rows) - len(published),
                "distinct_rule_count": len({pair["rule"] for pair in published}),
                "pair_counts_by_category": count_categories(published),
                "pair_counts_by_namespace": count_namespaces(published),
                "rule_counts_by_category": count_rules_by("category", published),
                "rule_counts_by_namespace": count_rules_by("namespace", published),
                "empty_field_counts": count_empty_fields(published),
            },
            "rule_complete_383": {
                "source_files": [
                    path.as_posix() for path in PUBLISHED_SOURCES + COMPLETION_SOURCES
                ],
                "raw_row_count": len(all_rows),
                "exact_unique_pair_count": len(complete),
                "exact_duplicate_row_count": len(all_rows) - len(complete),
                "distinct_rule_count": len({pair["rule"] for pair in complete}),
                "pair_counts_by_category": count_categories(complete),
                "pair_counts_by_namespace": count_namespaces(complete),
                "rule_counts_by_category": count_rules_by("category", complete),
                "rule_counts_by_namespace": count_rules_by("namespace", complete),
                "empty_field_counts": count_empty_fields(complete),
            },
        },
        "duplicate_group_count": len(duplicate_groups),
        "near_duplicate_review": {
            "threshold": args.near_duplicate_threshold,
            "score": "mean of problem-code and repair-code SequenceMatcher ratios",
            "candidate_count": len(near_duplicates),
            "automatic_removal": False,
        },
    }

    write_json(output_dir / "source_audit.json", audit)
    write_jsonl(output_dir / "canonical_pairs.jsonl", canonical)
    write_jsonl(output_dir / "published_378.jsonl", published)
    write_jsonl(output_dir / "rule_complete_383.jsonl", complete)
    write_json(output_dir / "exact_duplicate_groups.json", duplicate_groups)
    write_json(output_dir / "near_duplicate_candidates.json", near_duplicates)
    write_rule_coverage(output_dir / "rule_coverage.csv", canonical, published, complete)

    print(json.dumps(audit["views"], ensure_ascii=False, indent=2))
    print(f"duplicate groups: {len(duplicate_groups)}")
    print(f"near-duplicate review candidates: {len(near_duplicates)}")


if __name__ == "__main__":
    main()
