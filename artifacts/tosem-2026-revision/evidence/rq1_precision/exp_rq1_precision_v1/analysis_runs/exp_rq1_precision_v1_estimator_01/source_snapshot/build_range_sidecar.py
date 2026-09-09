#!/usr/bin/env python3
"""Align native HomeCheck ranges to the frozen CodeLinter annotation sample."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SAMPLE = (
    WORKSPACE_ROOT
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1/frozen_sample.jsonl"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1/annotation_range_sidecar.json"
)
DETECTOR_IDENTITY = "CodeLinter 6.0.240"
NATIVE_RANGE_SOURCES = {
    "ast-node",
    "ir-statement",
    "ir-operand",
    "declaration",
    "comment-block",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalize_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")


def report_relative_path(file_path: str, expected_paths: set[str]) -> str | None:
    normalized = normalize_path(file_path)
    matches = [
        relative
        for relative in expected_paths
        if normalized == relative or normalized.endswith(f"/{relative}")
    ]
    return matches[0] if len(matches) == 1 else None


def native_findings(
    project: str,
    report_path: Path,
    expected_paths: set[str],
) -> tuple[list[dict[str, Any]], int]:
    reports = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(reports, list):
        raise ValueError(f"HomeCheck report must be a JSON array: {report_path}")
    findings: list[dict[str, Any]] = []
    skipped_legacy = 0
    for file_report in reports:
        if not isinstance(file_report, dict):
            continue
        relative_path = report_relative_path(
            str(file_report.get("relativePath") or file_report.get("filePath") or ""),
            expected_paths,
        )
        if relative_path is None:
            continue
        rows = file_report.get("defects", file_report.get("messages", []))
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("rangeSource")
            if source not in NATIVE_RANGE_SOURCES:
                skipped_legacy += 1
                continue
            line = row.get("reportLine", row.get("line"))
            column = row.get("reportColumn", row.get("column"))
            start_line = row.get("rangeStartLine", row.get("range_start_line", line))
            start_column = row.get(
                "rangeStartColumn", row.get("range_start_column", column)
            )
            end_line = row.get("endLine", row.get("end_line"))
            end_column = row.get("endColumn", row.get("end_column"))
            rule = row.get("ruleId", row.get("rule"))
            message = row.get("description", row.get("message"))
            coordinates = (
                line,
                column,
                start_line,
                start_column,
                end_line,
                end_column,
            )
            if not all(isinstance(value, int) and value > 0 for value in coordinates):
                continue
            if end_line < start_line or (
                end_line == start_line and end_column <= start_column
            ):
                continue
            findings.append(
                {
                    "project": project,
                    "relative_path": relative_path,
                    "rule": rule,
                    "line": line,
                    "column": column,
                    "message": message,
                    "start_line": start_line,
                    "start_column": start_column,
                    "end_line": end_line,
                    "end_column": end_column,
                    "source": source,
                }
            )
    return findings, skipped_legacy


def identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(field)
        for field in ("project", "relative_path", "rule", "line", "column", "message")
    )


def build_sidecar(
    sample: list[dict[str, Any]],
    reports: dict[str, Path],
    homecheck_source_commit: str,
) -> dict[str, Any]:
    expected_by_project: dict[str, set[str]] = defaultdict(set)
    for record in sample:
        expected_by_project[record["project"]].add(normalize_path(record["relative_path"]))

    native: list[dict[str, Any]] = []
    skipped_legacy = 0
    for project, path in sorted(reports.items()):
        if project not in expected_by_project:
            raise ValueError(f"Report project is absent from frozen sample: {project}")
        rows, skipped = native_findings(project, path, expected_by_project[project])
        native.extend(rows)
        skipped_legacy += skipped

    by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for finding in native:
        by_identity[identity(finding)].append(finding)

    ranges: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, str]] = []
    for record in sample:
        matches = by_identity.get(identity(record), [])
        if len(matches) != 1:
            unmatched.append(
                {
                    "blind_id": record["blind_id"],
                    "reason": "no_native_match" if not matches else "ambiguous_native_match",
                }
            )
            continue
        match = matches[0]
        ranges[record["blind_id"]] = {
            "start_line": match["start_line"],
            "start_column": match["start_column"],
            "end_line": match["end_line"],
            "end_column": match["end_column"],
            "source": match["source"],
        }

    return {
        "schema_version": 1,
        "purpose": "annotation_display_only",
        "detector_unchanged": DETECTOR_IDENTITY,
        "homecheck_source_commit": homecheck_source_commit,
        "coordinate_convention": "1-based, end column exclusive",
        "sample_count": len(sample),
        "matched_count": len(ranges),
        "skipped_legacy_count": skipped_legacy,
        "ranges": ranges,
        "unmatched": unmatched,
    }


def parse_report(value: str) -> tuple[str, Path]:
    project, separator, raw_path = value.partition("=")
    if not separator or not project or not raw_path:
        raise argparse.ArgumentTypeError("report must be PROJECT=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"report does not exist: {path}")
    return project, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--homecheck-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reports = dict(args.report)
    if len(reports) != len(args.report):
        parser.error("each project may have only one report")
    sidecar = build_sidecar(read_jsonl(args.sample), reports, args.homecheck_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {sidecar['matched_count']}/{sidecar['sample_count']} native ranges",
        flush=True,
    )


if __name__ == "__main__":
    main()
