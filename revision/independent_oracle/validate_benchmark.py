#!/usr/bin/env python3
"""Run CodeLinter on the independent defective/repaired benchmark projects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CODELINTER = Path(
    "/home/zhihao/deveco/command-line-tools/codelinter/bin/codelinter"
)
DEFAULT_CONFIG = REPO_ROOT / "revision" / "code-linter-performance-security-arkts-eslint.json5"
DEFECTIVE_PROJECT = SCRIPT_DIR / "benchmark" / "defective_project"
REPAIRED_PROJECT = SCRIPT_DIR / "benchmark" / "repaired_project"
CASE_MANIFEST = SCRIPT_DIR / "cases_manifest.json"
CODELINTER_OVERLAY = SCRIPT_DIR / "codelinter_overlay.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codelinter",
        type=Path,
        default=Path(os.environ.get("CODELINTER_BIN", DEFAULT_CODELINTER)),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate authored cases without requiring all 63 cases to exist.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_codelinter_overlay(binary: Path, version: str) -> dict[str, Any]:
    overlay = json.loads(CODELINTER_OVERLAY.read_text(encoding="utf-8"))
    if version != overlay["base_codelinter_version"]:
        raise SystemExit(
            "CodeLinter version does not match overlay: "
            f"expected {overlay['base_codelinter_version']}, found {version}"
        )

    codelinter_root = binary.parent.parent
    verified_files: list[dict[str, str]] = []
    for relative_path, expected_sha256 in overlay["files"].items():
        path = codelinter_root / relative_path
        if not path.is_file():
            raise SystemExit(f"CodeLinter overlay file not found: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"CodeLinter overlay hash mismatch for {path}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
        verified_files.append(
            {"path": str(path), "sha256": actual_sha256}
        )

    return {
        "id": overlay["id"],
        "base_codelinter_version": overlay["base_codelinter_version"],
        "homecheck_source_commit": overlay["homecheck_source_commit"],
        "reason": overlay["reason"],
        "manifest": str(CODELINTER_OVERLAY),
        "manifest_sha256": sha256_file(CODELINTER_OVERLAY),
        "verified_files": verified_files,
    }


def run_linter(
    binary: Path,
    config: Path,
    project: Path,
    report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[list[str], int]:
    command = [
        str(binary),
        "--config",
        str(config),
        "--format",
        "json",
        "--output",
        str(report_path),
        str(project),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return command, result.returncode


def load_report(path: Path, project: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []
    for file_result in payload:
        absolute_path = Path(file_result["filePath"])
        try:
            relative_path = absolute_path.relative_to(project).as_posix()
        except ValueError:
            relative_path = absolute_path.as_posix()
        for message in file_result.get("messages", []):
            normalized.append({"relative_path": relative_path, **message})
    return normalized


def findings_for_case(
    findings: list[dict[str, Any]], case: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    target_files = set(case[f"{side}_files"])
    target_rule = case.get("codelinter_rule", case["rule"])
    return [
        finding
        for finding in findings
        if finding.get("rule") == target_rule
        and finding["relative_path"] in target_files
    ]


def main() -> None:
    args = parse_args()
    binary = args.codelinter.resolve()
    config = args.config.resolve()
    if not binary.is_file():
        raise SystemExit(f"CodeLinter executable not found: {binary}")
    if not config.is_file():
        raise SystemExit(f"CodeLinter config not found: {config}")

    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    authored_cases = [
        case
        for case in cases
        if case["defective_files"] and case["repaired_files"]
    ]
    if len(cases) != 63:
        raise SystemExit(f"Expected 63 manifest entries, found {len(cases)}")
    if not args.allow_incomplete and len(authored_cases) != 63:
        raise SystemExit(
            f"Expected 63 authored cases, found {len(authored_cases)}; "
            "use --allow-incomplete only for pilots"
        )

    run_dir = SCRIPT_DIR / "runs" / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    codelinter_overlay = validate_codelinter_overlay(binary, version)
    commands: dict[str, list[str]] = {}
    exit_codes: dict[str, int] = {}
    for side, project in (
        ("defective", DEFECTIVE_PROJECT),
        ("repaired", REPAIRED_PROJECT),
    ):
        command, exit_code = run_linter(
            binary,
            config,
            project,
            run_dir / f"{side}.json",
            run_dir / f"{side}.stdout.log",
            run_dir / f"{side}.stderr.log",
        )
        commands[side] = command
        exit_codes[side] = exit_code

    defective_findings = load_report(run_dir / "defective.json", DEFECTIVE_PROJECT)
    repaired_findings = load_report(run_dir / "repaired.json", REPAIRED_PROJECT)
    case_results: list[dict[str, Any]] = []
    for case in authored_cases:
        defective_target = findings_for_case(defective_findings, case, "defective")
        repaired_target = findings_for_case(repaired_findings, case, "repaired")
        case_results.append(
            {
                "case_id": case["case_id"],
                "rule": case["rule"],
                "codelinter_rule": case.get("codelinter_rule", case["rule"]),
                "defective_target_count": len(defective_target),
                "repaired_target_count": len(repaired_target),
                "passes_target_contract": bool(defective_target)
                and not repaired_target,
            }
        )

    incomplete_markers = (
        "Some error occurred during linting",
        "does not match that of the project",
    )
    scan_warnings = {}
    for side in ("defective", "repaired"):
        stdout = (run_dir / f"{side}.stdout.log").read_text(encoding="utf-8")
        scan_warnings[side] = [marker for marker in incomplete_markers if marker in stdout]

    summary = {
        "run_id": args.run_id,
        "experiment": "EXP-INDEP-63",
        "tier": "auxiliary/dev" if args.allow_incomplete else "main/test",
        "codelinter_version": version,
        "tool_identity": f"{version}+{codelinter_overlay['id']}",
        "codelinter_binary": str(binary),
        "codelinter_overlay": codelinter_overlay,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "commands": commands,
        "exit_codes": exit_codes,
        "scan_warnings": scan_warnings,
        "manifest_case_count": len(cases),
        "authored_case_count": len(authored_cases),
        "validated_case_count": sum(
            result["passes_target_contract"] for result in case_results
        ),
        "failed_case_count": sum(
            not result["passes_target_contract"] for result in case_results
        ),
        "all_scans_clean": all(code == 0 for code in exit_codes.values())
        and not any(scan_warnings.values()),
        "database_mutation": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest_snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "codelinter_overlay_snapshot.json").write_text(
        CODELINTER_OVERLAY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with (run_dir / "case_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(case_results[0]))
        writer.writeheader()
        writer.writerows(case_results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_scans_clean"] or summary["failed_case_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
