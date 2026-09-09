#!/usr/bin/env python3
"""Run the frozen HomeCheck configuration and measure candidate project size."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_MANIFEST = SCRIPT_DIR / "source_manifest.json"
DEFAULT_CODELINTER = Path("/home/zhihao/deveco/command-line-tools/codelinter/bin/codelinter")
DEFAULT_CONFIG = REPO_ROOT / "revision" / "code-linter.json5"
OVERLAY_PATH = REPO_ROOT / "revision" / "independent_oracle" / "codelinter_overlay.json"
SOURCE_SUFFIXES = {".ets", ".ts"}
EXCLUDED_DIRS = {".git", "build", "node_modules", "oh_modules", ".hvigor", ".idea"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_codelinter(binary: Path) -> dict[str, Any]:
    version_result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    version = version_result.stdout.strip()
    overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    if version_result.returncode != 0 or version != overlay["base_codelinter_version"]:
        raise SystemExit(
            f"CodeLinter identity mismatch: expected {overlay['base_codelinter_version']}, found {version!r}"
        )
    root = binary.parent.parent
    files = []
    for relative, expected in overlay["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"CodeLinter overlay mismatch: {path}")
        files.append({"path": str(path), "sha256": expected})
    return {
        "version": version,
        "overlay_id": overlay["id"],
        "overlay_manifest": str(OVERLAY_PATH),
        "overlay_manifest_sha256": sha256_file(OVERLAY_PATH),
        "homecheck_source_commit": overlay["homecheck_source_commit"],
        "verified_files": files,
    }


def source_metrics(project: Path) -> dict[str, int]:
    file_count = 0
    physical_loc = 0
    nonblank_loc = 0
    for root, dirs, files in os.walk(project):
        dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
        for name in files:
            path = Path(root) / name
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            file_count += 1
            physical_loc += len(lines)
            nonblank_loc += sum(bool(line.strip()) for line in lines)
    return {
        "source_file_count": file_count,
        "physical_source_loc": physical_loc,
        "nonblank_source_loc": nonblank_loc,
    }


def normalize_findings(report_path: Path, project: Path) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for file_result in payload:
        absolute = Path(file_result["filePath"])
        try:
            relative = absolute.resolve().relative_to(project.resolve()).as_posix()
        except ValueError:
            relative = absolute.as_posix()
        for message in file_result.get("messages", []):
            item = {
                "relative_path": relative,
                "line": int(message.get("line", 0)),
                "column": int(message.get("column", 0)),
                "end_line": int(message.get("endLine", message.get("line", 0))),
                "end_column": int(message.get("endColumn", message.get("column", 0))),
                "severity": message.get("severity", ""),
                "rule": message.get("rule", ""),
                "message": message.get("message", ""),
            }
            identity = tuple(item[key] for key in ("relative_path", "line", "column", "rule", "message"))
            if identity not in seen:
                seen.add(identity)
                findings.append(item)
    return sorted(
        findings,
        key=lambda item: (item["relative_path"], item["line"], item["column"], item["rule"], item["message"]),
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--codelinter", type=Path, default=DEFAULT_CODELINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--names", help="Optional comma-separated project names.")
    args = parser.parse_args()

    source_manifest_path = args.source_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    binary = args.codelinter.resolve()
    config = args.config.resolve()
    if not binary.is_file() or not config.is_file():
        raise SystemExit("CodeLinter binary or configuration is missing")
    identity = verify_codelinter(binary)
    selected_names = set(args.names.split(",")) if args.names else None
    projects = [item for item in source_manifest["projects"] if item["status"] == "ready"]
    if selected_names is not None:
        projects = [item for item in projects if item["name"] in selected_names]

    run_dir = SCRIPT_DIR / "scan_runs" / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Scan run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    findings_dir = run_dir / "findings"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    findings_dir.mkdir()
    logs_dir.mkdir()
    reports_dir.mkdir()
    manifest_path = run_dir / "scan_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "run_id": args.run_id,
        "started_at": utc_now(),
        "completed_at": None,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "codelinter": identity,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "loc_definition": "Nonblank physical lines in .ets and .ts files, excluding dependency, build, VCS, and IDE directories.",
        "projects": [],
    }
    write_manifest(manifest_path, manifest)

    for project_meta in sorted(projects, key=lambda item: item["name"]):
        name = project_meta["name"]
        project = Path(project_meta["source_path"]).resolve()
        report_path = reports_dir / f"{name}.json"
        stdout_path = logs_dir / f"{name}.stdout.log"
        stderr_path = logs_dir / f"{name}.stderr.log"
        print(f"[scan] {name}: {project}", flush=True)
        metrics = source_metrics(project)
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
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.monotonic() - started
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        status = "scan_failed"
        findings: list[dict[str, Any]] = []
        parse_error = ""
        if report_path.is_file():
            try:
                findings = normalize_findings(report_path, project)
                status = "scanned" if result.returncode == 0 else "scanned_nonzero_exit"
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                parse_error = str(error)
        finding_path = findings_dir / f"{name}.json"
        finding_path.write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        loc = metrics["nonblank_source_loc"]
        initial = len(findings)
        project_result = {
            "name": name,
            "status": status,
            "source_path": str(project),
            "namespace": project_meta["namespace"],
            "repo": project_meta["repo"],
            "repo_url": project_meta["repo_url"],
            "subpath": project_meta["subpath"],
            "commit": project_meta["commit"],
            "tree_oid": project_meta["tree_oid"],
            **metrics,
            "initial_alerts": initial,
            "alerts_per_kloc": (initial * 1000 / loc) if loc else None,
            "scan_seconds": elapsed,
            "exit_code": result.returncode,
            "command": command,
            "findings_path": str(finding_path),
            "findings_sha256": sha256_file(finding_path),
            "report_path": str(report_path) if report_path.is_file() else None,
            "parse_error": parse_error,
        }
        manifest["projects"].append(project_result)
        write_manifest(manifest_path, manifest)
        print(
            f"[{' ok ' if status.startswith('scanned') else 'fail'}] {name}: "
            f"{initial} alerts, {loc} nonblank LOC, {elapsed:.1f}s",
            flush=True,
        )

    manifest["completed_at"] = utc_now()
    manifest["scanned_project_count"] = sum(item["status"].startswith("scanned") for item in manifest["projects"])
    manifest["total_initial_alerts"] = sum(item["initial_alerts"] for item in manifest["projects"] if item["status"].startswith("scanned"))
    write_manifest(manifest_path, manifest)
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
