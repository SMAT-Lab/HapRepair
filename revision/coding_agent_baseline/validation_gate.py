#!/usr/bin/env python3
"""Budgeted HomeCheck scans for the EXP-AGENT-10 comparison."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scan_projects import (
    DEFAULT_CODELINTER,
    DEFAULT_CONFIG,
    normalize_findings,
    verify_codelinter,
)


SCHEMA_VERSION = 1
VALID_KINDS = {"initial", "validation", "final"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@contextmanager
def locked_state(state_dir: Path) -> Iterator[Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "state.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield state_dir / "state.json"
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def initialize_gate(
    state_dir: Path,
    workspace: Path,
    *,
    max_validation_scans: int = 5,
    codelinter: Path = DEFAULT_CODELINTER,
    config: Path = DEFAULT_CONFIG,
    codelinter_identity: dict[str, Any] | None = None,
    experiment: str = "EXP-AGENT-10",
) -> dict[str, Any]:
    """Create a fresh gate and pin the scanner identity."""
    state_dir = state_dir.resolve()
    workspace = workspace.resolve()
    codelinter = codelinter.resolve()
    config = config.resolve()
    if not workspace.is_dir():
        raise ValueError(f"Workspace does not exist: {workspace}")
    if max_validation_scans < 1:
        raise ValueError("max_validation_scans must be positive")
    if not codelinter.is_file() or not config.is_file():
        raise ValueError("CodeLinter binary or configuration is missing")
    identity = codelinter_identity or verify_codelinter(codelinter)
    with locked_state(state_dir) as state_path:
        if state_path.exists():
            raise FileExistsError(f"Validation gate already exists: {state_path}")
        state = {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment,
            "created_at": utc_now(),
            "workspace": str(workspace),
            "codelinter": str(codelinter),
            "codelinter_identity": identity,
            "config": str(config),
            "config_sha256": sha256_file(config),
            "max_validation_scans": max_validation_scans,
            "validation_attempts_consumed": 0,
            "initial_scan": None,
            "validation_scans": [],
            "final_scan": None,
        }
        write_json(state_path, state)
    return state


def load_state(state_dir: Path) -> dict[str, Any]:
    state_path = state_dir.resolve() / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Validation gate is not initialized: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _execute_scan(
    command: list[str], report_path: Path, workspace: Path
) -> tuple[subprocess.CompletedProcess[str], float, list[dict[str, Any]], str]:
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    findings: list[dict[str, Any]] = []
    parse_error = ""
    if report_path.is_file():
        try:
            findings = normalize_findings(report_path, workspace)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            parse_error = str(error)
    else:
        parse_error = "CodeLinter did not produce a report"
    return result, elapsed, findings, parse_error


def run_scan(state_dir: Path, kind: str) -> dict[str, Any]:
    """Run one scan; reserve validation budget before executing CodeLinter."""
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown scan kind: {kind}")
    state_dir = state_dir.resolve()
    with locked_state(state_dir) as state_path:
        if not state_path.is_file():
            raise FileNotFoundError(f"Validation gate is not initialized: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if kind == "initial":
            if state["initial_scan"] is not None:
                raise RuntimeError("Initial scan has already been attempted")
            ordinal = 0
            slot = "initial_scan"
        elif kind == "final":
            if state["final_scan"] is not None:
                raise RuntimeError("Final evaluator-only scan has already been attempted")
            ordinal = 0
            slot = "final_scan"
        else:
            consumed = int(state["validation_attempts_consumed"])
            maximum = int(state["max_validation_scans"])
            if consumed >= maximum:
                raise RuntimeError(
                    f"Validation budget exhausted: {consumed}/{maximum} attempts consumed"
                )
            ordinal = consumed + 1
            state["validation_attempts_consumed"] = ordinal
            slot = "validation_scans"

        attempt_id = f"{kind}_{ordinal:02d}" if ordinal else kind
        attempt = {
            "attempt_id": attempt_id,
            "kind": kind,
            "budget_consumed": kind == "validation",
            "status": "started",
            "started_at": utc_now(),
            "completed_at": None,
        }
        if slot == "validation_scans":
            state[slot].append(attempt)
        else:
            state[slot] = attempt
        write_json(state_path, state)

    workspace = Path(state["workspace"])
    scan_dir = state_dir / "scans" / attempt_id
    scan_dir.mkdir(parents=True, exist_ok=False)
    report_path = scan_dir / "report.json"
    findings_path = scan_dir / "findings.json"
    stdout_path = scan_dir / "stdout.log"
    stderr_path = scan_dir / "stderr.log"
    command = [
        state["codelinter"],
        "--config",
        state["config"],
        "--format",
        "json",
        "--output",
        str(report_path),
        str(workspace),
    ]

    try:
        result, elapsed, findings, parse_error = _execute_scan(
            command, report_path, workspace
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        write_json(findings_path, findings)
        status = "scanned" if not parse_error else "scan_failed"
        details = {
            "status": status,
            "completed_at": utc_now(),
            "exit_code": result.returncode,
            "elapsed_seconds": elapsed,
            "finding_count": len(findings),
            "findings_path": str(findings_path),
            "findings_sha256": sha256_file(findings_path),
            "report_path": str(report_path) if report_path.is_file() else None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "parse_error": parse_error,
            "command": command,
        }
    except BaseException as error:
        details = {
            "status": "scan_failed",
            "completed_at": utc_now(),
            "exit_code": None,
            "elapsed_seconds": None,
            "finding_count": None,
            "findings_path": None,
            "findings_sha256": None,
            "report_path": None,
            "stdout_path": None,
            "stderr_path": None,
            "parse_error": f"{type(error).__name__}: {error}",
            "command": command,
        }

    with locked_state(state_dir) as state_path:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if slot == "validation_scans":
            target = next(
                item for item in state[slot] if item["attempt_id"] == attempt_id
            )
            target.update(details)
            completed = target
        else:
            state[slot].update(details)
            completed = state[slot]
        write_json(state_path, state)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--state-dir", type=Path, required=True)
    init_parser.add_argument("--workspace", type=Path, required=True)
    init_parser.add_argument("--max-validation-scans", type=int, default=5)
    init_parser.add_argument("--codelinter", type=Path, default=DEFAULT_CODELINTER)
    init_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--state-dir", type=Path, required=True)
    scan_parser.add_argument("--kind", choices=sorted(VALID_KINDS), required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "init":
        result = initialize_gate(
            args.state_dir,
            args.workspace,
            max_validation_scans=args.max_validation_scans,
            codelinter=args.codelinter,
            config=args.config,
        )
    elif args.command == "scan":
        result = run_scan(args.state_dir, args.kind)
    else:
        result = load_state(args.state_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
