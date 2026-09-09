#!/usr/bin/env python3
"""Verification-guided candidate lifecycle for the HapRepair Skill."""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import homecheck


SCRIPT_DIR = Path(__file__).resolve().parent
DECLARATION_EXTRACTOR = SCRIPT_DIR / "extract_declarations.js"
PUBLIC_API_EXTRACTOR = SCRIPT_DIR / "extract_public_api.js"
ARKTS_DIAGNOSTIC_HOOK = SCRIPT_DIR / "arkts_ignore_diagnostics.js"
SOURCE_SUFFIXES = {".ets", ".ts"}
EDITABLE_SUFFIXES = SOURCE_SUFFIXES | {
    ".json",
    ".json5",
    ".yaml",
    ".yml",
    ".toml",
    ".properties",
}
EXCLUDED_DIRS = {
    ".git",
    ".haprepair",
    ".exp_agent",
    ".hvigor",
    ".idea",
    "build",
    "node_modules",
    "oh_modules",
}
INVALID_BLOCK_MARKERS = (
    "defer",
    "later",
    "no source edit",
    "time limit",
    "out of time",
    "too many",
)
AGENT_MODE_ENV = "HAPREPAIR_AGENT_MODE"
CONTROLLER_OPERATIONS = frozenset(
    {
        "init-session",
        "scan-initial",
        "begin-round",
        "record-completion",
        "preflight-round",
        "validate-round",
        "finalize",
    }
)
STATE_DECORATOR_PATTERN = re.compile(
    r"@(State|Prop|Link|ObjectLink|Provide|Consume|StorageLink|StorageProp|LocalStorageLink|LocalStorageProp)\b"
)
NAMESPACE_EXPORT_PATTERN = re.compile(
    r"^\s*export\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+['\"]([^'\"]+)['\"]\s*;?\s*$"
)
INDEX_OF_PATTERN = re.compile(r"\.indexOf\s*\(")


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def session_path(state_dir: Path) -> Path:
    return state_dir.resolve() / "session.json"


@contextmanager
def locked_session(state_dir: Path) -> Iterator[Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "session.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield session_path(state_dir)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_session(state_dir: Path) -> dict[str, Any]:
    path = session_path(state_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Repair session is not initialized: {path}")
    return read_json(path)


def save_session(state_dir: Path, session: dict[str, Any]) -> None:
    session["updated_at"] = utc_now()
    with locked_session(state_dir) as path:
        write_json(path, session)


def parse_command(value: str | None) -> list[str] | None:
    return shlex.split(value) if value else None


def enforce_agent_boundary(operation: str) -> None:
    """Keep evaluator session mutations in the evaluator process only.

    The coding agent receives the same Skill files as the evaluator, so prompt
    instructions alone are not a sufficient isolation boundary.  The runner
    sets this environment variable only for agent turns; evaluator invocations
    intentionally leave it unset.
    """
    if os.getenv(AGENT_MODE_ENV) == "1" and operation in CONTROLLER_OPERATIONS:
        raise PermissionError(
            f"{operation} is evaluator-only; the repair agent may edit source and "
            "write the completion report, but must not mutate the session controller"
        )


def iter_editable_files(root: Path) -> Iterable[Path]:
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in EDITABLE_SUFFIXES:
                yield path


def source_fingerprint(workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in iter_editable_files(workspace):
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(workspace).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def snapshot_workspace(workspace: Path, destination: Path) -> dict[str, Any]:
    files_dir = destination / "files"
    files_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, str] = {}
    for source in iter_editable_files(workspace):
        relative = source.relative_to(workspace).as_posix()
        target = files_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative] = sha256_file(source)
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "files_dir": str(files_dir),
        "manifest_path": str(manifest_path),
        "file_count": len(manifest),
    }


def restore_snapshot(workspace: Path, snapshot: dict[str, Any]) -> None:
    manifest = read_json(Path(snapshot["manifest_path"]))
    current = {
        path.relative_to(workspace).as_posix(): path
        for path in iter_editable_files(workspace)
    }
    for relative, path in current.items():
        if relative not in manifest:
            path.unlink()
    files_dir = Path(snapshot["files_dir"])
    for relative, expected in manifest.items():
        source = files_dir / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Snapshot corruption detected: {source}")
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def source_diff(original: Path, repaired: Path, output_path: Path) -> dict[str, Any]:
    original_files = {
        path.relative_to(original).as_posix(): path
        for path in iter_editable_files(original)
    }
    repaired_files = {
        path.relative_to(repaired).as_posix(): path
        for path in iter_editable_files(repaired)
    }
    changed: list[str] = []
    deleted: list[str] = []
    added: list[str] = []
    large_deletions: list[dict[str, Any]] = []
    patch: list[str] = []
    for relative in sorted(original_files.keys() | repaired_files.keys()):
        old_path = original_files.get(relative)
        new_path = repaired_files.get(relative)
        old_lines = (
            old_path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
            if old_path
            else []
        )
        new_lines = (
            new_path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
            if new_path
            else []
        )
        if old_lines == new_lines:
            continue
        changed.append(relative)
        if old_path is None:
            added.append(relative)
        if new_path is None:
            deleted.append(relative)
        removed = max(len(old_lines) - len(new_lines), 0)
        if old_lines and removed >= 10 and removed / len(old_lines) >= 0.5:
            large_deletions.append(
                {
                    "relative_path": relative,
                    "old_lines": len(old_lines),
                    "new_lines": len(new_lines),
                }
            )
        patch.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(patch), encoding="utf-8")
    changed_source = [item for item in changed if Path(item).suffix in SOURCE_SUFFIXES]
    return {
        "changed_files": changed,
        "changed_file_count": len(changed),
        "changed_source_files": changed_source,
        "changed_source_file_count": len(changed_source),
        "added_source_files": [
            item for item in added if Path(item).suffix in SOURCE_SUFFIXES
        ],
        "deleted_source_files": [
            item for item in deleted if Path(item).suffix in SOURCE_SUFFIXES
        ],
        "large_deletion_flags": large_deletions,
        "patch_path": str(output_path),
        "patch_sha256": sha256_file(output_path),
    }


def run_node_extractor(extractor: Path, workspace: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["node", str(extractor), str(workspace.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{extractor.name} failed with exit {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def declaration_inventory(workspace: Path) -> dict[str, set[str]]:
    payload = run_node_extractor(DECLARATION_EXTRACTOR, workspace)
    return {relative: set(names) for relative, names in payload["declarations"].items()}


def public_api_inventory(workspace: Path) -> dict[str, Any]:
    return run_node_extractor(PUBLIC_API_EXTRACTOR, workspace)


def compare_public_api(
    baseline: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    baseline_api = baseline["api"]
    observed_api = observed["api"]
    changes = {
        "removed": sorted(set(baseline_api) - set(observed_api)),
        "added": sorted(set(observed_api) - set(baseline_api)),
        "changed": sorted(
            key
            for key in set(baseline_api) & set(observed_api)
            if baseline_api[key] != observed_api[key]
        ),
    }
    return {
        "status": "passed" if not any(changes.values()) else "failed",
        "changes": changes,
    }


def structural_guard(
    baseline_root: Path, workspace: Path, output_dir: Path
) -> dict[str, Any]:
    diff = source_diff(baseline_root, workspace, output_dir / "source_changes.patch")
    before = declaration_inventory(baseline_root)
    after = declaration_inventory(workspace)
    removed = {
        relative: sorted(names - after.get(relative, set()))
        for relative, names in before.items()
        if names - after.get(relative, set())
    }
    status = (
        "failed"
        if diff["deleted_source_files"] or diff["large_deletion_flags"] or removed
        else "passed"
    )
    return {"status": status, "removed_declarations": removed, "diff": diff}


def source_pattern_count(root: Path, pattern: re.Pattern[str]) -> int:
    count = 0
    for path in iter_editable_files(root):
        if path.suffix in SOURCE_SUFFIXES:
            count += len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return count


def namespace_export_inventory(root: Path) -> dict[str, list[str]]:
    """Record namespace-barrel exports whose identity is part of the API."""
    inventory: dict[str, list[str]] = {}
    for path in iter_editable_files(root):
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        records = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = NAMESPACE_EXPORT_PATTERN.match(line)
            if match:
                records.append(f"{match.group(1)}::{match.group(2)}")
        if records:
            inventory[path.relative_to(root).as_posix()] = sorted(records)
    return inventory


def namespace_export_guard(
    baseline_root: Path, workspace: Path
) -> dict[str, Any]:
    """Reject changes that remove or redirect baseline `export * as` namespaces.

    Declaration fingerprints do not encode whether a caller receives a namespace
    object or a collection of named bindings. Keep that export shape stable while
    repairing interacting cycle/export findings.
    """
    before = namespace_export_inventory(baseline_root)
    after = namespace_export_inventory(workspace)
    missing: dict[str, list[str]] = {}
    changed: dict[str, dict[str, list[str]]] = {}
    for relative, records in before.items():
        observed = after.get(relative, [])
        missing_records = sorted(set(records) - set(observed))
        if missing_records:
            missing[relative] = missing_records
        elif records != observed:
            changed[relative] = {"before": records, "after": observed}
    failures: list[str] = []
    if missing:
        failures.append(
            "namespace export identity changed; preserve every baseline `export * as` barrel"
        )
    if changed:
        failures.append(
            "namespace export targets changed; do not redirect or wrap baseline namespace exports"
        )
    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "baseline": before,
        "observed": after,
        "missing": missing,
        "changed": changed,
    }


def completion_evidence_text(report: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in report.get("entity_repairs") or []:
        if not isinstance(item, dict):
            continue
        parts.extend(str(value) for value in item.get("invariants") or [])
        parts.extend(str(value) for value in item.get("evidence") or [])
        parts.append(str(item.get("transformation", "")))
    return " ".join(parts).lower()


def semantic_guard(
    baseline_root: Path,
    workspace: Path,
    completion_report: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    before_index = source_pattern_count(baseline_root, INDEX_OF_PATTERN)
    after_index = source_pattern_count(workspace, INDEX_OF_PATTERN)
    if after_index > before_index:
        failures.append(
            "new indexOf call detected; never synthesize a collection index with indexOf"
        )

    evidence = completion_evidence_text(completion_report)
    before_reusable = source_pattern_count(baseline_root, re.compile(r"@Reusable\b"))
    after_reusable = source_pattern_count(workspace, re.compile(r"@Reusable\b"))
    if after_reusable > before_reusable and not all(
        term in evidence for term in ("lifecycle", "reuse", "identity")
    ):
        failures.append(
            "new @Reusable requires lifecycle, reuse-state, and identity evidence"
        )

    before_decorators = source_pattern_count(baseline_root, STATE_DECORATOR_PATTERN)
    after_decorators = source_pattern_count(workspace, STATE_DECORATOR_PATTERN)
    if before_decorators != after_decorators and not all(
        term in evidence for term in ("ownership", "mutation", "ui")
    ):
        failures.append(
            "state-decorator changes require ownership, mutation, and UI-read evidence"
        )
    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "counts": {
            "index_of_before": before_index,
            "index_of_after": after_index,
            "reusable_before": before_reusable,
            "reusable_after": after_reusable,
            "state_decorators_before": before_decorators,
            "state_decorators_after": after_decorators,
        },
    }


def validation_environment(ignore_arkts_diagnostics: bool) -> dict[str, str]:
    env = os.environ.copy()
    if not ignore_arkts_diagnostics:
        return env
    if not ARKTS_DIAGNOSTIC_HOOK.is_file():
        raise FileNotFoundError(
            f"Missing ArkTS diagnostic hook: {ARKTS_DIAGNOSTIC_HOOK}"
        )
    option = f"--require={ARKTS_DIAGNOSTIC_HOOK}"
    current = env.get("NODE_OPTIONS", "").strip()
    env["NODE_OPTIONS"] = f"{current} {option}".strip()
    return env


def run_validation_command(
    command: list[str] | None,
    workspace: Path,
    output_dir: Path,
    *,
    label: str,
    timeout_seconds: int,
    ignore_arkts_diagnostics: bool,
) -> dict[str, Any]:
    if command is None:
        return {"label": label, "status": "not_available", "command": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            env=validation_environment(ignore_arkts_diagnostics),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        status = "passed" if result.returncode == 0 else "failed"
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        status = "failed"
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        exit_code = None
        timed_out = True
    stdout_path = output_dir / f"{label}.stdout.log"
    stderr_path = output_dir / f"{label}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "label": label,
        "status": status,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def scan_findings(scan: dict[str, Any]) -> list[dict[str, Any]]:
    if scan.get("status") != "scanned" or not scan.get("findings_path"):
        raise RuntimeError("No successful HomeCheck findings are available")
    return read_json(Path(scan["findings_path"]))


def homecheck_state(session: dict[str, Any]) -> dict[str, Any]:
    return homecheck.load_state(Path(session["homecheck_state_dir"]))


def validation_budget(session: dict[str, Any]) -> dict[str, int]:
    state = homecheck_state(session)
    return {
        "consumed": int(state["validation_scans_consumed"]),
        "maximum": int(state["max_validation_scans"]),
    }


def current_plan(session: dict[str, Any], output: Path) -> dict[str, Any]:
    result = homecheck.make_plan(
        argparse.Namespace(
            state_dir=Path(session["homecheck_state_dir"]), output=output
        )
    )
    return read_json(Path(result["path"]))


def init_session(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Project workspace does not exist: {workspace}")
    if state_dir.is_relative_to(workspace):
        raise ValueError("Repair state directory must be outside the project workspace")
    if state_dir.exists() and any(state_dir.iterdir()):
        raise FileExistsError(f"Repair state directory is not empty: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)
    initial_snapshot = snapshot_workspace(
        workspace, state_dir / "snapshots" / "initial"
    )
    public_api = public_api_inventory(workspace)
    public_api_path = state_dir / "public_api_baseline.json"
    write_json(public_api_path, public_api)
    homecheck_result = homecheck.init_session(
        argparse.Namespace(
            workspace=workspace,
            state_dir=state_dir / "homecheck",
            max_validation_scans=args.max_validation_scans,
            codelinter=args.codelinter,
            config=args.config,
            overlay=args.overlay,
            allow_unverified_codelinter=args.allow_unverified_codelinter,
        )
    )
    session = {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "initialized",
        "workspace": str(workspace),
        "state_dir": str(state_dir),
        "homecheck_state_dir": str(state_dir / "homecheck"),
        "validation": {
            "build_command": parse_command(args.build_command),
            "test_command": parse_command(args.test_command),
            "timeout_seconds": args.command_timeout,
            "ignore_arkts_diagnostics": not args.strict_arkts_diagnostics,
        },
        "public_api_baseline": str(public_api_path),
        "public_api_baseline_sha256": sha256_file(public_api_path),
        "initial_snapshot": initial_snapshot,
        "last_valid_snapshot": initial_snapshot,
        "last_valid_round": 0,
        "best_valid_snapshot": initial_snapshot,
        "best_valid_round": 0,
        "best_valid_score": None,
        "best_valid_metrics": None,
        "active_round": None,
        "rounds": [],
        "rule_interactions": [],
        "final_candidate_selection": None,
        "final_scan": None,
    }
    save_session(state_dir, session)
    return {
        "operation": "init_session",
        "status": "initialized",
        "workspace": str(workspace),
        "state_dir": str(state_dir),
        "homecheck": homecheck_result,
        "build_gate": "configured" if args.build_command else "not_available",
        "test_gate": "configured" if args.test_command else "not_available",
    }


def scan_initial(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    if session["status"] != "initialized":
        raise RuntimeError(
            "Initial scan is only valid immediately after initialization"
        )
    scan = homecheck.scan_session(
        argparse.Namespace(
            state_dir=Path(session["homecheck_state_dir"]), kind="initial"
        )
    )
    if scan["status"] != "scanned":
        session["status"] = "scan_failed"
        save_session(state_dir, session)
        return {"operation": "scan_initial", **scan}
    count = int(scan["target_finding_count"])
    session["status"] = "ready"
    session["best_valid_score"] = [count, 0, 0, 0]
    session["best_valid_metrics"] = scan["metrics"]
    save_session(state_dir, session)
    plan = current_plan(session, state_dir / "plans" / "initial.json")
    return {
        **scan,
        "operation": "scan_initial",
        "plan_path": str(state_dir / "plans" / "initial.json"),
        "plan": plan,
    }


def begin_round(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    if session["status"] not in {"ready", "repair_required"}:
        raise RuntimeError(f"Cannot begin a round from status {session['status']}")
    if session["active_round"] is not None:
        raise RuntimeError("A repair round is already active")
    budget = validation_budget(session)
    if budget["consumed"] >= budget["maximum"]:
        raise RuntimeError(
            f"Validation budget exhausted: {budget['consumed']}/{budget['maximum']}"
        )
    round_number = budget["consumed"] + 1
    round_dir = state_dir / "rounds" / f"round_{round_number:02d}"
    if round_dir.exists():
        raise FileExistsError(f"Round directory already exists: {round_dir}")
    snapshot = snapshot_workspace(Path(session["workspace"]), round_dir / "snapshot")
    plan_path = round_dir / "plan.json"
    plan = current_plan(session, plan_path)
    mode = "gate_recovery" if session["status"] == "repair_required" else "repair"
    active = {
        "round": round_number,
        "mode": mode,
        "status": "editing",
        "started_at": utc_now(),
        "round_dir": str(round_dir),
        "snapshot": snapshot,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "input_scan": homecheck_state(session)["current_scan"],
        "completion": None,
        "preflight": None,
        "validation_attempts": [],
    }
    session["active_round"] = active
    session["status"] = "round_active"
    save_session(state_dir, session)
    return {
        "operation": "begin_round",
        "status": "editing",
        "round": round_number,
        "mode": mode,
        "plan_path": str(plan_path),
        "plan": plan,
        "validation_budget": budget,
    }


def audit_completion(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    required_rules = {item["rule"] for item in plan["rules"]}
    selected = {str(item) for item in report.get("selected_rules", [])}
    problems = []
    if not required_rules <= selected:
        problems.append(f"missing required rules: {sorted(required_rules - selected)}")
    if report.get("blocked"):
        problems.append("blocked entries are not accepted in the v8 all-target protocol")
    if report.get("unresolved_external"):
        problems.append(
            "unresolved_external entries are not repaired and cannot pass round coverage"
        )
    consulted = report.get("consulted_specs") or {}
    if not isinstance(consulted, dict):
        problems.append("consulted_specs must be a rule-to-path-list object")
        consulted = {}

    repairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in report.get("entity_repairs") or []:
        if not isinstance(item, dict):
            problems.append("entity_repairs entries must be objects")
            continue
        rule = str(item.get("rule", ""))
        relative_path = str(item.get("relative_path", ""))
        status = str(item.get("status", ""))
        if status != "repaired":
            problems.append(f"{rule} in {relative_path} is not marked repaired")
        for field in ("entities", "invariants", "evidence", "locations"):
            value = item.get(field)
            if not isinstance(value, list) or not value:
                problems.append(
                    f"{rule} in {relative_path} requires non-empty {field} evidence"
                )
        if not str(item.get("transformation", "")).strip():
            problems.append(f"{rule} in {relative_path} requires a transformation")
        repairs.setdefault((rule, relative_path), []).append(item)

    for rule_record in plan["rules"]:
        rule = rule_record["rule"]
        spec = rule_record["semantic_spec"]
        required_specs = {spec["core_spec_path"]}
        required_specs.update(family["spec_path"] for family in spec["families"])
        observed_specs = {
            str(path) for path in consulted.get(rule, [])
        } if isinstance(consulted.get(rule, []), list) else set()
        if spec["covered"] and required_specs != observed_specs:
            problems.append(
                f"{rule} must consult exact semantic specs: {sorted(required_specs)}"
            )
        for file_record in rule_record["files"]:
            relative_path = file_record["relative_path"]
            group = repairs.get((rule, relative_path), [])
            if not group:
                problems.append(f"{rule} has unrepaired file: {relative_path}")
                continue
            expected_locations = {
                (int(item["line"]), int(item["column"]))
                for item in file_record["locations"]
            }
            reported_locations = {
                (int(location.get("line", -1)), int(location.get("column", -1)))
                for item in group
                for location in item.get("locations", [])
                if isinstance(location, dict)
            }
            if expected_locations != reported_locations:
                problems.append(
                    f"{rule} in {relative_path} must account for exact frozen locations: "
                    f"missing={sorted(expected_locations - reported_locations)}, "
                    f"unknown={sorted(reported_locations - expected_locations)}"
                )
    return {
        "complete": not problems,
        "problems": problems,
        "feedback": (
            "Every frozen alert location has semantic repair evidence."
            if not problems
            else "Round target coverage is incomplete: " + "; ".join(problems)
        ),
    }


def record_completion(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    active = session.get("active_round")
    if active is None:
        raise RuntimeError("No active round")
    report = read_json(args.report.resolve())
    plan_path = Path(active["plan_path"])
    if sha256_file(plan_path) != active["plan_sha256"]:
        raise RuntimeError("Frozen round plan drifted")
    audit = audit_completion(read_json(plan_path), report)
    destination = Path(active["round_dir"]) / "completion.json"
    write_json(destination, report)
    record = {
        **audit,
        "report_path": str(destination),
        "report_sha256": sha256_file(destination),
    }
    active["completion"] = record
    if not audit["complete"]:
        active["status"] = "coverage_required"
    save_session(state_dir, session)
    return {"operation": "record_completion", "round": active["round"], **record}


def preflight_round(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    active = session.get("active_round")
    if active is None:
        raise RuntimeError("No active round")
    completion = active.get("completion") or {}
    if not completion.get("complete"):
        return {
            "operation": "preflight_round",
            "round": active["round"],
            "status": "coverage_required",
            "scan_consumed": False,
            "feedback": completion.get(
                "feedback", "Record complete rule/file coverage before preflight."
            ),
            "validation_budget": validation_budget(session),
        }
    workspace = Path(session["workspace"])
    attempt_number = len(active.setdefault("preflight_attempts", [])) + 1
    attempt_dir = (
        Path(active["round_dir"]) / "preflight" / f"attempt_{attempt_number:02d}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    edit = source_diff(
        Path(active["snapshot"]["files_dir"]),
        workspace,
        attempt_dir / "round_edits.patch",
    )
    if edit["changed_source_file_count"] == 0:
        result = {
            "operation": "preflight_round",
            "round": active["round"],
            "status": "editing_required",
            "scan_consumed": False,
            "edit_evidence": edit,
            "candidate_source_fingerprint": source_fingerprint(workspace),
            "feedback": "No ArkTS/TypeScript source edit exists in the active round.",
            "validation_budget": validation_budget(session),
        }
    else:
        structure = structural_guard(
            Path(session["last_valid_snapshot"]["files_dir"]),
            workspace,
            attempt_dir,
        )
        write_json(attempt_dir / "structural_guard.json", structure)
        if structure["status"] == "passed":
            observed_api = public_api_inventory(workspace)
            write_json(attempt_dir / "public_api_observed.json", observed_api)
            public_api = compare_public_api(
                read_json(Path(session["public_api_baseline"])), observed_api
            )
        else:
            public_api = {
                "status": "skipped_structural_guard_failed",
                "changes": {},
            }
        write_json(attempt_dir / "public_api_guard.json", public_api)
        completion_report = read_json(Path(completion["report_path"]))
        semantic = semantic_guard(
            Path(active["snapshot"]["files_dir"]),
            workspace,
            completion_report,
        )
        write_json(attempt_dir / "semantic_guard.json", semantic)
        namespace_exports = namespace_export_guard(
            Path(active["snapshot"]["files_dir"]), workspace
        )
        write_json(attempt_dir / "namespace_export_guard.json", namespace_exports)
        failures = []
        if structure["status"] != "passed":
            failures.append("source deletion or declaration-removal guard failed")
            removed = structure.get("removed_declarations") or {}
            if removed:
                details = "; ".join(
                    f"{relative}: {', '.join(names)}"
                    for relative, names in sorted(removed.items())
                )
                failures.append(
                    "RESTORE these removed declarations before any other repair: "
                    + details
                )
            deleted = structure.get("diff", {}).get("deleted_source_files") or []
            if deleted:
                failures.append("RESTORE deleted source files: " + ", ".join(deleted))
            large = structure.get("diff", {}).get("large_deletion_flags") or []
            if large:
                failures.append(
                    "RECHECK large deletions in: "
                    + ", ".join(item["relative_path"] for item in large)
                )
        if public_api["status"] == "failed":
            failures.append("package-exported public API guard failed")
        failures.extend(semantic["failures"])
        failures.extend(namespace_exports["failures"])
        result = {
            "operation": "preflight_round",
            "round": active["round"],
            "status": "repair_required" if failures else "preflight_passed",
            "scan_consumed": False,
            "gate_failures": failures,
            "structural_guard": structure,
            "public_api_guard": public_api,
            "semantic_guard": semantic,
            "namespace_export_guard": namespace_exports,
            "edit_evidence": edit,
            "candidate_source_fingerprint": source_fingerprint(workspace),
            "feedback": (
                "Repair the retained candidate in place before validation. "
                + " ".join(failures)
                if failures
                else "Structural and public-API preflight passed."
            ),
            "validation_budget": validation_budget(session),
        }
    active["preflight_attempts"].append(result)
    active["preflight"] = result
    active["status"] = result["status"]
    save_session(state_dir, session)
    return result


def interaction_record(
    round_number: int, deltas: dict[str, list[dict[str, str]]], provisional: bool
) -> dict[str, Any] | None:
    eliminated = Counter(item["rule"] for item in deltas["eliminated"])
    introduced = Counter(item["rule"] for item in deltas["introduced"])
    if not eliminated or not introduced:
        return None
    manifest = homecheck.validate_specs()
    aliases = manifest.get("aliases", {})
    family_by_rule = {
        rule: family["id"]
        for family in manifest["families"]
        for rule in family["rules"]
    }
    related = {
        (family["id"], other)
        for family in manifest["families"]
        for other in [family["id"], *family["interacts_with"]]
    }
    sibling_edges = []
    for source in deltas["eliminated"]:
        source_rule = aliases.get(source["rule"], source["rule"])
        source_family = family_by_rule.get(source_rule)
        for target in deltas["introduced"]:
            target_rule = aliases.get(target["rule"], target["rule"])
            target_family = family_by_rule.get(target_rule)
            if (
                source["relative_path"] == target["relative_path"]
                and source_family
                and (source_family, target_family) in related
            ):
                sibling_edges.append(
                    {
                        "relative_path": source["relative_path"],
                        "from_rule": source["rule"],
                        "to_rule": target["rule"],
                    }
                )
    unique_sibling_edges = [
        dict(zip(("relative_path", "from_rule", "to_rule"), edge))
        for edge in sorted(
            {
                (item["relative_path"], item["from_rule"], item["to_rule"])
                for item in sibling_edges
            }
        )
    ]
    return {
        "round": round_number,
        "provisional": provisional,
        "eliminated_rules": dict(sorted(eliminated.items())),
        "introduced_rules": dict(sorted(introduced.items())),
        "sibling_edges": unique_sibling_edges,
        "has_unresolved_sibling_exchange": bool(unique_sibling_edges),
        "interpretation": (
            "Same-file related-family exchanges remain unfinished and cannot promote "
            "the candidate to best-valid. Other associations are descriptive only."
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


def validate_round(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    active = session.get("active_round")
    if active is None:
        raise RuntimeError("No active round")
    workspace = Path(session["workspace"])
    preflight = active.get("preflight") or {}
    fingerprint = source_fingerprint(workspace)
    if (
        preflight.get("status") != "preflight_passed"
        or preflight.get("candidate_source_fingerprint") != fingerprint
    ):
        return {
            "operation": "validate_round",
            "round": active["round"],
            "status": "preflight_required",
            "candidate_retained": True,
            "scan_consumed": False,
            "feedback": "Run a fresh passing preflight for the current source candidate.",
            "validation_budget": validation_budget(session),
        }
    round_dir = Path(active["round_dir"])
    attempt_number = len(active.setdefault("validation_attempts", [])) + 1
    command_dir = round_dir / "commands" / f"attempt_{attempt_number:02d}"
    policy = session["validation"]
    build = run_validation_command(
        policy["build_command"],
        workspace,
        command_dir,
        label="build",
        timeout_seconds=policy["timeout_seconds"],
        ignore_arkts_diagnostics=policy["ignore_arkts_diagnostics"],
    )
    test = (
        run_validation_command(
            policy["test_command"],
            workspace,
            command_dir,
            label="test",
            timeout_seconds=policy["timeout_seconds"],
            ignore_arkts_diagnostics=policy["ignore_arkts_diagnostics"],
        )
        if build["status"] in {"passed", "not_available"}
        else {
            "label": "test",
            "status": "skipped_build_failed",
            "command": policy["test_command"],
        }
    )
    gate_failures = []
    if build["status"] == "failed":
        gate_failures.append("configured build command failed")
    if test["status"] == "failed":
        gate_failures.append("configured test command failed")
    command_attempt = {
        "attempt": attempt_number,
        "completed_at": utc_now(),
        "candidate_source_fingerprint": fingerprint,
        "build_gate": build,
        "test_gate": test,
        "gate_failures": gate_failures,
    }
    active["validation_attempts"].append(command_attempt)
    if gate_failures:
        active["status"] = "repair_required"
        active["preflight"] = None
        session["status"] = "round_active"
        save_session(state_dir, session)
        return {
            "operation": "validate_round",
            "round": active["round"],
            "status": "repair_required",
            "candidate_retained": True,
            "scan_consumed": False,
            "gate_failures": gate_failures,
            "build_gate": build["status"],
            "test_gate": test["status"],
            "feedback": (
                "Repair the retained candidate in place, then rerun preflight and "
                "validation in this same round."
            ),
            "validation_budget": validation_budget(session),
        }
    validation = homecheck.scan_session(
        argparse.Namespace(
            state_dir=Path(session["homecheck_state_dir"]), kind="validation"
        )
    )
    if validation["status"] != "scanned":
        gate_failures.append("HomeCheck validation scan failed")
        round_metrics = None
        total_metrics = None
        round_deltas = {"eliminated": [], "remaining": [], "introduced": []}
    else:
        input_findings = scan_findings(active["input_scan"])
        current_findings = scan_findings(validation)
        round_metrics, round_deltas = homecheck.alert_metrics(
            input_findings, current_findings
        )
        total_metrics = validation["metrics"]
    repair_required = bool(gate_failures)
    total_edit = source_diff(
        Path(session["initial_snapshot"]["files_dir"]),
        workspace,
        round_dir / "candidate_total.patch",
    )
    score = (
        candidate_score(
            total_metrics, total_edit["changed_source_file_count"], active["round"]
        )
        if total_metrics is not None
        else None
    )
    interaction = interaction_record(
        active["round"], round_deltas, provisional=repair_required
    )
    deltas_path = round_dir / "round_alert_deltas.json"
    write_json(deltas_path, round_deltas)
    record = {
        **active,
        "status": "repair_required" if repair_required else "accepted",
        "completed_at": utc_now(),
        "build_gate": build,
        "test_gate": test,
        "gate_failures": gate_failures,
        "validation_scan": validation,
        "round_metrics": round_metrics,
        "total_metrics": total_metrics,
        "candidate_score": score,
        "candidate_total_edit_evidence": total_edit,
        "rule_interaction": interaction,
        "round_alert_deltas_path": str(deltas_path),
        "candidate_retained": repair_required,
    }
    best_updated = False
    if not repair_required:
        accepted = snapshot_workspace(
            workspace,
            state_dir / "snapshots" / f"accepted_round_{active['round']:02d}",
        )
        session["last_valid_snapshot"] = accepted
        session["last_valid_round"] = active["round"]
        unresolved_exchange = bool(
            interaction and interaction["has_unresolved_sibling_exchange"]
        )
        if (
            not unresolved_exchange
            and (session["best_valid_score"] is None or score < session["best_valid_score"])
        ):
            session["best_valid_snapshot"] = accepted
            session["best_valid_round"] = active["round"]
            session["best_valid_score"] = score
            session["best_valid_metrics"] = total_metrics
            best_updated = True
        if interaction is not None:
            interaction["provisional"] = False
            session["rule_interactions"].append(interaction)
    record["best_valid_updated"] = best_updated
    record["best_valid_round"] = session["best_valid_round"]
    record["best_valid_score"] = session["best_valid_score"]
    record["feedback"] = (
        "Candidate retained for in-place gate recovery; do not restore the original project."
        if repair_required
        else (
            "Round retained but a sibling-rule exchange remains unfinished; repair it "
            "in the next joint semantic-entity round."
            if interaction and interaction["has_unresolved_sibling_exchange"]
            else "Round accepted; repair every residual and introduced finding in the next round."
        )
    )
    write_json(round_dir / "round.json", record)
    session["rounds"].append(record)
    session["active_round"] = None
    session["status"] = "repair_required" if repair_required else "ready"
    save_session(state_dir, session)
    return {
        "operation": "validate_round",
        "round": active["round"],
        "status": record["status"],
        "candidate_retained": repair_required,
        "gate_failures": gate_failures,
        "build_gate": build["status"],
        "test_gate": test["status"],
        "round_metrics": round_metrics,
        "total_metrics": total_metrics,
        "rule_interaction": interaction,
        "best_valid_updated": best_updated,
        "best_valid_round": session["best_valid_round"],
        "best_valid_score": session["best_valid_score"],
        "remaining_findings_path": validation.get("findings_path"),
        "feedback": record["feedback"],
        "validation_budget": validation_budget(session),
        "round_record": str(round_dir / "round.json"),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    session = load_session(state_dir)
    if session["active_round"] is not None:
        raise RuntimeError("Finish or abandon the active round before finalization")
    if session["status"] == "completed":
        raise RuntimeError("Session is already finalized")
    workspace = Path(session["workspace"])
    current_invalid = session["status"] == "repair_required"
    latest_is_best = (
        not current_invalid
        and session["last_valid_round"] == session["best_valid_round"]
    )
    if current_invalid or not latest_is_best:
        archive = snapshot_workspace(
            workspace, state_dir / "snapshots" / "abandoned_final_candidate"
        )
        restore_snapshot(workspace, session["best_valid_snapshot"])
        selection = {
            "status": "restored_best_valid_candidate",
            "archived_candidate": archive,
            "latest_valid_round": session["last_valid_round"],
            "selected_round": session["best_valid_round"],
            "selected_score": session["best_valid_score"],
            "current_candidate_was_invalid": current_invalid,
        }
    else:
        selection = {
            "status": "latest_candidate_is_best_valid",
            "latest_valid_round": session["last_valid_round"],
            "selected_round": session["best_valid_round"],
            "selected_score": session["best_valid_score"],
            "current_candidate_was_invalid": False,
        }
    final_scan = homecheck.scan_session(
        argparse.Namespace(state_dir=Path(session["homecheck_state_dir"]), kind="final")
    )
    session["status"] = "completed"
    session["final_candidate_selection"] = selection
    session["final_scan"] = final_scan
    save_session(state_dir, session)
    return {
        "operation": "finalize",
        "status": "completed",
        "final_candidate_selection": selection,
        "final_scan": final_scan,
        "validation_budget": validation_budget(session),
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    session = load_session(args.state_dir.resolve())
    scanner = homecheck_state(session)
    current = scanner.get("current_scan")
    current_count = (
        len(scan_findings(current))
        if current and current.get("status") == "scanned"
        else None
    )
    return {
        "operation": "status",
        "status": session["status"],
        "workspace": session["workspace"],
        "current_finding_count": current_count,
        "completed_rounds": len(session["rounds"]),
        "active_round": session["active_round"],
        "last_valid_round": session["last_valid_round"],
        "best_valid_round": session["best_valid_round"],
        "best_valid_score": session["best_valid_score"],
        "rule_interactions": session["rule_interactions"],
        "validation_budget": validation_budget(session),
        "final_candidate_selection": session["final_candidate_selection"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    init = subparsers.add_parser("init-session")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--state-dir", type=Path, required=True)
    init.add_argument("--max-validation-scans", type=int, default=5)
    init.add_argument("--codelinter", type=Path)
    init.add_argument("--config", type=Path, default=homecheck.DEFAULT_CONFIG)
    init.add_argument("--overlay", type=Path, default=homecheck.DEFAULT_OVERLAY)
    init.add_argument("--allow-unverified-codelinter", action="store_true")
    init.add_argument("--build-command")
    init.add_argument("--test-command")
    init.add_argument("--command-timeout", type=int, default=1800)
    init.add_argument("--strict-arkts-diagnostics", action="store_true")
    init.set_defaults(handler=init_session)

    initial = subparsers.add_parser("scan-initial")
    initial.add_argument("--state-dir", type=Path, required=True)
    initial.set_defaults(handler=scan_initial)

    begin = subparsers.add_parser("begin-round")
    begin.add_argument("--state-dir", type=Path, required=True)
    begin.set_defaults(handler=begin_round)

    completion = subparsers.add_parser("record-completion")
    completion.add_argument("--state-dir", type=Path, required=True)
    completion.add_argument("--report", type=Path, required=True)
    completion.set_defaults(handler=record_completion)

    preflight = subparsers.add_parser("preflight-round")
    preflight.add_argument("--state-dir", type=Path, required=True)
    preflight.set_defaults(handler=preflight_round)

    validate = subparsers.add_parser("validate-round")
    validate.add_argument("--state-dir", type=Path, required=True)
    validate.set_defaults(handler=validate_round)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--state-dir", type=Path, required=True)
    finish.set_defaults(handler=finalize)

    state = subparsers.add_parser("status")
    state.add_argument("--state-dir", type=Path, required=True)
    state.set_defaults(handler=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        enforce_agent_boundary(args.operation)
        result = args.handler(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "operation": args.operation,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
