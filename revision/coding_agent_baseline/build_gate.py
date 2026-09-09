#!/usr/bin/env python3
"""Common evaluator-side build gate for both EXP-AGENT-10 conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from preflight_builds import (
    PREFLIGHT_ROOT,
    build_command,
    build_environment,
    dependency_command,
    run_logged,
)
from validation_gate import sha256_file, write_json


ARKTS_DIAGNOSTIC_HOOK = Path(__file__).resolve().parent / "arkts_ignore_diagnostics.js"


DEPENDENCY_METADATA_NAMES = {
    "oh-package-lock.json5",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}


def _dependency_metadata_snapshot(workspace: Path) -> dict[str, bytes]:
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
        and path.name in DEPENDENCY_METADATA_NAMES
        and not any(
            part in {".git", ".hvigor", "build", "node_modules", "oh_modules"}
            for part in path.relative_to(workspace).parts
        )
    }


def _restore_dependency_metadata(
    workspace: Path, snapshot: dict[str, bytes]
) -> None:
    current = {
        path.relative_to(workspace).as_posix(): path
        for path in workspace.rglob("*")
        if path.is_file()
        and path.name in DEPENDENCY_METADATA_NAMES
        and not any(
            part in {".git", ".hvigor", "build", "node_modules", "oh_modules"}
            for part in path.relative_to(workspace).parts
        )
    }
    for relative, path in current.items():
        if relative not in snapshot:
            path.unlink()
    for relative, content in snapshot.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def load_preflight(
    project_name: str, preflight_root: Path = PREFLIGHT_ROOT
) -> tuple[Path, dict[str, Any]]:
    result_path = preflight_root.resolve() / project_name / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing frozen build preflight: {result_path}")
    return result_path, json.loads(result_path.read_text(encoding="utf-8"))


def _relevant_environment(env: dict[str, str]) -> dict[str, str]:
    keys = (
        "NODE_HOME",
        "DEVECO_NODE_HOME",
        "OHOS_BASE_SDK_HOME",
        "OHOS_SDK_HOME",
        "DEVECO_SDK_HOME",
        "HVIGOR_USER_HOME",
        "NODE_PATH",
        "npm_config_registry",
        "npm_config_@ohos:registry",
        "NODE_OPTIONS",
    )
    return {key: env[key] for key in keys if key in env}


def arkts_diagnostic_environment(env: dict[str, str]) -> dict[str, str]:
    """Enable Hvigor's compiler-supported ignoreWarning mode for full builds."""
    if not ARKTS_DIAGNOSTIC_HOOK.is_file():
        raise FileNotFoundError(f"Missing ArkTS diagnostic hook: {ARKTS_DIAGNOSTIC_HOOK}")
    configured = env.copy()
    require_option = f"--require={ARKTS_DIAGNOSTIC_HOOK}"
    existing = configured.get("NODE_OPTIONS", "").strip()
    configured["NODE_OPTIONS"] = f"{existing} {require_option}".strip()
    return configured


def prepare_build_gate(
    project_name: str,
    workspace: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 1800,
    preflight_root: Path = PREFLIGHT_ROOT,
) -> dict[str, Any]:
    """Prepare dependencies and recheck an initially buildable project."""
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path, preflight = load_preflight(project_name, preflight_root)
    record: dict[str, Any] = {
        "schema_version": 1,
        "project": project_name,
        "preflight_result": str(result_path),
        "preflight_result_sha256": sha256_file(result_path),
        "preflight_build_status": preflight.get("build_status"),
        "available": preflight.get("build_status") == "passed",
        "test_available": preflight.get("test_status") == "passed",
        "validation_scope": "statically validated only",
        "dependency_setup": None,
        "initial_build_check": None,
        "arkts_diagnostic_policy": {
            "mode": "compiler_ignoreWarning",
            "scope": "ArkTS checker diagnostics",
            "hook_path": str(ARKTS_DIAGNOSTIC_HOOK),
            "hook_sha256": sha256_file(ARKTS_DIAGNOSTIC_HOOK),
            "hook_target": "any loaded @ohos/hvigor-ohos-plugin ArkCompile module",
            "note": (
                "The compiler continues through ArkTS checker diagnostics; later "
                "compiler, module-resolution, packaging, and process failures remain fatal."
            ),
        },
    }
    record_path = output_dir / "setup.json"
    if not record["available"]:
        write_json(record_path, record)
        return record

    env = arkts_diagnostic_environment(build_environment(project_name))
    dependency = dependency_command(workspace, project_name)
    command = build_command(workspace, project_name)
    frozen_dependency = (preflight.get("dependency_install") or {}).get("command")
    frozen_command = (preflight.get("build") or {}).get("command")
    if dependency != frozen_dependency or command != frozen_command:
        raise RuntimeError(
            f"Build command drift for {project_name}: "
            f"dependency={dependency!r}/{frozen_dependency!r}, "
            f"build={command!r}/{frozen_command!r}"
        )
    record["environment"] = _relevant_environment(env)
    record["build_command"] = command
    metadata_snapshot = _dependency_metadata_snapshot(workspace)
    try:
        if dependency is not None:
            record["dependency_setup"] = run_logged(
                dependency,
                workspace,
                env,
                output_dir / "logs" / "dependency-install.log",
                min(timeout_seconds, 600),
            )
            if record["dependency_setup"]["exit_code"] != 0:
                write_json(record_path, record)
                raise RuntimeError(
                    f"Frozen dependency setup failed for {project_name}; "
                    f"see {record['dependency_setup']['log_path']}"
                )
        record["initial_build_check"] = run_logged(
            command,
            workspace,
            env,
            output_dir / "logs" / "initial-build.log",
            timeout_seconds,
        )
    finally:
        _restore_dependency_metadata(workspace, metadata_snapshot)
    initial = record["initial_build_check"]
    record["initial_build_passed"] = bool(
        initial and not initial["timed_out"] and initial["exit_code"] == 0
    )
    write_json(record_path, record)
    if not record["initial_build_passed"]:
        raise RuntimeError(
            f"Frozen initial build gate failed for {project_name}; "
            f"see {(initial or {}).get('log_path')}"
        )
    return record


def run_build_gate(
    project_name: str,
    workspace: Path,
    output_dir: Path,
    setup: dict[str, Any],
    *,
    label: str,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run one cleanly recorded evaluator build without changing scan budget."""
    if not setup.get("available"):
        return {
            "label": label,
            "status": "not_available",
            "command": None,
            "build": None,
        }
    command = build_command(workspace, project_name)
    if command != setup.get("build_command"):
        raise RuntimeError(f"Prepared build command changed for {project_name}")
    build = run_logged(
        command,
        workspace.resolve(),
        arkts_diagnostic_environment(build_environment(project_name)),
        output_dir.resolve() / f"{label}.log",
        timeout_seconds,
    )
    status = (
        "passed"
        if not build["timed_out"] and build["exit_code"] == 0
        else "failed"
    )
    record = {"label": label, "status": status, "command": command, "build": build}
    write_json(output_dir.resolve() / f"{label}.json", record)
    return record
