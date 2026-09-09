#!/usr/bin/env python3
"""Freeze evaluator-side build availability for EXP-AGENT-10 projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SELECTED_PROJECTS = HERE / "selected_projects.json"
PREFLIGHT_ROOT = Path(
    "/home/zhihao/hdd/haprepair/baseline_data/exp_agent_10/build_preflight"
)
OHOS_SDK_HOME = Path(
    "/home/zhihao/hdd/haprepair/baseline_data/openharmony_sdk/openharmony"
)
LEGACY_API8_SDK_HOME = Path(
    "/home/zhihao/hdd/haprepair/baseline_data/openharmony_sdk/legacy_api8"
)
DEVECO_ROOT = Path("/home/zhihao/deveco/command-line-tools")
NODE_HOME = DEVECO_ROOT / "tool/node"
OHPM = DEVECO_ROOT / "bin/ohpm"
GLOBAL_HVIGOR = DEVECO_ROOT / "bin/hvigorw"
DEVECO_SDK_HOME = DEVECO_ROOT / "sdk"
HVIGOR_4_0_5_CACHE = (
    PREFLIGHT_ROOT
    / ".hvigor/project_caches/5efa8655cec37f88532ff7955e27664d/workspace"
    / "node_modules"
)
HVIGOR_4_0_5 = HVIGOR_4_0_5_CACHE / "@ohos/hvigor/bin/hvigor-simple.js"

DECLARED_SDKS = {
    "TextComponentTest": "OpenHarmony API 18",
    "ace_ets_module_swiper_api11": "OpenHarmony API 20",
    "acts_validator": "OpenHarmony API 20",
    "applications_photos": "OpenHarmony API 20",
    "bluetoothtest": "OpenHarmony API 26.0.0",
    "wifi_testapp": "OpenHarmony API 23",
    "HealthyPotAssistant": "OpenHarmony API 8",
    "asn1_ber": "OpenHarmony API 12",
    "applications_permission_manager": "OpenHarmony API 20",
    "ohos_cordova": "HarmonyOS 5.0.0 (compatible API 12; compile API implicit)",
}

GLOBAL_HVIGOR_PROJECTS: set[str] = set()

HVIGOR_4_0_5_PROJECTS = {
    "ace_ets_module_swiper_api11",
    "acts_validator",
}

COPY_IGNORES = shutil.ignore_patterns(
    ".git", ".hvigor", "build", "node_modules", "oh_modules"
)


def prepare_legacy_api8_sdk() -> None:
    for component in ("ets", "js", "toolchains"):
        source = OHOS_SDK_HOME / "8" / component
        metadata = json.loads((source / "oh-uni-package.json").read_text("utf-8"))
        destination = LEGACY_API8_SDK_HOME / component / metadata["version"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"Incorrect API 8 SDK link: {destination}")
            continue
        if destination.exists():
            raise RuntimeError(f"Refusing to replace API 8 SDK path: {destination}")
        destination.symlink_to(source, target_is_directory=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".git", ".hvigor", "build", "node_modules", "oh_modules"}
               for part in relative.parts):
            continue
        files.append((relative.as_posix(), path.stat().st_size, sha256_file(path)))
    digest = hashlib.sha256()
    for relative, size, checksum in sorted(files):
        digest.update(f"{relative}\0{size}\0{checksum}\n".encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def count_tests(root: Path) -> dict[str, int]:
    counts = {"local_test_files": 0, "device_test_files": 0}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".ets", ".ts", ".js"}:
            continue
        parts = path.relative_to(root).parts
        if "test" in parts:
            counts["local_test_files"] += 1
        if "ohosTest" in parts:
            counts["device_test_files"] += 1
    return counts


def run_logged(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    start = time.monotonic()
    timed_out = False
    return_code: int | None = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.monotonic() - start,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": return_code,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def dependency_command(workspace: Path, project_name: str) -> list[str] | None:
    if (workspace / "oh-package.json5").is_file():
        return [str(OHPM), "install", "--all", "--no-save"]
    if (workspace / "package.json").is_file():
        return [str(NODE_HOME / "bin/npm"), "install", "--no-audit", "--no-fund"]
    return None


def build_command(workspace: Path, project_name: str) -> list[str]:
    if project_name in HVIGOR_4_0_5_PROJECTS:
        if not HVIGOR_4_0_5.is_file():
            raise FileNotFoundError(f"Missing frozen Hvigor 4.0.5: {HVIGOR_4_0_5}")
        return [
            str(NODE_HOME / "bin/node"),
            str(HVIGOR_4_0_5),
            "assembleHap",
            "--no-daemon",
        ]
    if (
        project_name not in GLOBAL_HVIGOR_PROJECTS
        and (workspace / "hvigorw").is_file()
    ):
        return ["bash", "hvigorw", "assembleHap", "--no-daemon"]
    old_hvigor = workspace / "node_modules/@ohos/hvigor/bin/hvigor.js"
    if project_name == "HealthyPotAssistant" and old_hvigor.is_file():
        return [str(NODE_HOME / "bin/node"), str(old_hvigor), "assembleHap"]
    command = [str(GLOBAL_HVIGOR), "assembleHap", "--no-daemon"]
    if project_name == "ohos_cordova":
        command.extend(["-c", "properties.enableSignTask=false"])
    return command


def build_environment(project_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NODE_HOME": str(NODE_HOME),
            "DEVECO_NODE_HOME": str(NODE_HOME),
            "OHOS_BASE_SDK_HOME": str(OHOS_SDK_HOME),
            "OHOS_SDK_HOME": str(OHOS_SDK_HOME),
            "DEVECO_SDK_HOME": str(DEVECO_SDK_HOME),
            "HVIGOR_USER_HOME": str(PREFLIGHT_ROOT / ".hvigor"),
            "NODE_PATH": str(HVIGOR_4_0_5_CACHE),
            "npm_config_registry": "https://registry.npmmirror.com/",
            "npm_config_@ohos:registry": "https://repo.harmonyos.com/npm/",
            "PATH": f"{NODE_HOME / 'bin'}:{DEVECO_ROOT / 'bin'}:{env.get('PATH', '')}",
        }
    )
    if project_name == "HealthyPotAssistant":
        prepare_legacy_api8_sdk()
        env["OHOS_SDK_HOME"] = str(LEGACY_API8_SDK_HOME)
    return env


def preflight_project(
    project: dict[str, Any], timeout_seconds: int, refresh: bool
) -> dict[str, Any]:
    name = project["name"]
    project_root = PREFLIGHT_ROOT / name
    workspace = project_root / "workspace"
    result_path = project_root / "result.json"
    if result_path.is_file() and not refresh:
        return json.loads(result_path.read_text(encoding="utf-8"))

    if workspace.exists():
        shutil.rmtree(workspace)
    source_path = Path(project["source_path"])
    shutil.copytree(source_path, workspace, ignore=COPY_IGNORES, symlinks=True)
    source_before = source_inventory(workspace)
    env = build_environment(name)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "project": name,
        "commit": project["commit"],
        "tree_oid": project["tree_oid"],
        "source_path": str(source_path),
        "workspace": str(workspace),
        "declared_sdk": DECLARED_SDKS[name],
        "source_inventory_before": source_before,
        "test_inventory": count_tests(workspace),
        "dependency_install": None,
        "build": None,
        "build_status": "not_run",
        "test_status": "not_run",
        "test_command": None,
        "test_scope_note": (
            "Device-side ohosTest suites are not runnable in the evaluator's "
            "headless SDK environment; local test execution is frozen separately "
            "only if a project exposes a passing host-side task."
        ),
    }

    dependency = dependency_command(workspace, name)
    if dependency is not None:
        result["dependency_install"] = run_logged(
            dependency,
            workspace,
            env,
            project_root / "logs/dependency-install.log",
            min(timeout_seconds, 600),
        )
        if result["dependency_install"]["exit_code"] != 0:
            result["build_status"] = "dependency_install_failed"
            result["source_inventory_after"] = source_inventory(workspace)
            write_json(result_path, result)
            return result

    command = build_command(workspace, name)
    result["build"] = run_logged(
        command,
        workspace,
        env,
        project_root / "logs/build.log",
        timeout_seconds,
    )
    if result["build"]["timed_out"]:
        result["build_status"] = "timed_out"
    elif result["build"]["exit_code"] == 0:
        result["build_status"] = "passed"
    else:
        result["build_status"] = "failed"
    result["source_inventory_after"] = source_inventory(workspace)
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", dest="projects")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    selected = json.loads(SELECTED_PROJECTS.read_text(encoding="utf-8"))
    projects = selected["projects"]
    if args.projects:
        requested = set(args.projects)
        unknown = requested - {project["name"] for project in projects}
        if unknown:
            raise SystemExit(f"Unknown selected projects: {sorted(unknown)}")
        projects = [project for project in projects if project["name"] in requested]

    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for project in projects:
        print(f"[preflight] {project['name']}", flush=True)
        result = preflight_project(project, args.timeout, args.refresh)
        results.append(result)
        print(
            f"[{result['build_status']}] {project['name']} "
            f"{(result.get('build') or {}).get('duration_seconds', 0):.1f}s",
            flush=True,
        )
        write_json(
            PREFLIGHT_ROOT / "manifest.json",
            {
                "schema_version": 1,
                "experiment": "EXP-AGENT-10",
                "updated_at": utc_now(),
                "timeout_seconds": args.timeout,
                "sdk_manifest": (
                    "/home/zhihao/hdd/haprepair/baseline_data/openharmony_sdk/"
                    "install_manifest.json"
                ),
                "results": results,
            },
        )


if __name__ == "__main__":
    main()
