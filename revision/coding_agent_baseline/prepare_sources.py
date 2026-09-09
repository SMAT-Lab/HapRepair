#!/usr/bin/env python3
"""Recover and pin sparse upstream copies of the 35 candidate projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
TARGETS_PATH = REPO_ROOT / "revision" / "target_projects_haprepair.json"
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_agent_10"
NAMESPACE_URLS = {
    "OpenHarmony": "https://github.com/openharmony/{repo}.git",
    "OpenHarmony-TPC": "https://gitee.com/openharmony-tpc/{repo}.git",
    "OpenHarmony-SIG": "https://gitee.com/openharmony-sig/{repo}.git",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str], cwd: Path | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=environment,
    )


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_source(target: dict[str, Any]) -> dict[str, Any]:
    parts = Path(target["root_path"]).parts
    try:
        marker = parts.index("repo_new")
    except ValueError as exc:
        raise ValueError(f"Unexpected historical root: {target['root_path']}") from exc
    namespace = parts[marker + 1]
    repo = parts[marker + 2]
    subpath = Path(*parts[marker + 3 :]).as_posix() if len(parts) > marker + 3 else ""
    if namespace not in NAMESPACE_URLS:
        raise ValueError(f"Unsupported namespace: {namespace}")
    return {
        **target,
        "namespace": namespace,
        "repo": repo,
        "subpath": subpath,
        "repo_url": NAMESPACE_URLS[namespace].format(repo=repo),
    }


def clone_repository(url: str, destination: Path) -> tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git",
        "clone",
        "--filter=blob:none",
        "--depth",
        "1",
        "--no-checkout",
        url,
        str(destination),
    ]
    try:
        result = run(command, timeout=90)
    except subprocess.TimeoutExpired:
        return False, "clone timed out after 90 seconds"
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail


def checkout_repository(repo_dir: Path, subpaths: list[str]) -> tuple[bool, str]:
    nonempty = sorted({path for path in subpaths if path})
    full_checkout = any(not path for path in subpaths)
    commands: list[list[str]] = []
    if not full_checkout:
        commands.extend(
            [
                ["sparse-checkout", "init", "--cone"],
                ["sparse-checkout", "set", *nonempty],
            ]
        )
    commands.append(["checkout", "--force", "HEAD"])
    output: list[str] = []
    for command in commands:
        result = git(repo_dir, *command)
        output.append((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            return False, "\n".join(part for part in output if part)
    return True, "\n".join(part for part in output if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--names",
        help="Optional comma-separated project names for a bounded recovery pilot.",
    )
    args = parser.parse_args()

    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    selected_names = set(args.names.split(",")) if args.names else None
    parsed = [parse_source(target) for target in targets]
    if selected_names is not None:
        parsed = [target for target in parsed if target["name"] in selected_names]
        unknown = selected_names - {target["name"] for target in parsed}
        if unknown:
            raise SystemExit(f"Unknown project names: {sorted(unknown)}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for target in parsed:
        grouped[(target["namespace"], target["repo"], target["repo_url"])].append(target)

    source_root = args.source_root.resolve()
    repos_root = source_root / "repos"
    project_results: list[dict[str, Any]] = []
    repository_results: list[dict[str, Any]] = []

    for (namespace, repo_name, url), projects in sorted(grouped.items()):
        repo_dir = repos_root / namespace / repo_name
        print(f"[source] {namespace}/{repo_name} ({len(projects)} project(s))", flush=True)
        clone_detail = "reused existing checkout"
        if repo_dir.exists() and git(repo_dir, "rev-parse", "HEAD").returncode != 0:
            failed_root = source_root / "failed_clones"
            failed_root.mkdir(parents=True, exist_ok=True)
            failed_path = failed_root / f"{namespace}__{repo_name}__{int(time.time())}"
            shutil.move(str(repo_dir), str(failed_path))
            clone_detail = f"moved incomplete checkout to {failed_path}"
        if not (repo_dir / ".git").is_dir():
            ok, clone_detail = clone_repository(url, repo_dir)
            if not ok:
                print(f"[failed] clone {url}: {clone_detail[-300:]}", flush=True)
                repository_results.append(
                    {
                        "namespace": namespace,
                        "repo": repo_name,
                        "url": url,
                        "status": "clone_failed",
                        "detail": clone_detail,
                    }
                )
                for project in projects:
                    project_results.append({**project, "status": "source_unavailable"})
                continue

        ok, checkout_detail = checkout_repository(repo_dir, [item["subpath"] for item in projects])
        commit_result = git(repo_dir, "rev-parse", "HEAD")
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else ""
        date_result = git(repo_dir, "show", "-s", "--format=%cI", "HEAD")
        commit_date = date_result.stdout.strip() if date_result.returncode == 0 else ""
        repository_results.append(
            {
                "namespace": namespace,
                "repo": repo_name,
                "url": url,
                "local_path": str(repo_dir),
                "status": "ready" if ok and commit else "checkout_failed",
                "commit": commit,
                "commit_date": commit_date,
                "detail": "\n".join(part for part in (clone_detail, checkout_detail) if part),
            }
        )

        for project in projects:
            project_path = repo_dir / project["subpath"] if project["subpath"] else repo_dir
            tree_spec = f"HEAD:{project['subpath']}" if project["subpath"] else "HEAD^{tree}"
            tree_result = git(repo_dir, "rev-parse", tree_spec)
            status = "ready" if ok and project_path.is_dir() and tree_result.returncode == 0 else "project_path_missing"
            project_results.append(
                {
                    **project,
                    "status": status,
                    "commit": commit,
                    "commit_date": commit_date,
                    "tree_oid": tree_result.stdout.strip() if tree_result.returncode == 0 else "",
                    "source_path": str(project_path),
                }
            )
            print(f"[{' ok ' if status == 'ready' else 'miss'}] {project['name']}: {project_path}", flush=True)

    manifest = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "created_at": utc_now(),
        "candidate_manifest": str(TARGETS_PATH),
        "candidate_manifest_sha256": sha256_file(TARGETS_PATH),
        "source_root": str(source_root),
        "requested_project_count": len(parsed),
        "ready_project_count": sum(item["status"] == "ready" for item in project_results),
        "repositories": repository_results,
        "projects": sorted(project_results, key=lambda item: item["name"]),
    }
    output_path = SCRIPT_DIR / "source_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {output_path} ({manifest['ready_project_count']}/{len(parsed)} ready)")


if __name__ == "__main__":
    main()
