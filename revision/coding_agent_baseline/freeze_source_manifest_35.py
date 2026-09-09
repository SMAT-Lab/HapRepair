#!/usr/bin/env python3
"""Freeze a 35-project source manifest after GitCode source recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
BASE_MANIFEST = SCRIPT_DIR / "source_manifest.json"
OUTPUT_MANIFEST = SCRIPT_DIR / "source_manifest_35_gitcode.json"

RECOVERED: dict[str, dict[str, str]] = {
    "JS_dialog_box_static": {
        "repo_dir": "baseline_data/source_recovery/xts_acts_gitcode",
        "repo_url": "https://gitcode.com/openharmony/xts_acts.git",
        "commit": "0e1b66cb7dcfd4c77d3caab43bcbdb2c61a2ab3b",
        "subpath": "web/page_interaction/JS_dialog_box_static",
    },
    "ace_ets_module_navigation1": {
        "repo_dir": "baseline_data/source_recovery/xts_acts_gitcode",
        "repo_url": "https://gitcode.com/openharmony/xts_acts.git",
        "commit": "0e1b66cb7dcfd4c77d3caab43bcbdb2c61a2ab3b",
        "subpath": "arkui/ace_ets_module_ui/ace_ets_module_navigation/ace_ets_module_navigation1",
    },
    "ace_ets_module_nowear_waterflow": {
        "repo_dir": "baseline_data/source_recovery/xts_acts_gitcode",
        "repo_url": "https://gitcode.com/openharmony/xts_acts.git",
        "commit": "0e1b66cb7dcfd4c77d3caab43bcbdb2c61a2ab3b",
        "subpath": "arkui/ace_ets_module_ui/ace_ets_module_scroll/ace_ets_module_nowear_waterflow",
    },
    "ace_ets_module_router1": {
        "repo_dir": "baseline_data/source_recovery/xts_acts_gitcode",
        "repo_url": "https://gitcode.com/openharmony/xts_acts.git",
        "commit": "0e1b66cb7dcfd4c77d3caab43bcbdb2c61a2ab3b",
        "subpath": "arkui/ace_ets_module_ui/ace_ets_module_RouteManagement/ace_ets_module_router1",
    },
    "ace_ets_module_swiper": {
        "repo_dir": "baseline_data/source_recovery/xts_acts_gitcode",
        "repo_url": "https://gitcode.com/openharmony/xts_acts.git",
        "commit": "0e1b66cb7dcfd4c77d3caab43bcbdb2c61a2ab3b",
        "subpath": "arkui/ace_ets_module_ui/ace_ets_module_swiper",
    },
    "PullLinking": {
        "repo_dir": "baseline_data/source_recovery/applications_app_samples_gitcode",
        "repo_url": "https://gitcode.com/openharmony/applications_app_samples.git",
        "commit": "f10120f8787d0c5c48c81dab380b2d72f155e46e",
        "subpath": "code/DocsSample/Ability/PullLinking",
    },
    "audio_suite": {
        "repo_dir": "baseline_data/source_recovery/multimedia_audio_framework_gitcode",
        "repo_url": "https://gitcode.com/openharmony/multimedia_audio_framework.git",
        "commit": "12421e142788bfa77008a2c305d0b30b271fed78",
        "subpath": "test/demo/audio_suite",
    },
    "CanvasTest": {
        "repo_dir": "baseline_data/source_recovery/kmptpc_oh_render_gitcode",
        "repo_url": "https://gitcode.com/openharmony-sig/kmptpc_oh_render.git",
        "commit": "bab7df7818c3110903912fe8fbddeca461abe3f3",
        "subpath": "test/CanvasTest",
    },
    "flutter_embedding": {
        "repo_dir": "baseline_data/source_recovery/flutter_engine_gitcode",
        "repo_url": "https://gitcode.com/openharmony-tpc/flutter_engine.git",
        "commit": "586a3d4e88a3c01c09d749c744d25df9f75b7eb5",
        "subpath": "shell/platform/ohos/flutter_embedding",
    },
}


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_project(project: dict[str, Any]) -> dict[str, Any]:
    recovery = RECOVERED[project["name"]]
    repo = WORKSPACE_ROOT / recovery["repo_dir"]
    commit = run_git(repo, "rev-parse", "HEAD")
    if commit != recovery["commit"]:
        raise SystemExit(f"Commit drift for {project['name']}: {commit}")
    if run_git(repo, "status", "--porcelain"):
        raise SystemExit(f"Dirty recovery checkout for {project['name']}: {repo}")
    subpath = recovery["subpath"]
    tree_oid = run_git(repo, "rev-parse", f"HEAD:{subpath}")
    source_path = repo / subpath
    if not source_path.is_dir():
        raise SystemExit(f"Missing recovered source directory: {source_path}")
    return {
        **project,
        "status": "ready",
        "repo_url": recovery["repo_url"],
        "subpath": subpath,
        "commit": commit,
        "commit_date": run_git(repo, "show", "-s", "--format=%cI", "HEAD"),
        "tree_oid": tree_oid,
        "source_path": str(source_path),
        "source_provider": "GitCode",
        "source_recovery": "fresh_revised_snapshot",
    }


def main() -> None:
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    projects = []
    for project in base["projects"]:
        if project["name"] in RECOVERED:
            projects.append(recover_project(project))
        elif project["status"] == "ready":
            projects.append(
                {
                    **project,
                    "source_provider": "existing_upstream_checkout",
                    "source_recovery": "fresh_revised_snapshot",
                }
            )
        else:
            raise SystemExit(f"Unresolved project remains: {project['name']}")

    names = [project["name"] for project in projects]
    if len(projects) != 35 or len(set(names)) != 35:
        raise SystemExit(f"Expected 35 distinct projects, found {len(projects)}/{len(set(names))}")

    recovered_repositories: dict[tuple[str, str], dict[str, Any]] = {}
    for project in projects:
        if project["name"] not in RECOVERED:
            continue
        key = (project["repo_url"], project["commit"])
        recovered_repositories[key] = {
            "namespace": project["namespace"],
            "repo": project["repo"],
            "url": project["repo_url"],
            "local_path": str(Path(project["source_path"]).parents[len(Path(project["subpath"]).parts) - 1]),
            "status": "ready",
            "commit": project["commit"],
            "commit_date": project["commit_date"],
            "source_provider": "GitCode",
        }

    manifest = {
        "schema_version": 2,
        "experiment": "TOSEM-MAJOR-REVISION-RQ1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "35 fresh revised snapshots; not reconstructed historical server commits",
        "base_manifest": str(BASE_MANIFEST),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "requested_project_count": 35,
        "ready_project_count": 35,
        "recovered_project_count": len(RECOVERED),
        "recovered_repositories": sorted(
            recovered_repositories.values(), key=lambda item: (item["namespace"], item["repo"])
        ),
        "projects": sorted(projects, key=lambda item: item["name"]),
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT_MANIFEST}: 35/35 ready, sha256={sha256_file(OUTPUT_MANIFEST)}")


if __name__ == "__main__":
    main()
