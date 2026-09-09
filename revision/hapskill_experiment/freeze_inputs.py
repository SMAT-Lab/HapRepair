#!/usr/bin/env python3
"""Freeze fresh target findings for the ten pinned EXP-HAPSKILL-10 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
SKILL_SCRIPT = HAPREPAIR_ROOT / "skills" / "haprepair-openharmony-repair" / "scripts" / "haprepair_skill.py"
DEFAULT_SELECTION = HERE.parent / "coding_agent_baseline" / "selected_projects.json"
DEFAULT_OUTPUT = HERE / "formal_inputs.json"
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "input_freeze" / "formal_01"
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_agent_baseline import tree_manifest  # type: ignore  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_git_identity(project: dict[str, Any]) -> dict[str, Any]:
    source = Path(project["source_path"]).resolve()
    subpath = Path(project.get("subpath") or "")
    repository = source
    for _ in subpath.parts:
        repository = repository.parent
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tree_spec = f"HEAD:{subpath.as_posix()}" if subpath.parts else "HEAD^{tree}"
    tree_oid = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", tree_spec],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != project["commit"] or tree_oid != project["tree_oid"]:
        raise RuntimeError(
            f"Pinned Git identity drift for {project['name']}: "
            f"HEAD {head}/{project['commit']}, tree {tree_oid}/{project['tree_oid']}"
        )
    return {"repository": str(repository), "head": head, "tree_oid": tree_oid}


def run_skill(*arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(payload.get("error", "HapRepair operation failed"))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    output_path = args.output.resolve()
    artifact_root = args.artifact_root.resolve()
    if output_path.exists() or artifact_root.exists():
        raise SystemExit("refusing to overwrite an existing formal input freeze")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    artifact_root.mkdir(parents=True)
    projects = []
    for project in selection["projects"]:
        name = project["name"]
        git_identity = verify_git_identity(project)
        input_tree = tree_manifest(Path(project["source_path"]).resolve())
        state = artifact_root / "state" / name
        run_skill(
            "init-session",
            "--workspace",
            str(Path(project["source_path"]).resolve()),
            "--state-dir",
            str(state),
            "--max-validation-scans",
            "5",
        )
        scan = run_skill("scan-project", "--state-dir", str(state), "--kind", "initial")
        frozen_path = artifact_root / "findings" / f"{name}.json"
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scan["target_findings_path"], frozen_path)
        enriched = dict(project)
        enriched["frozen_target_findings_path"] = str(frozen_path)
        enriched["frozen_target_findings_sha256"] = sha256_file(frozen_path)
        enriched["frozen_target_finding_count"] = scan["findings"]["finding_count"]
        enriched["frozen_raw_finding_count"] = scan["raw_finding_count"]
        enriched["frozen_excluded_non_target_finding_count"] = scan[
            "excluded_non_target_finding_count"
        ]
        enriched["frozen_input_file_count"] = len(input_tree)
        enriched["frozen_input_tree_sha256"] = sha256_tree(input_tree)
        enriched["verified_git_identity"] = git_identity
        projects.append(enriched)
        print(f"[frozen] {name}: {enriched['frozen_target_finding_count']} target alerts", flush=True)
    payload = {
        "schema_version": 1,
        "experiment": "EXP-HAPSKILL-10",
        "status": "frozen",
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "projects": projects,
    }
    write_json(output_path, payload)
    write_json(artifact_root / "manifest.json", payload)
    print(output_path)


if __name__ == "__main__":
    main()
