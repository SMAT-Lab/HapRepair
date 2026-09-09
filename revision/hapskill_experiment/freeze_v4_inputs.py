#!/usr/bin/env python3
"""Freeze the 35-project Skill inputs and unchanged 10-project reference subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
DEFAULT_SOURCE = BASELINE_DIR / "source_manifest_35_gitcode.json"
DEFAULT_SCAN = (
    BASELINE_DIR
    / "scan_runs"
    / "candidate_scan_35_gitcode_01"
    / "scan_manifest.json"
)
DEFAULT_REFERENCE = BASELINE_DIR / "selected_projects.json"
DEFAULT_OUTPUT_35 = HERE / "formal_inputs_hapskill_35_v4.json"
DEFAULT_OUTPUT_10 = HERE / "formal_inputs_agent_ref_10_v4.json"
DEFAULT_ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "input_freeze" / "v4_formal_01"
)
PREFLIGHT_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_agent_10" / "build_preflight"
TARGET_PREFIXES = ("@performance/", "@security/", "@hw-ets-eslint/")

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_agent_baseline import tree_manifest  # type: ignore  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode("utf-8"))
    return digest.hexdigest()


def git_identity(project: dict[str, Any]) -> dict[str, str]:
    source = Path(project["source_path"]).resolve()
    repository = Path(
        subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    relative = source.relative_to(repository).as_posix()
    tree_spec = "HEAD^{tree}" if relative == "." else f"HEAD:{relative}"
    tree_oid = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", tree_spec],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != project["commit"] or tree_oid != project["tree_oid"]:
        raise RuntimeError(
            f"Git identity drift for {project['name']}: "
            f"HEAD={head}/{project['commit']} tree={tree_oid}/{project['tree_oid']}"
        )
    return {
        "repository": str(repository),
        "head": head,
        "tree_oid": tree_oid,
        "source_relative_path": relative,
    }


def validation_availability(project_name: str) -> dict[str, Any]:
    result_path = PREFLIGHT_ROOT / project_name / "result.json"
    if not result_path.is_file():
        return {
            "build": "not_configured",
            "test": "not_configured",
            "scope": "statically validated only",
            "preflight_result": None,
            "preflight_result_sha256": None,
            "arkts_diagnostic_policy": "not_applicable_without_build_gate",
            "note": "No frozen evaluator build/test preflight exists for this project.",
        }
    result = read_json(result_path)
    build = "available" if result.get("build_status") == "passed" else "unavailable"
    test = "available" if result.get("test_status") == "passed" else "unavailable"
    return {
        "build": build,
        "test": test,
        "scope": (
            "build/test and static validation" if build == "available" else "statically validated only"
        ),
        "preflight_result": str(result_path.resolve()),
        "preflight_result_sha256": sha256_file(result_path),
        "preflight_build_status": result.get("build_status"),
        "preflight_test_status": result.get("test_status"),
        "arkts_diagnostic_policy": (
            "compiler_ignoreWarning_for_ArkTS_checker_diagnostics"
            if build == "available"
            else "not_applicable_without_build_gate"
        ),
    }


def freeze_project(
    source: dict[str, Any], scan: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    if scan.get("status") != "scanned" or scan.get("parse_error"):
        raise RuntimeError(f"Invalid scan record for {source['name']}")
    source_path = Path(source["source_path"]).resolve()
    if not source_path.is_dir() or source_path != Path(scan["source_path"]).resolve():
        raise RuntimeError(f"Source path mismatch for {source['name']}")
    for key in ("commit", "tree_oid"):
        if source[key] != scan[key]:
            raise RuntimeError(f"{key} mismatch for {source['name']}")

    findings_source = Path(scan["findings_path"]).resolve()
    if sha256_file(findings_source) != scan["findings_sha256"]:
        raise RuntimeError(f"Finding artifact hash drift for {source['name']}")
    raw_findings = read_json(findings_source)
    target_findings = [
        item
        for item in raw_findings
        if str(item.get("rule", "")).startswith(TARGET_PREFIXES)
    ]
    frozen_path = artifact_root / "findings" / f"{source['name']}.json"
    write_json(frozen_path, target_findings)

    input_tree = tree_manifest(source_path)
    category_counts = Counter(
        str(item["rule"]).split("/", 1)[0].lstrip("@") for item in target_findings
    )
    physical_loc = int(scan["physical_source_loc"])
    nonblank_loc = int(scan["nonblank_source_loc"])
    return {
        **source,
        "status": "frozen",
        "source_file_count": int(scan["source_file_count"]),
        "physical_source_loc": physical_loc,
        "nonblank_source_loc": nonblank_loc,
        "frozen_target_finding_count": len(target_findings),
        "frozen_raw_finding_count": len(raw_findings),
        "frozen_excluded_non_target_finding_count": len(raw_findings)
        - len(target_findings),
        "target_alerts_per_nonblank_kloc": (
            len(target_findings) / (nonblank_loc / 1000) if nonblank_loc else None
        ),
        "target_category_counts": dict(sorted(category_counts.items())),
        "frozen_target_findings_path": str(frozen_path),
        "frozen_target_findings_sha256": sha256_file(frozen_path),
        "frozen_input_file_count": len(input_tree),
        "frozen_input_tree_sha256": sha256_tree(input_tree),
        "verified_git_identity": git_identity(source),
        "validation_availability": validation_availability(source["name"]),
    }


def make_payload(
    *, experiment: str, projects: list[dict[str, Any]], provenance: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment": experiment,
        "status": "frozen",
        "provenance": provenance,
        "project_count": len(projects),
        "target_finding_count": sum(
            project["frozen_target_finding_count"] for project in projects
        ),
        "raw_finding_count": sum(
            project["frozen_raw_finding_count"] for project in projects
        ),
        "excluded_non_target_finding_count": sum(
            project["frozen_excluded_non_target_finding_count"] for project in projects
        ),
        "projects": projects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scan-manifest", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--reference-selection", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-35", type=Path, default=DEFAULT_OUTPUT_35)
    parser.add_argument("--output-10", type=Path, default=DEFAULT_OUTPUT_10)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    paths = [
        args.source_manifest.resolve(),
        args.scan_manifest.resolve(),
        args.reference_selection.resolve(),
    ]
    output_35 = args.output_35.resolve()
    output_10 = args.output_10.resolve()
    artifact_root = args.artifact_root.resolve()
    if output_35.exists() or output_10.exists() or artifact_root.exists():
        raise SystemExit("refusing to overwrite an existing v4 formal input freeze")

    source_payload, scan_payload, reference_payload = map(read_json, paths)
    source_projects = {item["name"]: item for item in source_payload["projects"]}
    scan_projects = {item["name"]: item for item in scan_payload["projects"]}
    if len(source_projects) != 35 or source_projects.keys() != scan_projects.keys():
        raise RuntimeError("The source and scan manifests do not define the same 35 projects")

    reference_names = [item["name"] for item in reference_payload["projects"]]
    if len(reference_names) != 10 or len(set(reference_names)) != 10:
        raise RuntimeError("The frozen reference selection must contain ten unique projects")
    if "wifi_testapp" not in reference_names or not set(reference_names) <= source_projects.keys():
        raise RuntimeError("The unchanged reference subset must include wifi_testapp")

    artifact_root.mkdir(parents=True)
    projects = []
    for name in sorted(source_projects):
        project = freeze_project(source_projects[name], scan_projects[name], artifact_root)
        projects.append(project)
        print(
            f"[verified] {name}: {project['frozen_target_finding_count']} target alerts",
            flush=True,
        )

    provenance = {
        "source_manifest": str(paths[0]),
        "source_manifest_sha256": sha256_file(paths[0]),
        "scan_manifest": str(paths[1]),
        "scan_manifest_sha256": sha256_file(paths[1]),
        "reference_selection": str(paths[2]),
        "reference_selection_sha256": sha256_file(paths[2]),
    }
    payload_35 = make_payload(
        experiment="EXP-HAPSKILL-35", projects=projects, provenance=provenance
    )
    by_name = {item["name"]: item for item in projects}
    reference_projects = [by_name[name] for name in reference_names]
    payload_10 = make_payload(
        experiment="EXP-AGENT-REF-10",
        projects=reference_projects,
        provenance={
            **provenance,
            "parent_35_manifest": str(output_35),
        },
    )
    write_json(output_35, payload_35)
    payload_10["provenance"]["parent_35_manifest_sha256"] = sha256_file(output_35)
    write_json(output_10, payload_10)
    write_json(artifact_root / "manifest_hapskill_35.json", payload_35)
    write_json(artifact_root / "manifest_agent_ref_10.json", payload_10)
    print(output_35)
    print(output_10)


if __name__ == "__main__":
    main()
