#!/usr/bin/env python3
"""Run updated standalone HomeCheck for exact-ID annotation range rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SAMPLE = (
    WORKSPACE_ROOT
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1/frozen_sample.jsonl"
)
DEFAULT_SCAN_MANIFEST = (
    WORKSPACE_ROOT
    / "HapRepair/revision/coding_agent_baseline/scan_runs/"
    "candidate_scan_35_gitcode_01/scan_manifest.json"
)
DEFAULT_HOMECHECK = WORKSPACE_ROOT / "homecheck"
DEFAULT_OUTPUT = SCRIPT_DIR / "range_scan_runs/native_range_exact_v10"
DEFAULT_OHOS_SDK = Path(
    "/home/zhihao/deveco/command-line-tools/sdk/default/openharmony/ets"
)
DEFAULT_HMS_SDK = Path(
    "/home/zhihao/deveco/command-line-tools/sdk/default/hms/ets"
)
EXACT_RANGE_RULES = {
    "@performance/avoid-overusing-custom-component-check",
    "@performance/foreach-args-check",
    "@performance/init-list-component",
    "@performance/hp-arkui-set-cache-count-for-lazyforeach-grid",
    "@security/no-commented-code",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_scan(
    homecheck: Path,
    project: dict[str, Any],
    rules: set[str],
    output_dir: Path,
    ohos_sdk: Path,
    hms_sdk: Path,
) -> dict[str, Any]:
    project_output = output_dir / "reports" / project["name"]
    project_output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="haprepair-range-scan-") as temp_name:
        temp = Path(temp_name)
        rule_config = temp / "rules.json"
        project_config = temp / "project.json"
        write_json(
            rule_config,
            {
                "files": ["**/*.ets", "**/*.ts"],
                "ignore": [
                    "**/ohosTest/**/*",
                    "**/node_modules/**/*",
                    "**/build/**/*",
                    "**/hvigorfile/**/*",
                    "**/oh_modules/**/*",
                    "**/.preview/**/*",
                ],
                "rules": {rule: "warn" for rule in sorted(rules)},
                "ruleSet": [],
                "overrides": [],
                "extRuleSet": [],
            },
        )
        write_json(
            project_config,
            {
                "projectName": project["name"],
                "projectPath": project["source_path"],
                "logPath": str(project_output / "HomeCheck.log"),
                "ohosSdkPath": str(ohos_sdk),
                "hmsSdkPath": str(hms_sdk),
                "checkPath": "",
                "sdkVersion": 18,
                "fix": "false",
                "npmPath": "",
                "npmInstallDir": str(homecheck),
                "reportDir": str(project_output),
                "arkCheckPath": str(homecheck),
                "product": "default",
                "language": "en",
                "sdksThirdParty": [],
            },
        )
        command = [
            "node",
            str(homecheck / "lib/run.js"),
            f"--projectConfigPath={project_config}",
            f"--configPath={rule_config}",
        ]
        result = subprocess.run(
            command,
            cwd=homecheck,
            capture_output=True,
            text=True,
            check=False,
        )
    (project_output / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (project_output / "stderr.log").write_text(result.stderr, encoding="utf-8")
    homecheck_log = project_output / "HomeCheck.log"
    log_text = homecheck_log.read_text(encoding="utf-8") if homecheck_log.is_file() else ""
    invalid_rules = [
        line.rsplit("Invalid rule name:", 1)[1].strip()
        for line in log_text.splitlines()
        if "Invalid rule name:" in line
    ]
    report = project_output / "issuesReport.json"
    return {
        "project": project["name"],
        "source_path": project["source_path"],
        "commit": project["commit"],
        "tree_oid": project["tree_oid"],
        "rules": sorted(rules),
        "command": command,
        "exit_code": result.returncode,
        "invalid_rules": invalid_rules,
        "report_path": str(report),
        "report_sha256": sha256(report) if report.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--scan-manifest", type=Path, default=DEFAULT_SCAN_MANIFEST)
    parser.add_argument("--homecheck", type=Path, default=DEFAULT_HOMECHECK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ohos-sdk", type=Path, default=DEFAULT_OHOS_SDK)
    parser.add_argument("--hms-sdk", type=Path, default=DEFAULT_HMS_SDK)
    parser.add_argument("--project", action="append", help="scan only the named project")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    for required in (args.sample, args.scan_manifest, args.homecheck / "lib/run.js"):
        if not required.is_file():
            parser.error(f"required file does not exist: {required}")

    sample = read_jsonl(args.sample)
    rules_by_project: dict[str, set[str]] = {}
    for record in sample:
        if record["rule"] in EXACT_RANGE_RULES:
            rules_by_project.setdefault(record["project"], set()).add(record["rule"])
    if args.project:
        unknown = sorted(set(args.project) - set(rules_by_project))
        if unknown:
            parser.error(f"project is absent from selected sample rules: {', '.join(unknown)}")
        rules_by_project = {
            name: rules_by_project[name] for name in args.project
        }
    projects = {
        item["name"]: item for item in read_json(args.scan_manifest)["projects"]
    }
    args.output_dir.mkdir(parents=True)
    results = [
        run_scan(
            args.homecheck,
            projects[name],
            rules,
            args.output_dir,
            args.ohos_sdk,
            args.hms_sdk,
        )
        for name, rules in sorted(rules_by_project.items())
    ]
    homecheck_diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", "src"],
        cwd=args.homecheck,
        capture_output=True,
        check=True,
    ).stdout
    manifest = {
        "schema_version": 1,
        "purpose": "annotation_display_only",
        "sample_path": str(args.sample.resolve()),
        "sample_sha256": sha256(args.sample),
        "homecheck_path": str(args.homecheck.resolve()),
        "homecheck_source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.homecheck,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "homecheck_source_diff_sha256": hashlib.sha256(homecheck_diff).hexdigest(),
        "detector_unchanged": "CodeLinter 6.0.240",
        "results": results,
    }
    write_json(args.output_dir / "scan_manifest.json", manifest)
    failures = [
        item["project"]
        for item in results
        if item["exit_code"] or item["invalid_rules"] or not item["report_sha256"]
    ]
    if failures:
        raise SystemExit(f"HomeCheck range scans failed: {', '.join(failures)}")
    print(f"wrote {args.output_dir} with {len(results)} project reports", flush=True)


if __name__ == "__main__":
    main()
