#!/usr/bin/env python3
"""Prepare, smoke-test, execute, and verify final-v14 EXP-INDEP-63 runs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent
REVISION_DIR = ORACLE_DIR.parent
REPO_ROOT = REVISION_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
BASELINE_HELPERS = REVISION_DIR / "coding_agent_baseline"
if str(BASELINE_HELPERS) not in sys.path:
    sys.path.insert(0, str(BASELINE_HELPERS))

from run_agent_baseline import (  # type: ignore  # noqa: E402
    HOST_CODEX_HOME,
    extract_commands,
    extract_thread_id,
    extract_usage,
    parse_codex_events,
    prepare_codex_home,
    remove_empty_auth_mount_placeholder,
)


DEFAULT_PROTOCOL = SCRIPT_DIR / "protocol_v14_static_skill.json"
DEFAULT_PROMPT = SCRIPT_DIR / "prompt_template_v14_static_skill.txt"
RUNS_DIR = ORACLE_DIR / "generation_runs"
SMOKE_RUNS_DIR = ORACLE_DIR / "generation_smoke_runs"
SOURCE_SUFFIXES = {".ets", ".ts"}
IGNORED_NAMES = {".git", "__pycache__", ".ruff_cache"}
RESTRICTED_COMMAND_MARKERS = (
    "homecheck",
    "codelinter",
    "repair_session.py",
    "knowledge_base",
    "rule_complete_383",
    "repaired_project",
    "generation_runs",
    "adjudication_packages",
    "reference_patch",
    "/baseline_data/",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_protocol(path_arg: str | None) -> tuple[dict[str, Any], Path]:
    path = Path(path_arg).resolve() if path_arg else DEFAULT_PROTOCOL
    if not path.is_file():
        raise SystemExit(f"Protocol file does not exist: {path}")
    protocol = read_json(path)
    if protocol.get("status") != "frozen":
        raise SystemExit("The final-v14 protocol must be frozen before use")
    return protocol, path


def resolve_repo_path(relative: str) -> Path:
    path = REPO_ROOT / relative
    if not path.exists():
        raise SystemExit(f"Required frozen path does not exist: {path}")
    return path


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"Frozen {label} hash mismatch: expected {expected}, found {actual}"
        )


def iter_tree_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
        and path.suffix != ".pyc"
    )


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in iter_tree_files(root)
    }


def tree_digest(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode())
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def validate_frozen_inputs(protocol: dict[str, Any]) -> dict[str, Path]:
    benchmark = protocol["benchmark"]
    leakage = protocol["leakage_gate"]
    method = protocol["method"]
    runtime = protocol["runtime"]
    paths = {
        "case_manifest": resolve_repo_path(benchmark["case_manifest"]),
        "defective_project": resolve_repo_path(benchmark["defective_project"]),
        "validation_summary": resolve_repo_path(benchmark["validation_summary"]),
        "defective_report": resolve_repo_path(benchmark["defective_report"]),
        "case_results": resolve_repo_path(benchmark["case_results"]),
        "leakage_report": resolve_repo_path(leakage["report"]),
        "skill": resolve_repo_path(method["skill_source"]),
        "prompt_template": resolve_repo_path(method["prompt_template"]),
    }
    checks = (
        ("case_manifest", benchmark["case_manifest_sha256"], "case manifest"),
        (
            "validation_summary",
            benchmark["validation_summary_sha256"],
            "validation summary",
        ),
        ("defective_report", benchmark["defective_report_sha256"], "defective report"),
        ("case_results", benchmark["case_results_sha256"], "case results"),
        ("leakage_report", leakage["report_sha256"], "leakage report"),
        ("prompt_template", method["prompt_template_sha256"], "prompt template"),
    )
    for key, expected, label in checks:
        assert_hash(paths[key], expected, label)

    skill_manifest = tree_manifest(paths["skill"])
    observed_skill_tree = tree_digest(skill_manifest)
    if observed_skill_tree != method["skill_tree_sha256"]:
        raise SystemExit(
            "Frozen Skill tree mismatch: "
            f"expected {method['skill_tree_sha256']}, found {observed_skill_tree}"
        )
    if len(skill_manifest) != method["skill_file_count"]:
        raise SystemExit("Frozen Skill file count differs")
    for relative, expected in method["skill_file_hashes"].items():
        assert_hash(paths["skill"] / relative, expected, f"Skill file {relative}")
    assert_hash(Path(__file__), method["runner_sha256"], "final-v14 runner")

    summary = read_json(paths["validation_summary"])
    leakage_report = read_json(paths["leakage_report"])
    if not (
        summary.get("validated_case_count") == benchmark["case_count"]
        and summary.get("failed_case_count") == 0
        and summary.get("all_scans_clean") is True
    ):
        raise SystemExit("Frozen two-sided CodeLinter benchmark gate is not satisfied")
    if not leakage_report.get("automatic_gate_passed"):
        raise SystemExit("Frozen benchmark leakage audit is not satisfied")
    if method.get("dynamic_retrieval") is not False:
        raise SystemExit("Final-v14 protocol must disable dynamic retrieval")
    if method.get("knowledge_base_mounted") is not False:
        raise SystemExit("Raw knowledge base must not be mounted")
    if benchmark.get("reference_project_mounted") is not False:
        raise SystemExit("Human reference project must not be mounted")

    image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            runtime["container_image"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if image.returncode != 0:
        raise SystemExit(
            f"Frozen container image is unavailable: {image.stderr.strip()}"
        )
    if image.stdout.strip() != runtime["container_image_id"]:
        raise SystemExit("Frozen container image identity differs")
    return paths


def validate_protocol(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args.protocol)
    paths = validate_frozen_inputs(protocol)
    manifest = read_json(paths["case_manifest"])
    cases = manifest["cases"]
    taxonomy = {
        "performance": sum(case["rule"].startswith("@performance/") for case in cases),
        "arkts_eslint": sum(
            case["rule"].startswith("@hw-ets-eslint/") for case in cases
        ),
        "security": sum(case["rule"].startswith("@security/") for case in cases),
    }
    checks = {
        "case_count": len(cases) == protocol["benchmark"]["case_count"] == 63,
        "rule_count": len({case["rule"] for case in cases})
        == protocol["benchmark"]["rule_count"]
        == 63,
        "taxonomy": taxonomy == protocol["benchmark"]["taxonomy"],
        "dynamic_retrieval_disabled": protocol["method"]["dynamic_retrieval"] is False,
        "embedding_absent": protocol["method"]["embedding_model"] is None,
        "ranking_absent": protocol["method"]["similarity_ranking"] is False,
        "top_k_absent": protocol["method"]["top_k"] is None,
        "knowledge_base_not_mounted": protocol["method"]["knowledge_base_mounted"]
        is False,
        "reference_project_not_mounted": protocol["benchmark"][
            "reference_project_mounted"
        ]
        is False,
        "human_repair_hidden": protocol["leakage_gate"]["human_repair_hidden"] is True,
        "historical_outputs_hidden": protocol["leakage_gate"][
            "historical_generation_outputs_hidden"
        ]
        is True,
        "single_invocation": protocol["model"]["codex_agent_invocations_per_case"] == 1,
        "single_candidate": protocol["model"]["accepted_candidates_per_case"] == 1,
        "no_automatic_retry": protocol["model"]["automatic_retries"] == 0,
    }
    result = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "run_id": protocol["run_id"],
        "validated_at": utc_now(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "prompt_sha256": sha256_file(paths["prompt_template"]),
        "skill_tree_sha256": tree_digest(tree_manifest(paths["skill"])),
        "checks": checks,
        "all_passed": all(checks.values()),
        "model_calls": 0,
    }
    output = Path(args.output).resolve() if args.output else None
    if output:
        write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(2)


def finding_index(
    report_path: Path, defective_root: Path
) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for file_report in read_json(report_path):
        source_path = Path(file_report["filePath"])
        try:
            relative = (
                source_path.resolve().relative_to(defective_root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise SystemExit(
                f"Frozen finding is outside defective project: {source_path}"
            ) from exc
        indexed[relative] = file_report.get("messages", [])
    return indexed


def case_findings(
    case: dict[str, Any], indexed: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    expected_rule = case.get("codelinter_rule", case["rule"])
    findings = []
    for relative in case["defective_files"]:
        for finding in indexed.get(relative, []):
            if finding.get("rule") == expected_rule:
                findings.append({"file": relative, **finding})
    if not findings:
        raise SystemExit(f"No frozen target finding for {case['case_id']}")
    return findings


def make_blind_ids(cases: list[dict[str, Any]], seed: int) -> dict[str, str]:
    ordered = sorted(case["case_id"] for case in cases)
    blind_ids = [f"V14-{index:03d}" for index in range(1, len(ordered) + 1)]
    random.Random(seed).shuffle(blind_ids)
    return dict(zip(ordered, blind_ids, strict=True))


def install_skill(source: Path, codex_home: Path) -> dict[str, Any]:
    destination = codex_home / "skills" / "haprepair-openharmony-repair"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache", "*.pyc"),
    )
    manifest = tree_manifest(destination)
    return {
        "path": str(destination),
        "file_count": len(manifest),
        "tree_sha256": tree_digest(manifest),
        "skill_md_sha256": sha256_file(destination / "SKILL.md"),
    }


def copy_case_sources(
    source_root: Path, relative_paths: list[str], destination: Path
) -> list[dict[str, Any]]:
    records = []
    for relative in relative_paths:
        source = source_root / relative
        if not source.is_file():
            raise SystemExit(f"Missing defective source file: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return records


def source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
    }


def source_diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    chunks = []
    for relative in sorted(set(before) | set(after)):
        old = (
            before.get(relative, b"").decode("utf-8", errors="replace").splitlines(True)
        )
        new = (
            after.get(relative, b"").decode("utf-8", errors="replace").splitlines(True)
        )
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def render_task(
    *, case: dict[str, Any], blind_id: str, findings: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "EXP-INDEP-63-FINAL-V14",
        "blind_id": blind_id,
        "target_rule": case["rule"],
        "codelinter_rule": case.get("codelinter_rule", case["rule"]),
        "rule_description": case["rule_description"],
        "target_findings": findings,
        "input_files": case["defective_files"],
        "controller_feedback_available": False,
        "homecheck_scan_available": False,
        "human_reference_available": False,
        "accepted_candidates": 1,
    }


def prepare(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args.protocol)
    if args.run_id != protocol["run_id"]:
        raise SystemExit(f"Formal run ID must be {protocol['run_id']}")
    paths = validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    manifest = read_json(paths["case_manifest"])
    cases = manifest["cases"]
    if len(cases) != protocol["benchmark"]["case_count"]:
        raise SystemExit("Unexpected benchmark case count")
    blind_map = make_blind_ids(cases, protocol["blind_id_seed"])
    indexed = finding_index(paths["defective_report"], paths["defective_project"])
    prompt = paths["prompt_template"].read_text(encoding="utf-8")
    prepared = []

    for case in sorted(cases, key=lambda item: item["case_id"]):
        blind_id = blind_map[case["case_id"]]
        case_dir = run_dir / "cases" / blind_id
        baseline = case_dir / "baseline"
        workspace = case_dir / "workspace"
        codex_home = case_dir / "codex_home"
        run_state = case_dir / "run_state"
        baseline.mkdir(parents=True)
        workspace.mkdir(parents=True)
        (run_state / "home").mkdir(parents=True)
        baseline_files = copy_case_sources(
            paths["defective_project"], case["defective_files"], baseline
        )
        workspace_files = copy_case_sources(
            paths["defective_project"], case["defective_files"], workspace
        )
        if baseline_files != workspace_files:
            raise SystemExit(f"Baseline/workspace copy mismatch for {case['case_id']}")
        task = render_task(
            case=case,
            blind_id=blind_id,
            findings=case_findings(case, indexed),
        )
        task_path = workspace / "HAPREPAIR_TASK.json"
        write_json(task_path, task)
        prompt_path = case_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        prepare_codex_home(codex_home)
        installed = install_skill(paths["skill"], codex_home)
        if installed["tree_sha256"] != protocol["method"]["skill_tree_sha256"]:
            raise SystemExit(f"Installed Skill drift for {case['case_id']}")
        prepared.append(
            {
                "case_id": case["case_id"],
                "blind_id": blind_id,
                "rule": case["rule"],
                "case_dir": case_dir.relative_to(run_dir).as_posix(),
                "input_files": workspace_files,
                "task_sha256": sha256_file(task_path),
                "prompt_sha256": sha256_file(prompt_path),
                "skill_tree_sha256": installed["tree_sha256"],
                "reference_project_mounted": False,
                "knowledge_base_mounted": False,
                "historical_outputs_mounted": False,
            }
        )

    write_json(run_dir / "protocol_snapshot.json", protocol)
    write_json(run_dir / "blind_map.json", blind_map)
    write_json(run_dir / "input_manifest.json", prepared)
    write_json(
        run_dir / "environment.prepare.json",
        {
            "prepared_at": utc_now(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short"),
            "case_count": len(prepared),
            "unique_rule_count": len({item["rule"] for item in prepared}),
            "dynamic_retrieval": False,
        },
    )
    print(json.dumps({"status": "prepared", "case_count": len(prepared)}, indent=2))


def is_restricted_command(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in RESTRICTED_COMMAND_MARKERS)


def build_docker_command(
    *, protocol: dict[str, Any], workspace: Path, codex_home: Path, run_state: Path
) -> list[str]:
    model = protocol["model"]
    runtime = protocol["runtime"]
    codex_command = [
        "codex",
        "exec",
        "--json",
        "--model",
        model["requested_id"],
        "-c",
        f'model_provider="{model["provider"]}"',
        "-c",
        f'model_reasoning_effort="{model["reasoning_effort"]}"',
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        "/workspace",
        "-",
    ]
    return [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "bridge",
        "--workdir",
        "/workspace",
        "--env",
        "CODEX_HOME=/codex-home",
        "--env",
        "HAPREPAIR_AGENT_MODE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/run-state/home",
        "--volume",
        f"{workspace}:/workspace",
        "--volume",
        f"{codex_home}:/codex-home",
        "--volume",
        f"{run_state}:/run-state",
        "--volume",
        f"{HOST_CODEX_HOME / 'auth.json'}:/codex-home/auth.json:ro",
        runtime["container_image"],
        *codex_command,
    ]


def execute_case(
    *, protocol: dict[str, Any], run_dir: Path, item: dict[str, Any]
) -> dict[str, Any]:
    case_dir = run_dir / item["case_dir"]
    baseline = case_dir / "baseline"
    workspace = case_dir / "workspace"
    codex_home = case_dir / "codex_home"
    run_state = case_dir / "run_state"
    result_path = case_dir / "result.json"
    if result_path.exists() or (case_dir / "attempt.json").exists():
        raise RuntimeError(f"Refusing a second request for {item['blind_id']}")
    task_path = workspace / "HAPREPAIR_TASK.json"
    task_hash_before = sha256_file(task_path)
    before = source_snapshot(baseline)
    attempt = {
        "event": "request_started",
        "timestamp": utc_now(),
        "case_id": item["case_id"],
        "blind_id": item["blind_id"],
        "attempt": 1,
        "requested_model": protocol["model"]["requested_id"],
        "prompt_sha256": item["prompt_sha256"],
    }
    write_json(case_dir / "attempt.json", attempt)
    append_jsonl(run_dir / "attempts.jsonl", attempt)
    command = build_docker_command(
        protocol=protocol,
        workspace=workspace,
        codex_home=codex_home,
        run_state=run_state,
    )
    started = time.monotonic()
    process = subprocess.run(
        command,
        input=(case_dir / "prompt.txt").read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
    )
    elapsed = time.monotonic() - started
    remove_empty_auth_mount_placeholder(codex_home)
    trace_path = case_dir / "trace.jsonl"
    stderr_path = case_dir / "stderr.log"
    trace_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    events = parse_codex_events(process.stdout)
    commands = extract_commands(events)
    restricted = [
        record["command"]
        for record in commands
        if is_restricted_command(record["command"])
    ]
    after = source_snapshot(workspace)
    diff_text = source_diff(before, after)
    (case_dir / "candidate.patch").write_text(diff_text, encoding="utf-8")
    deleted_inputs = sorted(set(before) - set(after))
    missing_required = sorted(
        record["path"] for record in item["input_files"] if record["path"] not in after
    )
    task_unchanged = task_path.is_file() and sha256_file(task_path) == task_hash_before
    accepted = all(
        (
            process.returncode == 0,
            bool(events),
            bool(diff_text),
            not deleted_inputs,
            not missing_required,
            task_unchanged,
            not restricted,
        )
    )
    output_files = [
        {
            "path": relative,
            "sha256": sha256_bytes(content),
            "size_bytes": len(content),
            "content": content.decode("utf-8", errors="replace"),
        }
        for relative, content in sorted(after.items())
    ]
    result = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "run_id": protocol["run_id"],
        "case_id": item["case_id"],
        "blind_id": item["blind_id"],
        "attempt": 1,
        "status": "accepted" if accepted else "rejected",
        "requested_model": protocol["model"]["requested_id"],
        "thread_id": extract_thread_id(events),
        "usage": extract_usage(events),
        "wall_seconds": elapsed,
        "exit_code": process.returncode,
        "systemic_failure": process.returncode != 0,
        "event_count": len(events),
        "commands": commands,
        "restricted_commands": restricted,
        "task_unchanged": task_unchanged,
        "source_diff_present": bool(diff_text),
        "source_diff_sha256": sha256_bytes(diff_text.encode("utf-8")),
        "deleted_input_paths": deleted_inputs,
        "missing_required_paths": missing_required,
        "output_files": output_files,
        "trace_sha256": sha256_file(trace_path),
        "stderr_sha256": sha256_file(stderr_path),
        "finished_at": utc_now(),
        "rejection_reasons": [
            reason
            for reason, failed in (
                ("codex_nonzero_exit", process.returncode != 0),
                ("no_json_events", not events),
                ("no_source_diff", not diff_text),
                ("input_source_deleted", bool(deleted_inputs)),
                ("required_source_missing", bool(missing_required)),
                ("task_file_modified", not task_unchanged),
                ("restricted_command_observed", bool(restricted)),
            )
            if failed
        ],
    }
    write_json(result_path, result)
    append_jsonl(
        run_dir / "attempts.jsonl",
        {
            "event": "request_finished",
            "timestamp": result["finished_at"],
            "case_id": item["case_id"],
            "blind_id": item["blind_id"],
            "attempt": 1,
            "status": result["status"],
            "wall_seconds": elapsed,
        },
    )
    return result


def summarize_run(run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    prepared = read_json(run_dir / "input_manifest.json")
    results = []
    for item in prepared:
        path = run_dir / item["case_dir"] / "result.json"
        if path.is_file():
            results.append(read_json(path))
    accepted = [result for result in results if result["status"] == "accepted"]
    rejected = [result for result in results if result["status"] == "rejected"]
    usage_keys = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
    )
    metrics = {
        "run_id": protocol["run_id"],
        "status": "complete" if len(results) == len(prepared) else "partial",
        "prepared_case_count": len(prepared),
        "attempted_case_count": len(results),
        "accepted_candidate_count": len(accepted),
        "rejected_case_count": len(rejected),
        "pending_case_count": len(prepared) - len(results),
        "one_request_per_attempted_case": all(
            result["attempt"] == 1 for result in results
        ),
        "dynamic_retrieval": False,
        "homecheck_feedback": False,
        "semantic_correctness_status": "not_adjudicated",
        "usage": {
            key: sum(result.get("usage", {}).get(key, 0) for result in results)
            for key in usage_keys
        },
        "wall_seconds_sum": sum(result["wall_seconds"] for result in results),
        "api_cost_usd": None,
        "cost_note": "Provider billing is unavailable and is not inferred.",
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def execute(args: argparse.Namespace) -> None:
    protocol, _ = load_protocol(args.protocol)
    if args.run_id != protocol["run_id"]:
        raise SystemExit("Run ID must match the frozen protocol")
    validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    prepared = read_json(run_dir / "input_manifest.json")
    selected = prepared
    if args.case_id:
        selected = [item for item in prepared if item["case_id"] == args.case_id]
        if not selected:
            raise SystemExit(f"Unknown case ID: {args.case_id}")
    for index, item in enumerate(selected, start=1):
        try:
            result = execute_case(protocol=protocol, run_dir=run_dir, item=item)
        except Exception as exc:
            print(f"{item['blind_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise SystemExit(2) from exc
        else:
            print(
                f"generation progress {index}/{len(selected)}: "
                f"{item['blind_id']} {result['status']}",
                flush=True,
            )
            if result["systemic_failure"]:
                summarize_run(run_dir, protocol)
                raise SystemExit(2)
            if result["status"] != "accepted" and args.fail_fast:
                raise SystemExit(2)
    print(json.dumps(summarize_run(run_dir, protocol), indent=2))


def prepare_smoke_workspace(
    smoke_dir: Path, protocol: dict[str, Any], skill_source: Path
) -> dict[str, Any]:
    workspace = smoke_dir / "workspace"
    baseline = smoke_dir / "baseline"
    codex_home = smoke_dir / "codex_home"
    run_state = smoke_dir / "run_state"
    relative = "entry/src/main/ets/SmokeCase.ets"
    content = (
        "@Entry\n@Component\nstruct SmokeCase {\n"
        "  @State private fixedValue: number = 1\n\n"
        "  build() {\n    Text(`${this.fixedValue}`)\n  }\n}\n"
    )
    for root in (workspace, baseline):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (run_state / "home").mkdir(parents=True)
    write_json(
        workspace / "HAPREPAIR_TASK.json",
        {
            "schema_version": 1,
            "experiment": "EXP-INDEP-63-FINAL-V14-SMOKE",
            "blind_id": "SMOKE-NONBENCHMARK-001",
            "target_rule": "@performance/hp-arkui-remove-unchanged-state-var",
            "codelinter_rule": "@performance/hp-arkui-remove-unchanged-state-var",
            "rule_description": "Avoid reactive state for a value that is never changed.",
            "target_findings": [
                {
                    "file": relative,
                    "line": 4,
                    "column": 3,
                    "rule": "@performance/hp-arkui-remove-unchanged-state-var",
                }
            ],
            "input_files": [relative],
            "controller_feedback_available": False,
            "homecheck_scan_available": False,
            "human_reference_available": False,
            "accepted_candidates": 1,
        },
    )
    prepare_codex_home(codex_home)
    installed = install_skill(skill_source, codex_home)
    return {
        "case_id": "smoke_nonbenchmark_001",
        "blind_id": "SMOKE-NONBENCHMARK-001",
        "rule": "@performance/hp-arkui-remove-unchanged-state-var",
        "case_dir": ".",
        "input_files": [
            {
                "path": relative,
                "sha256": sha256_file(baseline / relative),
                "size_bytes": len(content.encode()),
            }
        ],
        "task_sha256": sha256_file(workspace / "HAPREPAIR_TASK.json"),
        "prompt_sha256": sha256_file(DEFAULT_PROMPT),
        "skill_tree_sha256": installed["tree_sha256"],
    }


def smoke(args: argparse.Namespace) -> None:
    protocol, _ = load_protocol(args.protocol)
    paths = validate_frozen_inputs(protocol)
    smoke_dir = SMOKE_RUNS_DIR / args.smoke_id
    if smoke_dir.exists():
        raise SystemExit(f"Smoke directory already exists: {smoke_dir}")
    smoke_dir.mkdir(parents=True)
    item = prepare_smoke_workspace(smoke_dir, protocol, paths["skill"])
    (smoke_dir / "prompt.txt").write_text(
        paths["prompt_template"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = execute_case(protocol=protocol, run_dir=smoke_dir, item=item)
    summary = {
        "smoke_id": args.smoke_id,
        "benchmark_case": False,
        "status": "passed" if result["status"] == "accepted" else "failed",
        "requested_model": protocol["model"]["requested_id"],
        "source_diff_present": result["source_diff_present"],
        "restricted_commands": result["restricted_commands"],
        "semantic_correctness_evidence": False,
    }
    write_json(smoke_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(2)


def verify(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args.protocol)
    paths = validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    prepared = read_json(run_dir / "input_manifest.json")
    checks = {
        "protocol_hash_matches_snapshot": sha256_file(protocol_path)
        == sha256_file(run_dir / "protocol_snapshot.json"),
        "case_count_63": len(prepared) == 63,
        "unique_case_ids_63": len({item["case_id"] for item in prepared}) == 63,
        "unique_blind_ids_63": len({item["blind_id"] for item in prepared}) == 63,
        "unique_rules_63": len({item["rule"] for item in prepared}) == 63,
        "taxonomy_42_1_20": (
            sum(item["rule"].startswith("@performance/") for item in prepared) == 42
            and sum(item["rule"].startswith("@hw-ets-eslint/") for item in prepared)
            == 1
            and sum(item["rule"].startswith("@security/") for item in prepared) == 20
        ),
        "no_dynamic_retrieval": protocol["method"]["dynamic_retrieval"] is False,
        "raw_knowledge_base_not_mounted": all(
            item["knowledge_base_mounted"] is False for item in prepared
        ),
        "human_reference_not_mounted": all(
            item["reference_project_mounted"] is False for item in prepared
        ),
        "historical_outputs_not_mounted": all(
            item["historical_outputs_mounted"] is False for item in prepared
        ),
        "all_task_hashes_match": all(
            sha256_file(
                run_dir / item["case_dir"] / "workspace" / "HAPREPAIR_TASK.json"
            )
            == item["task_sha256"]
            for item in prepared
        ),
        "all_input_hashes_match": all(
            sha256_file(run_dir / item["case_dir"] / "baseline" / record["path"])
            == record["sha256"]
            for item in prepared
            for record in item["input_files"]
        ),
        "all_installed_skill_hashes_match": all(
            tree_digest(
                tree_manifest(
                    run_dir
                    / item["case_dir"]
                    / "codex_home"
                    / "skills"
                    / "haprepair-openharmony-repair"
                )
            )
            == protocol["method"]["skill_tree_sha256"]
            for item in prepared
        ),
        "frozen_benchmark_and_leakage_inputs_valid": bool(paths),
    }
    output_errors = []
    if args.require_complete:
        results = []
        for item in prepared:
            result_path = run_dir / item["case_dir"] / "result.json"
            if not result_path.is_file():
                output_errors.append(f"missing result: {item['blind_id']}")
                continue
            results.append(read_json(result_path))
        checks.update(
            {
                "all_63_results_present": len(results) == 63,
                "all_63_candidates_accepted": len(results) == 63
                and all(result["status"] == "accepted" for result in results),
                "exactly_one_attempt_per_case": len(results) == 63
                and all(result["attempt"] == 1 for result in results),
                "all_source_diffs_present": len(results) == 63
                and all(result["source_diff_present"] for result in results),
                "no_restricted_commands": len(results) == 63
                and all(not result["restricted_commands"] for result in results),
                "no_input_deletions": len(results) == 63
                and all(not result["deleted_input_paths"] for result in results),
                "task_files_unchanged": len(results) == 63
                and all(result["task_unchanged"] for result in results),
            }
        )
    verification = {
        "run_id": args.run_id,
        "checks": checks,
        "all_passed": all(checks.values()),
        "output_errors": output_errors,
    }
    write_json(run_dir / "verification.json", verification)
    print(json.dumps(verification, indent=2))
    if not verification["all_passed"]:
        raise SystemExit(2)


def identities(args: argparse.Namespace) -> None:
    skill = resolve_repo_path(args.skill_source)
    values = {
        "skill_tree_sha256": tree_digest(tree_manifest(skill)),
        "skill_file_count": len(tree_manifest(skill)),
        "prompt_template_sha256": sha256_file(DEFAULT_PROMPT),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    print(json.dumps(values, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identities_parser = subparsers.add_parser("identities")
    identities_parser.add_argument(
        "--skill-source",
        default="skills/haprepair-openharmony-repair-v14-dev",
    )
    identities_parser.set_defaults(func=identities)

    validate_parser = subparsers.add_parser("validate-protocol")
    validate_parser.add_argument("--protocol")
    validate_parser.add_argument("--output")
    validate_parser.set_defaults(func=validate_protocol)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.set_defaults(func=prepare)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--protocol")
    smoke_parser.add_argument("--smoke-id", required=True)
    smoke_parser.set_defaults(func=smoke)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--protocol")
    execute_parser.add_argument("--run-id", required=True)
    execute_parser.add_argument("--case-id")
    execute_parser.add_argument("--fail-fast", action="store_true")
    execute_parser.set_defaults(func=execute)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--protocol")
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.add_argument("--require-complete", action="store_true")
    verify_parser.set_defaults(func=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
