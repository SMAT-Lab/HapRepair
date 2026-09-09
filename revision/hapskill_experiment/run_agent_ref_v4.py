#!/usr/bin/env python3
"""Run one isolated EXP-AGENT-REF-10 v4 reference-agent condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
SKILL_SCRIPTS = HAPREPAIR_ROOT / "skills/haprepair-openharmony-repair-v14-dev/scripts"
for import_dir in (BASELINE_DIR, SKILL_SCRIPTS):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import homecheck  # type: ignore  # noqa: E402
import repair_session as guards  # type: ignore  # noqa: E402
from agent_ref_session import (  # noqa: E402
    audit_completion,
    candidate_score,
    completion_schema,
    describe_interactions,
    is_restricted_command,
    make_reference_plan,
    read_json,
    workspace_container_path,
    write_json,
)
from build_gate import prepare_build_gate, run_build_gate  # type: ignore  # noqa: E402
from run_agent_baseline import (  # type: ignore  # noqa: E402
    HOST_CODEX_HOME,
    RULES_ROOT,
    compute_alert_metrics,
    copy_project,
    extract_commands,
    extract_thread_id,
    extract_usage,
    parse_codex_events,
    prepare_codex_home,
    provider_endpoint_fingerprint,
    remove_empty_auth_mount_placeholder,
    sha256_file,
    source_diff,
    tree_manifest,
    utc_now,
    verify_initial_findings,
)


DEFAULT_PROTOCOL = HERE / "protocol-agent-ref-10-v4.json"
DEFAULT_INPUTS = HERE / "formal_inputs_agent_ref_10_v4.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data/exp_agent_ref_10/runs"
DEFAULT_IMAGE = "hybrid-gym-codex:0.146.0"
DEFAULT_AUTHORIZATION = (
    WORKSPACE_ROOT / "paper/rebuttal/gates/exp_agent_ref_10_v4_static_g2_20260806.json"
)
COMMAND_LINE_TOOLS = Path("/home/zhihao/deveco/command-line-tools")
SDK_ROOT = WORKSPACE_ROOT / "baseline_data/openharmony_sdk"
HOST_JAVA_HOME = Path("/home/zhihao/.jdks/jdk-17.0.18+8")
RUNTIME_LIB_DIR = WORKSPACE_ROOT / "baseline_data/haprepair_runtime_libs/lib"
MAX_AGENT_TURNS_PER_ROUND = 6


def validate_authorization(
    protocol_path: Path,
    inputs_path: Path,
    run_id: str,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Refuse a formal launch unless G2 pins the exact current harness."""
    protocol = read_json(protocol_path)
    if authorization_path is None:
        configured = protocol.get("formal_run", {}).get("authorization_artifact")
        authorization_path = (
            Path(configured).resolve() if configured else DEFAULT_AUTHORIZATION
        )
    if not authorization_path.is_file():
        raise RuntimeError(f"Static G2 authorization is missing: {authorization_path}")
    authorization = read_json(authorization_path)
    authorized_run_ids = authorization.get("authorized_run_ids")
    run_id_authorized = (
        run_id in authorized_run_ids
        if isinstance(authorized_run_ids, list)
        else authorization.get("authorized_run_id") == run_id
    )
    if (
        authorization.get("status") != "closed"
        or authorization.get("passed") is not True
        or authorization.get("formal_execution_authorized") is not True
        or not run_id_authorized
        or any(not check.get("passed") for check in authorization.get("checks", []))
    ):
        raise RuntimeError("Static G2 authorization is not closed and passing")
    configured_artifacts = authorization.get("runner_artifact_paths")
    if isinstance(configured_artifacts, dict):
        artifacts = {
            name: Path(path).resolve() for name, path in configured_artifacts.items()
        }
    else:
        artifacts = {
            "agent_ref_session.py": HERE / "agent_ref_session.py",
            "run_agent_ref_v4.py": HERE / "run_agent_ref_v4.py",
            "run_formal_agent_ref_v4.py": HERE / "run_formal_agent_ref_v4.py",
            "test_agent_ref_v4.py": HERE / "test_agent_ref_v4.py",
            "audit_agent_ref_v4.py": HERE / "audit_agent_ref_v4.py",
        }
    expected = authorization.get("runner_artifacts") or {}
    drift = {
        name: {"expected": expected.get(name), "observed": sha256_file(path)}
        for name, path in artifacts.items()
        if expected.get(name) != sha256_file(path)
    }
    contract = next(
        (
            check
            for check in authorization["checks"]
            if check.get("name") == "frozen_contract_and_exact_order"
        ),
        {},
    )
    if contract.get("protocol_sha256") != sha256_file(protocol_path):
        drift["protocol"] = {
            "expected": contract.get("protocol_sha256"),
            "observed": sha256_file(protocol_path),
        }
    if contract.get("inputs_sha256") != sha256_file(inputs_path):
        drift["inputs"] = {
            "expected": contract.get("inputs_sha256"),
            "observed": sha256_file(inputs_path),
        }
    if drift:
        raise RuntimeError(f"Harness drifted after static G2 authorization: {drift}")
    return {
        "path": str(authorization_path),
        "sha256": sha256_file(authorization_path),
        "authorized_run_id": run_id,
        "runner_artifacts": expected,
    }


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode())
    return digest.hexdigest()


def load_contract(
    protocol_path: Path, input_path: Path, project_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(protocol_path)
    inputs = read_json(input_path)
    if protocol.get("status") != "frozen" or inputs.get("status") != "frozen":
        raise ValueError("Protocol and formal inputs must be frozen")
    expected_input_experiment = protocol["dataset"].get(
        "formal_inputs_experiment", protocol["experiment"]
    )
    if inputs.get("experiment") != expected_input_experiment:
        raise ValueError("Experiment identity mismatch")
    if protocol["dataset"]["formal_inputs_sha256"] != sha256_file(input_path):
        raise ValueError("Formal input manifest hash differs from the protocol")
    projects = inputs.get("projects")
    expected_project_count = int(protocol["dataset"]["project_count"])
    if not isinstance(projects, list) or len(projects) != expected_project_count:
        raise ValueError(
            "Reference manifest project count differs from the frozen protocol"
        )
    matches = [item for item in projects if item.get("name") == project_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen project named {project_name!r}")
    expected_order = [
        item.split(":", 1)[0] for item in protocol["formal_run"]["condition_order"]
    ]
    if [item["name"] for item in projects] != expected_order:
        raise ValueError("Formal input order differs from frozen condition order")
    return protocol, matches[0]


def validate_runtime(protocol: dict[str, Any], image: str) -> dict[str, Any]:
    runtime = protocol["runtime"]
    if image != runtime["container_image"]:
        raise RuntimeError("Container image name differs from the protocol")
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    image_id = inspect.stdout.strip()
    if inspect.returncode or image_id != runtime["container_image_id"]:
        raise RuntimeError("Container image ID differs from the protocol")
    versions = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image,
            "-lc",
            "codex --version",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if versions.returncode:
        raise RuntimeError("Cannot verify the frozen container runtime")
    output = versions.stdout + versions.stderr
    host_python = subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True, check=False
    )
    python_output = host_python.stdout + host_python.stderr
    if (
        runtime["codex_cli"] not in output
        or host_python.returncode
        or runtime["python"] not in python_output
    ):
        raise RuntimeError(f"Container tool versions differ: {output.strip()}")
    codelinter = COMMAND_LINE_TOOLS / "codelinter/bin/codelinter"
    version = subprocess.run(
        [str(codelinter), "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode or version.stdout.strip() != runtime["codelinter"]:
        raise RuntimeError("CodeLinter version differs from the protocol")
    node_bin = runtime.get("node_bin")
    if node_bin:
        node = subprocess.run(
            [str(Path(node_bin) / "node"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if node.returncode or node.stdout.strip() != runtime.get("node"):
            raise RuntimeError("DevEco Node version differs from the retry protocol")
    artifacts = {
        "codelinter_config_sha256": homecheck.DEFAULT_CONFIG,
        "codelinter_overlay_sha256": homecheck.DEFAULT_OVERLAY,
        "openharmony_sdk_manifest_sha256": SDK_ROOT / "install_manifest.json",
        "sdk_build_components_sha256": SDK_ROOT
        / "install_manifest_build_components.json",
        "sdk_native_api20_sha256": SDK_ROOT / "install_manifest_native_api20.json",
    }
    observed = {
        key: sha256_file(path) if path.is_file() else None
        for key, path in artifacts.items()
    }
    drift = {
        key: {"expected": runtime.get(key), "observed": checksum}
        for key, checksum in observed.items()
        if runtime.get(key) != checksum
    }
    if drift:
        raise RuntimeError(f"Frozen scanner/SDK artifacts drifted: {drift}")
    return {
        "image": image,
        "image_id": image_id,
        "versions": {
            "container_codex": output.strip(),
            "host_python": python_output.strip(),
        },
        "artifact_hashes": observed,
        "host_node": runtime.get("node"),
    }


def configure_host_runtime(protocol: dict[str, Any]) -> dict[str, str]:
    """Pin evaluator subprocesses to the audited DevEco Node when specified."""
    node_bin = protocol["runtime"].get("node_bin")
    if not node_bin:
        return {"path_policy": "inherited"}
    node_dir = Path(node_bin).resolve()
    if not (node_dir / "node").is_file():
        raise RuntimeError(f"Frozen DevEco Node is missing: {node_dir / 'node'}")
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{node_dir}:{existing}"
    return {"path_policy": "prepended_frozen_deveco_node", "node_bin": str(node_dir)}


def findings_from_scan(scan: dict[str, Any]) -> list[dict[str, Any]]:
    if scan.get("status") != "scanned":
        raise RuntimeError(f"HomeCheck scan failed: {scan}")
    return read_json(Path(scan["findings_path"]))


def per_rule_deltas(deltas: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counts = {
        key: Counter(item["rule"] for item in values) for key, values in deltas.items()
    }
    rules = set().union(*(counter.keys() for counter in counts.values()))
    return [
        {
            "rule": rule,
            "eliminated_alerts": counts["eliminated"][rule],
            "remaining_alerts": counts["remaining"][rule],
            "introduced_alerts": counts["introduced"][rule],
        }
        for rule in sorted(rules)
    ]


def prompt_for_attempt(
    project: dict[str, Any],
    round_number: int,
    attempt: int,
    workspace: Path,
    plan: Path,
    completion: Path,
    schema: Path,
    feedback: str,
    build_available: bool,
) -> str:
    return f"""
Repair the OpenHarmony ArkTS/TypeScript project in /workspace using the evaluator's
current HomeCheck findings. This is independent coding-agent evaluation: do not use
or search for HapRepair, Skills, repair guides, semantic specifications, retrieval
output, reference patches, independent-oracle data, prior experiment runs, or the
host repository. You may inspect /rules, which is the permitted HomeCheck checker
implementation and documentation, and the mounted SDK/toolchain.

Project: {project["name"]}; frozen commit: {project.get("commit")}
Active round: {round_number}; attempt in this round: {attempt}
Plan: {workspace_container_path(plan, workspace)}
Completion report: {workspace_container_path(completion, workspace)}
Exact completion JSON Schema: {workspace_container_path(schema, workspace)}
Evaluator feedback: {feedback}
Build gate: {"available" if build_available else "unavailable; static validation only"}

Inspect every listed rule/file group and every exact location before validation.
Account for same-file multi-rule clusters jointly. Preserve declarations, exports,
public APIs, behavior, resource schemas, and initialization order. Do not delete or
rename code merely to silence a finding. Retained edits from a failed preflight or
build remain the current candidate and must be repaired in place.

Read the completion schema and obey it exactly. Write JSON at the completion path
with selected_rules, entity_repairs, blocked, and unresolved_external. Each
entity_repairs entry must use the exact key "evidence" whose value is a non-empty
JSON array of repository-evidence strings. Do not use "repository_evidence" and do
not use a scalar string. Include rule, relative_path, every exact location,
non-empty entities, a concrete transformation, and status "repaired".
selected_rules must exactly match the plan. blocked and unresolved_external must
be empty. Make the edits and schema-valid report now.
""".strip()


def run_codex_turn(
    *,
    workspace: Path,
    codex_home: Path,
    build_home: Path,
    prompt: str,
    trace_path: Path,
    stderr_path: Path,
    model: dict[str, Any],
    thread_id: str | None,
    image: str,
) -> dict[str, Any]:
    common = [
        "--json",
        "--model",
        model["requested_id"],
        "-c",
        f'model_provider="{model["provider"]}"',
        "-c",
        f'model_reasoning_effort="{model["reasoning_effort"]}"',
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    codex = (
        ["codex", "exec", *common, "--cd", "/workspace", "-"]
        if thread_id is None
        else ["codex", "exec", "resume", *common, thread_id, "-"]
    )
    command = [
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
        "HOME=/build-home",
        "--env",
        "HVIGOR_USER_HOME=/build-home",
        "--env",
        "LD_LIBRARY_PATH=/opt/haprepair-runtime",
        "--volume",
        f"{workspace}:/workspace:rw",
        "--volume",
        f"{codex_home}:/codex-home:rw",
        "--volume",
        f"{HOST_CODEX_HOME / 'auth.json'}:/codex-home/auth.json:ro",
        "--volume",
        f"{build_home}:/build-home:rw",
        "--volume",
        f"{RULES_ROOT}:/rules:ro",
        "--volume",
        f"{COMMAND_LINE_TOOLS}:{COMMAND_LINE_TOOLS}:ro",
        "--volume",
        f"{SDK_ROOT}:{SDK_ROOT}:ro",
        "--volume",
        f"{HOST_JAVA_HOME}:/opt/java:ro",
        "--volume",
        f"{RUNTIME_LIB_DIR}:/opt/haprepair-runtime:ro",
        image,
        *codex,
    ]
    started = time.monotonic()
    result = subprocess.run(
        command, input=prompt, capture_output=True, text=True, check=False
    )
    remove_empty_auth_mount_placeholder(codex_home)
    trace_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    events = parse_codex_events(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Codex turn exited {result.returncode}; see {stderr_path}")
    return {
        "elapsed_seconds": time.monotonic() - started,
        "thread_id": extract_thread_id(events) or thread_id,
        "usage": extract_usage(events),
        "commands": extract_commands(events),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "stderr_path": str(stderr_path),
    }


def initialize_scanner(workspace: Path, state: Path, maximum: int) -> dict[str, Any]:
    return homecheck.init_session(
        argparse.Namespace(
            workspace=workspace,
            state_dir=state,
            max_validation_scans=maximum,
            codelinter=COMMAND_LINE_TOOLS / "codelinter/bin/codelinter",
            config=homecheck.DEFAULT_CONFIG,
            overlay=homecheck.DEFAULT_OVERLAY,
            allow_unverified_codelinter=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    inputs_path = args.project_manifest.resolve()
    protocol, project = load_contract(protocol_path, inputs_path, args.project)
    if args.run_id != protocol["formal_run"]["run_id"]:
        raise SystemExit("Run ID differs from the frozen protocol")
    source = Path(project["source_path"]).resolve()
    frozen_findings = Path(project["frozen_target_findings_path"]).resolve()
    source_hash = sha256_tree(tree_manifest(source))
    finding_hash = sha256_file(frozen_findings)
    identity = {
        "project": args.project,
        "source_tree_sha256": source_hash,
        "frozen_source_tree_sha256": project["frozen_input_tree_sha256"],
        "findings_sha256": finding_hash,
        "frozen_findings_sha256": project["frozen_target_findings_sha256"],
        "byte_identical_input": source_hash == project["frozen_input_tree_sha256"],
        "finding_identity_matches": finding_hash
        == project["frozen_target_findings_sha256"],
    }
    if not identity["byte_identical_input"] or not identity["finding_identity_matches"]:
        raise RuntimeError(f"Frozen input drift for {args.project}: {identity}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_verified",
                    "experiment": protocol["experiment"],
                    "run_id": args.run_id,
                    "condition": "vanilla",
                    **identity,
                    "model": protocol["model"],
                    "isolation": {
                        "agent_mounts": [
                            "workspace:rw",
                            "run-local Codex home:rw",
                            "build home:rw",
                            "HomeCheck rules:ro",
                            "SDK/toolchain:ro",
                        ],
                        "haprepair_repository_mounted": False,
                        "skill_state_mounted": False,
                        "corpus_mounted": False,
                        "other_runs_mounted": False,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    authorization = validate_authorization(protocol_path, inputs_path, args.run_id)
    runtime = validate_runtime(protocol, args.container_image)
    host_runtime = configure_host_runtime(protocol)
    run_dir = args.run_root.resolve() / args.run_id / "vanilla" / args.project
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "run_id": args.run_id,
        "condition": "vanilla",
        "project": args.project,
        "status": "preparing",
        "started_at": utc_now(),
        **identity,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "inputs": str(inputs_path),
        "inputs_sha256": sha256_file(inputs_path),
        "runtime": runtime,
        "host_runtime": host_runtime,
        "static_g2_authorization": authorization,
        "model": protocol["model"],
        "rounds": [],
    }
    write_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        workspace = run_dir / "workspace"
        copy_project(source, workspace)
        codex_home = run_dir / "codex_home"
        prepare_codex_home(codex_home)
        build_home = run_dir / "build_home"
        build_home.mkdir()
        traces = run_dir / "traces"
        traces.mkdir()
        evaluator_state = run_dir / "evaluator_state"
        evaluator_state.mkdir()
        baseline_snapshot = guards.snapshot_workspace(
            workspace, evaluator_state / "snapshots/initial"
        )
        public_baseline = guards.public_api_inventory(workspace)
        write_json(evaluator_state / "public_api_baseline.json", public_baseline)
        availability = project["validation_availability"]["build"] == "available"
        build_setup = (
            prepare_build_gate(args.project, workspace, run_dir / "build_setup")
            if availability
            else {"available": False, "test_available": False}
        )
        common_contract = protocol["common_contract"]
        maximum = int(
            common_contract.get(
                "maximum_validation_scans",
                common_contract.get("maximum_post_edit_validation_scans"),
            )
        )
        maximum_agent_turns = int(
            protocol["scheduling"].get(
                "maximum_agent_turns_per_round", MAX_AGENT_TURNS_PER_ROUND
            )
        )
        if maximum_agent_turns < 1:
            raise RuntimeError("Per-round agent-turn allowance must be positive")
        manifest["maximum_validation_scans"] = maximum
        manifest["maximum_agent_turns_per_round"] = maximum_agent_turns
        scanner_dir = evaluator_state / "homecheck"
        initialize_scanner(workspace, scanner_dir, maximum)
        initial_scan = homecheck.scan_session(
            argparse.Namespace(state_dir=scanner_dir, kind="initial")
        )
        initial_findings = findings_from_scan(initial_scan)
        exact, exact_detail = verify_initial_findings(
            read_json(frozen_findings), initial_findings
        )
        if not exact:
            raise RuntimeError(
                f"Initial localization differs from frozen input: {exact_detail}"
            )

        last_valid = baseline_snapshot
        last_valid_round = 0
        best_valid = baseline_snapshot
        best_valid_round = 0
        best_score = [len(initial_findings), 0, 0, 0]
        all_commands: list[dict[str, Any]] = []
        usage = Counter()
        coverage_retry_count = 0
        no_diff_retry_count = 0
        repair_required_round_count = 0
        interactions: list[dict[str, Any]] = []
        current_findings = initial_findings

        for round_number in range(1, maximum + 1):
            if not current_findings:
                break
            round_dir = evaluator_state / "rounds" / f"round_{round_number:02d}"
            round_dir.mkdir(parents=True)
            round_snapshot = guards.snapshot_workspace(
                workspace, round_dir / "snapshot"
            )
            plan = make_reference_plan(current_findings)
            control = workspace / ".exp_agent" / f"round_{round_number:02d}"
            control.mkdir(parents=True, exist_ok=True)
            plan_path = control / "plan.json"
            completion_path = control / "completion.json"
            schema_path = control / "completion-schema.json"
            write_json(plan_path, plan)
            write_json(schema_path, completion_schema())
            thread_id: str | None = None
            attempts = []
            feedback = "Complete the frozen plan."
            passed = False
            round_no_diff = 0
            round_repair_required = False
            for attempt in range(1, maximum_agent_turns + 1):
                completion_path.unlink(missing_ok=True)
                turn = run_codex_turn(
                    workspace=workspace,
                    codex_home=codex_home,
                    build_home=build_home,
                    prompt=prompt_for_attempt(
                        project,
                        round_number,
                        attempt,
                        workspace,
                        plan_path,
                        completion_path,
                        schema_path,
                        feedback,
                        availability,
                    ),
                    trace_path=traces
                    / f"round_{round_number:02d}_attempt_{attempt:02d}.jsonl",
                    stderr_path=traces
                    / f"round_{round_number:02d}_attempt_{attempt:02d}.stderr.log",
                    model=protocol["model"],
                    thread_id=thread_id,
                    image=args.container_image,
                )
                thread_id = turn["thread_id"]
                all_commands.extend(turn["commands"])
                usage.update(turn["usage"])
                restricted = [
                    item
                    for item in turn["commands"]
                    if is_restricted_command(item["command"])
                ]
                if restricted:
                    raise RuntimeError(f"Restricted artifact access: {restricted}")
                report = read_json(completion_path) if completion_path.is_file() else {}
                coverage = audit_completion(plan, report)
                edit = guards.source_diff(
                    Path(round_snapshot["files_dir"]),
                    workspace,
                    round_dir / f"attempt_{attempt:02d}.patch",
                )
                attempt_record: dict[str, Any] = {
                    "attempt": attempt,
                    "thread_id": thread_id,
                    "turn": turn,
                    "coverage": coverage,
                    "edit": edit,
                }
                if not coverage["complete"]:
                    coverage_retry_count += 1
                    feedback = coverage["feedback"]
                    attempts.append(attempt_record)
                    continue
                if edit["changed_source_file_count"] == 0:
                    round_no_diff += 1
                    no_diff_retry_count += 1
                    feedback = "No ArkTS/TypeScript source diff exists; make the required source repairs."
                    attempts.append(attempt_record)
                    if round_no_diff > int(
                        protocol["scheduling"]["maximum_no_diff_retries_per_round"]
                    ):
                        raise RuntimeError("Bounded no-diff retries exhausted")
                    continue
                preflight_dir = round_dir / f"preflight_{attempt:02d}"
                preflight_dir.mkdir()
                structure = guards.structural_guard(
                    Path(last_valid["files_dir"]), workspace, preflight_dir
                )
                observed_api = (
                    guards.public_api_inventory(workspace)
                    if structure["status"] == "passed"
                    else {"api": {}}
                )
                public = (
                    guards.compare_public_api(public_baseline, observed_api)
                    if structure["status"] == "passed"
                    else {"status": "skipped"}
                )
                namespace = guards.namespace_export_guard(
                    Path(round_snapshot["files_dir"]), workspace
                )
                failures = []
                if structure["status"] != "passed":
                    failures.append("structural/declaration guard failed")
                if public["status"] != "passed":
                    failures.append("public API guard failed")
                if namespace["status"] != "passed":
                    failures.extend(namespace["failures"])
                attempt_record["preflight"] = {
                    "structure": structure,
                    "public_api": public,
                    "namespace_export": namespace,
                    "failures": failures,
                }
                if failures:
                    round_repair_required = True
                    feedback = "Repair the retained candidate in place: " + "; ".join(
                        failures
                    )
                    attempts.append(attempt_record)
                    continue
                build = run_build_gate(
                    args.project,
                    workspace,
                    round_dir / "build",
                    build_setup,
                    label=f"attempt_{attempt:02d}",
                )
                attempt_record["build_gate"] = build
                if build["status"] == "failed":
                    round_repair_required = True
                    feedback = "Evaluator build failed. Repair the retained candidate in place using the recorded build log."
                    attempts.append(attempt_record)
                    continue
                passed = True
                attempts.append(attempt_record)
                break
            if not passed:
                repair_required_round_count += 1
                write_json(
                    round_dir / "round.json",
                    {
                        "round": round_number,
                        "status": "bounded_edit_retry_exhausted",
                        "candidate_retained": True,
                        "attempts": attempts,
                    },
                )
                raise RuntimeError(
                    f"Bounded edit retries exhausted in round {round_number}"
                )
            if round_repair_required:
                repair_required_round_count += 1

            validation = homecheck.scan_session(
                argparse.Namespace(state_dir=scanner_dir, kind="validation")
            )
            next_findings = findings_from_scan(validation)
            round_metrics, round_deltas = compute_alert_metrics(
                current_findings, next_findings
            )
            total_metrics, _ = compute_alert_metrics(initial_findings, next_findings)
            total_diff = source_diff(
                source, workspace, round_dir / "candidate_total.patch"
            )
            score = candidate_score(
                total_metrics, total_diff["changed_source_file_count"], round_number
            )
            accepted = guards.snapshot_workspace(
                workspace,
                evaluator_state / "snapshots" / f"accepted_{round_number:02d}",
            )
            last_valid, last_valid_round = accepted, round_number
            best_updated = score < best_score
            if best_updated:
                best_valid, best_valid_round, best_score = accepted, round_number, score
            interaction = describe_interactions(round_number, round_deltas)
            if interaction:
                interactions.append(interaction)
            record = {
                "round": round_number,
                "status": "accepted",
                "thread_id": thread_id,
                "attempts": attempts,
                "validation_scan": validation,
                "round_metrics": round_metrics,
                "total_metrics": total_metrics,
                "candidate_score": score,
                "best_valid_updated": best_updated,
                "interaction": interaction,
            }
            manifest["rounds"].append(record)
            write_json(round_dir / "round.json", record)
            write_json(manifest_path, manifest)
            current_findings = next_findings

        latest_is_best = last_valid_round == best_valid_round
        if not latest_is_best:
            guards.snapshot_workspace(
                workspace, evaluator_state / "snapshots/abandoned_final_candidate"
            )
            guards.restore_snapshot(workspace, best_valid)
            selection = {
                "status": "restored_best_valid_candidate",
                "latest_valid_round": last_valid_round,
                "selected_round": best_valid_round,
                "selected_score": best_score,
            }
        else:
            selection = {
                "status": "latest_candidate_is_best_valid",
                "latest_valid_round": last_valid_round,
                "selected_round": best_valid_round,
                "selected_score": best_score,
            }
        final_scan = homecheck.scan_session(
            argparse.Namespace(state_dir=scanner_dir, kind="final")
        )
        final_findings = findings_from_scan(final_scan)
        final_metrics, final_deltas = compute_alert_metrics(
            initial_findings, final_findings
        )
        scanner_state = read_json(scanner_dir / "state.json")
        restricted = [
            item for item in all_commands if is_restricted_command(item["command"])
        ]
        final_diff = source_diff(source, workspace, run_dir / "source_changes.patch")
        manifest.update(
            {
                "status": "protocol_violation" if restricted else "completed",
                "completed_at": utc_now(),
                "wall_clock_seconds": time.monotonic() - started,
                "alert_metrics": final_metrics,
                "per_rule_alert_deltas": per_rule_deltas(final_deltas),
                "validation_scan_count": scanner_state["validation_scans_consumed"],
                "coverage_retry_count": coverage_retry_count,
                "no_diff_retry_count": no_diff_retry_count,
                "repair_required_round_count": repair_required_round_count,
                "best_valid_round": best_valid_round,
                "best_valid_score": best_score,
                "final_candidate_selection": selection,
                "observed_rule_interactions": interactions,
                "build_status": manifest["rounds"][-1]["attempts"][-1]["build_gate"][
                    "status"
                ]
                if manifest["rounds"]
                else "not_run",
                "test_status": "not_available",
                "input_tokens": usage["input_tokens"],
                "cached_input_tokens": usage["cached_input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "api_cost_if_traceable": None,
                "restricted_accesses": restricted,
                "source_diff": final_diff,
                "isolation": {
                    "haprepair_repository_mounted": False,
                    "skill_state_mounted": False,
                    "corpus_mounted": False,
                    "other_runs_mounted": False,
                },
                "provider_endpoint_fingerprint": provider_endpoint_fingerprint(
                    codex_home, protocol["model"]["provider"]
                ),
            }
        )
        write_json(manifest_path, manifest)
        if restricted:
            raise RuntimeError("Restricted access detected after trace audit")
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "failure": f"{type(error).__name__}: {error}",
                "wall_clock_seconds": time.monotonic() - started,
            }
        )
        write_json(manifest_path, manifest)
        raise
    print(manifest_path)


if __name__ == "__main__":
    main()
