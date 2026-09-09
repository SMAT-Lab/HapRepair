#!/usr/bin/env python3
"""Run one isolated HapRepair Skill or coding-agent reference condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HAPREPAIR_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
BASELINE_DIR = HERE.parent / "coding_agent_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from build_gate import prepare_build_gate  # type: ignore  # noqa: E402
from preflight_builds import build_environment  # type: ignore  # noqa: E402
from run_agent_baseline import (  # type: ignore  # noqa: E402
    HOST_CODEX_HOME,
    RULES_ROOT,
    classify_command,
    command_status,
    copy_project,
    docker_image_identity,
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
)
from validation_gate import write_json  # type: ignore  # noqa: E402


DEFAULT_PROTOCOL = HERE / "protocol-hapskill-35-v5-image-diagnostic.json"
DEFAULT_SELECTION = BASELINE_DIR / "selected_projects.json"
DEFAULT_CANDIDATES = (
    BASELINE_DIR / "scan_runs" / "candidate_scan_01" / "scan_manifest.json"
)
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "runs"
DEFAULT_IMAGE = "hybrid-gym-codex:0.146.0"
DEFAULT_RETRIEVAL_PYTHON = Path("/data/zhihao/miniconda3/bin/python")
SKILL_SCRIPT = (
    HAPREPAIR_ROOT
    / "skills"
    / "haprepair-openharmony-repair-v5-dev"
    / "scripts"
    / "haprepair_skill.py"
)
SKILL_PROTOCOL = (
    HAPREPAIR_ROOT
    / "skills"
    / "haprepair-openharmony-repair-v5-dev"
    / "references"
    / "protocol-v5.json"
)
BROKER_SCRIPT = HERE / "skill_broker.py"
CLIENT_SCRIPT = HERE / "skill_client.py"
MODEL_CACHE = WORKSPACE_ROOT / "model_cache" / "huggingface"
EMBEDDING_CACHE = WORKSPACE_ROOT / "model_cache" / "haprepair"
CORPUS = HAPREPAIR_ROOT / "revision" / "knowledge_base" / "rule_complete_383.jsonl"
CODELINTER = Path("/home/zhihao/deveco/command-line-tools/codelinter/bin/codelinter")
DEVECO_NODE_BIN = Path("/home/zhihao/deveco/command-line-tools/tool/node/bin")
CODELINTER_CONFIG = HAPREPAIR_ROOT / "revision" / "code-linter.json5"
CODELINTER_OVERLAY = (
    HAPREPAIR_ROOT / "revision" / "independent_oracle" / "codelinter_overlay.json"
)

CONDITIONS = {"vanilla", "hapskill"}
RESTRICTED_MARKERS = (
    "/haprepair/",
    "/baseline_data/",
    "rule_complete_383",
    "knowledge_base",
    "independent_oracle",
    "reference_patch",
    "exp_agent_10",
)
RETRIEVAL_RULE_ALIASES = {
    "@performance/init-list-component": "@hw-ets-eslint/init-list-component",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_skill_protocol(protocol: dict[str, Any], protocol_path: Path) -> Path:
    configured = protocol.get("retrieval", {}).get("skill_protocol_path")
    path = (
        (protocol_path.parent / configured).resolve()
        if configured
        else SKILL_PROTOCOL.resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"Skill protocol is missing: {path}")
    expected = protocol.get("retrieval", {}).get("skill_protocol_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError("Skill protocol hash does not match the experiment protocol")
    return path


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode("utf-8"))
    return digest.hexdigest()


def project_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("projects", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("project manifest has no projects/results list")


def load_project(name: str, manifest_path: Path) -> dict[str, Any]:
    matches = [
        item
        for item in project_list(read_json(manifest_path))
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one project named {name!r} in {manifest_path}")
    return matches[0]


def run_skill(state_dir: Path | None, *arguments: str) -> dict[str, Any]:
    command = [sys.executable, str(SKILL_SCRIPT), *arguments]
    environment = os.environ.copy()
    environment["PATH"] = f"{DEVECO_NODE_BIN}:{environment.get('PATH', '')}"
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"HapRepair operation returned invalid JSON: {result.stderr[-2000:]}"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            payload.get("error", f"HapRepair operation exited {result.returncode}")
        )
    return payload


def load_target_findings(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(Path(path))
    if not isinstance(value, list):
        raise ValueError(f"finding artifact is not a JSON list: {path}")
    return value


def validate_frozen_runtime(protocol: dict[str, Any], image: str) -> dict[str, str]:
    runtime = protocol.get("runtime")
    identity = docker_image_identity(image)
    if not isinstance(runtime, dict):
        return identity
    if image != runtime.get("container_image"):
        raise RuntimeError("Container image name differs from the frozen protocol")
    if identity["image_id"] != runtime.get("container_image_id"):
        raise RuntimeError("Container image ID differs from the frozen protocol")
    pinned_files = {
        "codelinter_config_sha256": CODELINTER_CONFIG,
        "codelinter_overlay_sha256": CODELINTER_OVERLAY,
    }
    for key, path in pinned_files.items():
        if not path.is_file() or sha256_file(path) != runtime.get(key):
            raise RuntimeError(f"Frozen runtime artifact drift: {path}")
    version = subprocess.run(
        [str(CODELINTER), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != runtime.get("codelinter"):
        raise RuntimeError("CodeLinter version differs from the frozen protocol")
    node = subprocess.run(
        [str(DEVECO_NODE_BIN / "node"), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if runtime.get("node") and (
        node.returncode != 0 or node.stdout.strip() != runtime["node"]
    ):
        raise RuntimeError("DevEco Node version differs from the frozen protocol")
    return {
        **identity,
        "codelinter": version.stdout.strip(),
        "node": node.stdout.strip() if node.returncode == 0 else "unavailable",
        "node_bin": str(DEVECO_NODE_BIN),
    }


def prepare_frozen_validation_gate(
    project: dict[str, Any], workspace: Path, output_dir: Path
) -> dict[str, Any]:
    """Honor the input freeze's build/test availability classification."""
    availability = project.get("validation_availability")
    if not isinstance(availability, dict):
        return prepare_build_gate(project["name"], workspace, output_dir)
    if availability.get("build") == "available":
        return prepare_build_gate(project["name"], workspace, output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    setup = {
        "schema_version": 1,
        "project": project["name"],
        "available": False,
        "test_available": availability.get("test") == "available",
        "validation_scope": availability.get("scope", "statically validated only"),
        "frozen_availability": availability,
        "note": (
            "No evaluator build command is configured for this frozen input; "
            "the run remains statically validated only."
        ),
    }
    write_json(output_dir / "setup.json", setup)
    return setup


def validation_wrapper(
    run_dir: Path, project_name: str, build_setup: dict[str, Any]
) -> str | None:
    if not build_setup.get("available"):
        return None
    wrapper = run_dir / "evaluator_build.sh"
    environment = build_environment(project_name)
    environment.update(build_setup.get("environment") or {})
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
        "PATH",
    )
    assignments = [f"{key}={environment[key]}" for key in keys if key in environment]
    lines = [
        "#!/bin/sh",
        "set -eu",
        "exec env " + shlex.join([*assignments, *build_setup["build_command"]]),
    ]
    wrapper.write_text("\n".join(lines) + "\n", encoding="utf-8")
    wrapper.chmod(0o700)
    return str(wrapper)


def validate_retrieval_runtime(python: Path) -> dict[str, str]:
    resolved = python.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Retrieval Python is missing: {resolved}")
    probe = subprocess.run(
        [
            str(resolved),
            "-c",
            (
                "import json, sys, torch, transformers; "
                "print(json.dumps({'executable': sys.executable, "
                "'python': sys.version.split()[0], 'torch': torch.__version__, "
                "'transformers': transformers.__version__}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()[-2000:]
        raise RuntimeError(
            f"Stella dependency preflight failed for {resolved}: {detail}"
        )
    try:
        identity = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Retrieval Python returned invalid dependency identity: {probe.stdout!r}"
        ) from error
    identity["requested_executable"] = str(resolved)
    return identity


def start_broker(
    run_dir: Path, state_dir: Path, device: str, retrieval_python: Path
) -> tuple[subprocess.Popen[bytes], Path, Any, Any, tempfile.TemporaryDirectory[str]]:
    broker_dir = run_dir / "broker"
    broker_dir.mkdir()
    socket_runtime = tempfile.TemporaryDirectory(prefix="hapskill_broker_")
    socket_path = Path(socket_runtime.name) / "broker.sock"
    stdout_stream = (broker_dir / "stdout.log").open("wb")
    stderr_stream = (broker_dir / "stderr.log").open("wb")
    command = [
        str(retrieval_python.resolve()),
        str(BROKER_SCRIPT),
        "--socket",
        str(socket_path),
        "--state-dir",
        str(state_dir),
        "--skill-script",
        str(SKILL_SCRIPT),
        "--model-cache",
        str(MODEL_CACHE),
        "--embedding-cache",
        str(EMBEDDING_CACHE),
        "--device",
        device,
        "--log",
        str(broker_dir / "broker_trace.jsonl"),
    ]
    broker_environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = subprocess.Popen(
        command,
        stdout=stdout_stream,
        stderr=stderr_stream,
        env=broker_environment,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if socket_path.exists():
            return process, socket_path, stdout_stream, stderr_stream, socket_runtime
        if process.poll() is not None:
            break
        time.sleep(0.05)
    stdout_stream.close()
    stderr_stream.close()
    socket_runtime.cleanup()
    raise RuntimeError("HapRepair broker failed to create its socket")


def stop_broker(
    process: subprocess.Popen[bytes],
    socket_path: Path,
    stdout_stream: Any,
    stderr_stream: Any,
    socket_runtime: tempfile.TemporaryDirectory[str],
) -> None:
    if process.poll() is None and socket_path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(b'{"operation":"shutdown"}\n')
                client.shutdown(socket.SHUT_WR)
                client.recv(4096)
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)
    stdout_stream.close()
    stderr_stream.close()
    socket_runtime.cleanup()


def run_codex_turn(
    workspace: Path,
    codex_home: Path,
    prompt: str,
    trace_path: Path,
    stderr_path: Path,
    *,
    model: dict[str, Any],
    thread_id: str | None,
    image: str,
    condition: str,
    broker_socket: Path | None,
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
    codex_command = (
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
        "--volume",
        f"{workspace}:/workspace:rw",
        "--volume",
        f"{codex_home}:/codex-home:rw",
        "--volume",
        f"{HOST_CODEX_HOME / 'auth.json'}:/codex-home/auth.json:ro",
        "--volume",
        f"{RULES_ROOT}:/rules:ro",
    ]
    if condition == "hapskill":
        if broker_socket is None:
            raise RuntimeError("HapRepair condition requires a broker socket")
        command.extend(
            [
                "--volume",
                f"{broker_socket.parent}:/hapskill-broker:rw",
                "--volume",
                f"{CLIENT_SCRIPT}:/hapskill-client.py:ro",
            ]
        )
    command.extend([image, *codex_command])
    started = time.monotonic()
    result = subprocess.run(
        command,
        input=prompt,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    remove_empty_auth_mount_placeholder(codex_home)
    trace_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    events = parse_codex_events(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex turn failed with exit {result.returncode}; see {stderr_path}"
        )
    return {
        "elapsed_seconds": time.monotonic() - started,
        "thread_id": extract_thread_id(events) or thread_id,
        "usage": extract_usage(events),
        "commands": extract_commands(events),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "stderr_path": str(stderr_path),
    }


def prompt_for_round(
    *,
    project: dict[str, Any],
    condition: str,
    round_number: int,
    max_rounds: int,
    localization_path: Path,
    round_plan_path: Path | None = None,
    round_completion_path: Path | None = None,
    previous_feedback: str,
    build_available: bool,
    edit_attempt: int = 1,
    experiment_name: str = "EXP-HAPSKILL-10",
    retrieval_required: bool = True,
    recovery_only: bool = False,
    preflight_evidence: str | None = None,
) -> str:
    condition_text = (
        "You are the vanilla coding-agent condition. No HapRepair retrieval capability is available."
        if condition == "vanilla"
        else f"""You are the agent-integrated HapRepair condition. A restricted HapRepair broker is available through:
  python3 /hapskill-client.py inspect-rule --rule <rule-id>
  python3 /hapskill-client.py retrieve-repairs --request-file <json-file>
For retrieval, write a JSON list containing objects with `rule`, `context`, and `top_k: 1`. Gather the context adaptively from the repository first, and batch distinct-rule requests when practical. {"For every corpus-covered rule listed by the round plan, request one Top-1 example before validation, even if repository evidence later blocks editing that rule." if retrieval_required else "Retrieval is optional: request one Top-1 example only when rule-specific evidence would help the repair."} That one successful same-rule retrieval may guide all inspected homogeneous instances of the rule in this round; do not retrieve per alert. Before editing, state whether each returned example applies and why; adapt or reject it when repository evidence or a reported sibling-rule interaction conflicts. Retrieved examples are demonstrations, not patches to copy blindly. Only report retrieval when the broker actually returned an example. The HapRepair repository and corpus are intentionally not mounted."""
    )
    mode_text = (
        "This is a gate-recovery-only turn. Repair every reported structural or public-API "
        "violation in the retained candidate. Do not work on a new HomeCheck rule."
        if recovery_only
        else "This is a complete-coverage repair turn. Attempt every rule and affected file in the frozen round plan before validation."
    )
    return f"""
You are repairing HomeCheck performance and security alerts in an isolated
OpenHarmony/ArkTS project for {experiment_name}. This is edit round
{round_number} of at most {max_rounds}, edit attempt {edit_attempt} within this round.

Current evaluator localization is at:
  /workspace/{localization_path.relative_to(localization_path.parents[1]).as_posix()}

{condition_text}

Evaluator feedback from the preceding round:
{previous_feedback}

Task mode:
{mode_text}
{f"Preflight evidence: {preflight_evidence}" if preflight_evidence else ""}

Round target plan:
{f"  /workspace/{round_plan_path.relative_to(round_plan_path.parents[1]).as_posix()}" if round_plan_path else "  not configured"}

Round completion report:
{f"  /workspace/{round_completion_path.relative_to(round_completion_path.parents[1]).as_posix()}" if round_completion_path else "  not configured"}

Constraints:
- Inspect affected project files before editing. Use rg and targeted reads to
  follow definitions, references, imports, resources, state ownership, and call sites.
- Count alerts by rule and file before choosing work. Prioritize the largest
  high-confidence homogeneous rule clusters under the remaining validation budget.
  Treat strongly co-located sibling rules as one repair cluster. The round plan
  contains every currently localized rule. For every required rule, inspect every
  affected file and handle every localized instance in that file; do not stop after
  a convenient subset. The smallest behavior-preserving repair applies per instance;
  it is not a limit on the number of files handled.
- Use a mechanical batch edit when readable checker logic and representative files
  establish one homogeneous transformation. Review the generated diff and exceptions
  instead of re-deriving the same predicate independently for every instance.
- Before stopping, write the completion report as JSON with `selected_rules`,
  `completed_files` (rule to relative-path list), and `blocked` entries containing
  `rule`, `relative_path`, and a concrete `reason`. A completed file means every
  localized instance of that selected rule in the file was inspected. Missing file
  coverage keeps this edit round open and consumes no HomeCheck validation scan.
  In gate-recovery-only mode, restore the reported structural/API invariants instead;
  the existing alert-coverage report remains unchanged.
- Rule documentation and checker implementations are under /rules.
- Work only in /workspace. Do not use network lookup for project or repair evidence.
- Do not run CodeLinter/HomeCheck. The evaluator owns all scans and will provide
  the residual and introduced alert set after this turn.
- Do not access reference patches, held-out cases, another condition's workspace,
  traces, outputs, or any unmounted HapRepair artifact.
- Builds and existing tests may be run as useful. Do not create tests that encode
  the supplied alert list.
- Do not delete an enclosing function, class, or component to suppress an alert.
- Preserve package-exported signatures and interfaces.
- Focus on the supplied alerts and avoid unrelated refactoring.
- When the preceding feedback says that repair is required, the evaluator has
  intentionally retained that failed candidate. Repair the current workspace in
  place; do not reconstruct the pinned source or discard otherwise useful edits.

Make one concrete source repair pass now, then stop and summarize only edits and
build/test commands supported by this turn's tool trace. Do not claim retrieval
without a successful broker result or alert elimination without evaluator validation.

Pinned project: {project["name"]}
Commit: {project.get("commit")}
Tree OID: {project.get("tree_oid")}
Evaluator build gate: {"available" if build_available else "not available; static validation only"}
""".strip()


def per_rule_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("rule", "")) for item in findings).items()))


def build_round_plan(
    findings: list[dict[str, Any]], *, overlap_threshold: float = 0.5
) -> dict[str, Any]:
    """Freeze every current rule and affected file for complete-round coverage."""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    locations: dict[str, set[tuple[str, int]]] = {}
    for finding in findings:
        rule = str(finding["rule"])
        by_rule.setdefault(rule, []).append(finding)
        locations.setdefault(rule, set()).add(
            (str(finding["relative_path"]), int(finding["line"]))
        )
    if not by_rule:
        return {
            "required_rules": [],
            "rule_groups": {},
            "overlap_threshold": overlap_threshold,
        }
    primary = min(by_rule, key=lambda rule: (-len(by_rule[rule]), rule))
    required = set(by_rule)
    groups: dict[str, Any] = {}
    for rule in sorted(required):
        file_counts = Counter(str(item["relative_path"]) for item in by_rule[rule])
        groups[rule] = {
            "alert_count": len(by_rule[rule]),
            "affected_file_count": len(file_counts),
            "files": dict(sorted(file_counts.items())),
        }
    return {
        "primary_rule": primary,
        "required_rules": sorted(
            required, key=lambda rule: (-len(by_rule[rule]), rule)
        ),
        "rule_groups": groups,
        "overlap_threshold": overlap_threshold,
        "completion_contract": (
            "For every currently localized rule, account for every affected file as "
            "completed or blocked before evaluator validation."
        ),
    }


def audit_round_completion(plan: dict[str, Any], report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return {
            "complete": False,
            "feedback": f"Missing required round completion report: {report_path}",
        }
    try:
        report = read_json(report_path)
    except (json.JSONDecodeError, OSError) as error:
        return {"complete": False, "feedback": f"Invalid completion report: {error}"}
    selected = set(report.get("selected_rules") or [])
    required = set(plan["required_rules"])
    problems = []
    if not required <= selected:
        problems.append(f"missing required rules: {sorted(required - selected)}")
    completed_raw = report.get("completed_files") or {}
    completed = {
        str(rule): {str(path) for path in paths}
        for rule, paths in completed_raw.items()
        if isinstance(paths, list)
    }
    blocked: dict[str, set[str]] = {}
    for item in report.get("blocked") or []:
        if not isinstance(item, dict) or not str(item.get("reason", "")).strip():
            problems.append("every blocked group needs a non-empty reason")
            continue
        blocked.setdefault(str(item.get("rule", "")), set()).add(
            str(item.get("relative_path", ""))
        )
    for rule in sorted(required):
        expected = set(plan["rule_groups"][rule]["files"])
        accounted = completed.get(rule, set()) | blocked.get(rule, set())
        missing = expected - accounted
        unknown = accounted - expected
        if missing:
            problems.append(f"{rule} has {len(missing)} unaccounted files")
        if unknown:
            problems.append(f"{rule} reports {len(unknown)} unknown files")
    return {
        "complete": not problems,
        "feedback": (
            "Round target coverage accepted."
            if not problems
            else "Round target coverage is incomplete: " + "; ".join(problems)
        ),
        "report": report,
    }


def per_rule_deltas(deltas: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    eliminated = Counter(str(item.get("rule", "")) for item in deltas["eliminated"])
    remaining = Counter(str(item.get("rule", "")) for item in deltas["remaining"])
    introduced = Counter(str(item.get("rule", "")) for item in deltas["introduced"])
    return [
        {
            "rule": rule,
            "eliminated_alerts": eliminated[rule],
            "remaining_alerts": remaining[rule],
            "introduced_alerts": introduced[rule],
        }
        for rule in sorted(eliminated.keys() | remaining.keys() | introduced.keys())
    ]


def audit_retrieval_trace(trace_path: Path) -> dict[str, Any]:
    entries = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retrievals = [
        item for item in entries if item.get("operation") == "retrieve-repairs"
    ]
    valid = []
    successful_rules: set[str] = set()
    for entry in retrievals:
        output_path = Path(entry["output_path"])
        if (
            not output_path.is_file()
            or sha256_file(output_path) != entry["output_sha256"]
        ):
            continue
        output = read_json(output_path)
        backend = output.get("backend", {})
        if (
            entry.get("status") != "error"
            and backend.get("backend") == "stella"
            and backend.get("paper_comparable") is True
        ):
            valid.append(entry)
            for result in output.get("results") or []:
                if result.get("available") and result.get("examples"):
                    successful_rules.add(str(result.get("requested_rule")))
    return {
        "attempt_count": len(retrievals),
        "valid_stella_call_count": len(valid),
        "valid_output_paths": [item["output_path"] for item in valid],
        "successful_rules": sorted(successful_rules),
    }


def covered_target_rules(
    rules: list[str] | set[str], corpus_rules: set[str]
) -> set[str]:
    return {
        rule for rule in rules if RETRIEVAL_RULE_ALIASES.get(rule, rule) in corpus_rules
    }


def materialize_preflight_evidence(
    preflight: dict[str, Any], control: Path, round_number: int, attempt: int
) -> str:
    evidence_path = control / (
        f"preflight_round_{round_number:02d}_attempt_{attempt:02d}.json"
    )
    write_json(evidence_path, preflight)
    patch_value = ((preflight.get("structural_guard") or {}).get("diff") or {}).get(
        "patch_path"
    )
    paths = [f"/workspace/.exp_agent/{evidence_path.name}"]
    if patch_value and Path(patch_value).is_file():
        patch_path = control / (
            f"preflight_round_{round_number:02d}_attempt_{attempt:02d}.patch"
        )
        shutil.copy2(patch_value, patch_path)
        paths.append(f"/workspace/.exp_agent/{patch_path.name}")
    return ", ".join(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--retrieval-python", type=Path, default=DEFAULT_RETRIEVAL_PYTHON
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    manifest_path = args.project_manifest.resolve()
    protocol = read_json(protocol_path)
    skill_protocol = resolve_skill_protocol(protocol, protocol_path)
    project = load_project(args.project, manifest_path)
    max_rounds = int(
        protocol["pilot"]["maximum_post_edit_validation_scans"]
        if args.pilot
        else protocol["common_contract"]["maximum_post_edit_validation_scans"]
    )
    if not args.pilot and "frozen_target_findings_path" not in project:
        raise SystemExit(
            "Formal execution requires a fresh frozen_target_findings_path; run the input freezer first"
        )
    agent_runtime = validate_frozen_runtime(protocol, args.container_image)
    retrieval_runtime = (
        validate_retrieval_runtime(args.retrieval_python)
        if args.condition == "hapskill" and not args.dry_run
        else None
    )

    run_dir = args.run_root.resolve() / args.run_id / args.condition / args.project
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    state_dir = run_dir / "skill_state"
    traces = run_dir / "traces"
    traces.mkdir()
    codex_home = run_dir / "codex_home"
    control = workspace / ".exp_agent"
    source = Path(project["source_path"]).resolve()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": protocol.get("experiment", "EXP-HAPSKILL-10"),
        "paper_facing": bool(protocol.get("paper_facing", not args.pilot)),
        "run_id": args.run_id,
        "condition": args.condition,
        "project": args.project,
        "status": "preparing",
        "started_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "skill_protocol": str(skill_protocol),
        "skill_protocol_sha256": sha256_file(skill_protocol),
        "project_manifest": str(manifest_path),
        "project_manifest_sha256": sha256_file(manifest_path),
        "source": str(source),
        "commit": project.get("commit"),
        "tree_oid": project.get("tree_oid"),
        "model": protocol["model"],
        "max_validation_scans": max_rounds,
        "rounds": [],
        "agent_runtime": agent_runtime,
        "retrieval_runtime": retrieval_runtime,
        "isolation": {
            "haprepair_repository_mounted": False,
            "corpus_mounted": False,
            "reference_artifacts_mounted": False,
            "broker_exposed": args.condition == "hapskill",
        },
    }
    run_manifest_path = run_dir / "run_manifest.json"
    write_json(run_manifest_path, manifest)

    copy_project(source, workspace)
    control.mkdir()
    prepare_codex_home(codex_home)
    manifest["agent_runtime"]["model_endpoint_sha256"] = provider_endpoint_fingerprint(
        codex_home, protocol["model"]["provider"]
    )
    source_tree = tree_manifest(source)
    workspace_tree = tree_manifest(workspace)
    manifest["input_file_count"] = len(source_tree)
    manifest["input_tree_sha256"] = sha256_tree(source_tree)
    manifest["byte_identical_input"] = source_tree == workspace_tree
    if not manifest["byte_identical_input"]:
        manifest["status"] = "copy_verification_failed"
        write_json(run_manifest_path, manifest)
        raise RuntimeError("isolated workspace differs from the pinned source")
    frozen_tree_hash = project.get("frozen_input_tree_sha256")
    if frozen_tree_hash and frozen_tree_hash != manifest["input_tree_sha256"]:
        manifest["status"] = "frozen_input_tree_mismatch"
        write_json(run_manifest_path, manifest)
        raise RuntimeError("pinned source tree differs from the formal input freeze")

    build_setup = (
        prepare_frozen_validation_gate(project, workspace, run_dir / "build_gate")
        if not args.pilot
        else {
            "available": False,
            "test_available": False,
            "validation_scope": "statically validated only",
        }
    )
    build_command = validation_wrapper(run_dir, args.project, build_setup)
    init_arguments = [
        "init-session",
        "--workspace",
        str(workspace),
        "--state-dir",
        str(state_dir),
        "--protocol",
        str(skill_protocol),
        "--max-validation-scans",
        str(max_rounds),
    ]
    if build_command:
        init_arguments.extend(["--build-command", build_command])
    init_result = run_skill(None, *init_arguments)
    initial_result = run_skill(
        state_dir, "scan-project", "--state-dir", str(state_dir), "--kind", "initial"
    )
    initial_findings = load_target_findings(initial_result["target_findings_path"])
    corpus_rules = {
        json.loads(line)["rule"]
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    covered_initial_rules = sorted(
        covered_target_rules(
            {str(item["rule"]) for item in initial_findings}, corpus_rules
        )
    )
    manifest["build_gate_setup"] = build_setup
    manifest["skill_init"] = init_result
    manifest["initial_scan"] = initial_result
    manifest["initial_target_findings_sha256"] = sha256_file(
        Path(initial_result["target_findings_path"])
    )
    manifest["initial_per_rule"] = per_rule_counts(initial_findings)
    frozen_path = project.get("frozen_target_findings_path")
    if frozen_path:
        frozen = Path(frozen_path).resolve()
        manifest["frozen_initial_sha256"] = sha256_file(frozen)
        if read_json(frozen) != initial_findings:
            manifest["status"] = "initial_scan_mismatch"
            write_json(run_manifest_path, manifest)
            raise RuntimeError(
                "observed initial target findings differ from the frozen input"
            )
    if args.dry_run:
        manifest["status"] = "dry_run_verified"
        manifest["completed_at"] = utc_now()
        write_json(run_manifest_path, manifest)
        print(run_manifest_path)
        return

    broker = None
    broker_socket = None
    broker_streams = None
    broker_runtime = None
    if args.condition == "hapskill":
        broker, broker_socket, broker_stdout, broker_stderr, broker_runtime = (
            start_broker(run_dir, state_dir, args.device, args.retrieval_python)
        )
        broker_streams = (broker_stdout, broker_stderr)

    thread_id: str | None = None
    total_usage: Counter[str] = Counter()
    all_commands: list[dict[str, Any]] = []
    previous_feedback = "No preceding round."
    previous_validation_status = "accepted"
    maximum_edit_attempts = int(
        protocol.get("scheduling", {}).get("maximum_agent_turns_per_round", 10)
    )
    experiment_started = time.monotonic()
    try:
        for round_number in range(1, max_rounds + 1):
            begin = run_skill(state_dir, "begin-round", "--state-dir", str(state_dir))
            current_findings = load_target_findings(begin["input_findings_path"])
            localization = control / f"findings_round_{round_number:02d}.json"
            write_json(localization, current_findings)
            localization_hash = sha256_file(localization)
            round_plan = build_round_plan(current_findings)
            round_plan["corpus_covered_rules"] = sorted(
                covered_target_rules(set(round_plan["required_rules"]), corpus_rules)
            )
            round_plan["retrieval_required"] = bool(
                args.condition == "hapskill"
                and protocol.get("retrieval", {}).get(
                    "require_success_for_covered_rules", True
                )
            )
            round_plan_path = control / f"round_plan_{round_number:02d}.json"
            round_completion_path = (
                control / f"round_completion_{round_number:02d}.json"
            )
            write_json(round_plan_path, round_plan)
            edit_attempts: list[dict[str, Any]] = []
            recovery_only = previous_validation_status == "repair_required"
            preflight_evidence = None
            if not recovery_only:
                round_completion_path.unlink(missing_ok=True)
            for edit_attempt in range(1, maximum_edit_attempts + 1):
                prompt = prompt_for_round(
                    project=project,
                    condition=args.condition,
                    round_number=round_number,
                    max_rounds=max_rounds,
                    localization_path=localization,
                    round_plan_path=round_plan_path,
                    round_completion_path=round_completion_path,
                    previous_feedback=previous_feedback,
                    build_available=bool(build_setup.get("available")),
                    edit_attempt=edit_attempt,
                    experiment_name=protocol.get("experiment", "EXP-HAPSKILL-10"),
                    retrieval_required=bool(
                        protocol.get("retrieval", {}).get(
                            "require_success_for_covered_rules", True
                        )
                    ),
                    recovery_only=recovery_only,
                    preflight_evidence=preflight_evidence,
                )
                suffix = "" if edit_attempt == 1 else f".attempt_{edit_attempt:02d}"
                prompt_path = traces / f"round_{round_number:02d}{suffix}.prompt.txt"
                prompt_path.write_text(prompt + "\n", encoding="utf-8")
                print(
                    f"[{args.condition}] {args.project} round {round_number}/{max_rounds} "
                    f"attempt {edit_attempt}: {len(current_findings)} alerts",
                    flush=True,
                )
                turn = run_codex_turn(
                    workspace,
                    codex_home,
                    prompt,
                    traces / f"round_{round_number:02d}{suffix}.jsonl",
                    traces / f"round_{round_number:02d}{suffix}.stderr.log",
                    model=protocol["model"],
                    thread_id=thread_id,
                    image=args.container_image,
                    condition=args.condition,
                    broker_socket=broker_socket,
                )
                thread_id = turn["thread_id"]
                if sha256_file(localization) != localization_hash:
                    raise RuntimeError("agent modified evaluator localization input")
                for key, value in turn["usage"].items():
                    total_usage[key] += value
                all_commands.extend(turn["commands"])
                completion = (
                    {
                        "complete": True,
                        "feedback": "Gate-recovery turn does not change the frozen alert-coverage contract.",
                        "mode": "gate_recovery",
                    }
                    if recovery_only
                    else audit_round_completion(round_plan, round_completion_path)
                )
                if not recovery_only and not completion["complete"]:
                    previous_feedback = completion["feedback"]
                    edit_attempts.append(
                        {
                            "attempt": edit_attempt,
                            "prompt_path": str(prompt_path),
                            "turn": turn,
                            "completion_audit": completion,
                            "validation": {
                                "status": "coverage_required",
                                "feedback": previous_feedback,
                                "scan_consumed": False,
                            },
                        }
                    )
                    manifest["active_edit_round"] = {
                        "round": round_number,
                        "input_alerts": len(current_findings),
                        "localization_path": str(localization),
                        "localization_sha256": localization_hash,
                        "round_plan_path": str(round_plan_path),
                        "round_plan": round_plan,
                        "edit_attempts": edit_attempts,
                    }
                    manifest["thread_id"] = thread_id
                    write_json(run_manifest_path, manifest)
                    continue

                required_retrieval = (
                    covered_target_rules(
                        set(round_plan["required_rules"]), corpus_rules
                    )
                    if args.condition == "hapskill"
                    and bool(
                        protocol.get("retrieval", {}).get(
                            "require_success_for_covered_rules", True
                        )
                    )
                    and not recovery_only
                    else set()
                )
                retrieval = audit_retrieval_trace(state_dir / "tool_trace.jsonl")
                missing_retrieval = required_retrieval - set(
                    retrieval["successful_rules"]
                )
                if missing_retrieval:
                    previous_feedback = (
                        "Required Top-1 Stella retrieval is missing for covered target rules: "
                        + ", ".join(sorted(missing_retrieval))
                        + ". Retrieve each rule once using repository context, assess whether "
                        "the example applies, update the cumulative completion report, and do "
                        "not request validation yet."
                    )
                    edit_attempts.append(
                        {
                            "attempt": edit_attempt,
                            "prompt_path": str(prompt_path),
                            "turn": turn,
                            "completion_audit": completion,
                            "retrieval_audit": retrieval,
                            "validation": {
                                "status": "retrieval_required",
                                "feedback": previous_feedback,
                                "scan_consumed": False,
                            },
                        }
                    )
                    manifest["active_edit_round"] = {
                        "round": round_number,
                        "input_alerts": len(current_findings),
                        "round_plan": round_plan,
                        "edit_attempts": edit_attempts,
                    }
                    manifest["thread_id"] = thread_id
                    write_json(run_manifest_path, manifest)
                    continue

                preflight = run_skill(
                    state_dir, "preflight-round", "--state-dir", str(state_dir)
                )
                if preflight["status"] != "preflight_passed":
                    preflight_evidence = materialize_preflight_evidence(
                        preflight, control, round_number, edit_attempt
                    )
                    previous_feedback = (
                        preflight["feedback"]
                        + " Evaluator evidence is available at "
                        + preflight_evidence
                        + ". Repair only these gate violations before doing any new alert work."
                    )
                    recovery_only = True
                    edit_attempts.append(
                        {
                            "attempt": edit_attempt,
                            "prompt_path": str(prompt_path),
                            "turn": turn,
                            "completion_audit": completion,
                            "retrieval_audit": retrieval,
                            "preflight": preflight,
                            "validation": {
                                "status": "preflight_repair_required",
                                "feedback": previous_feedback,
                                "scan_consumed": False,
                            },
                        }
                    )
                    manifest["active_edit_round"] = {
                        "round": round_number,
                        "input_alerts": len(current_findings),
                        "round_plan": round_plan,
                        "edit_attempts": edit_attempts,
                    }
                    manifest["thread_id"] = thread_id
                    write_json(run_manifest_path, manifest)
                    continue

                validation = run_skill(
                    state_dir, "validate-round", "--state-dir", str(state_dir)
                )
                previous_feedback = validation["feedback"]
                edit_attempts.append(
                    {
                        "attempt": edit_attempt,
                        "prompt_path": str(prompt_path),
                        "turn": turn,
                        "completion_audit": completion,
                        "retrieval_audit": retrieval,
                        "preflight": preflight,
                        "validation": validation,
                    }
                )
                manifest["active_edit_round"] = {
                    "round": round_number,
                    "input_alerts": len(current_findings),
                    "localization_path": str(localization),
                    "localization_sha256": localization_hash,
                    "round_plan_path": str(round_plan_path),
                    "round_plan": round_plan,
                    "round_completion_path": str(round_completion_path),
                    "edit_attempts": edit_attempts,
                }
                manifest["thread_id"] = thread_id
                write_json(run_manifest_path, manifest)
                if validation["status"] not in {
                    "editing_required",
                    "preflight_required",
                }:
                    break
            else:
                raise RuntimeError(
                    f"active round did not reach evaluator validation after "
                    f"{maximum_edit_attempts} agent turns in round {round_number}"
                )
            manifest["rounds"].append(
                {
                    "round": round_number,
                    "input_alerts": len(current_findings),
                    "input_per_rule": per_rule_counts(current_findings),
                    "localization_path": str(localization),
                    "localization_sha256": localization_hash,
                    "round_plan_path": str(round_plan_path),
                    "round_plan": round_plan,
                    "round_completion_path": str(round_completion_path),
                    "prompt_path": str(prompt_path),
                    "turn": turn,
                    "validation": validation,
                    "edit_attempts": edit_attempts,
                }
            )
            manifest.pop("active_edit_round", None)
            manifest["thread_id"] = thread_id
            write_json(run_manifest_path, manifest)
            previous_validation_status = validation["status"]
            if (
                validation["status"] == "accepted"
                and validation["total_metrics"]["final_alerts"] == 0
            ):
                break
            if protocol.get("scheduling", {}).get("thread_scope") == "active_round":
                thread_id = None
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["completed_at"] = utc_now()
        write_json(run_manifest_path, manifest)
        raise
    finally:
        if (
            broker is not None
            and broker_socket is not None
            and broker_streams is not None
            and broker_runtime is not None
        ):
            stop_broker(broker, broker_socket, *broker_streams, broker_runtime)

    final_result = run_skill(
        state_dir, "scan-project", "--state-dir", str(state_dir), "--kind", "final"
    )
    final_findings = load_target_findings(final_result["target_findings_path"])
    final_deltas = read_json(Path(final_result["alert_deltas_path"]))
    diff = source_diff(source, workspace, run_dir / "source_changes.patch")
    restricted = []
    for command in all_commands:
        text = command["command"].lower()
        if any(marker.lower() in text for marker in RESTRICTED_MARKERS):
            restricted.append(command)
    agent_builds = [
        item for item in all_commands if classify_command(item["command"]) == "build"
    ]
    agent_tests = [
        item for item in all_commands if classify_command(item["command"]) == "test"
    ]
    session = read_json(state_dir / "session.json")
    tool_trace = state_dir / "tool_trace.jsonl"
    retrieval_audit = audit_retrieval_trace(tool_trace)
    missing_covered_retrieval = sorted(
        set(covered_initial_rules) - set(retrieval_audit["successful_rules"])
    )
    retrieval_violation = (
        args.condition == "vanilla" and retrieval_audit["attempt_count"] != 0
    ) or (
        args.condition == "hapskill"
        and bool(
            protocol.get("retrieval", {}).get("require_success_for_covered_rules", True)
        )
        and bool(missing_covered_retrieval)
    )
    manifest.update(
        {
            "status": "protocol_violation"
            if restricted or retrieval_violation
            else "completed",
            "completed_at": utc_now(),
            "wall_clock_seconds": time.monotonic() - experiment_started,
            "final_scan": final_result,
            "alert_metrics": final_result["metrics"],
            "final_per_rule": per_rule_counts(final_findings),
            "per_rule_alert_deltas": per_rule_deltas(final_deltas),
            "validation_scan_count": len(session["rounds"]),
            "no_diff_retry_count": sum(
                item["validation"]["status"] == "editing_required"
                for round_item in manifest["rounds"]
                for item in round_item.get("edit_attempts", [])
            ),
            "coverage_retry_count": sum(
                item["validation"]["status"] == "coverage_required"
                for round_item in manifest["rounds"]
                for item in round_item.get("edit_attempts", [])
            ),
            "repair_required_round_count": sum(
                item["status"] == "repair_required" for item in session["rounds"]
            ),
            "final_candidate_abandonment": final_result.get(
                "final_candidate_abandonment"
            ),
            "last_valid_round": session["last_valid_round"],
            "best_valid_round": session.get("best_valid_round"),
            "best_valid_score": session.get("best_valid_score"),
            "observed_rule_interactions": session.get("rule_interactions", []),
            "final_candidate_selection": final_result.get("final_candidate_selection"),
            "build_status": (
                "passed"
                if build_setup.get("available")
                and (
                    session["last_valid_round"] == 0
                    or session["rounds"][session["last_valid_round"] - 1]["build_gate"][
                        "status"
                    ]
                    == "passed"
                )
                else "not_available"
                if not build_setup.get("available")
                else "failed"
            ),
            "test_status": command_status(all_commands, "test"),
            "agent_build_count": len(agent_builds),
            "agent_test_count": len(agent_tests),
            "build_test_commands": agent_builds + agent_tests,
            "input_tokens": total_usage.get("input_tokens", 0),
            "cached_input_tokens": total_usage.get("cached_input_tokens", 0),
            "output_tokens": total_usage.get("output_tokens", 0),
            "total_tokens": total_usage.get("total_tokens", 0),
            "api_cost": None,
            "api_cost_note": "Provider billing was not exposed in the Codex trace.",
            "restricted_accesses": restricted,
            "covered_initial_retrieval_rules": covered_initial_rules,
            "retrieval_audit": retrieval_audit,
            "missing_covered_retrieval_rules": missing_covered_retrieval,
            "retrieval_protocol_violation": retrieval_violation,
            "source_diff": diff,
            "skill_tool_trace": str(tool_trace),
            "skill_tool_trace_sha256": sha256_file(tool_trace),
        }
    )
    write_json(run_manifest_path, manifest)
    print(
        f"[done] {args.condition} {args.project}: "
        f"{manifest['alert_metrics']['eliminated_alerts']} eliminated, "
        f"{manifest['alert_metrics']['introduced_alerts']} introduced",
        flush=True,
    )


if __name__ == "__main__":
    main()
