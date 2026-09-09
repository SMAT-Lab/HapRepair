#!/usr/bin/env python3
"""Run one isolated HapRepair v13 barrel-cycle-safe installed-Skill condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
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
    classify_command,
    command_status,
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
)
from validation_gate import write_json  # type: ignore  # noqa: E402


DEFAULT_PROTOCOL = HERE / "protocol-hapskill-35-v13-barrel-cycle-safe.json"
DEFAULT_MANIFEST = HERE / "formal_inputs_hapskill_35_v4.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_hapskill" / "runs"
DEFAULT_IMAGE = "hybrid-gym-codex:0.146.0"
SKILL_DIR = HAPREPAIR_ROOT / "skills" / "haprepair-openharmony-repair-v13-dev"
SKILL_PROTOCOL = SKILL_DIR / "references" / "protocol.json"
GUIDE_MANIFEST = SKILL_DIR / "references" / "repair-guides" / "manifest.json"
SPEC_MANIFEST = SKILL_DIR / "references" / "rule-specs" / "manifest.json"
COMMAND_LINE_TOOLS = Path("/home/zhihao/deveco/command-line-tools")
BASELINE_SDK_ROOT = WORKSPACE_ROOT / "baseline_data" / "openharmony_sdk"
HOST_JAVA_HOME = Path("/home/zhihao/.jdks/jdk-17.0.18+8")
HOST_RUNTIME_LIB_DIR = (
    WORKSPACE_ROOT / "baseline_data" / "haprepair_runtime_libs" / "lib"
)
CONTAINER_RUNTIME_LIB_DIR = "/opt/haprepair-runtime"
PNPM_SOURCE = Path(
    "/home/zhihao/.nvm/versions/node/v20.19.0/lib/node_modules/pnpm"
)
CONTAINER_SKILL = Path("/codex-home/skills/haprepair-openharmony-repair")
CONTAINER_STATE = Path("/run-state")
CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_TOOL_SCRIPT = CONTAINER_SKILL / "scripts" / "repair_session.py"
CONTAINER_RESULT_DIR = Path(
    "/home/zhihao/deveco/command-line-tools/codelinter/linter/result"
)
RESTRICTED_MARKERS = (
    "/baseline_data/",
    "rule_complete_383",
    "knowledge_base",
    "independent_oracle",
    "reference_patch",
    "exp_agent_10",
)


def is_restricted_command(command: str) -> bool:
    """Reject hidden-evidence paths while allowing the pinned SDK inspection.

    The formal container deliberately mounts the OpenHarmony SDK read-only so an
    agent can resolve framework/API details.  Its host path contains the generic
    ``/baseline_data/`` component, so remove only that exact allowlisted prefix
    before applying the evidence-access markers.  Any other baseline path in the
    same command remains restricted.
    """
    normalized = command.lower()
    normalized = normalized.replace(str(BASELINE_SDK_ROOT).lower(), "")
    return any(marker.lower() in normalized for marker in RESTRICTED_MARKERS)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_tree(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(manifest.items()):
        digest.update(f"{relative}\0{checksum}\n".encode())
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


def docker_image_identity(image: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Codex container image is unavailable: {image}")
    return {"image": image, "image_id": result.stdout.strip()}


def validate_runtime(protocol: dict[str, Any], image: str) -> dict[str, str]:
    identity = docker_image_identity(image)
    runtime = protocol["runtime"]
    if image != runtime["container_image"]:
        raise RuntimeError("Container image name differs from the protocol")
    if identity["image_id"] != runtime["container_image_id"]:
        raise RuntimeError("Container image ID differs from the protocol")
    version = subprocess.run(
        [str(COMMAND_LINE_TOOLS / "codelinter/bin/codelinter"), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    node = subprocess.run(
        [str(COMMAND_LINE_TOOLS / "tool/node/bin/node"), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != runtime["codelinter"]:
        raise RuntimeError("CodeLinter version differs from the protocol")
    if node.returncode != 0 or node.stdout.strip() != runtime["node"]:
        raise RuntimeError("DevEco Node version differs from the protocol")
    required_runtime_libs = (
        "libGL.so.1",
        "libGLX.so.0",
        "libGLdispatch.so.0",
    )
    missing_runtime_libs = [
        name for name in required_runtime_libs if not (HOST_RUNTIME_LIB_DIR / name).exists()
    ]
    if missing_runtime_libs:
        raise RuntimeError(
            "Container-compatible OpenGL runtime is missing: "
            + ", ".join(missing_runtime_libs)
        )
    observed_hashes = {
        name: sha256_file(HOST_RUNTIME_LIB_DIR / name)
        for name in required_runtime_libs
    }
    runtime_manifest = runtime.get("container_runtime_libs")
    if isinstance(runtime_manifest, dict):
        expected_hashes = runtime_manifest.get("sha256") or {}
        if expected_hashes != observed_hashes:
            raise RuntimeError(
                "Container-compatible OpenGL runtime hashes differ from the protocol"
            )
    return {
        **identity,
        "codelinter": version.stdout.strip(),
        "node": node.stdout.strip(),
        "command_line_tools": str(COMMAND_LINE_TOOLS),
        "container_runtime_lib_dir": str(HOST_RUNTIME_LIB_DIR),
        "container_runtime_lib_hashes": observed_hashes,
    }


def validate_method_artifacts(protocol: dict[str, Any]) -> None:
    expected = protocol["repair_references"]
    observed = {
        "skill_protocol_sha256": sha256_file(SKILL_PROTOCOL),
        "skill_md_sha256": sha256_file(SKILL_DIR / "SKILL.md"),
        "homecheck_tool_sha256": sha256_file(SKILL_DIR / "scripts" / "homecheck.py"),
        "candidate_session_tool_sha256": sha256_file(
            SKILL_DIR / "scripts" / "repair_session.py"
        ),
        "runner_sha256": sha256_file(Path(__file__)),
        "guide_manifest_sha256": sha256_file(GUIDE_MANIFEST),
        "semantic_spec_manifest_sha256": sha256_file(SPEC_MANIFEST),
    }
    drift = {
        key: {"expected": expected.get(key), "observed": value}
        for key, value in observed.items()
        if expected.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Frozen v13 method artifacts drifted: {drift}")


def install_skill(codex_home: Path) -> dict[str, Any]:
    destination = codex_home / "skills" / "haprepair-openharmony-repair"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SKILL_DIR,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache", "*.pyc"),
    )
    guide_manifest = read_json(
        destination / "references" / "repair-guides" / "manifest.json"
    )
    spec_manifest = read_json(
        destination / "references" / "rule-specs" / "manifest.json"
    )
    return {
        "path": str(destination),
        "skill_tree_sha256": sha256_tree(tree_manifest(destination)),
        "skill_md_sha256": sha256_file(destination / "SKILL.md"),
        "protocol_sha256": sha256_file(destination / "references" / "protocol.json"),
        "guide_manifest_sha256": sha256_file(
            destination / "references" / "repair-guides" / "manifest.json"
        ),
        "guide_pair_count": guide_manifest["pair_count"],
        "guide_rule_count": guide_manifest["rule_count"],
        "spec_manifest_sha256": sha256_file(
            destination / "references" / "rule-specs" / "manifest.json"
        ),
        "spec_rule_count": sum(
            len(family["rules"]) for family in spec_manifest["families"]
        ),
    }


def prepare_frozen_validation_gate(
    project: dict[str, Any], workspace: Path, output_dir: Path
) -> dict[str, Any]:
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
    }
    write_json(output_dir / "setup.json", setup)
    return setup


def validation_wrapper(
    control: Path, project_name: str, build_setup: dict[str, Any]
) -> str | None:
    if not build_setup.get("available"):
        return None
    wrapper = control / "evaluator_build.sh"
    environment = build_environment(project_name)
    environment.update(build_setup.get("environment") or {})
    # Build preflight records host paths. The evaluator runs in a container, so
    # keep the pinned SDK mounted, but place writable Hvigor state in this
    # condition's mounted state directory and load the hook from the installed
    # Skill. The offline pnpm cache is staged before this wrapper is generated.
    environment["HVIGOR_USER_HOME"] = "/run-state/build-home"
    # Older project-local hvigor wrappers derive their cache and npmrc paths from
    # os.homedir() and ignore HVIGOR_USER_HOME. Keep both paths inside the mounted
    # per-condition state directory so those projects cannot fall back to /.
    environment["HOME"] = "/run-state/build-home"
    environment["npm_config_userconfig"] = "/run-state/build-home/.npmrc"
    environment["NPM_CONFIG_USERCONFIG"] = "/run-state/build-home/.npmrc"
    environment["NODE_PATH"] = ":".join(
        (
            "/workspace/oh_modules",
            "/home/zhihao/deveco/command-line-tools/hvigor/hvigor/node_modules",
            "/home/zhihao/deveco/command-line-tools/hvigor/hvigor-ohos-plugin/node_modules",
        )
    )
    environment["NODE_OPTIONS"] = (
        "--require=/codex-home/skills/haprepair-openharmony-repair/"
        "scripts/arkts_ignore_diagnostics.js"
    )
    # Candidate validation runs in a minimal Codex image. Mount the same host
    # JDK read-only and expose it explicitly for Hvigor's PackageHap task.
    environment["JAVA_HOME"] = "/opt/java"
    environment["PATH"] = "/opt/java/bin:" + environment.get("PATH", "")
    keys = (
        "NODE_HOME",
        "DEVECO_NODE_HOME",
        "OHOS_BASE_SDK_HOME",
        "OHOS_SDK_HOME",
        "DEVECO_SDK_HOME",
        "HVIGOR_USER_HOME",
        "HOME",
        "NODE_PATH",
        "npm_config_registry",
        "npm_config_@ohos:registry",
        "npm_config_userconfig",
        "NPM_CONFIG_USERCONFIG",
        "NODE_OPTIONS",
        "JAVA_HOME",
        "PATH",
    )
    assignments = [f"{key}={environment[key]}" for key in keys if key in environment]
    build_setup["container_environment"] = {
        key: environment[key] for key in keys if key in environment
    }
    wrapper.write_text(
        "#!/bin/sh\nset -eu\nexec env "
        + shlex.join([*assignments, *build_setup["build_command"]])
        + "\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return "/workspace/.exp_agent/evaluator_build.sh"


def common_mounts(
    workspace: Path,
    state_dir: Path,
    codex_home: Path,
    codelinter_result: Path,
) -> list[str]:
    return [
        "--volume",
        f"{workspace}:/workspace:rw",
        "--volume",
        f"{state_dir}:/run-state:rw",
        "--volume",
        f"{codex_home}:/codex-home:rw",
        "--volume",
        f"{COMMAND_LINE_TOOLS}:{COMMAND_LINE_TOOLS}:ro",
        "--volume",
        f"{HOST_JAVA_HOME}:/opt/java:ro",
        "--volume",
        f"{BASELINE_SDK_ROOT}:{BASELINE_SDK_ROOT}:ro",
        "--volume",
        f"{codelinter_result}:{CONTAINER_RESULT_DIR}:rw",
        "--volume",
        f"{HOST_RUNTIME_LIB_DIR}:{CONTAINER_RUNTIME_LIB_DIR}:ro",
    ]


def stage_offline_pnpm(state_dir: Path) -> dict[str, Any]:
    """Stage a pinned pnpm package in the mounted per-condition Hvigor cache."""
    if not PNPM_SOURCE.is_dir():
        raise RuntimeError(f"Pinned offline pnpm package is missing: {PNPM_SOURCE}")
    build_home = state_dir / "build-home"
    build_home.mkdir(parents=True, exist_ok=True)
    npmrc = build_home / ".npmrc"
    if not npmrc.exists():
        npmrc.write_text(
            "registry=https://registry.npmmirror.com/\n"
            "@ohos:registry=https://repo.harmonyos.com/npm/\n",
            encoding="utf-8",
        )
    tools = build_home / "wrapper" / "tools"
    package = tools / "node_modules" / "pnpm"
    if not package.exists():
        package.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PNPM_SOURCE, package)
    bin_dir = tools / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "pnpm"
    if shim.exists() or shim.is_symlink():
        shim.unlink()
    shim.symlink_to("../pnpm/bin/pnpm.cjs")
    version = json.loads((package / "package.json").read_text(encoding="utf-8"))["version"]
    (tools / "package.json").write_text(
        json.dumps({"dependencies": {"pnpm": version}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "package_source": str(PNPM_SOURCE),
        "version": version,
        "container_home": "/run-state/build-home",
        "npmrc": "/run-state/build-home/.npmrc",
    }


def run_skill(
    *,
    workspace: Path,
    state_dir: Path,
    codex_home: Path,
    codelinter_result: Path,
    image: str,
    arguments: list[str],
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "none",
        "--workdir",
        "/workspace",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/run-state/build-home",
        "--env",
        "HVIGOR_USER_HOME=/run-state/build-home",
        "--env",
        "npm_config_userconfig=/run-state/build-home/.npmrc",
        "--env",
        "NPM_CONFIG_USERCONFIG=/run-state/build-home/.npmrc",
        "--env",
        f"LD_LIBRARY_PATH={CONTAINER_RUNTIME_LIB_DIR}",
        *common_mounts(workspace, state_dir, codex_home, codelinter_result),
        image,
        str(CONTAINER_TOOL_SCRIPT),
        *arguments,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"HapRepair v13 returned invalid JSON: {result.stderr[-2000:]}"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(payload.get("error", f"Skill exited {result.returncode}"))
    return payload


def run_codex_turn(
    *,
    workspace: Path,
    state_dir: Path,
    codex_home: Path,
    codelinter_result: Path,
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
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HAPREPAIR_AGENT_MODE=1",
        "--env",
        "HOME=/run-state/build-home",
        "--env",
        "HVIGOR_USER_HOME=/run-state/build-home",
        "--env",
        "npm_config_userconfig=/run-state/build-home/.npmrc",
        "--env",
        "NPM_CONFIG_USERCONFIG=/run-state/build-home/.npmrc",
        "--env",
        f"LD_LIBRARY_PATH={CONTAINER_RUNTIME_LIB_DIR}",
        *common_mounts(workspace, state_dir, codex_home, codelinter_result),
        "--volume",
        f"{HOST_CODEX_HOME / 'auth.json'}:/codex-home/auth.json:ro",
        image,
        *codex_command,
    ]
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


def container_to_host(
    value: str | Path, *, workspace: Path, state_dir: Path, codex_home: Path
) -> Path:
    path = PurePosixPath(str(value))
    mappings = (
        (PurePosixPath("/workspace"), workspace),
        (PurePosixPath("/run-state"), state_dir),
        (PurePosixPath("/codex-home"), codex_home),
    )
    for container_root, host_root in mappings:
        try:
            relative = path.relative_to(container_root)
        except ValueError:
            continue
        return host_root.joinpath(*relative.parts)
    raise ValueError(f"Unmapped container path: {value}")


def current_findings(
    *, workspace: Path, state_dir: Path, codex_home: Path
) -> list[dict[str, Any]]:
    scanner = read_json(state_dir / "homecheck" / "state.json")
    scan = scanner.get("current_scan")
    if not scan or scan.get("status") != "scanned":
        raise RuntimeError("No successful current HomeCheck scan")
    path = container_to_host(
        scan["findings_path"],
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
    )
    return read_json(path)


def exact_spec_paths(plan: dict[str, Any]) -> dict[str, list[str]]:
    return {
        item["rule"]: [
            item["semantic_spec"]["core_spec_path"],
            *[
                family["spec_path"]
                for family in item["semantic_spec"]["families"]
            ],
        ]
        for item in plan["rules"]
        if item["semantic_spec"]["covered"]
    }


def audit_spec_access(
    commands: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    command_text = "\n".join(str(item.get("command", "")) for item in commands)
    expected = exact_spec_paths(plan)
    accessed = sorted(
        rule
        for rule, paths in expected.items()
        if all(Path(path).name in command_text for path in paths)
    )
    return {
        "required_rules": sorted(expected),
        "accessed_rules": accessed,
        "missing_rules": sorted(set(expected) - set(accessed)),
        "complete": set(expected) <= set(accessed),
    }


def prompt_for_attempt(
    *,
    project: dict[str, Any],
    round_number: int,
    maximum_scans: int,
    attempt: int,
    findings_path: Path,
    plan_path: Path,
    completion_path: Path,
    feedback: str,
    gate_recovery: bool,
    build_available: bool,
) -> str:
    mode = (
        "Repair the retained candidate's reported gate failures in place. Preserve all "
        "otherwise useful edits and do not start a different repair round."
        if gate_recovery
        else (
            "Repair every frozen alert location in every file. Merge same-entity and "
            "interacting-rule-family findings before editing."
        )
    )
    return f"""
Use $haprepair-openharmony-repair for this evaluator-controlled OpenHarmony repair
round. The Skill is installed at /codex-home/skills/haprepair-openharmony-repair.
Read its SKILL.md and follow its evaluator-controlled mode. The evaluator invokes the
Skill's bundled candidate-session controller; do not invoke HomeCheck or create a
second session yourself.

HIGHEST-PRIORITY SOURCE-PRESERVATION RULE: never delete or replace an existing
struct, class, component, function, method, build method, exported symbol, or public
API, even if it appears unused or a different implementation seems cleaner. A
component must not be replaced by a builder that removes the original component.
Keep the original declaration and signature, and make behavior-preserving edits
inside it or add a compatibility wrapper/builder alongside it. Do not delete a
function, component, field, import, or container merely to make a finding disappear.
The evaluator's structural guard rejects declaration removal; if feedback names a
removed declaration, restore it before doing any other repair and preserve all
other useful edits.

IMPORT-GRAPH AND PUBLIC-API SAFETY: for @security/no-cycle, inspect the complete
directed import graph and classify every cycle edge as runtime, type-only, or
side-effect-bearing before editing. Use only a behavior-preserving transformation:
move genuinely shared definitions to a lowest common module, invert the dependency
through an existing interface, or use import type only when the mounted ArkTS
compiler accepts the syntax and the edge is genuinely type-only. Never delete an
import/export, replace it with import *, copy a definition, add a dynamic import,
change a package entry point, or rename an exported symbol to hide a cycle. Treat
export *, explicit re-exports, no-use-any-import, and hp-arkts-no-use-any-export-*
as one interaction family. Preserve the baseline package API and initialization
order; a public-API guard failure is unfinished work and must be repaired in place.

TOOLCHAIN BOUNDARY: do not edit hvigorw, hvigor/hvigor-wrapper.js, .hvigor,
generated dependency stores, SDK files, or evaluator-only environment files to
work around npmrc, HOME, HVIGOR_USER_HOME, cache-permission, libGL, or similar
infrastructure errors. Those paths are evaluator-owned. Repair only project
source/configuration files that are part of the localized finding; compiler
warning-class ArkTS diagnostics such as `any` are handled by the evaluator hook,
while module-resolution, compilation, packaging, process, and resource-schema
failures remain real gate failures.

DARK RESOURCE SAFETY: for @performance/dark-color-mode-check, inspect the
module's existing base resource keys and SDK-supported schema. Never create an
empty dark/element/color.json, an empty color object/array, or an unverified
colorMode property. A valid repair must contain a non-empty schema-valid dark
resource with corresponding keys and preserved product colors.

CONTROLLER ISOLATION: this is an agent turn, not an evaluator turn. Do not invoke
`repair_session.py` operations (`init-session`, `scan-initial`, `begin-round`,
`record-completion`, `preflight-round`, `validate-round`, or `finalize`), and do not
invoke HomeCheck scans or make a new plan. The evaluator owns those operations.
Only edit project source/configuration files and write the requested completion
report, then stop. Failed command attempts caused by the agent-mode guard are
expected evidence of this boundary and must not be worked around.

Project: {project["name"]}
Commit: {project.get("commit")}
Round: {round_number}, validation budget: at most {maximum_scans}
Edit attempt in this active round: {attempt}
Current findings: /workspace/{findings_path.relative_to(findings_path.parents[1]).as_posix()}
Frozen plan: /workspace/{plan_path.relative_to(plan_path.parents[1]).as_posix()}
Completion report: /workspace/{completion_path.relative_to(completion_path.parents[1]).as_posix()}
Evaluator feedback: {feedback}
Task mode: {mode}

Inspect every planned affected file and every localized instance. Open the exact core
and family semantic-spec paths from each rule record. The 383 repair-guide examples
are optional provenance and are never patch templates. Use rg and targeted reads to
follow declarations, mutations, imports, calls, resources, state ownership,
lifecycle, persisted formats, protocols, and configuration. Repair high-risk rules
with deeper evidence; do not omit them.

Write selected_rules, consulted_specs, entity_repairs, and unresolved_external to the
completion report. Each entity_repair must contain its rule, relative_path, every
frozen line/column location, owning entities, applicable invariants, repository
evidence, transformation summary, and status "repaired". Ordinary blocked entries
are invalid, and unresolved_external does not pass coverage. Preserve declarations
and exported interfaces. Do not delete code to silence a finding. The frozen
evaluator build gate is {"available" if build_available else "unavailable"}.

Do not access the network for repair evidence, hidden reference patches, another
condition, or the HapRepair source repository. The bundled specifications and
examples are not RAG, retrieval, Top-k output, embeddings, or similarity-ranked
results. Make every planned repair and the complete report now, then stop for
evaluator validation.
""".strip()


def per_rule_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item["rule"]) for item in findings).items()))


def per_rule_deltas(deltas: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    eliminated = Counter(str(item["rule"]) for item in deltas["eliminated"])
    remaining = Counter(str(item["rule"]) for item in deltas["remaining"])
    introduced = Counter(str(item["rule"]) for item in deltas["introduced"])
    return [
        {
            "rule": rule,
            "eliminated_alerts": eliminated[rule],
            "remaining_alerts": remaining[rule],
            "introduced_alerts": introduced[rule],
        }
        for rule in sorted(eliminated.keys() | remaining.keys() | introduced.keys())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    project_manifest_path = args.project_manifest.resolve()
    protocol = read_json(protocol_path)
    validate_method_artifacts(protocol)
    project = load_project(args.project, project_manifest_path)
    maximum_scans = int(protocol["common_contract"]["maximum_validation_scans"])
    maximum_attempts = int(protocol["scheduling"]["maximum_agent_turns_per_round"])
    runtime = validate_runtime(protocol, args.container_image)

    run_dir = args.run_root.resolve() / args.run_id / "hapskill" / args.project
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    state_dir = run_dir / "skill_state"
    state_dir.mkdir()
    traces = run_dir / "traces"
    traces.mkdir()
    codex_home = run_dir / "codex_home"
    control = workspace / ".exp_agent"
    codelinter_result = run_dir / "codelinter_result"
    codelinter_result.mkdir()
    source = Path(project["source_path"]).resolve()
    manifest_path = run_dir / "run_manifest.json"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "paper_facing": bool(protocol["paper_facing"]),
        "run_id": args.run_id,
        "condition": "hapskill",
        "project": args.project,
        "status": "preparing",
        "started_at": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "skill_protocol": str(SKILL_PROTOCOL),
        "skill_protocol_sha256": sha256_file(SKILL_PROTOCOL),
        "project_manifest": str(project_manifest_path),
        "project_manifest_sha256": sha256_file(project_manifest_path),
        "source": str(source),
        "commit": project.get("commit"),
        "tree_oid": project.get("tree_oid"),
        "model": protocol["model"],
        "maximum_validation_scans": maximum_scans,
        "agent_runtime": runtime,
        "rounds": [],
        "isolation": {
            "haprepair_repository_mounted": False,
            "source_corpus_mounted": False,
            "installed_skill_only": True,
            "command_line_tools_read_only": True,
            "per_condition_codelinter_log_overlay": str(codelinter_result),
            "dynamic_retrieval_exposed": False,
        },
    }
    write_json(manifest_path, manifest)

    copy_project(source, workspace)
    control.mkdir()
    prepare_codex_home(codex_home)
    manifest["installed_skill"] = install_skill(codex_home)
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
        write_json(manifest_path, manifest)
        raise RuntimeError("Isolated workspace differs from the pinned source")
    frozen_tree = project.get("frozen_input_tree_sha256")
    if frozen_tree and frozen_tree != manifest["input_tree_sha256"]:
        manifest["status"] = "frozen_input_tree_mismatch"
        write_json(manifest_path, manifest)
        raise RuntimeError("Pinned source differs from the frozen input tree")

    build_setup = prepare_frozen_validation_gate(
        project, workspace, run_dir / "build_gate"
    )
    build_command = validation_wrapper(control, args.project, build_setup)
    init_arguments = [
        "init-session",
        "--workspace",
        "/workspace",
        "--state-dir",
        "/run-state",
        "--max-validation-scans",
        str(maximum_scans),
    ]
    if build_command:
        init_arguments.extend(["--build-command", build_command])
    init = run_skill(
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
        codelinter_result=codelinter_result,
        image=args.container_image,
        arguments=init_arguments,
    )
    # The candidate-session controller requires an empty state directory at
    # initialization time. Stage the offline package only after that state
    # snapshot has been created; the validation wrapper resolves it lazily from
    # /run-state/build-home when a build gate runs.
    if build_setup.get("available"):
        build_setup["offline_pnpm"] = stage_offline_pnpm(state_dir)
    initial = run_skill(
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
        codelinter_result=codelinter_result,
        image=args.container_image,
        arguments=["scan-initial", "--state-dir", "/run-state"],
    )
    initial_findings = current_findings(
        workspace=workspace, state_dir=state_dir, codex_home=codex_home
    )
    manifest["build_gate_setup"] = build_setup
    manifest["skill_init"] = init
    manifest["initial_scan"] = initial
    manifest["initial_per_rule"] = per_rule_counts(initial_findings)
    initial_findings_path = container_to_host(
        initial["findings_path"],
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
    )
    manifest["initial_target_findings_sha256"] = sha256_file(initial_findings_path)
    frozen_path = project.get("frozen_target_findings_path")
    if frozen_path:
        frozen = Path(frozen_path).resolve()
        manifest["frozen_initial_sha256"] = sha256_file(frozen)
        if read_json(frozen) != initial_findings:
            manifest["status"] = "initial_scan_mismatch"
            write_json(manifest_path, manifest)
            raise RuntimeError("Initial findings differ from the frozen input")
    if args.dry_run:
        manifest["status"] = "dry_run_verified"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(manifest_path)
        return

    total_usage: Counter[str] = Counter()
    all_commands: list[dict[str, Any]] = []
    experiment_started = time.monotonic()
    try:
        for _ in range(maximum_scans):
            if not current_findings(
                workspace=workspace, state_dir=state_dir, codex_home=codex_home
            ):
                break
            begin = run_skill(
                workspace=workspace,
                state_dir=state_dir,
                codex_home=codex_home,
                codelinter_result=codelinter_result,
                image=args.container_image,
                arguments=["begin-round", "--state-dir", "/run-state"],
            )
            round_number = int(begin["round"])
            findings = current_findings(
                workspace=workspace, state_dir=state_dir, codex_home=codex_home
            )
            localization_path = control / f"findings_round_{round_number:02d}.json"
            plan_path = control / f"round_plan_{round_number:02d}.json"
            completion_path = control / f"round_completion_{round_number:02d}.json"
            write_json(localization_path, findings)
            write_json(plan_path, begin["plan"])
            localization_sha = sha256_file(localization_path)
            plan_sha = sha256_file(plan_path)
            thread_id: str | None = None
            round_commands: list[dict[str, Any]] = []
            attempts: list[dict[str, Any]] = []
            feedback = "No preceding gate failure in this active round."
            gate_recovery = False
            for attempt in range(1, maximum_attempts + 1):
                prompt = prompt_for_attempt(
                    project=project,
                    round_number=round_number,
                    maximum_scans=maximum_scans,
                    attempt=attempt,
                    findings_path=localization_path,
                    plan_path=plan_path,
                    completion_path=completion_path,
                    feedback=feedback,
                    gate_recovery=gate_recovery,
                    build_available=bool(build_setup.get("available")),
                )
                suffix = "" if attempt == 1 else f".attempt_{attempt:02d}"
                prompt_path = traces / f"round_{round_number:02d}{suffix}.prompt.txt"
                prompt_path.write_text(prompt + "\n", encoding="utf-8")
                print(
                    f"[hapskill] {args.project} round {round_number} attempt {attempt}: "
                    f"{len(findings)} alerts",
                    flush=True,
                )
                turn = run_codex_turn(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    prompt=prompt,
                    trace_path=traces / f"round_{round_number:02d}{suffix}.jsonl",
                    stderr_path=traces / f"round_{round_number:02d}{suffix}.stderr.log",
                    model=protocol["model"],
                    thread_id=thread_id,
                    image=args.container_image,
                )
                thread_id = turn["thread_id"]
                round_commands.extend(turn["commands"])
                all_commands.extend(turn["commands"])
                for key, value in turn["usage"].items():
                    total_usage[key] += value
                if sha256_file(localization_path) != localization_sha:
                    raise RuntimeError("Agent modified frozen localization")
                if sha256_file(plan_path) != plan_sha:
                    raise RuntimeError("Agent modified frozen round plan")

                if not completion_path.is_file():
                    feedback = "Required completion report is missing."
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "status": "coverage_required",
                            "feedback": feedback,
                        }
                    )
                    continue
                completion = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=args.container_image,
                    arguments=[
                        "record-completion",
                        "--state-dir",
                        "/run-state",
                        "--report",
                        f"/workspace/.exp_agent/{completion_path.name}",
                    ],
                )
                spec_access = audit_spec_access(round_commands, begin["plan"])
                if not completion["complete"] or not spec_access["complete"]:
                    problems = list(completion["problems"])
                    if not spec_access["complete"]:
                        problems.append(
                            "trace lacks exact normative spec access for: "
                            + ", ".join(spec_access["missing_rules"])
                        )
                    feedback = "Coverage evidence is incomplete: " + "; ".join(problems)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "completion": completion,
                            "spec_access": spec_access,
                            "status": "coverage_required",
                            "feedback": feedback,
                        }
                    )
                    continue

                preflight = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=args.container_image,
                    arguments=["preflight-round", "--state-dir", "/run-state"],
                )
                if preflight["status"] != "preflight_passed":
                    feedback = preflight["feedback"]
                    gate_recovery = True
                    attempts.append(
                        {
                            "attempt": attempt,
                            "turn": turn,
                            "completion": completion,
                            "spec_access": spec_access,
                            "preflight": preflight,
                            "status": preflight["status"],
                            "feedback": feedback,
                        }
                    )
                    continue

                validation = run_skill(
                    workspace=workspace,
                    state_dir=state_dir,
                    codex_home=codex_home,
                    codelinter_result=codelinter_result,
                    image=args.container_image,
                    arguments=["validate-round", "--state-dir", "/run-state"],
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "turn": turn,
                        "completion": completion,
                        "spec_access": spec_access,
                        "preflight": preflight,
                        "validation": validation,
                        "status": validation["status"],
                    }
                )
                session = read_json(state_dir / "session.json")
                if (
                    validation["status"] == "accepted"
                    or session["active_round"] is None
                ):
                    break
                feedback = validation["feedback"]
                gate_recovery = True
            else:
                raise RuntimeError(
                    f"Round {round_number} did not reach HomeCheck validation after "
                    f"{maximum_attempts} agent turns"
                )

            manifest["rounds"].append(
                {
                    "round": round_number,
                    "input_alerts": len(findings),
                    "input_per_rule": per_rule_counts(findings),
                    "localization_path": str(localization_path),
                    "localization_sha256": localization_sha,
                    "plan_path": str(plan_path),
                    "plan_sha256": plan_sha,
                    "completion_path": str(completion_path),
                    "attempts": attempts,
                    "validation": validation,
                }
            )
            write_json(manifest_path, manifest)
            if (
                validation["status"] == "accepted"
                and validation["total_metrics"]["final_alerts"] == 0
            ):
                break
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    final = run_skill(
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
        codelinter_result=codelinter_result,
        image=args.container_image,
        arguments=["finalize", "--state-dir", "/run-state"],
    )
    final_scan = final["final_scan"]
    final_findings_path = container_to_host(
        final_scan["findings_path"],
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
    )
    final_deltas_path = container_to_host(
        final_scan["alert_deltas_path"],
        workspace=workspace,
        state_dir=state_dir,
        codex_home=codex_home,
    )
    final_findings = read_json(final_findings_path)
    final_deltas = read_json(final_deltas_path)
    session = read_json(state_dir / "session.json")
    scanner = read_json(state_dir / "homecheck" / "state.json")
    restricted = [
        item
        for item in all_commands
        if is_restricted_command(item["command"])
    ]
    spec_access = [
        attempt.get("spec_access")
        for round_item in manifest["rounds"]
        for attempt in round_item["attempts"]
        if attempt.get("spec_access")
    ]
    spec_violation = any(not item["complete"] for item in spec_access[-1:])
    agent_builds = [
        item for item in all_commands if classify_command(item["command"]) == "build"
    ]
    agent_tests = [
        item for item in all_commands if classify_command(item["command"]) == "test"
    ]
    diff = source_diff(source, workspace, run_dir / "source_changes.patch")
    manifest.update(
        {
            "status": "protocol_violation"
            if restricted or spec_violation
            else "completed",
            "completed_at": utc_now(),
            "wall_clock_seconds": time.monotonic() - experiment_started,
            "finalization": final,
            "alert_metrics": final_scan["metrics"],
            "final_per_rule": per_rule_counts(final_findings),
            "per_rule_alert_deltas": per_rule_deltas(final_deltas),
            "validation_scan_count": scanner["validation_scans_consumed"],
            "last_valid_round": session["last_valid_round"],
            "best_valid_round": session["best_valid_round"],
            "best_valid_score": session["best_valid_score"],
            "observed_rule_interactions": session["rule_interactions"],
            "final_candidate_selection": session["final_candidate_selection"],
            "build_status": (
                "not_available"
                if not build_setup.get("available")
                else session["rounds"][session["last_valid_round"] - 1]["build_gate"][
                    "status"
                ]
                if session["last_valid_round"]
                else "not_run"
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
            "spec_access_audits": spec_access,
            "spec_protocol_violation": spec_violation,
            "source_diff": diff,
        }
    )
    write_json(manifest_path, manifest)
    print(
        f"[done] hapskill {args.project}: "
        f"{manifest['alert_metrics']['eliminated_alerts']} eliminated, "
        f"{manifest['alert_metrics']['introduced_alerts']} introduced",
        flush=True,
    )


if __name__ == "__main__":
    main()
