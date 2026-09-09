#!/usr/bin/env python3
"""Run the EXP-AGENT-10 coding-agent condition on pinned project copies."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from build_gate import prepare_build_gate, run_build_gate
from public_api_guard import prepare_public_api_guard, run_public_api_guard
from validation_gate import initialize_gate, load_state, run_scan, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SELECTION = SCRIPT_DIR / "selected_projects.json"
DEFAULT_PROTOCOL = SCRIPT_DIR / "protocol.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_agent_10" / "runs"
DEFAULT_CODEX_IMAGE = "hybrid-gym-codex:0.146.0"
HOST_CODEX_HOME = Path.home() / ".codex"
RULES_ROOT = Path(
    "/home/zhihao/deveco/command-line-tools/codelinter/linter/arkPerfCheck"
)
SOURCE_SUFFIXES = {".ets", ".ts"}
SNAPSHOT_SUFFIXES = SOURCE_SUFFIXES | {
    ".json",
    ".json5",
    ".yaml",
    ".yml",
    ".toml",
    ".properties",
}
EXCLUDED_DIRS = {".git", ".exp_agent", "build", "node_modules", "oh_modules", ".hvigor"}
RESTRICTED_PATH_MARKERS = (
    "/revision/knowledge_base",
    "/revision/independent_oracle",
    "/revision/fixed_projects",
    "/data/pairs",
    "/data/security",
    "/multiple_round",
    "/deepseek/round_",
    "/qwen2.5-72b-instruct/round_",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_endpoint_fingerprint(codex_home: Path, provider: str) -> str:
    config_path = codex_home / "config.toml"
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except PermissionError:
        # Docker may tighten the copied file to root-only after a run. The
        # source config is the byte source used by prepare_codex_home.
        config_text = (HOST_CODEX_HOME / "config.toml").read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    base_url = config["model_providers"][provider]["base_url"].rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()


def sha256_json(path: Path) -> str:
    return sha256_file(path)


def iter_project_files(project: Path, suffixes: set[str] | None = None) -> Iterable[Path]:
    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if suffixes is None or path.suffix in suffixes:
                yield path


def tree_manifest(project: Path) -> dict[str, str]:
    return {
        path.relative_to(project).as_posix(): sha256_file(path)
        for path in iter_project_files(project)
    }


def copy_project(source: Path, destination: Path) -> None:
    def ignore(_root: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED_DIRS}

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def canonical_finding(finding: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        finding.get(key)
        for key in (
            "relative_path",
            "line",
            "column",
            "end_line",
            "end_column",
            "severity",
            "rule",
            "message",
        )
    )


def verify_initial_findings(
    frozen: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    frozen_counter = Counter(canonical_finding(item) for item in frozen)
    observed_counter = Counter(canonical_finding(item) for item in observed)
    missing = frozen_counter - observed_counter
    extra = observed_counter - frozen_counter
    return not missing and not extra, {
        "frozen_count": len(frozen),
        "observed_count": len(observed),
        "missing_count": sum(missing.values()),
        "extra_count": sum(extra.values()),
    }


def normalize_message(message: str) -> str:
    return " ".join(str(message).split())


def alert_identity(finding: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(finding.get("relative_path", "")),
        str(finding.get("rule", "")),
        normalize_message(str(finding.get("message", ""))),
    )


def compute_alert_metrics(
    initial: list[dict[str, Any]], final: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    initial_counter = Counter(alert_identity(item) for item in initial)
    final_counter = Counter(alert_identity(item) for item in final)
    eliminated_counter = initial_counter - final_counter
    introduced_counter = final_counter - initial_counter
    remaining_counter = initial_counter & final_counter

    def expand(counter: Counter[tuple[str, str, str]]) -> list[dict[str, Any]]:
        return [
            {"relative_path": key[0], "rule": key[1], "message": key[2]}
            for key in sorted(counter)
            for _ in range(counter[key])
        ]

    metrics = {
        "initial_alerts": len(initial),
        "final_alerts": len(final),
        "eliminated_alerts": sum(eliminated_counter.values()),
        "remaining_alerts": sum(remaining_counter.values()),
        "introduced_alerts": sum(introduced_counter.values()),
        "net_reduction": len(initial) - len(final),
    }
    deltas = {
        "eliminated": expand(eliminated_counter),
        "remaining": expand(remaining_counter),
        "introduced": expand(introduced_counter),
    }
    return metrics, deltas


def load_findings(scan: dict[str, Any]) -> list[dict[str, Any]]:
    path = scan.get("findings_path")
    if scan.get("status") != "scanned" or not path:
        raise RuntimeError(f"HomeCheck scan failed: {scan.get('parse_error', 'unknown error')}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initial_prompt(
    project: dict[str, Any],
    findings_path: Path,
    round_number: int,
    max_rounds: int,
    evaluator_build_available: bool,
) -> str:
    docs_root = Path("/rules/docs")
    implementation_root = Path("/rules/node_modules/homecheck/lib/checker")
    container_findings_path = Path("/workspace") / findings_path.relative_to(
        findings_path.parents[1]
    )
    return f"""
You are the coding-agent comparator in the EXP-AGENT-10 controlled study. Repair
HomeCheck performance and security alerts in the current OpenHarmony/ArkTS
project. This is edit round {round_number} of at most {max_rounds}.

The evaluator-provided localization JSON is at:
  {container_findings_path}
It contains file, line, column, rule, severity, and message for every current
alert. Inspect the affected project files before editing. You may inspect rule
documentation under {docs_root} and checker implementations under
{implementation_root}.

Study constraints:
- Work only in the current project copy.
- Do not inspect HapRepair's knowledge base, defect-repair pairs, held-out or
  reference patches, independent-oracle package, or any HapRepair-generated
  repair output. Do not search the HapRepair repository or experiment run roots.
- Do not use network access. All permitted project and rule material is local.
- Do not run CodeLinter/HomeCheck yourself. The external evaluator owns the
  validation scan and will send its results after this turn.
- You may run builds and existing tests as often as useful. Do not create tests
  that encode the supplied alert list.
- Do not delete an entire function or component to suppress an alert.
- Do not change package-exported public method signatures or exported interface
  contracts. The evaluator enforces this mechanically and will reject the whole
  round even if the change is justified in prose.
- If an edit causes a build or test failure, fix it or roll it back before ending
  this turn.
- Focus only on the supplied alerts. Avoid unrelated refactoring.

Make a concrete repair pass now. When the edits for this round are complete,
stop and summarize changed files, targeted rule IDs, and build/test commands.
Do not claim an alert is eliminated until the evaluator reports it.

Pinned project: {project['name']}
Commit: {project['commit']}
Tree OID: {project['tree_oid']}
Evaluator build gate: {'available and run after this turn' if evaluator_build_available else 'not available on the pinned initial version; static validation only'}
""".strip()


def feedback_prompt(
    findings_path: Path,
    round_number: int,
    max_rounds: int,
    evaluator_build_available: bool,
    previous_feedback: str,
) -> str:
    container_findings_path = Path("/workspace") / findings_path.relative_to(
        findings_path.parents[1]
    )
    return f"""
The evaluator completed validation scan {round_number - 1}. The current
residual and newly introduced HomeCheck alerts are at:
  {container_findings_path}

Evaluator feedback for the rejected or accepted candidate from the preceding round:
{previous_feedback}

Continue with edit round {round_number} of at most {max_rounds}. Inspect
the current source, repair the supplied alerts, and retain all prior study
constraints: do not run CodeLinter yourself, do not access HapRepair retrieval
data or reference/generated patches, preserve behavior, do not change any
package-exported public signature or interface contract,
and resolve or roll back build/test failures before stopping for validation.
The evaluator-side build gate is {'available and will run after this turn' if evaluator_build_available else 'unavailable on the pinned initial version; this project remains statically validated only'}.
""".strip()


def format_evaluator_feedback(
    public_api_gate: dict[str, Any],
    evaluator_build: dict[str, Any],
    *,
    rejected_outputs: list[str] | None = None,
) -> str:
    parts: list[str] = []
    if public_api_gate.get("status") == "failed":
        changed = public_api_gate.get("changes", {})
        affected = changed.get("changed", []) + changed.get("removed", []) + changed.get("added", [])
        parts.append(
            "The entire round was rolled back because it changed package-exported APIs: "
            + ", ".join(affected[:20])
            + ". Preserve those contracts exactly."
        )
    if evaluator_build.get("status") == "failed":
        log_path = (evaluator_build.get("build") or {}).get("log_path")
        errors: list[str] = []
        if log_path and Path(log_path).is_file():
            for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                if "Error Message:" in clean and clean not in errors:
                    errors.append(clean)
        parts.append(
            "The entire round was rolled back because the evaluator build failed. "
            + ("Compiler feedback: " + " | ".join(errors[:12]) if errors else "Do not repeat the candidate edits.")
        )
    if rejected_outputs:
        parts.append(
            "HapRepair rejected non-code or syntactically invalid full-file responses for: "
            + ", ".join(rejected_outputs[:20])
            + ". Return only syntactically valid complete ArkTS source."
        )
    if not parts:
        parts.append("The preceding candidate passed evaluator guards; use the new HomeCheck localization for remaining work.")
    return "\n".join(parts)


def parse_codex_events(raw_output: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def extract_thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") in {"thread.started", "session.started"}:
            for key in ("thread_id", "session_id", "id"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    return value
        thread = event.get("thread")
        if isinstance(thread, dict) and isinstance(thread.get("id"), str):
            return thread["id"]
    return None


def extract_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for event in events:
        candidate = event.get("usage")
        if not isinstance(candidate, dict):
            continue
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
            value = candidate.get(key)
            if isinstance(value, int):
                usage[key] = value
    if "total_tokens" not in usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return usage


def extract_commands(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") not in {
            "command_execution",
            "shell_command",
        }:
            continue
        command = item.get("command") or item.get("cmd")
        if isinstance(command, list):
            command = " ".join(str(part) for part in command)
        if not isinstance(command, str):
            continue
        commands.append(
            {
                "command": command,
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
                "duration_seconds": item.get("duration_seconds"),
                "category": classify_command(command),
                "restricted_path_reference": any(
                    marker in command for marker in RESTRICTED_PATH_MARKERS
                ),
            }
        )
    return commands


def classify_command(command: str) -> str:
    lowered = command.lower()
    boundary = r"(?:^|&&\s*|\|\|\s*|;\s*|(?:bash|sh)\s+-lc\s+['\"]\s*)"
    test_patterns = (
        boundary + r"(?:\./)?npm\s+(?:run\s+)?test\b",
        boundary + r"(?:\./)?pnpm\s+(?:run\s+)?test\b",
        boundary + r"(?:\./)?yarn\s+test\b",
        boundary + r"(?:\./)?pytest\b",
        boundary + r"(?:\./)?ctest\b",
        boundary + r"(?:\./)?hvigorw?\s+[^;&|]*(?:test|check)\b",
    )
    build_patterns = (
        boundary + r"(?:\./)?hvigorw?\s+[^;&|]*(?:assemble|build|compile|hap)\b",
        boundary + r"(?:\./)?npm\s+run\s+(?:build|compile)\b",
        boundary + r"(?:\./)?pnpm\s+(?:run\s+)?(?:build|compile)\b",
        boundary + r"(?:\./)?yarn\s+(?:build|compile)\b",
    )
    if any(re.search(pattern, lowered) for pattern in test_patterns):
        return "test"
    if any(re.search(pattern, lowered) for pattern in build_patterns):
        return "build"
    return "other"


def command_status(commands: list[dict[str, Any]], category: str) -> str:
    relevant = [item for item in commands if item["category"] == category]
    if not relevant:
        return "not_run"
    last = relevant[-1]
    if last.get("exit_code") == 0 and last.get("status") not in {"failed", "error"}:
        return "passed"
    return "failed"


def snapshot_editable_files(project: Path) -> dict[str, bytes]:
    return {
        path.relative_to(project).as_posix(): path.read_bytes()
        for path in iter_project_files(project, SNAPSHOT_SUFFIXES)
        if path.is_file()
    }


def restore_editable_files(project: Path, snapshot: dict[str, bytes]) -> None:
    current = {
        path.relative_to(project).as_posix(): path
        for path in iter_project_files(project, SNAPSHOT_SUFFIXES)
        if path.is_file()
    }
    for relative, path in current.items():
        if relative not in snapshot:
            path.unlink()
    for relative, content in snapshot.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def source_diff(
    original: Path, repaired: Path, output_path: Path
) -> dict[str, Any]:
    original_files = {
        path.relative_to(original).as_posix(): path
        for path in iter_project_files(original, SNAPSHOT_SUFFIXES)
    }
    repaired_files = {
        path.relative_to(repaired).as_posix(): path
        for path in iter_project_files(repaired, SNAPSHOT_SUFFIXES)
    }
    changed: list[str] = []
    deleted: list[str] = []
    added: list[str] = []
    large_deletion_flags: list[dict[str, Any]] = []
    public_interface_flags: list[str] = []
    patch_parts: list[str] = []
    for relative in sorted(original_files.keys() | repaired_files.keys()):
        old_path = original_files.get(relative)
        new_path = repaired_files.get(relative)
        old_lines = (
            old_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if old_path
            else []
        )
        new_lines = (
            new_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if new_path
            else []
        )
        if old_lines == new_lines:
            continue
        changed.append(relative)
        if old_path is None:
            added.append(relative)
        if new_path is None:
            deleted.append(relative)
        removed = max(len(old_lines) - len(new_lines), 0)
        if old_lines and removed >= 10 and removed / len(old_lines) >= 0.5:
            large_deletion_flags.append(
                {"relative_path": relative, "old_lines": len(old_lines), "new_lines": len(new_lines)}
            )
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
        if Path(relative).suffix in SOURCE_SUFFIXES and any(
            line.startswith(("-export ", "+export ", "-  export ", "+  export "))
            for line in diff
        ):
            public_interface_flags.append(relative)
        patch_parts.extend(diff)
    output_path.write_text("".join(patch_parts), encoding="utf-8")
    changed_source = [path for path in changed if Path(path).suffix in SOURCE_SUFFIXES]
    changed_resources = [path for path in changed if Path(path).suffix not in SOURCE_SUFFIXES]
    added_source = [path for path in added if Path(path).suffix in SOURCE_SUFFIXES]
    deleted_source = [path for path in deleted if Path(path).suffix in SOURCE_SUFFIXES]
    return {
        "changed_files": changed,
        "changed_file_count": len(changed),
        "changed_source_files": changed_source,
        "changed_source_file_count": len(changed_source),
        "changed_resource_or_config_files": changed_resources,
        "added_source_files": added_source,
        "deleted_source_files": deleted_source,
        "large_deletion_flags": large_deletion_flags,
        "public_interface_change_flags": public_interface_flags,
        "patch_path": str(output_path),
        "patch_sha256": sha256_file(output_path),
    }


def run_codex_turn(
    workspace: Path,
    codex_home: Path,
    prompt: str,
    trace_path: Path,
    stderr_path: Path,
    *,
    model: str,
    provider: str,
    effort: str,
    thread_id: str | None,
    container_image: str,
) -> dict[str, Any]:
    common = [
        "--json",
        "--model",
        model,
        "-c",
        f'model_provider="{provider}"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if thread_id is None:
        codex_command = ["codex", "exec", *common, "--cd", "/workspace", "-"]
    else:
        codex_command = ["codex", "exec", "resume", *common, thread_id, "-"]
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
        container_image,
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
    elapsed = time.monotonic() - started
    trace_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    events = parse_codex_events(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex turn failed with exit {result.returncode}; see {stderr_path}"
        )
    return {
        "command": command,
        "elapsed_seconds": elapsed,
        "thread_id": extract_thread_id(events) or thread_id,
        "usage": extract_usage(events),
        "commands": extract_commands(events),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "stderr_path": str(stderr_path),
    }


def prepare_codex_home(destination: Path) -> None:
    """Create a run-local Codex home without exposing prior sessions or memories."""
    destination.mkdir(parents=True, exist_ok=False)
    required = ("config.toml",)
    optional = ("models.json", "models_cache.json", "installation_id")
    for name in required:
        source = HOST_CODEX_HOME / name
        if not source.is_file():
            raise FileNotFoundError(f"Required Codex credential/config file is missing: {source}")
        shutil.copy2(source, destination / name)
    for name in optional:
        source = HOST_CODEX_HOME / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def remove_empty_auth_mount_placeholder(codex_home: Path) -> None:
    """Remove Docker's empty nested-bind placeholder, never credential data."""
    placeholder = codex_home / "auth.json"
    if not placeholder.exists():
        return
    if not placeholder.is_file() or placeholder.stat().st_size != 0:
        raise RuntimeError(f"Refusing to remove non-empty run-local auth path: {placeholder}")
    placeholder.unlink()


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


def select_project(selection: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in selection["projects"] if item["name"] == name]
    if not matches:
        raise ValueError(f"Project is not in the frozen selection: {name}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-image", default=DEFAULT_CODEX_IMAGE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the isolated copy and verify its initial scan without invoking Codex.",
    )
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    protocol_path = args.protocol.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    project_meta = select_project(selection, args.project)
    model = protocol["model"]
    max_rounds = int(protocol["homecheck"]["max_post_edit_validation_scans"])

    run_dir = args.run_root.resolve() / args.run_id / "coding_agent" / args.project
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    gate_dir = run_dir / "validation_gate"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir()
    codex_home = run_dir / "codex_home"
    control_dir = workspace / ".exp_agent"
    source = Path(project_meta["source_path"]).resolve()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "condition": "coding_agent",
        "run_id": args.run_id,
        "project": args.project,
        "status": "preparing",
        "started_at": utc_now(),
        "completed_at": None,
        "selection": str(selection_path),
        "selection_sha256": sha256_json(selection_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_json(protocol_path),
        "source": str(source),
        "commit": project_meta["commit"],
        "tree_oid": project_meta["tree_oid"],
        "model": model,
        "agent_runtime": docker_image_identity(args.container_image),
        "isolation": {
            "writable_mounts": ["project workspace", "run-local Codex home"],
            "read_only_mounts": ["HomeCheck rule documentation and implementation"],
            "haprepair_repository_mounted": False,
        },
        "rounds": [],
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    copy_started = time.monotonic()
    copy_project(source, workspace)
    control_dir.mkdir()
    prepare_codex_home(codex_home)
    manifest["agent_runtime"]["model_endpoint_sha256"] = provider_endpoint_fingerprint(
        codex_home, model["provider"]
    )
    source_tree = tree_manifest(source)
    workspace_tree = tree_manifest(workspace)
    manifest["copy_seconds"] = time.monotonic() - copy_started
    manifest["source_file_count"] = len(source_tree)
    manifest["byte_identical_input"] = source_tree == workspace_tree
    if not manifest["byte_identical_input"]:
        manifest["status"] = "copy_verification_failed"
        write_json(manifest_path, manifest)
        raise SystemExit("Workspace copy differs from the pinned source tree")

    initialize_gate(gate_dir, workspace, max_validation_scans=max_rounds)
    print(f"[initial] scanning {args.project}", flush=True)
    initial_scan = run_scan(gate_dir, "initial")
    observed_initial = load_findings(initial_scan)
    frozen_initial = json.loads(Path(project_meta["findings_path"]).read_text(encoding="utf-8"))
    initial_match, initial_verification = verify_initial_findings(
        frozen_initial, observed_initial
    )
    manifest["initial_scan"] = initial_scan
    manifest["initial_verification"] = initial_verification
    if not initial_match:
        manifest["status"] = "initial_scan_mismatch"
        write_json(manifest_path, manifest)
        raise SystemExit(f"Initial scan does not match frozen findings: {initial_verification}")
    print(f"[initial] verified {len(observed_initial)} alerts", flush=True)

    if args.dry_run:
        manifest["status"] = "dry_run_verified"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(f"[done] dry-run manifest: {manifest_path}", flush=True)
        return

    build_gate_dir = run_dir / "build_gate"
    try:
        build_gate_setup = prepare_build_gate(
            args.project, workspace, build_gate_dir
        )
    except RuntimeError as error:
        manifest["status"] = "build_gate_setup_failed"
        manifest["completed_at"] = utc_now()
        manifest["failure"] = str(error)
        write_json(manifest_path, manifest)
        raise
    manifest["build_gate_setup"] = build_gate_setup
    public_api_setup = prepare_public_api_guard(
        workspace, run_dir / "public_api_guard"
    )
    manifest["public_api_guard_setup"] = public_api_setup
    write_json(manifest_path, manifest)

    thread_id: str | None = None
    current_findings = observed_initial
    all_commands: list[dict[str, Any]] = []
    total_usage: Counter[str] = Counter()
    previous_feedback = "No preceding round."
    experiment_started = time.monotonic()
    for round_number in range(1, max_rounds + 1):
        localization_path = control_dir / f"findings_round_{round_number:02d}.json"
        write_json(localization_path, current_findings)
        input_alert_count = len(current_findings)
        localization_sha256 = sha256_file(localization_path)
        prompt = (
            initial_prompt(
                project_meta,
                localization_path,
                round_number,
                max_rounds,
                bool(build_gate_setup["available"]),
            )
            if round_number == 1
            else feedback_prompt(
                localization_path,
                round_number,
                max_rounds,
                bool(build_gate_setup["available"]),
                previous_feedback,
            )
        )
        prompt_path = trace_dir / f"round_{round_number:02d}.prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        snapshot = snapshot_editable_files(workspace)
        print(
            f"[agent] round {round_number}/{max_rounds}: {len(current_findings)} localized alerts",
            flush=True,
        )
        try:
            turn = run_codex_turn(
                workspace,
                codex_home,
                prompt,
                trace_dir / f"round_{round_number:02d}.jsonl",
                trace_dir / f"round_{round_number:02d}.stderr.log",
                model=model["requested_id"],
                provider=model["provider"],
                effort=model["reasoning_effort"],
                thread_id=thread_id,
                container_image=args.container_image,
            )
        except RuntimeError as error:
            manifest["status"] = "agent_environment_failed"
            manifest["completed_at"] = utc_now()
            manifest["failure"] = str(error)
            write_json(manifest_path, manifest)
            raise
        thread_id = turn["thread_id"]
        if sha256_file(localization_path) != localization_sha256:
            manifest["status"] = "protocol_violation"
            manifest["completed_at"] = utc_now()
            manifest["failure"] = "Agent modified evaluator-provided localization input"
            write_json(manifest_path, manifest)
            raise RuntimeError(manifest["failure"])
        for key, value in turn["usage"].items():
            total_usage[key] += value
        all_commands.extend(turn["commands"])
        agent_build_status = command_status(turn["commands"], "build")
        agent_test_status = command_status(turn["commands"], "test")
        try:
            public_api_gate = run_public_api_guard(
                workspace,
                run_dir / "public_api_guard" / "rounds",
                public_api_setup,
                label=f"round_{round_number:02d}",
            )
            rollback_reason = None
            rollback_verification = None
            public_api_rollback_verification = None
            if public_api_gate["status"] == "failed":
                evaluator_build = {
                    "label": f"round_{round_number:02d}",
                    "status": "skipped_public_api_guard_failed",
                    "command": None,
                    "build": None,
                }
                rollback_reason = (
                    "The evaluator-side public API guard detected a changed exported "
                    "interface; all editable-file changes from this round were restored."
                )
            else:
                evaluator_build = run_build_gate(
                    args.project,
                    workspace,
                    build_gate_dir / "rounds",
                    build_gate_setup,
                    label=f"round_{round_number:02d}",
                )
            if rollback_reason or evaluator_build["status"] == "failed":
                if rollback_reason is None:
                    rollback_reason = (
                        "The common evaluator-side build gate failed; all editable-file "
                        "changes from this round were restored before HomeCheck validation."
                    )
                restore_editable_files(workspace, snapshot)
                public_api_rollback_verification = run_public_api_guard(
                    workspace,
                    run_dir / "public_api_guard" / "rounds",
                    public_api_setup,
                    label=f"round_{round_number:02d}_rollback",
                )
                if public_api_rollback_verification["status"] != "passed":
                    raise RuntimeError(
                        f"Round {round_number} rollback did not restore the public API"
                    )
                rollback_verification = run_build_gate(
                    args.project,
                    workspace,
                    build_gate_dir / "rounds",
                    build_gate_setup,
                    label=f"round_{round_number:02d}_rollback",
                )
                if build_gate_setup["available"] and rollback_verification["status"] != "passed":
                    raise RuntimeError(
                        f"Round {round_number} rollback did not restore the frozen build"
                    )
        except Exception as error:
            manifest["status"] = "build_gate_failed"
            manifest["completed_at"] = utc_now()
            manifest["failure"] = f"{type(error).__name__}: {error}"
            write_json(manifest_path, manifest)
            raise
        validation_scan = run_scan(gate_dir, "validation")
        current_findings = load_findings(validation_scan)
        round_record = {
            "round": round_number,
            "input_alerts": input_alert_count,
            "localization_sha256": localization_sha256,
            "prompt_path": str(prompt_path),
            "turn": turn,
            "agent_build_status": agent_build_status,
            "agent_test_status": agent_test_status,
            "public_api_guard": public_api_gate,
            "public_api_rollback_verification": public_api_rollback_verification,
            "evaluator_build_gate": evaluator_build,
            "rollback_reason": rollback_reason,
            "rollback_verification": rollback_verification,
            "validation_scan": validation_scan,
        }
        manifest["rounds"].append(round_record)
        manifest["thread_id"] = thread_id
        previous_feedback = format_evaluator_feedback(
            public_api_gate, evaluator_build
        )
        round_record["feedback_for_next_round"] = previous_feedback
        write_json(manifest_path, manifest)
        print(
            f"[validation] round {round_number}: {len(current_findings)} alerts remain",
            flush=True,
        )
        if not current_findings:
            break

    print("[final] evaluator-only HomeCheck scan", flush=True)
    final_scan = run_scan(gate_dir, "final")
    final_findings = load_findings(final_scan)
    alert_metrics, alert_deltas = compute_alert_metrics(
        observed_initial, final_findings
    )
    deltas_path = run_dir / "alert_deltas.json"
    write_json(deltas_path, alert_deltas)
    diff_summary = source_diff(source, workspace, run_dir / "source_changes.patch")
    gate_state = load_state(gate_dir)
    restricted_accesses = [
        item for item in all_commands if item["restricted_path_reference"]
    ]
    build_commands = [item for item in all_commands if item["category"] == "build"]
    test_commands = [item for item in all_commands if item["category"] == "test"]
    evaluator_builds = [
        record
        for round_record in manifest["rounds"]
        for record in (
            round_record["evaluator_build_gate"],
            round_record.get("rollback_verification"),
        )
        if record and record.get("build")
    ]

    def measured_seconds(commands: list[dict[str, Any]]) -> float | None:
        durations = [item.get("duration_seconds") for item in commands]
        if not durations or not all(isinstance(value, (int, float)) for value in durations):
            return None
        return float(sum(durations))

    agent_build_seconds = measured_seconds(build_commands)
    evaluator_build_seconds = sum(
        item["build"]["duration_seconds"] for item in evaluator_builds
    )
    total_build_seconds = (
        None
        if build_commands and agent_build_seconds is None
        else (agent_build_seconds or 0.0) + evaluator_build_seconds
    )

    manifest.update(
        {
            "status": "completed" if not restricted_accesses else "protocol_violation",
            "completed_at": utc_now(),
            "wall_clock_seconds": time.monotonic() - experiment_started,
            "final_scan": final_scan,
            "alert_metrics": alert_metrics,
            "alert_deltas_path": str(deltas_path),
            "validation_scan_count": gate_state["validation_attempts_consumed"],
            "build_status": (
                "passed" if build_gate_setup["available"] else "not_available"
            ),
            "test_status": command_status(all_commands, "test"),
            "build_count": len(build_commands) + len(evaluator_builds),
            "test_count": len(test_commands),
            "build_execution_seconds": total_build_seconds,
            "test_execution_seconds": measured_seconds(test_commands),
            "agent_build_count": len(build_commands),
            "agent_test_count": len(test_commands),
            "evaluator_build_count": len(evaluator_builds),
            "evaluator_build_execution_seconds": evaluator_build_seconds,
            "validation_scope": build_gate_setup["validation_scope"],
            "command_timing_note": "Null when the Codex JSONL event did not expose command duration.",
            "build_test_commands": build_commands + test_commands,
            "input_tokens": total_usage.get("input_tokens", 0),
            "cached_input_tokens": total_usage.get("cached_input_tokens", 0),
            "output_tokens": total_usage.get("output_tokens", 0),
            "total_tokens": total_usage.get("total_tokens", 0),
            "api_cost": None,
            "api_cost_note": "Not reported because no traceable provider price or billing record was available.",
            "restricted_accesses": restricted_accesses,
            "source_diff": diff_summary,
        }
    )
    write_json(manifest_path, manifest)
    print(
        f"[done] {alert_metrics['eliminated_alerts']} eliminated, "
        f"{alert_metrics['introduced_alerts']} introduced; {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
