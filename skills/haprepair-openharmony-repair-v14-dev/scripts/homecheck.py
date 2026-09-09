#!/usr/bin/env python3
"""Deterministic HomeCheck operations bundled with the HapRepair Skill."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SKILL_DIR / "assets" / "homecheck-config.json5"
DEFAULT_OVERLAY = SKILL_DIR / "assets" / "homecheck-overlay.json"
GUIDE_DIR = SKILL_DIR / "references" / "repair-guides"
GUIDE_MANIFEST = GUIDE_DIR / "manifest.json"
GUIDE_INDEX = GUIDE_DIR / "index.md"
SPEC_DIR = SKILL_DIR / "references" / "rule-specs"
SPEC_MANIFEST = SPEC_DIR / "manifest.json"
SPEC_INDEX = SPEC_DIR / "index.md"
EXPECTED_GUIDE_MANIFEST_SHA256 = (
    "ec958a60597fd0d0fd8f404962ab3a3621aee370ccd0a78487159eb66bfce7f7"
)
EXPECTED_GUIDE_INDEX_SHA256 = (
    "ef35051d73a404906d7ab87ffb219f9c16110448fe2e4b4e600f03905a627732"
)
EXPECTED_PAIR_COUNT = 383
EXPECTED_RULE_COUNT = 63
EXPECTED_SPEC_MANIFEST_SHA256 = (
    "2d3ad8e88365e35666473f17c06cedbc77979434bf544c3fc60ef06dfea55f6b"
)
EXPECTED_SPEC_INDEX_SHA256 = (
    "d09e1aaea03aca518a9a66cf3b57021269362db5ad6ceffbe2986f24806de770"
)
EXPECTED_SPEC_FILE_SHA256 = {
    "collections-reuse.md": "8c6fb523268e8b4e8e5a90998cd9b6536b4eadbe281373af9fa6ec86d9d73fb4",
    "computation-api.md": "bda6f84b591d0bd6ab6ed7b863f23d691c80687654d735b506971b0bda4f41cf",
    "core-invariants.md": "91353341db10149875a7c77b548f8637830d5fa21633e40e745f225a5c0b7818",
    "layout-animation-media.md": "6e0ddefe73e04ac4eb7d51f931bab3122ff75b3900d251e92db3e8af64db9581",
    "security-crypto.md": "193c5e17cf1755dd5a6b4aee487393cdb21d31276ca47c5693ad44bbd30fad33",
    "security-structure.md": "d57e3089a8e76ec1da1cbde73e9e46dd3e69e6b1d4184ea6b0d9e5f8bd1b2052",
    "state-reactivity.md": "63a28de781edecce770ca03aad3d8cf120f189ae57eaec373586d85f1bdfb0cf",
}
TARGET_RULE_PREFIXES = ("@performance/", "@security/", "@hw-ets-eslint/")
GUIDE_ALIASES = {
    "@performance/init-list-component": "@hw-ets-eslint/init-list-component",
    "@performance/hp-arkui-reduce-pangesture-distance": "@performance/hp-arkui-reduce-pan-gesture-distance",
}
DEFAULT_CODELINTER_CANDIDATES = (
    Path("/home/zhihao/deveco/command-line-tools/codelinter/bin/codelinter"),
)
AGENT_MODE_ENV = "HAPREPAIR_AGENT_MODE"
AGENT_FORBIDDEN_OPERATIONS = frozenset({"init-session", "scan", "make-plan"})


def enforce_agent_boundary(operation: str) -> None:
    """Prevent an agent turn from creating scans or replacing evaluator state."""
    if os.getenv(AGENT_MODE_ENV) == "1" and operation in AGENT_FORBIDDEN_OPERATIONS:
        raise PermissionError(
            f"{operation} is evaluator-only; the repair agent must use the frozen "
            "findings and plan supplied by the evaluator"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def locked_state(state_dir: Path) -> Iterator[Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "state.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield state_dir / "state.json"
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_state(state_dir: Path) -> dict[str, Any]:
    state_path = state_dir.resolve() / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"HomeCheck session is not initialized: {state_path}")
    return read_json(state_path)


def resolve_codelinter(value: Path | None) -> Path:
    candidates: list[Path] = []
    if value is not None:
        candidates.append(value)
    configured = os.getenv("HAPREPAIR_CODELINTER")
    if configured:
        candidates.append(Path(configured))
    on_path = shutil.which("codelinter")
    if on_path:
        candidates.append(Path(on_path))
    candidates.extend(DEFAULT_CODELINTER_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "CodeLinter was not found; pass --codelinter or set HAPREPAIR_CODELINTER"
    )


def validate_guides() -> dict[str, Any]:
    if sha256_file(GUIDE_MANIFEST) != EXPECTED_GUIDE_MANIFEST_SHA256:
        raise RuntimeError("Bundled repair-guide manifest hash mismatch")
    if sha256_file(GUIDE_INDEX) != EXPECTED_GUIDE_INDEX_SHA256:
        raise RuntimeError("Bundled repair-guide index hash mismatch")
    manifest = read_json(GUIDE_MANIFEST)
    if (
        manifest.get("pair_count") != EXPECTED_PAIR_COUNT
        or manifest.get("rule_count") != EXPECTED_RULE_COUNT
        or len(manifest.get("rules", [])) != EXPECTED_RULE_COUNT
    ):
        raise RuntimeError(
            "Bundled repair guides are not the frozen 383-pair/63-rule set"
        )
    for item in manifest["rules"]:
        guide = GUIDE_DIR / item["file"]
        if not guide.is_file() or sha256_file(guide) != item["sha256"]:
            raise RuntimeError(f"Bundled repair guide drifted: {guide}")
    return manifest


def validate_specs() -> dict[str, Any]:
    if sha256_file(SPEC_MANIFEST) != EXPECTED_SPEC_MANIFEST_SHA256:
        raise RuntimeError("Bundled rule-spec manifest hash mismatch")
    if sha256_file(SPEC_INDEX) != EXPECTED_SPEC_INDEX_SHA256:
        raise RuntimeError("Bundled rule-spec index hash mismatch")
    manifest = read_json(SPEC_MANIFEST)
    rules = [rule for family in manifest.get("families", []) for rule in family["rules"]]
    if len(rules) != 71 or len(set(rules)) != 71:
        raise RuntimeError("Bundled rule specs must cover 71 unique rule identities")
    for name, expected in EXPECTED_SPEC_FILE_SHA256.items():
        path = SPEC_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Bundled rule specification drifted: {path}")
    return manifest


def guide_record(rule: str) -> dict[str, Any]:
    value = rule.strip()
    if not value.startswith("@") and "/" in value:
        value = f"@{value}"
    canonical = GUIDE_ALIASES.get(value, value)
    manifest = validate_guides()
    matches = [item for item in manifest["rules"] if item["rule"] == canonical]
    if not matches and "/" not in value:
        suffix_matches = [
            item for item in manifest["rules"] if item["rule"].endswith(f"/{value}")
        ]
        if len(suffix_matches) == 1:
            matches = suffix_matches
            canonical = matches[0]["rule"]
        elif len(suffix_matches) > 1:
            raise ValueError(f"Ambiguous rule suffix {value!r}")
    if not matches:
        return {
            "requested_rule": rule,
            "canonical_rule": canonical,
            "covered": False,
            "guide_path": None,
            "example_count": 0,
        }
    item = matches[0]
    return {
        "requested_rule": rule,
        "canonical_rule": canonical,
        "covered": True,
        "guide_path": str((GUIDE_DIR / item["file"]).resolve()),
        "example_count": item["example_count"],
        "guide_sha256": item["sha256"],
    }


def semantic_record(rule: str) -> dict[str, Any]:
    value = rule.strip()
    if not value.startswith("@") and "/" in value:
        value = f"@{value}"
    manifest = validate_specs()
    canonical = manifest.get("aliases", {}).get(value, value)
    matches = [family for family in manifest["families"] if canonical in family["rules"]]
    if not matches:
        return {
            "requested_rule": rule,
            "canonical_rule": canonical,
            "covered": False,
            "core_spec_path": str((SPEC_DIR / manifest["core"]).resolve()),
            "families": [],
        }
    return {
        "requested_rule": rule,
        "canonical_rule": canonical,
        "covered": True,
        "core_spec_path": str((SPEC_DIR / manifest["core"]).resolve()),
        "families": [
            {
                "id": family["id"],
                "spec_path": str((SPEC_DIR / family["spec"]).resolve()),
                "interacts_with": family["interacts_with"],
            }
            for family in matches
        ],
    }


def interaction_clusters(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = validate_specs()
    window = int(manifest["interaction_window_lines"])
    families: dict[str, str] = {}
    related: set[tuple[str, str]] = set()
    aliases = manifest.get("aliases", {})
    for family in manifest["families"]:
        family_id = family["id"]
        for rule in family["rules"]:
            families[rule] = family_id
        related.add((family_id, family_id))
        for other in family["interacts_with"]:
            related.add((family_id, other))
            related.add((other, family_id))

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        normalized = dict(finding)
        normalized["canonical_rule"] = aliases.get(finding["rule"], finding["rule"])
        normalized["family"] = families.get(normalized["canonical_rule"])
        by_file[finding["relative_path"]].append(normalized)

    clusters: list[dict[str, Any]] = []
    for relative_path, items in sorted(by_file.items()):
        ordered = sorted(items, key=lambda item: (int(item["line"]), item["rule"]))
        parents = list(range(len(ordered)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left, source in enumerate(ordered):
            for right in range(left + 1, len(ordered)):
                target = ordered[right]
                if int(target["line"]) - int(source["line"]) > window:
                    break
                if source["family"] and (source["family"], target["family"]) in related:
                    union(left, right)

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(ordered):
            groups[find(index)].append(item)
        for group in groups.values():
            rules = sorted({item["rule"] for item in group})
            if len(rules) < 2:
                continue
            clusters.append(
                {
                    "relative_path": relative_path,
                    "first_line": min(int(item["line"]) for item in group),
                    "last_line": max(int(item["line"]) for item in group),
                    "alert_count": len(group),
                    "rules": rules,
                    "families": sorted({item["family"] for item in group if item["family"]}),
                    "locations": [
                        {
                            "line": item["line"],
                            "column": item["column"],
                            "rule": item["rule"],
                        }
                        for item in group
                    ],
                }
            )
    return clusters


def verify_codelinter(
    binary: Path, overlay_path: Path, *, allow_unverified: bool
) -> dict[str, Any]:
    overlay = read_json(overlay_path)
    version_result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    version = version_result.stdout.strip()
    issues: list[str] = []
    if version_result.returncode != 0:
        issues.append(f"version command exited {version_result.returncode}")
    if version != overlay["base_codelinter_version"]:
        issues.append(
            f"expected CodeLinter {overlay['base_codelinter_version']}, found {version!r}"
        )
    root = binary.parent.parent
    verified_files = []
    for relative, expected in overlay["files"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        verified_files.append(
            {"path": str(path), "expected_sha256": expected, "sha256": observed}
        )
        if observed != expected:
            issues.append(f"overlay mismatch: {path}")
    if issues and not allow_unverified:
        raise RuntimeError("; ".join(issues))
    return {
        "binary": str(binary),
        "version": version,
        "overlay_id": overlay["id"],
        "overlay_manifest": str(overlay_path),
        "overlay_manifest_sha256": sha256_file(overlay_path),
        "homecheck_source_commit": overlay["homecheck_source_commit"],
        "verified": not issues,
        "verification_issues": issues,
        "verified_files": verified_files,
    }


def normalize_findings(report_path: Path, project: Path) -> list[dict[str, Any]]:
    payload = read_json(report_path)
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("files", []))
    if not isinstance(payload, list):
        raise ValueError("CodeLinter report is not a file-result list")
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for file_result in payload:
        absolute = Path(str(file_result["filePath"]))
        try:
            relative = absolute.resolve().relative_to(project.resolve()).as_posix()
        except ValueError:
            relative = absolute.as_posix()
        for message in file_result.get("messages", []):
            item = {
                "relative_path": relative,
                "line": int(message.get("line", 0)),
                "column": int(message.get("column", 0)),
                "end_line": int(message.get("endLine", message.get("line", 0))),
                "end_column": int(message.get("endColumn", message.get("column", 0))),
                "severity": message.get("severity", ""),
                "rule": message.get("rule", ""),
                "message": message.get("message", ""),
            }
            identity = tuple(
                item[key]
                for key in ("relative_path", "line", "column", "rule", "message")
            )
            if identity not in seen:
                seen.add(identity)
                findings.append(item)
    return sorted(
        findings,
        key=lambda item: (
            item["relative_path"],
            item["line"],
            item["column"],
            item["rule"],
            item["message"],
        ),
    )


def target_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in findings
        if str(item.get("rule", "")).startswith(TARGET_RULE_PREFIXES)
    ]


def alert_identity(finding: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(finding.get("relative_path", "")),
        str(finding.get("rule", "")),
        " ".join(str(finding.get("message", "")).split()),
    )


def alert_metrics(
    initial: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, list[dict[str, str]]]]:
    initial_counter = Counter(alert_identity(item) for item in initial)
    current_counter = Counter(alert_identity(item) for item in current)
    counters = {
        "eliminated": initial_counter - current_counter,
        "remaining": initial_counter & current_counter,
        "introduced": current_counter - initial_counter,
    }

    def expand(counter: Counter[tuple[str, str, str]]) -> list[dict[str, str]]:
        return [
            {"relative_path": key[0], "rule": key[1], "message": key[2]}
            for key in sorted(counter)
            for _ in range(counter[key])
        ]

    metrics = {
        "initial_alerts": len(initial),
        "final_alerts": len(current),
        "eliminated_alerts": sum(counters["eliminated"].values()),
        "remaining_alerts": sum(counters["remaining"].values()),
        "introduced_alerts": sum(counters["introduced"].values()),
        "net_reduction": len(initial) - len(current),
    }
    return metrics, {key: expand(value) for key, value in counters.items()}


def execute_scan(
    *,
    binary: Path,
    config: Path,
    project: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    findings_path = output_dir / "findings.json"
    all_findings_path = output_dir / "all_findings.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    command = [
        str(binary),
        "--config",
        str(config),
        "--format",
        "json",
        "--output",
        str(report_path),
        str(project),
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    parse_error = None
    all_findings: list[dict[str, Any]] = []
    if report_path.is_file():
        try:
            all_findings = normalize_findings(report_path, project)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            parse_error = str(error)
    else:
        parse_error = "CodeLinter did not produce a report"
    findings = target_findings(all_findings)
    write_json(all_findings_path, all_findings)
    write_json(findings_path, findings)
    return {
        "status": "scanned" if parse_error is None else "scan_failed",
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "raw_finding_count": len(all_findings),
        "target_finding_count": len(findings),
        "excluded_non_target_finding_count": len(all_findings) - len(findings),
        "findings_path": str(findings_path),
        "findings_sha256": sha256_file(findings_path),
        "all_findings_path": str(all_findings_path),
        "report_path": str(report_path) if report_path.is_file() else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "parse_error": parse_error,
    }


def init_session(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    state_dir = args.state_dir.resolve()
    if not workspace.is_dir():
        raise ValueError(f"Project does not exist: {workspace}")
    try:
        state_dir.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise ValueError("Keep HomeCheck state outside the project workspace")
    if args.max_validation_scans < 1:
        raise ValueError("--max-validation-scans must be positive")
    binary = resolve_codelinter(args.codelinter)
    config = args.config.resolve()
    overlay = args.overlay.resolve()
    if not config.is_file() or not overlay.is_file():
        raise FileNotFoundError(
            "Bundled HomeCheck config or identity manifest is missing"
        )
    guides = validate_guides()
    specs = validate_specs()
    identity = verify_codelinter(
        binary, overlay, allow_unverified=args.allow_unverified_codelinter
    )
    with locked_state(state_dir) as state_path:
        if state_path.exists():
            raise FileExistsError(f"HomeCheck session already exists: {state_path}")
        state = {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "workspace": str(workspace),
            "codelinter": str(binary),
            "codelinter_identity": identity,
            "config": str(config),
            "config_sha256": sha256_file(config),
            "guide_manifest": str(GUIDE_MANIFEST),
            "guide_manifest_sha256": sha256_file(GUIDE_MANIFEST),
            "guide_pair_count": guides["pair_count"],
            "guide_rule_count": guides["rule_count"],
            "spec_manifest": str(SPEC_MANIFEST),
            "spec_manifest_sha256": sha256_file(SPEC_MANIFEST),
            "spec_rule_count": sum(
                len(family["rules"]) for family in specs["families"]
            ),
            "max_validation_scans": args.max_validation_scans,
            "validation_scans_consumed": 0,
            "initial_scan": None,
            "validation_scans": [],
            "final_scan": None,
            "current_scan": None,
        }
        write_json(state_path, state)
    return {
        "operation": "init_session",
        "status": "initialized",
        "state_dir": str(state_dir),
        "workspace": str(workspace),
        "codelinter_identity": identity,
        "config_sha256": sha256_file(config),
        "repair_references": {
            "pair_count": guides["pair_count"],
            "rule_count": guides["rule_count"],
            "manifest_sha256": sha256_file(GUIDE_MANIFEST),
        },
        "semantic_specs": {
            "rule_count": sum(len(family["rules"]) for family in specs["families"]),
            "manifest_sha256": sha256_file(SPEC_MANIFEST),
        },
        "validation_budget": {
            "consumed": 0,
            "maximum": args.max_validation_scans,
        },
    }


def scan_session(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    with locked_state(state_dir) as state_path:
        if not state_path.is_file():
            raise FileNotFoundError(
                f"HomeCheck session is not initialized: {state_path}"
            )
        state = read_json(state_path)
        if sha256_file(Path(state["config"])) != state["config_sha256"]:
            raise RuntimeError("Frozen HomeCheck config drifted")
        if sha256_file(GUIDE_MANIFEST) != state["guide_manifest_sha256"]:
            raise RuntimeError("Frozen repair-guide manifest drifted")
        if sha256_file(SPEC_MANIFEST) != state["spec_manifest_sha256"]:
            raise RuntimeError("Frozen rule-spec manifest drifted")
        kind = args.kind
        if kind == "initial":
            if state["initial_scan"] is not None:
                raise RuntimeError("Initial scan has already completed")
            attempt_id = "initial"
        elif kind == "final":
            if state["initial_scan"] is None:
                raise RuntimeError("Run the initial scan before the final scan")
            if state["final_scan"] is not None:
                raise RuntimeError("Final scan has already completed")
            attempt_id = "final"
        else:
            if state["initial_scan"] is None:
                raise RuntimeError("Run the initial scan before validation")
            consumed = int(state["validation_scans_consumed"])
            maximum = int(state["max_validation_scans"])
            if consumed >= maximum:
                raise RuntimeError(f"Validation budget exhausted: {consumed}/{maximum}")
            consumed += 1
            state["validation_scans_consumed"] = consumed
            attempt_id = f"validation_{consumed:02d}"
        placeholder = {
            "attempt_id": attempt_id,
            "kind": kind,
            "status": "started",
            "started_at": utc_now(),
            "budget_consumed": kind == "validation",
        }
        if kind == "initial":
            state["initial_scan"] = placeholder
        elif kind == "final":
            state["final_scan"] = placeholder
        else:
            state["validation_scans"].append(placeholder)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    try:
        scan = execute_scan(
            binary=Path(state["codelinter"]),
            config=Path(state["config"]),
            project=Path(state["workspace"]),
            output_dir=state_dir / "scans" / attempt_id,
        )
    except BaseException as error:
        scan = {
            "status": "scan_failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    scan.update(
        {
            "attempt_id": attempt_id,
            "kind": kind,
            "completed_at": utc_now(),
            "budget_consumed": kind == "validation",
        }
    )
    with locked_state(state_dir) as state_path:
        state = read_json(state_path)
        if kind == "initial":
            state["initial_scan"] = scan
        elif kind == "final":
            state["final_scan"] = scan
        else:
            state["validation_scans"][-1] = scan
        if scan["status"] == "scanned":
            state["current_scan"] = scan
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    response = {"operation": "scan", **scan}
    if scan["status"] == "scanned" and state["initial_scan"]["status"] == "scanned":
        initial = read_json(Path(state["initial_scan"]["findings_path"]))
        current = read_json(Path(scan["findings_path"]))
        metrics, deltas = alert_metrics(initial, current)
        deltas_path = state_dir / "scans" / attempt_id / "alert_deltas.json"
        write_json(deltas_path, deltas)
        response["metrics"] = metrics
        response["alert_deltas_path"] = str(deltas_path)
    response["validation_budget"] = {
        "consumed": state["validation_scans_consumed"],
        "maximum": state["max_validation_scans"],
    }
    return response


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    state = load_state(args.state_dir)
    scan = state.get("current_scan")
    if not scan or scan.get("status") != "scanned":
        raise RuntimeError("No successful current HomeCheck scan is available")
    findings = read_json(Path(scan["findings_path"]))
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for finding in findings:
        grouped[finding["rule"]][finding["relative_path"]].append(finding)
    rules = []
    for rule, files in grouped.items():
        guide = guide_record(rule)
        semantic_spec = semantic_record(rule)
        file_records = [
            {
                "relative_path": relative,
                "alert_count": len(items),
                "locations": [
                    {"line": item["line"], "column": item["column"]} for item in items
                ],
            }
            for relative, items in sorted(files.items())
        ]
        rules.append(
            {
                "rule": rule,
                "alert_count": sum(item["alert_count"] for item in file_records),
                "affected_file_count": len(file_records),
                "guide": guide,
                "semantic_spec": semantic_spec,
                "files": file_records,
            }
        )
    rules.sort(key=lambda item: (-item["alert_count"], item["rule"]))
    plan = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_scan": scan["attempt_id"],
        "source_findings_sha256": scan["findings_sha256"],
        "total_alerts": len(findings),
        "required_rule_count": len(rules),
        "required_rule_file_groups": sum(item["affected_file_count"] for item in rules),
        "interaction_clusters": interaction_clusters(findings),
        "rules": rules,
    }
    output = (
        args.output.resolve() if args.output else args.state_dir / "current_plan.json"
    )
    write_json(output, plan)
    return {"operation": "make_plan", "status": "planned", "path": str(output), **plan}


def session_status(args: argparse.Namespace) -> dict[str, Any]:
    state = load_state(args.state_dir)
    current = state.get("current_scan")
    response = {
        "operation": "status",
        "status": "completed" if state.get("final_scan") else "active",
        "workspace": state["workspace"],
        "codelinter_identity": state["codelinter_identity"],
        "current_scan": current,
        "validation_budget": {
            "consumed": state["validation_scans_consumed"],
            "maximum": state["max_validation_scans"],
        },
    }
    if current and current.get("status") == "scanned":
        initial = read_json(Path(state["initial_scan"]["findings_path"]))
        findings = read_json(Path(current["findings_path"]))
        response["metrics"], _ = alert_metrics(initial, findings)
    return response


def verify_operation(args: argparse.Namespace) -> dict[str, Any]:
    guides = validate_guides()
    specs = validate_specs()
    binary = resolve_codelinter(args.codelinter)
    identity = verify_codelinter(
        binary,
        args.overlay.resolve(),
        allow_unverified=args.allow_unverified_codelinter,
    )
    return {
        "operation": "verify",
        "status": "verified" if identity["verified"] else "unverified",
        "codelinter_identity": identity,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "repair_references": {
            "path": str(GUIDE_DIR),
            "pair_count": guides["pair_count"],
            "rule_count": guides["rule_count"],
            "manifest_sha256": sha256_file(GUIDE_MANIFEST),
            "index_sha256": sha256_file(GUIDE_INDEX),
        },
        "semantic_specs": {
            "path": str(SPEC_DIR),
            "rule_count": sum(len(family["rules"]) for family in specs["families"]),
            "manifest_sha256": sha256_file(SPEC_MANIFEST),
            "index_sha256": sha256_file(SPEC_INDEX),
        },
    }


def guide_operation(args: argparse.Namespace) -> dict[str, Any]:
    return {"operation": "guide", **guide_record(args.rule)}


def spec_operation(args: argparse.Namespace) -> dict[str, Any]:
    return {"operation": "spec", **semantic_record(args.rule)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--codelinter", type=Path)
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    verify.add_argument("--allow-unverified-codelinter", action="store_true")
    verify.set_defaults(handler=verify_operation)

    init = subparsers.add_parser("init-session")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--state-dir", type=Path, required=True)
    init.add_argument("--max-validation-scans", type=int, default=5)
    init.add_argument("--codelinter", type=Path)
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    init.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    init.add_argument("--allow-unverified-codelinter", action="store_true")
    init.set_defaults(handler=init_session)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--state-dir", type=Path, required=True)
    scan.add_argument(
        "--kind", choices=("initial", "validation", "final"), required=True
    )
    scan.set_defaults(handler=scan_session)

    plan = subparsers.add_parser("make-plan")
    plan.add_argument("--state-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    plan.set_defaults(handler=make_plan)

    guide = subparsers.add_parser("guide")
    guide.add_argument("--rule", required=True)
    guide.set_defaults(handler=guide_operation)

    spec = subparsers.add_parser("spec")
    spec.add_argument("--rule", required=True)
    spec.set_defaults(handler=spec_operation)

    status = subparsers.add_parser("status")
    status.add_argument("--state-dir", type=Path, required=True)
    status.set_defaults(handler=session_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        enforce_agent_boundary(args.operation)
        result = args.handler(args)
    except Exception as error:
        result = {
            "operation": args.operation,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
