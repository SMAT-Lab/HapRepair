#!/usr/bin/env python3
"""Run the final read-only G0/G2 preflight for EXP-HAPSKILL-10."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HAPREPAIR_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = HAPREPAIR_ROOT.parent
OUTPUT = (
    WORKSPACE_ROOT
    / "paper"
    / "rebuttal"
    / "gates"
    / "exp_hapskill_10_preflight_v2_20260803.json"
)
FORMAL_RUN = (
    WORKSPACE_ROOT
    / "baseline_data"
    / "exp_hapskill"
    / "runs"
    / "exp_hapskill_10_luna_formal_03"
)
WRAPPER_SMOKE = (
    WORKSPACE_ROOT
    / "baseline_data"
    / "exp_hapskill"
    / "smoke"
    / "wrapper_fix_ohos_cordova_01"
    / "vanilla"
    / "ohos_cordova"
)

EXPECTED_HASHES = {
    "protocol": "4997ba5201e3399202e54f678fcd6ce2b3a458b122876af13051cf34d4670c62",
    "formal_inputs": "4d6a86c4babdd684f3c4ff547b55ea713fc9b6613297eeb5d4eca95a4c0c0a87",
    "skill_protocol": "d83a398eea6f2fc5dfae495138f2aa203976677c9256f731a2660b1bb0c45101",
    "formal_02_method_change": "4c869b16ee0efc005f89719795acd2ce7c4152a831a2dc425786981caf6375f7",
    "corpus": "85c2f6f17799bed681201efec8ad3bf204d768d8150226e25369906ad10ae714",
    "codelinter_config": "91de02b03d0c5df48d3da615eb0b17615f2f9ad0e0f7f575119c964ae3b9cc95",
    "codelinter_overlay": "ca0555024524ad76c0d6f058bb98a36729a92c8b701f373b776bdb7e677f625c",
    "sdk": "dd1f23fab928cca17b8a3daeb5a6f0b8566b460d9c85b24684b39da80f84cde7",
    "sdk_build_components": "ba07957c4def67422612d794593e1c5fc7f0991afaf289e26483c895544aa49d",
    "sdk_native_api20": "a9bbeae7b72c80ae573d5439bb88e9d5cbe202d260f8394f24ae37669911d4f4",
    "build_preflight": "b00dbda76a840d3ce78d062ac820d8b62bdf78a810a78710e8f2bb04118c4155",
}
EXPECTED_IMAGE_ID = (
    "sha256:9cc027cd2f9404ccd58c61e367fe712895188a926c07d464ea0936e893b9a89a"
)

ARTIFACTS = {
    "protocol": SCRIPT_DIR / "protocol.json",
    "formal_inputs": SCRIPT_DIR / "formal_inputs.json",
    "skill_protocol": HAPREPAIR_ROOT
    / "skills"
    / "haprepair-openharmony-repair"
    / "references"
    / "protocol.json",
    "corpus": HAPREPAIR_ROOT
    / "revision"
    / "knowledge_base"
    / "rule_complete_383.jsonl",
    "codelinter_config": HAPREPAIR_ROOT / "revision" / "code-linter.json5",
    "codelinter_overlay": HAPREPAIR_ROOT
    / "revision"
    / "independent_oracle"
    / "codelinter_overlay.json",
    "sdk": WORKSPACE_ROOT
    / "baseline_data"
    / "openharmony_sdk"
    / "install_manifest.json",
    "sdk_build_components": WORKSPACE_ROOT
    / "baseline_data"
    / "openharmony_sdk"
    / "install_manifest_build_components.json",
    "sdk_native_api20": WORKSPACE_ROOT
    / "baseline_data"
    / "openharmony_sdk"
    / "install_manifest_native_api20.json",
    "build_preflight": WORKSPACE_ROOT
    / "baseline_data"
    / "exp_agent_10"
    / "build_preflight"
    / "manifest.json",
    "formal_02_method_change": WORKSPACE_ROOT
    / "paper"
    / "rebuttal"
    / "gates"
    / "exp_hapskill_10_formal_02_method_change_20260803.json",
}

HARNESS_FILES = [
    SCRIPT_DIR / "run_condition.py",
    SCRIPT_DIR / "run_paired.py",
    SCRIPT_DIR / "skill_broker.py",
    SCRIPT_DIR / "skill_client.py",
    HAPREPAIR_ROOT / "revision" / "coding_agent_baseline" / "validation_gate.py",
    HAPREPAIR_ROOT / "revision" / "coding_agent_baseline" / "build_gate.py",
    HAPREPAIR_ROOT / "revision" / "coding_agent_baseline" / "public_api_guard.py",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any
) -> None:
    checks.append({"name": name, "passed": passed, **evidence})


def main() -> None:
    checks: list[dict[str, Any]] = []
    for name, path in ARTIFACTS.items():
        actual = sha256_file(path) if path.is_file() else None
        add_check(
            checks,
            f"frozen_hash:{name}",
            actual == EXPECTED_HASHES[name],
            path=str(path),
            expected_sha256=EXPECTED_HASHES[name],
            actual_sha256=actual,
        )

    protocol = json.loads(ARTIFACTS["protocol"].read_text(encoding="utf-8"))
    formal_inputs = json.loads(ARTIFACTS["formal_inputs"].read_text(encoding="utf-8"))
    skill_protocol = json.loads(ARTIFACTS["skill_protocol"].read_text(encoding="utf-8"))
    method_change = json.loads(
        ARTIFACTS["formal_02_method_change"].read_text(encoding="utf-8")
    )
    add_check(
        checks,
        "formal_contract",
        protocol["model"]["requested_id"] == "gpt-5.6-luna"
        and protocol["model"]["provider"] == "xsp"
        and protocol["common_contract"]["maximum_post_edit_validation_scans"] == 5
        and protocol["formal_run"]["run_id"] == "exp_hapskill_10_luna_formal_03"
        and protocol["formal_run"]["supersedes_run_id"]
        == "exp_hapskill_10_luna_formal_02"
        and "candidate_retention" in protocol["common_contract"]
        and "last_valid_restoration" in protocol["common_contract"]
        and protocol["retrieval"]["skill_protocol_id"] == "haprepair-skill-v2-20260803"
        and len(protocol["formal_run"]["condition_order"]) == 20,
        model=protocol["model"],
        condition_order=protocol["formal_run"]["condition_order"],
    )
    add_check(
        checks,
        "retained_candidate_contract",
        skill_protocol["protocol_id"] == "haprepair-skill-v2-20260803"
        and set(skill_protocol["candidate_policy"])
        == {"accepted", "repair_required", "final_abandonment"}
        and "vanilla" in skill_protocol["fairness"]
        and "HapRepair-skill" in skill_protocol["fairness"]
        and method_change["status"] == "excluded_method_development_run"
        and method_change["superseding_run"]["run_id"]
        == "exp_hapskill_10_luna_formal_03",
        skill_protocol_id=skill_protocol["protocol_id"],
        candidate_policy=skill_protocol["candidate_policy"],
        method_change_artifact=str(ARTIFACTS["formal_02_method_change"]),
    )
    add_check(
        checks,
        "formal_inputs",
        formal_inputs["status"] == "frozen"
        and len(formal_inputs["projects"]) == 10
        and sum(
            item["frozen_target_finding_count"] for item in formal_inputs["projects"]
        )
        == 4393
        and all(item["status"] == "scanned" for item in formal_inputs["projects"]),
        project_count=len(formal_inputs["projects"]),
        target_alert_total=sum(
            item["frozen_target_finding_count"] for item in formal_inputs["projects"]
        ),
    )

    dry_root = (
        WORKSPACE_ROOT
        / "baseline_data"
        / "exp_hapskill"
        / "smoke"
        / "formal_input_dry_04"
    )
    dry_manifests = {
        condition: json.loads(
            (
                dry_root
                / condition
                / "applications_permission_manager"
                / "run_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for condition in ("vanilla", "hapskill")
    }
    vanilla = dry_manifests["vanilla"]
    hapskill = dry_manifests["hapskill"]
    add_check(
        checks,
        "paired_formal_dry_run",
        vanilla["status"] == hapskill["status"] == "dry_run_verified"
        and vanilla["input_tree_sha256"] == hapskill["input_tree_sha256"]
        and vanilla["initial_target_findings_sha256"]
        == hapskill["initial_target_findings_sha256"]
        and vanilla["protocol_sha256"]
        == hapskill["protocol_sha256"]
        == EXPECTED_HASHES["protocol"],
        conditions={
            name: {
                "path": str(
                    dry_root
                    / name
                    / "applications_permission_manager"
                    / "run_manifest.json"
                ),
                "sha256": sha256_file(
                    dry_root
                    / name
                    / "applications_permission_manager"
                    / "run_manifest.json"
                ),
                "status": manifest["status"],
                "input_tree_sha256": manifest["input_tree_sha256"],
                "initial_target_findings_sha256": manifest[
                    "initial_target_findings_sha256"
                ],
            }
            for name, manifest in dry_manifests.items()
        },
    )

    source_35 = (
        HAPREPAIR_ROOT
        / "revision"
        / "coding_agent_baseline"
        / "source_manifest_35_gitcode.json"
    )
    scan_35 = (
        HAPREPAIR_ROOT
        / "revision"
        / "coding_agent_baseline"
        / "scan_runs"
        / "candidate_scan_35_gitcode_01"
        / "scan_manifest.json"
    )
    rq1 = (
        WORKSPACE_ROOT
        / "paper"
        / "rebuttal"
        / "rq1_population"
        / "population_35_summary.json"
    )
    source_payload = json.loads(source_35.read_text(encoding="utf-8"))
    scan_payload = json.loads(scan_35.read_text(encoding="utf-8"))
    rq1_payload = json.loads(rq1.read_text(encoding="utf-8"))
    add_check(
        checks,
        "g0_fresh_35_project_population",
        source_payload["ready_project_count"] == 35
        and scan_payload["scanned_project_count"] == 35
        and rq1_payload["project_count"] == 35
        and rq1_payload["target_finding_count"] == 9884,
        artifacts={
            str(source_35): sha256_file(source_35),
            str(scan_35): sha256_file(scan_35),
            str(rq1): sha256_file(rq1),
            rq1_payload["population_path"]: rq1_payload["population_sha256"],
        },
        raw_findings=rq1_payload["raw_finding_count"],
        target_findings=rq1_payload["target_finding_count"],
        category_counts=rq1_payload["category_counts"],
    )

    image = run(
        [
            "docker",
            "image",
            "inspect",
            "hybrid-gym-codex:0.146.0",
            "--format",
            "{{.Id}}",
        ],
        WORKSPACE_ROOT,
    )
    image_id = image.stdout.strip()
    add_check(
        checks,
        "container_image",
        image.returncode == 0 and image_id == EXPECTED_IMAGE_ID,
        image="hybrid-gym-codex:0.146.0",
        expected_id=EXPECTED_IMAGE_ID,
        actual_id=image_id,
        stderr=image.stderr.strip(),
    )

    wrapper = WRAPPER_SMOKE / "evaluator_build.sh"
    wrapper_workspace = WRAPPER_SMOKE / "workspace"
    wrapper_result = run([str(wrapper)], wrapper_workspace)
    add_check(
        checks,
        "evaluator_wrapper_real_build_smoke",
        wrapper.is_file()
        and wrapper_result.returncode == 0
        and "BUILD SUCCESSFUL" in wrapper_result.stdout,
        command=[str(wrapper)],
        cwd=str(wrapper_workspace),
        wrapper_sha256=sha256_file(wrapper) if wrapper.is_file() else None,
        returncode=wrapper_result.returncode,
        stdout=wrapper_result.stdout,
        stderr=wrapper_result.stderr,
    )

    add_check(
        checks,
        "formal_run_directory_absent",
        not FORMAL_RUN.exists(),
        path=str(FORMAL_RUN),
    )
    add_check(
        checks,
        "harness_files_frozen",
        all(path.is_file() for path in HARNESS_FILES),
        files={str(path): sha256_file(path) for path in HARNESS_FILES},
    )

    test_jobs = [
        (
            "hapskill_harness",
            ["python3", "-m", "unittest", "-v", "test_harness.py"],
            SCRIPT_DIR,
            7,
        ),
        (
            "skill",
            ["python3", "-m", "unittest", "-v", "test_haprepair_skill.py"],
            HAPREPAIR_ROOT / "skills" / "haprepair-openharmony-repair" / "scripts",
            16,
        ),
        (
            "legacy_shared_gates",
            ["python3", "-m", "unittest", "-v", "test_baseline_runner.py"],
            HAPREPAIR_ROOT / "revision" / "coding_agent_baseline",
            22,
        ),
    ]
    for name, command, cwd, expected_count in test_jobs:
        result = run(command, cwd)
        combined = result.stdout + result.stderr
        lifecycle_covered = name != "skill" or all(
            marker in combined
            for marker in (
                "test_repair_required_candidate_can_be_fixed_and_accepted_in_place",
                "test_final_scan_restores_last_valid_candidate",
                "test_structural_guard_remains_anchored_to_last_valid_candidate",
            )
        )
        add_check(
            checks,
            f"tests:{name}",
            result.returncode == 0
            and f"Ran {expected_count} tests" in combined
            and lifecycle_covered,
            command=command,
            cwd=str(cwd),
            expected_test_count=expected_count,
            returncode=result.returncode,
            lifecycle_covered=lifecycle_covered,
            output=combined,
        )

    passed = all(check["passed"] for check in checks)
    result = {
        "schema_version": 1,
        "experiment": "EXP-HAPSKILL-10",
        "audit": "G0/G2 read-only final preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "formal_execution_authorized_by_preflight": passed,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUTPUT}: {'PASS' if passed else 'FAIL'} sha256={sha256_file(OUTPUT)}")
    if not passed:
        for check in checks:
            if not check["passed"]:
                print(f"FAILED: {check['name']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
