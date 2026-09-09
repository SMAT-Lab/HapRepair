#!/usr/bin/env python3
"""Audit a completed frozen E4b localization run without calling any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run_localization as runner


DEFAULT_RUN_ID = "exp_loc_18_three_model_v1"
DEFAULT_OUTPUT = (
    runner.WORKSPACE_ROOT
    / "paper/rebuttal/e4b_localization/exp_loc_18_three_model_v1/formal_run_audit.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recursive_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from recursive_values(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_values(item, key)


def response_set_sha256(run_dir: Path, paths: list[Path]) -> str:
    lines = [
        f"{path.relative_to(run_dir)}\t{sha256_file(path)}\n"
        for path in paths
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: Any
) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def audit_run(protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    benchmark = runner.load_benchmark(protocol_path)
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    manifest = runner.load_json(manifest_path)
    stored_summary = runner.load_json(summary_path)
    response_paths = sorted((run_dir / "responses").glob("*/*.json"))
    records = [runner.load_json(path) for path in response_paths]
    models = tuple(model["id"] for model in benchmark.protocol["models"])
    expected_conditions = {
        (model_id, relative)
        for model_id in models
        for relative in benchmark.files
    }
    actual_conditions = [
        (record.get("model_requested"), record.get("file"))
        for record in records
    ]

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "protocol_and_ground_truth_hashes",
        manifest.get("protocol_sha256") == benchmark.protocol_sha256
        and manifest.get("ground_truth_sha256") == benchmark.gt_sha256,
        {
            "protocol_sha256": benchmark.protocol_sha256,
            "ground_truth_sha256": benchmark.gt_sha256,
        },
    )
    add_check(
        checks,
        "manifest_contract",
        manifest.get("mode") == "formal"
        and manifest.get("models") == list(models)
        and manifest.get("files") == list(benchmark.files)
        and manifest.get("expected_condition_count") == len(expected_conditions),
        {
            "mode": manifest.get("mode"),
            "models": manifest.get("models"),
            "file_count": len(manifest.get("files", [])),
            "expected_condition_count": manifest.get("expected_condition_count"),
        },
    )
    add_check(
        checks,
        "exact_condition_partition",
        len(records) == len(expected_conditions)
        and len(set(actual_conditions)) == len(actual_conditions)
        and set(actual_conditions) == expected_conditions,
        {
            "expected": len(expected_conditions),
            "observed": len(records),
            "unique": len(set(actual_conditions)),
            "missing": sorted(expected_conditions - set(actual_conditions)),
            "extra": sorted(set(actual_conditions) - expected_conditions),
        },
    )

    record_errors: list[dict[str, Any]] = []
    model_identity_mismatches: list[dict[str, str | None]] = []
    reasoning_content_values: list[Any] = []
    reasoning_token_values: list[Any] = []
    finish_reasons: Counter[str] = Counter()
    transport_retry_count = 0
    non_success_attempts = 0
    parse_failures: list[dict[str, Any]] = []
    api_failures: list[dict[str, Any]] = []
    out_of_range_predictions: list[dict[str, Any]] = []
    token_usage: dict[str, Counter[str]] = defaultdict(Counter)

    for path, record in zip(response_paths, records):
        model_id = record.get("model_requested")
        relative = record.get("file")
        if model_id != record.get("model_reported"):
            model_identity_mismatches.append(
                {
                    "file": relative,
                    "requested": model_id,
                    "reported": record.get("model_reported"),
                }
            )
        source_path = benchmark.source_root / relative
        code = source_path.read_text(encoding="utf-8")
        base_prompt = runner.build_prompt(code, relative, benchmark.rules)
        prompt_variant = record.get("prompt_variant", "frozen")
        if prompt_variant == "bounded_format_retry":
            effective_prompt = runner.add_bounded_retry_instruction(
                base_prompt, len(code.splitlines())
            )
        else:
            effective_prompt = base_prompt
        expected_prompt_hash = runner.sha256_bytes(
            effective_prompt.encode("utf-8")
        )
        expected_base_prompt_hash = runner.sha256_bytes(
            base_prompt.encode("utf-8")
        )
        problems = []
        if record.get("formal") is not True:
            problems.append("not_formal")
        if record.get("run_id") != run_dir.name:
            problems.append("wrong_run_id")
        if record.get("protocol_sha256") != benchmark.protocol_sha256:
            problems.append("wrong_protocol_sha256")
        if record.get("ground_truth_sha256") != benchmark.gt_sha256:
            problems.append("wrong_ground_truth_sha256")
        if record.get("source_sha256") != runner.sha256_file(source_path):
            problems.append("wrong_source_sha256")
        if record.get("prompt_sha256") != expected_prompt_hash:
            problems.append("wrong_prompt_sha256")
        if record.get("base_prompt_sha256", expected_base_prompt_hash) != (
            expected_base_prompt_hash
        ):
            problems.append("wrong_base_prompt_sha256")
        if prompt_variant not in {"frozen", "bounded_format_retry"}:
            problems.append("unknown_prompt_variant")
        if problems:
            record_errors.append(
                {"path": str(path.relative_to(run_dir)), "problems": problems}
            )

        if record.get("status") != "completed":
            api_failures.append(
                {"model": model_id, "file": relative, "status": record.get("status")}
            )
        elif record.get("parse_ok") is not True:
            parse_failures.append(
                {
                    "model": model_id,
                    "file": relative,
                    "parse_error": record.get("parse_error"),
                    "finish_reason": record.get("raw_response", {})
                    .get("choices", [{}])[0]
                    .get("finish_reason"),
                    "completion_tokens": (record.get("usage") or {}).get(
                        "completion_tokens"
                    ),
                }
            )

        raw_response = record.get("raw_response", {})
        reasoning_content_values.extend(
            recursive_values(raw_response, "reasoning_content")
        )
        reasoning_token_values.extend(
            recursive_values(raw_response, "reasoning_tokens")
        )
        reason = (
            raw_response.get("choices", [{}])[0].get("finish_reason") or "missing"
        )
        finish_reasons[reason] += 1
        attempts = record.get("attempts", [])
        transport_retry_count += max(0, len(attempts) - 1)
        non_success_attempts += sum(
            attempt.get("result") != "success" for attempt in attempts
        )

        max_line = len(code.splitlines())
        for defect in record.get("predicted_defects", []):
            if defect["line"] > max_line:
                out_of_range_predictions.append(
                    {
                        "model": model_id,
                        "file": relative,
                        "rule": defect["rule"],
                        "line": defect["line"],
                        "max_line": max_line,
                    }
                )
        usage = record.get("usage") or {}
        token_usage[model_id]["prompt_tokens"] += usage.get("prompt_tokens", 0)
        token_usage[model_id]["completion_tokens"] += usage.get(
            "completion_tokens", 0
        )
        token_usage[model_id]["total_tokens"] += usage.get("total_tokens", 0)

    add_check(
        checks,
        "record_hash_and_prompt_binding",
        not record_errors,
        {"error_count": len(record_errors), "errors": record_errors},
    )
    add_check(
        checks,
        "requested_reported_model_identity",
        not model_identity_mismatches,
        {
            "mismatch_count": len(model_identity_mismatches),
            "mismatches": model_identity_mismatches,
        },
    )
    nonempty_reasoning = [
        value for value in reasoning_content_values if value not in (None, "", [])
    ]
    nonzero_reasoning_tokens = [
        value for value in reasoning_token_values if value not in (None, 0)
    ]
    add_check(
        checks,
        "thinking_disabled",
        benchmark.protocol["generation"].get("reasoning_effort") == "none"
        and not nonempty_reasoning
        and not nonzero_reasoning_tokens,
        {
            "reasoning_effort": benchmark.protocol["generation"].get(
                "reasoning_effort"
            ),
            "reasoning_content_field_count": len(reasoning_content_values),
            "nonempty_reasoning_content_count": len(nonempty_reasoning),
            "reasoning_tokens_field_count": len(reasoning_token_values),
            "nonzero_reasoning_tokens_count": len(nonzero_reasoning_tokens),
        },
    )
    add_check(
        checks,
        "api_completion",
        not api_failures,
        {
            "api_failure_count": len(api_failures),
            "failures": api_failures,
            "transport_retry_count": transport_retry_count,
            "non_success_attempt_count": non_success_attempts,
        },
    )

    recomputed_summary = runner.evaluate_records(benchmark, records)
    summary_matches = (
        stored_summary.get("protocol_sha256") == benchmark.protocol_sha256
        and stored_summary.get("ground_truth_sha256") == benchmark.gt_sha256
        and stored_summary.get("model_summaries")
        == recomputed_summary.get("model_summaries")
    )
    add_check(
        checks,
        "summary_exact_recomputation",
        summary_matches,
        {"model_summaries_match": summary_matches},
    )

    passed = all(check["passed"] for check in checks)
    metrics = {
        model_id: {
            "overall": stored_summary["model_summaries"][model_id]["strata"][
                "overall"
            ],
            "single_defect_files": stored_summary["model_summaries"][model_id][
                "strata"
            ]["single_defect_files"],
            "multi_defect_files": stored_summary["model_summaries"][model_id][
                "strata"
            ]["multi_defect_files"],
            "rule_only_recall": stored_summary["model_summaries"][model_id][
                "rule_only_recall"
            ],
            "within_one_line_recall": stored_summary["model_summaries"][model_id][
                "within_one_line_recall"
            ],
        }
        for model_id in models
    }
    return {
        "schema_version": 1,
        "audit_id": f"{run_dir.name}_formal_run_audit",
        "generated_at": utc_now(),
        "status": (
            "passed_with_recorded_parse_failure"
            if passed and parse_failures
            else "passed" if passed else "failed"
        ),
        "passed": passed,
        "run_id": run_dir.name,
        "checks": checks,
        "observations": {
            "response_count": len(records),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "parse_failure_count": len(parse_failures),
            "parse_failures": parse_failures,
            "parse_failure_disposition": (
                "Preserved as zero valid predictions under the frozen protocol; "
                "no semantic resampling."
            ),
            "out_of_range_prediction_count": len(out_of_range_predictions),
            "out_of_range_predictions": out_of_range_predictions,
            "token_usage": {
                model_id: dict(token_usage[model_id]) for model_id in models
            },
        },
        "metrics": metrics,
        "hashes": {
            "protocol": benchmark.protocol_sha256,
            "ground_truth": benchmark.gt_sha256,
            "run_manifest": sha256_file(manifest_path),
            "summary": sha256_file(summary_path),
            "response_set": response_set_sha256(run_dir, response_paths),
        },
        "reporting_boundary": benchmark.protocol["evaluation"][
            "reporting_boundary"
        ],
        "next_action": "Use this audited LLM run in the joint E4b evaluation.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=runner.DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=runner.DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = (args.run_root / args.run_id).resolve()
    audit = audit_run(args.protocol.resolve(), run_dir)
    runner.write_json_replace(args.output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
