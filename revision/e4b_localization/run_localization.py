#!/usr/bin/env python3
"""Run and evaluate the frozen EXP-LOC-18 three-model localization protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
HAPREPAIR_ROOT = WORKSPACE_ROOT / "HapRepair"
DEFAULT_PROTOCOL = (
    WORKSPACE_ROOT
    / "paper/rebuttal/e4b_localization/exp_loc_18_three_model_v1/protocol.json"
)
DEFAULT_RUN_ROOT = HAPREPAIR_ROOT / "revision/e4b_localization/runs"
ALLOWED_RULE_PREFIX = "@performance/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_replace(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Benchmark:
    protocol: dict[str, Any]
    protocol_path: Path
    protocol_sha256: str
    source_root: Path
    gt_path: Path
    gt_sha256: str
    gt: dict[str, Any]
    files: tuple[str, ...]
    rules: tuple[str, ...]


def resolve_workspace_path(value: str) -> Path:
    path = (WORKSPACE_ROOT / value).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"Path escapes workspace: {value}")
    return path


def load_benchmark(protocol_path: Path) -> Benchmark:
    protocol_path = protocol_path.resolve()
    protocol = load_json(protocol_path)
    benchmark = protocol["benchmark"]
    source_root = resolve_workspace_path(benchmark["source_root"])
    gt_path = resolve_workspace_path(benchmark["ground_truth"])
    actual_gt_sha = sha256_file(gt_path)
    if actual_gt_sha != benchmark["ground_truth_sha256"]:
        raise RuntimeError(
            f"Ground-truth hash drift: {actual_gt_sha} != "
            f"{benchmark['ground_truth_sha256']}"
        )
    gt = load_json(gt_path)
    files = tuple(sorted(gt))
    rules = tuple(
        sorted(
            {
                defect["rule"]
                for meta in gt.values()
                for defect in meta.get("defects", [])
            }
        )
    )
    defect_count = sum(len(meta.get("defects", [])) for meta in gt.values())
    if len(files) != benchmark["file_count"]:
        raise RuntimeError("Frozen file count mismatch")
    if defect_count != benchmark["defect_count"]:
        raise RuntimeError("Frozen defect count mismatch")
    if len(rules) != benchmark["rule_count"]:
        raise RuntimeError("Frozen rule count mismatch")
    for relative in files:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
    return Benchmark(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=sha256_file(protocol_path),
        source_root=source_root,
        gt_path=gt_path,
        gt_sha256=actual_gt_sha,
        gt=gt,
        files=files,
        rules=rules,
    )


def build_prompt(code: str, relative_path: str, rules: tuple[str, ...]) -> str:
    rule_lines = "\n".join(f"- {rule}" for rule in rules)
    return f"""You are an ArkTS performance-defect detector in a fixed closed-set evaluation.

Analyze only the source text supplied below. Do not use tools, retrieve files, or ask for more context. Find every violation of the listed rule IDs, including multiple violations of the same rule. Use 1-based source line numbers and report the line where the violating construct begins.

Rules of interest:
{rule_lines}

Return exactly one JSON object with this shape and no markdown or explanation:
{{"defects":[{{"rule":"@performance/example-rule","line":123}}]}}

If none of the listed rules is violated, return {{"defects":[]}}.

File: {relative_path}

```arkts
{code}
```
"""


def add_bounded_retry_instruction(prompt: str, source_line_count: int) -> str:
    return (
        prompt
        + "\nFORMAT RETRY: Return one item per distinct violating construct, not one "
        "item per line inside a multi-line construct. For a multi-line construct, "
        "report only its first line. This source has "
        f"{source_line_count} lines, so every line value must be between 1 and "
        f"{source_line_count}.\n"
    )


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("Response choice has no textual content")


def parse_prediction(content: str, allowed_rules: set[str]) -> tuple[list[dict[str, Any]], str | None]:
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return [], f"json_decode_error:{exc.msg}"
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("defects"), list):
        items = parsed["defects"]
    else:
        return [], "schema_error:expected_object_with_defects"
    defects: set[tuple[str, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            return [], "schema_error:defect_not_object"
        rule = item.get("rule")
        line = item.get("line")
        if isinstance(line, str) and line.isdigit():
            line = int(line)
        if not isinstance(rule, str) or rule not in allowed_rules:
            return [], "schema_error:unknown_rule"
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            return [], "schema_error:invalid_line"
        defects.add((rule, line))
    return [
        {"rule": rule, "line": line}
        for rule, line in sorted(defects, key=lambda item: (item[1], item[0]))
    ], None


def endpoint_url(base_url: str, path: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/" + path.strip("/")
    if normalized.endswith(suffix):
        return normalized
    return normalized + suffix


def call_model(
    model: dict[str, Any],
    prompt: str,
    generation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key_name = model["api_key_env"]
    base_name = model["api_base_env"]
    api_key = os.getenv(key_name)
    api_base = os.getenv(base_name)
    if not api_key or not api_base:
        raise RuntimeError(f"Missing required environment: {key_name}/{base_name}")
    url = endpoint_url(api_base, model["chat_completions_path"])
    payload = {
        "model": model["id"],
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": generation["temperature"],
        "reasoning_effort": generation["reasoning_effort"],
        "max_tokens": generation["max_output_tokens"],
    }
    if generation.get("response_format") is not None:
        payload["response_format"] = generation["response_format"]
    if generation.get("frequency_penalty") is not None:
        payload["frequency_penalty"] = generation["frequency_penalty"]
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts: list[dict[str, Any]] = []
    maximum_attempts = int(generation["transport_retries"]) + 1
    for attempt in range(1, maximum_attempts + 1):
        started = time.monotonic()
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "haprepair-exp-loc-18/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
                status = response.status
            elapsed = time.monotonic() - started
            attempts.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "result": "success",
                }
            )
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("API response is not a JSON object")
            return decoded, attempts
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            error_body = exc.read().decode("utf-8", errors="replace")[:2000]
            attempts.append(
                {
                    "attempt": attempt,
                    "status": exc.code,
                    "elapsed_seconds": elapsed,
                    "result": "http_error",
                    "error": error_body,
                }
            )
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt == maximum_attempts:
                raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            elapsed = time.monotonic() - started
            attempts.append(
                {
                    "attempt": attempt,
                    "status": None,
                    "elapsed_seconds": elapsed,
                    "result": "transport_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt == maximum_attempts:
                raise RuntimeError(f"Transport failure: {exc}") from exc
        time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("Unreachable retry loop")


def result_path(run_dir: Path, model_id: str, relative_path: str) -> Path:
    return run_dir / "responses" / slug(model_id) / f"{slug(relative_path)}.json"


def execute_one(
    benchmark: Benchmark,
    run_dir: Path,
    model: dict[str, Any],
    relative_path: str,
    formal: bool,
    max_output_tokens: int | None = None,
    json_object_output: bool = False,
    strict_json_schema: bool = False,
    frequency_penalty: float | None = None,
    bounded_retry_prompt: bool = False,
) -> dict[str, Any]:
    output = result_path(run_dir, model["id"], relative_path)
    if output.exists():
        existing = load_json(output)
        if (
            existing.get("protocol_sha256") != benchmark.protocol_sha256
            or existing.get("source_sha256")
            != sha256_file(benchmark.source_root / relative_path)
        ):
            raise RuntimeError(f"Existing result drifted: {output}")
        return existing
    source_path = benchmark.source_root / relative_path
    code = source_path.read_text(encoding="utf-8")
    base_prompt = build_prompt(code, relative_path, benchmark.rules)
    prompt = (
        add_bounded_retry_instruction(base_prompt, len(code.splitlines()))
        if bounded_retry_prompt
        else base_prompt
    )
    generation = dict(benchmark.protocol["generation"])
    if max_output_tokens is not None:
        generation["max_output_tokens"] = max_output_tokens
    if json_object_output:
        generation["response_format"] = {"type": "json_object"}
    if strict_json_schema:
        generation["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "defect_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "defects": {
                            "type": "array",
                            "maxItems": len(code.splitlines()),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rule": {
                                        "type": "string",
                                        "enum": list(benchmark.rules),
                                    },
                                    "line": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": len(code.splitlines()),
                                    },
                                },
                                "required": ["rule", "line"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["defects"],
                    "additionalProperties": False,
                },
            },
        }
    if frequency_penalty is not None:
        generation["frequency_penalty"] = frequency_penalty
    started_at = utc_now()
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": 1,
        "formal": formal,
        "run_id": run_dir.name,
        "model_requested": model["id"],
        "file": relative_path,
        "source_sha256": sha256_file(source_path),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "base_prompt_sha256": sha256_bytes(base_prompt.encode("utf-8")),
        "prompt_variant": "bounded_format_retry" if bounded_retry_prompt else "frozen",
        "protocol_sha256": benchmark.protocol_sha256,
        "ground_truth_sha256": benchmark.gt_sha256,
        "effective_generation": {
            "temperature": generation["temperature"],
            "reasoning_effort": generation["reasoning_effort"],
            "max_output_tokens": generation["max_output_tokens"],
            "response_format": generation.get("response_format"),
            "frequency_penalty": generation.get("frequency_penalty"),
        },
        "started_at": started_at,
        "provider_route": {
            "api_base_env": model["api_base_env"],
            "api_key_env": model["api_key_env"],
            "api_base_sha256": sha256_bytes(
                os.environ[model["api_base_env"]].encode("utf-8")
            ),
        },
    }
    try:
        response, attempts = call_model(
            model,
            prompt,
            generation,
        )
        content = extract_message_content(response)
        defects, parse_error = parse_prediction(content, set(benchmark.rules))
        record.update(
            {
                "status": "completed",
                "model_reported": response.get("model"),
                "response_id": response.get("id"),
                "usage": response.get("usage"),
                "attempts": attempts,
                "content": content,
                "predicted_defects": defects,
                "parse_ok": parse_error is None,
                "parse_error": parse_error,
                "raw_response": response,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "api_failure",
                "model_reported": None,
                "usage": None,
                "attempts": [],
                "content": None,
                "predicted_defects": [],
                "parse_ok": False,
                "parse_error": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["completed_at"] = utc_now()
    record["elapsed_seconds"] = time.monotonic() - started
    write_json_new(output, record)
    return record


def metric_counts(
    gt: set[tuple[str, int]], pred: set[tuple[str, int]]
) -> dict[str, Any]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_records(
    benchmark: Benchmark, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["model_requested"]].append(record)
    summaries: dict[str, Any] = {}
    for model_id, model_records in sorted(by_model.items()):
        strata: dict[str, dict[str, set[tuple[str, str, int]]]] = defaultdict(
            lambda: {"gt": set(), "pred": set()}
        )
        parse_failures = 0
        api_failures = 0
        exact_file_count = 0
        within_one_line_tp: set[tuple[str, str, int]] = set()
        rule_only_tp: set[tuple[str, str, int]] = set()
        per_file = []
        for record in sorted(model_records, key=lambda item: item["file"]):
            relative = record["file"]
            gt_pairs = {
                (item["rule"], int(item["line"]))
                for item in benchmark.gt[relative].get("defects", [])
            }
            pred_pairs = {
                (item["rule"], int(item["line"]))
                for item in record.get("predicted_defects", [])
            }
            if record["status"] != "completed":
                api_failures += 1
            elif not record.get("parse_ok"):
                parse_failures += 1
            stratum = "single_defect_files" if len(gt_pairs) == 1 else "multi_defect_files"
            for rule, line in gt_pairs:
                key = (relative, rule, line)
                strata["overall"]["gt"].add(key)
                strata[stratum]["gt"].add(key)
                strata[f"rule:{rule}"]["gt"].add(key)
                if any(pred_rule == rule for pred_rule, _ in pred_pairs):
                    rule_only_tp.add(key)
                if any(
                    pred_rule == rule and abs(pred_line - line) <= 1
                    for pred_rule, pred_line in pred_pairs
                ):
                    within_one_line_tp.add(key)
            for rule, line in pred_pairs:
                key = (relative, rule, line)
                strata["overall"]["pred"].add(key)
                strata[stratum]["pred"].add(key)
                strata[f"rule:{rule}"]["pred"].add(key)
            if gt_pairs == pred_pairs:
                exact_file_count += 1
            per_file.append(
                {
                    "file": relative,
                    "gt_count": len(gt_pairs),
                    "predicted_count": len(pred_pairs),
                    "parse_ok": record.get("parse_ok", False),
                    "status": record["status"],
                    **metric_counts(gt_pairs, pred_pairs),
                }
            )
        evaluated = {
            name: metric_counts(values["gt"], values["pred"])
            for name, values in sorted(strata.items())
        }
        total_gt = len(strata["overall"]["gt"])
        summaries[model_id] = {
            "file_count": len(model_records),
            "completed_file_count": sum(
                record["status"] == "completed" for record in model_records
            ),
            "parse_failure_count": parse_failures,
            "api_failure_count": api_failures,
            "exact_match_file_count": exact_file_count,
            "strata": evaluated,
            "rule_only_recall": len(rule_only_tp) / total_gt if total_gt else 0.0,
            "within_one_line_recall": (
                len(within_one_line_tp) / total_gt if total_gt else 0.0
            ),
            "per_file": per_file,
        }
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "protocol_sha256": benchmark.protocol_sha256,
        "ground_truth_sha256": benchmark.gt_sha256,
        "model_summaries": summaries,
    }


def evaluate_file_rule_records(
    benchmark: Benchmark, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate detection by unique (file, rule), deliberately ignoring lines."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["model_requested"]].append(record)
    summaries: dict[str, Any] = {}
    for model_id, model_records in sorted(by_model.items()):
        strata: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
            lambda: {"gt": set(), "pred": set()}
        )
        parse_failures = 0
        api_failures = 0
        exact_file_count = 0
        per_file = []
        for record in sorted(model_records, key=lambda item: item["file"]):
            relative = record["file"]
            gt_rules = {
                item["rule"] for item in benchmark.gt[relative].get("defects", [])
            }
            pred_rules = {
                item["rule"] for item in record.get("predicted_defects", [])
            }
            if record["status"] != "completed":
                api_failures += 1
            elif not record.get("parse_ok"):
                parse_failures += 1
            defect_count = len(benchmark.gt[relative].get("defects", []))
            stratum = "single_defect_files" if defect_count == 1 else "multi_defect_files"
            for rule in gt_rules:
                key = (relative, rule)
                strata["overall"]["gt"].add(key)
                strata[stratum]["gt"].add(key)
                strata[f"rule:{rule}"]["gt"].add(key)
            for rule in pred_rules:
                key = (relative, rule)
                strata["overall"]["pred"].add(key)
                strata[stratum]["pred"].add(key)
                strata[f"rule:{rule}"]["pred"].add(key)
            if gt_rules == pred_rules:
                exact_file_count += 1
            per_file.append(
                {
                    "file": relative,
                    "gt_rule_count": len(gt_rules),
                    "predicted_rule_count": len(pred_rules),
                    "parse_ok": record.get("parse_ok", False),
                    "status": record["status"],
                    **metric_counts(gt_rules, pred_rules),
                }
            )
        summaries[model_id] = {
            "file_count": len(model_records),
            "completed_file_count": sum(
                record["status"] == "completed" for record in model_records
            ),
            "parse_failure_count": parse_failures,
            "api_failure_count": api_failures,
            "exact_match_file_count": exact_file_count,
            "strata": {
                name: metric_counts(values["gt"], values["pred"])
                for name, values in sorted(strata.items())
            },
            "per_file": per_file,
        }
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "identity": "unique (relative_file, rule); source line ignored",
        "protocol_sha256": benchmark.protocol_sha256,
        "ground_truth_sha256": benchmark.gt_sha256,
        "model_summaries": summaries,
    }


def select_models(benchmark: Benchmark, requested: list[str] | None) -> list[dict[str, Any]]:
    models = benchmark.protocol["models"]
    if not requested:
        return models
    known = {model["id"]: model for model in models}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown frozen model(s): {unknown}")
    return [known[model_id] for model_id in requested]


def collect_records(run_dir: Path) -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted((run_dir / "responses").glob("*/*.json"))]


def write_run_manifest(
    benchmark: Benchmark,
    run_dir: Path,
    mode: str,
    models: list[dict[str, Any]],
    files: list[str],
) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "mode": mode,
        "created_at": utc_now(),
        "protocol_path": str(benchmark.protocol_path.relative_to(WORKSPACE_ROOT)),
        "protocol_sha256": benchmark.protocol_sha256,
        "ground_truth_path": str(benchmark.gt_path.relative_to(WORKSPACE_ROOT)),
        "ground_truth_sha256": benchmark.gt_sha256,
        "source_root": str(benchmark.source_root.relative_to(WORKSPACE_ROOT)),
        "source_tree": benchmark.protocol["benchmark"]["source_tree"],
        "models": [model["id"] for model in models],
        "files": files,
        "expected_condition_count": len(models) * len(files),
        "python": sys.version,
        "secrets_persisted": False,
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        existing = load_json(path)
        comparable = {
            key: existing[key]
            for key in (
                "mode",
                "protocol_sha256",
                "ground_truth_sha256",
                "models",
                "files",
                "expected_condition_count",
            )
        }
        expected = {key: manifest[key] for key in comparable}
        if comparable != expected:
            raise RuntimeError(f"Existing run manifest is incompatible: {path}")
        return
    write_json_new(path, manifest)


def execute(args: argparse.Namespace) -> int:
    benchmark = load_benchmark(args.protocol)
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "status": "valid",
                    "protocol_sha256": benchmark.protocol_sha256,
                    "ground_truth_sha256": benchmark.gt_sha256,
                    "file_count": len(benchmark.files),
                    "defect_count": sum(
                        len(meta.get("defects", [])) for meta in benchmark.gt.values()
                    ),
                    "rule_count": len(benchmark.rules),
                },
                sort_keys=True,
            )
        )
        return 0

    models = select_models(benchmark, args.models)
    run_dir = (args.run_root / args.run_id).resolve()
    if not run_dir.is_relative_to(args.run_root.resolve()):
        raise ValueError("Run ID escapes run root")

    if args.mode == "summarize":
        records = collect_records(run_dir)
        summary = evaluate_records(benchmark, records)
        write_json_replace(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "smoke":
        if args.files:
            raise ValueError("--files is only supported in formal mode")
        files = [args.smoke_file or benchmark.files[0]]
        if files[0] not in benchmark.files:
            raise ValueError("Smoke file is not in the frozen benchmark")
        formal = False
    elif args.mode == "formal":
        files = list(args.files or benchmark.files)
        unknown_files = sorted(set(files) - set(benchmark.files))
        if unknown_files:
            raise ValueError(f"Unknown frozen file(s): {unknown_files}")
        if len(set(files)) != len(files):
            raise ValueError("Duplicate --files values are not allowed")
        formal = True
        if benchmark.protocol.get("status") != "frozen_before_formal_execution":
            raise RuntimeError("Protocol is not frozen for formal execution")
    else:
        raise ValueError(args.mode)

    write_run_manifest(benchmark, run_dir, args.mode, models, files)
    work = [(model, relative) for model in models for relative in files]
    records = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(work))) as executor:
        futures = {
            executor.submit(
                execute_one,
                benchmark,
                run_dir,
                model,
                relative,
                formal,
                args.max_output_tokens,
                args.json_object_output,
                args.strict_json_schema,
                args.frequency_penalty,
                args.bounded_retry_prompt,
            ): (model["id"], relative)
            for model, relative in work
        }
        for future in as_completed(futures):
            model_id, relative = futures[future]
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "file": relative,
                        "status": record["status"],
                        "parse_ok": record.get("parse_ok"),
                        "predicted": len(record.get("predicted_defects", [])),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = evaluate_records(benchmark, records)
    write_json_replace(run_dir / "summary.json", summary)
    return 1 if any(record["status"] != "completed" for record in records) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "formal", "summarize"),
        required=True,
    )
    parser.add_argument("--run-id", default="exp_loc_18_three_model_v1")
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Run an explicit frozen-file subset in formal mode (for audited retries).",
    )
    parser.add_argument("--smoke-file")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Audited retry override; the effective value is persisted per response.",
    )
    parser.add_argument(
        "--json-object-output",
        action="store_true",
        help="Use the provider JSON-object response mode for an audited retry.",
    )
    parser.add_argument(
        "--strict-json-schema",
        action="store_true",
        help="Constrain an audited retry to the frozen output schema and file bounds.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        help="Provider-supported repetition control for an audited retry.",
    )
    parser.add_argument(
        "--bounded-retry-prompt",
        action="store_true",
        help="Append line-bound and one-item-per-construct instructions for a format retry.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.frequency_penalty is not None and not -2 <= args.frequency_penalty <= 2:
        parser.error("--frequency-penalty must be between -2 and 2")
    return args


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
