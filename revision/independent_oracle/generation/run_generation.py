#!/usr/bin/env python3
"""Prepare, execute, summarize, and verify EXP-INDEP-63 generation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
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
PROTOCOL_PATH = SCRIPT_DIR / "protocol.json"
PROMPT_TEMPLATE_PATH = SCRIPT_DIR / "prompt_template.txt"
RUNS_DIR = ORACLE_DIR / "generation_runs"
SYSTEM_PROMPT = (
    "You are an expert ArkTS repair system. Follow the supplied same-rule "
    "demonstration and return only the requested JSON patch object."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    path = Path(args.protocol).resolve() if args.protocol else PROTOCOL_PATH
    if not path.is_file():
        raise SystemExit(f"Protocol file does not exist: {path}")
    return read_json(path), path


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_repo_path(relative: str) -> Path:
    path = REPO_ROOT / relative
    if not path.exists():
        raise SystemExit(f"Required path does not exist: {path}")
    return path


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"Frozen {label} hash mismatch: expected {expected}, found {actual}"
        )


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def endpoint_fingerprint(base_url: str) -> str:
    return sha256_bytes(base_url.rstrip("/").encode("utf-8"))


def load_dotenv_credentials() -> tuple[str, str, str]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("python-dotenv is required") from exc
    env_path = WORKSPACE_ROOT / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    if not api_key or not api_base:
        raise SystemExit("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    if not api_base.rstrip("/").endswith("/v1"):
        api_base = f"{api_base.rstrip('/')}/v1"
    return api_key, api_base, sha256_file(env_path)


def validate_frozen_inputs(protocol: dict[str, Any]) -> dict[str, Path]:
    benchmark = protocol["benchmark"]
    retrieval = protocol["retrieval"]
    leakage_gate = protocol["leakage_gate"]
    paths = {
        "case_manifest": resolve_repo_path(benchmark["case_manifest"]),
        "defective_project": resolve_repo_path(benchmark["defective_project"]),
        "knowledge_base": resolve_repo_path(retrieval["corpus"]),
        "leakage_report": resolve_repo_path(leakage_gate["report"]),
    }
    validation_dir = ORACLE_DIR / "runs" / benchmark["validation_run"]
    paths.update(
        {
            "validation_summary": validation_dir / "summary.json",
            "defective_report": validation_dir / "defective.json",
            "case_results": validation_dir / "case_results.csv",
        }
    )
    assert_hash(
        paths["case_manifest"], benchmark["case_manifest_sha256"], "case manifest"
    )
    assert_hash(
        paths["knowledge_base"], retrieval["corpus_sha256"], "knowledge base"
    )
    assert_hash(
        paths["validation_summary"],
        benchmark["validation_summary_sha256"],
        "validation summary",
    )
    assert_hash(
        paths["defective_report"],
        benchmark["defective_report_sha256"],
        "defective report",
    )
    assert_hash(
        paths["case_results"], benchmark["case_results_sha256"], "case results"
    )
    assert_hash(
        paths["leakage_report"], leakage_gate["report_sha256"], "leakage report"
    )

    summary = read_json(paths["validation_summary"])
    leakage = read_json(paths["leakage_report"])
    if not (
        summary.get("validated_case_count") == benchmark["case_count"]
        and summary.get("failed_case_count") == 0
        and summary.get("all_scans_clean") is True
    ):
        raise SystemExit("Frozen CodeLinter validation gate is not satisfied")
    if leakage_gate["required_pass"] and not leakage.get("automatic_gate_passed"):
        raise SystemExit("Frozen benchmark leakage gate is not satisfied")
    return paths


def finding_index(report_path: Path, defective_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for file_report in read_json(report_path):
        path = Path(file_report["filePath"])
        try:
            relative = path.resolve().relative_to(defective_root.resolve()).as_posix()
        except ValueError as exc:
            raise SystemExit(f"Finding outside defective project: {path}") from exc
        result[relative] = file_report.get("messages", [])
    return result


def assemble_files(project: Path, relative_paths: list[str]) -> tuple[str, list[dict[str, str]]]:
    files = []
    rendered = []
    for relative in relative_paths:
        content = (project / relative).read_text(encoding="utf-8")
        files.append({"path": relative, "content": content})
        rendered.append(f"File: {relative}\n```arkts\n{content}\n```")
    return "\n\n".join(rendered), files


def make_blind_ids(cases: list[dict[str, Any]], seed: int) -> dict[str, str]:
    ordered = sorted(case["case_id"] for case in cases)
    blind_ids = [f"HPR-{index:03d}" for index in range(1, len(ordered) + 1)]
    random.Random(seed).shuffle(blind_ids)
    return dict(zip(ordered, blind_ids, strict=True))


def load_encoder(
    model_name: str, device_name: str, cache_dir: str
) -> tuple[Any, Any, str]:
    try:
        import torch
        import transformers.utils.import_utils as transformers_import_utils
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("torch and transformers are required for retrieval") from exc
    # This text-only encoder does not use torchvision. Disabling the optional
    # import avoids an unrelated torch/torchvision binary mismatch on the host.
    transformers_import_utils._torchvision_available = False
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"Requested retrieval device is unavailable: {device_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    model.to(torch.device(device_name))
    revision = getattr(model.config, "_commit_hash", None) or "unavailable"
    return model, tokenizer, revision


def embed_one(text: str, model: Any, tokenizer: Any) -> Any:
    import torch

    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().float().cpu()


def cosine(left: Any, right: Any) -> float:
    import torch

    score = torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0))
    return float(score.item())


def render_prompt(
    template: str,
    case: dict[str, Any],
    blind_id: str,
    findings: list[dict[str, Any]],
    target_files: str,
    demo: dict[str, Any],
) -> str:
    return template.format(
        demo_rule=demo["rule"],
        demo_description=demo["description"],
        demo_problem_code=demo["problem_code"],
        demo_repair_code=demo["repair_code"],
        demo_explanation=demo["problem_explanation"],
        blind_id=blind_id,
        target_rule=case["rule"],
        target_description=case["rule_description"],
        target_findings_json=json.dumps(findings, ensure_ascii=False, indent=2),
        target_files=target_files,
    )


def prepare(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args)
    if args.run_id != protocol["run_id"]:
        raise SystemExit(f"Formal run ID must be {protocol['run_id']}")
    paths = validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    if run_dir.exists() and not args.resume_prepare:
        raise SystemExit(f"Run directory already exists: {run_dir}")
    if run_dir.exists() and (run_dir / "input_manifest.json").exists():
        raise SystemExit("Preparation already completed; refusing to overwrite it")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    (run_dir / "errors").mkdir(exist_ok=True)

    manifest = read_json(paths["case_manifest"])
    cases = manifest["cases"]
    pairs = load_jsonl(paths["knowledge_base"])
    if len(cases) != protocol["benchmark"]["case_count"]:
        raise SystemExit("Unexpected benchmark case count")
    if len(pairs) != protocol["retrieval"]["pair_count"]:
        raise SystemExit("Unexpected retrieval-pair count")
    case_rules = {case["rule"] for case in cases}
    pair_rules = {pair["rule"] for pair in pairs}
    if case_rules != pair_rules or len(case_rules) != protocol["benchmark"]["rule_count"]:
        raise SystemExit("Benchmark and retrieval rule sets differ")

    blind_map = make_blind_ids(cases, protocol["blind_id_seed"])
    findings_by_file = finding_index(
        paths["defective_report"], paths["defective_project"]
    )
    pair_vectors: dict[str, Any] = {}
    preparation_attempts = run_dir / "preparation_attempts.jsonl"
    attempt = {
        "event": "preparation_started",
        "timestamp": utc_now(),
        "device": args.device,
        "cache_dir": str(Path(args.cache_dir).resolve()),
    }
    append_jsonl(preparation_attempts, attempt)
    try:
        model, tokenizer, encoder_revision = load_encoder(
            protocol["retrieval"]["encoder"], args.device, args.cache_dir
        )
    except Exception as exc:
        append_jsonl(
            preparation_attempts,
            {
                "event": "preparation_failed",
                "timestamp": utc_now(),
                "failure_layer": "environment",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    started = time.monotonic()
    for index, pair in enumerate(pairs, start=1):
        pair_vectors[pair["pair_id"]] = embed_one(
            pair["problem_code"], model, tokenizer
        )
        if index % 25 == 0:
            print(f"retrieval corpus embeddings: {index}/{len(pairs)}", flush=True)

    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    prepared = []
    retrieval_rows = []
    for index, case in enumerate(sorted(cases, key=lambda item: item["case_id"]), start=1):
        target_rendered, target_file_records = assemble_files(
            paths["defective_project"], case["defective_files"]
        )
        case_findings = []
        codelinter_rule = case.get("codelinter_rule", case["rule"])
        for relative in case["defective_files"]:
            for finding in findings_by_file.get(relative, []):
                if finding.get("rule") == codelinter_rule:
                    case_findings.append({"file": relative, **finding})
        if not case_findings:
            raise SystemExit(f"No frozen target finding for {case['case_id']}")

        query_vector = embed_one(target_rendered, model, tokenizer)
        candidates = []
        for pair in pairs:
            if pair["rule"] != case["rule"]:
                continue
            candidates.append(
                (cosine(query_vector, pair_vectors[pair["pair_id"]]), pair["pair_id"], pair)
            )
        if not candidates:
            raise SystemExit(f"No same-rule retrieval candidate for {case['case_id']}")
        candidates.sort(key=lambda item: (-item[0], item[1]))
        score, _, demo = candidates[0]
        if not math.isfinite(score):
            raise SystemExit(f"Non-finite retrieval score for {case['case_id']}")

        blind_id = blind_map[case["case_id"]]
        user_prompt = render_prompt(
            template, case, blind_id, case_findings, target_rendered, demo
        )
        prompt_record = {
            "schema_version": 1,
            "experiment": protocol["experiment"],
            "run_id": args.run_id,
            "case_id": case["case_id"],
            "blind_id": blind_id,
            "rule": case["rule"],
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "prompt_sha256": sha256_bytes(
                (SYSTEM_PROMPT + "\n" + user_prompt).encode("utf-8")
            ),
            "target_files": target_file_records,
            "target_findings": case_findings,
            "retrieval": {
                "rank": 1,
                "pair_id": demo["pair_id"],
                "rule": demo["rule"],
                "score": score,
            },
            "benchmark_reference_included": False,
        }
        prompt_path = run_dir / "prompts" / f"{blind_id}.json"
        write_json(prompt_path, prompt_record)
        prepared.append(
            {
                "case_id": case["case_id"],
                "blind_id": blind_id,
                "rule": case["rule"],
                "prompt": prompt_path.relative_to(run_dir).as_posix(),
                "prompt_sha256": prompt_record["prompt_sha256"],
                "retrieved_pair_id": demo["pair_id"],
                "retrieval_score": score,
            }
        )
        for rank, (candidate_score, candidate_pair_id, _) in enumerate(candidates, start=1):
            retrieval_rows.append(
                {
                    "case_id": case["case_id"],
                    "blind_id": blind_id,
                    "rule": case["rule"],
                    "rank": rank,
                    "pair_id": candidate_pair_id,
                    "score": candidate_score,
                }
            )
        print(f"prepared prompts: {index}/{len(cases)}", flush=True)

    write_json(run_dir / "protocol_snapshot.json", protocol)
    write_json(run_dir / "blind_map.json", blind_map)
    write_json(run_dir / "input_manifest.json", prepared)
    with (run_dir / "retrieval_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(retrieval_rows[0]))
        writer.writeheader()
        writer.writerows(retrieval_rows)
    environment = {
        "prepared_at": utc_now(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "retrieval_device": args.device,
        "retrieval_encoder": protocol["retrieval"]["encoder"],
        "retrieval_encoder_revision": encoder_revision,
        "preparation_wall_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "environment.prepare.json", environment)
    write_json(
        run_dir / "preparation_summary.json",
        {
            "run_id": args.run_id,
            "status": "prepared",
            "case_count": len(prepared),
            "unique_rule_count": len({item["rule"] for item in prepared}),
            "unique_blind_id_count": len({item["blind_id"] for item in prepared}),
            "unique_prompt_count": len({item["prompt_sha256"] for item in prepared}),
            "retrieval_top_k": protocol["retrieval"]["top_k"],
            "reference_project_access_during_generation": False,
        },
    )
    append_jsonl(
        preparation_attempts,
        {
            "event": "preparation_succeeded",
            "timestamp": utc_now(),
            "case_count": len(prepared),
        },
    )
    print(json.dumps(read_json(run_dir / "preparation_summary.json"), indent=2))


def openai_client() -> tuple[Any, str, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai is required") from exc
    api_key, api_base, env_hash = load_dotenv_credentials()
    return (
        OpenAI(api_key=api_key, base_url=api_base, max_retries=0, timeout=300.0),
        endpoint_fingerprint(api_base),
        env_hash,
    )


def normalize_chat_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
        raw_type = type(response).__name__
    elif isinstance(response, dict):
        payload = response
        raw_type = "dict"
    elif isinstance(response, str):
        raw_type = "str"
        try:
            decoded = json.loads(response)
            payload = decoded if isinstance(decoded, dict) else {"direct_content": response}
        except json.JSONDecodeError:
            payload = {"direct_content": response}
    else:
        raise TypeError(f"Unsupported chat response type: {type(response).__name__}")

    choices = payload.get("choices")
    content = None
    finish_reason = None
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
    if content is None:
        content = payload.get("direct_content") or payload.get("output_text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat endpoint returned no non-empty text content")
    if content.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("Chat endpoint returned an HTML application page, not a model response")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = None
    return {
        "payload": payload,
        "raw_type": raw_type,
        "content": content,
        "finish_reason": finish_reason,
        "reported_model": payload.get("model") or "not_reported",
        "response_id": payload.get("id") or "not_reported",
        "created": payload.get("created"),
        "usage": usage,
    }


def parse_patch_object(content: str) -> tuple[dict[str, Any], str]:
    text = content.strip()
    try:
        value = json.loads(text)
        mode = "direct_json"
    except json.JSONDecodeError:
        import re

        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
        if not match:
            raise ValueError("Response is not a JSON object")
        value = json.loads(match.group(1))
        mode = "fenced_json"
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError("Response JSON must contain a files array")
    return value, mode


def validate_patch_files(
    patch: dict[str, Any], required_paths: set[str]
) -> tuple[list[dict[str, str]], list[str]]:
    normalized_files = []
    output_paths = []
    for file_record in patch["files"]:
        if not isinstance(file_record, dict):
            raise ValueError("Every files entry must be an object")
        path = file_record.get("path")
        content = file_record.get("content")
        if not isinstance(path, str) or not path:
            raise ValueError("Every files entry must have a non-empty path")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Output file has empty content: {path}")
        parsed_path = Path(path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError(f"Output file path is unsafe: {path}")
        output_paths.append(path)
        normalized_files.append({"path": path, "content": content})
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Response contains duplicate output paths")
    missing = required_paths - set(output_paths)
    if missing:
        raise ValueError(f"Response omits input files: {sorted(missing)}")
    return normalized_files, sorted(set(output_paths) - required_paths)


def smoke(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args)
    model_id = args.model or protocol["model"]["requested_id"]
    if model_id != protocol["model"]["requested_id"]:
        raise SystemExit("Smoke test must use the frozen exact model ID")
    client, endpoint_hash, env_hash = openai_client()
    smoke_dir = RUNS_DIR / args.smoke_id
    if smoke_dir.exists():
        raise SystemExit(f"Smoke directory already exists: {smoke_dir}")
    smoke_dir.mkdir(parents=True)
    request = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Return only this JSON object: {\"status\":\"ok\"}"},
        ],
        "temperature": protocol["model"]["temperature"],
    }
    write_json(
        smoke_dir / "request.json",
        {
            **request,
            "protocol_sha256": sha256_file(protocol_path),
            "endpoint_sha256": endpoint_hash,
            "env_file_sha256": env_hash,
        },
    )
    started_at = utc_now()
    started = time.monotonic()
    try:
        response = client.chat.completions.create(**request)
        normalized = normalize_chat_response(response)
        write_json(smoke_dir / "response.json", normalized)
        result = {
            "status": "success",
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "requested_model": model_id,
            "reported_model": normalized["reported_model"],
            "raw_response_type": normalized["raw_type"],
        }
        write_json(smoke_dir / "summary.json", result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        result = {
            "status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "requested_model": model_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(smoke_dir / "summary.json", result)
        print(json.dumps(result, indent=2))
        raise SystemExit(2)


def execute(args: argparse.Namespace) -> None:
    protocol, protocol_path = load_protocol(args)
    model_id = args.model or protocol["model"]["requested_id"]
    if args.run_id != protocol["run_id"] or model_id != protocol["model"]["requested_id"]:
        raise SystemExit("Run ID and model must match the frozen protocol")
    validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    prepared = read_json(run_dir / "input_manifest.json")
    if len(prepared) != protocol["benchmark"]["case_count"]:
        raise SystemExit("Prepared input manifest is incomplete")
    client, endpoint_hash, env_hash = openai_client()
    write_json(
        run_dir / "environment.execute.json",
        {
            "started_at": utc_now(),
            "python": sys.version,
            "openai_endpoint_sha256": endpoint_hash,
            "env_file_sha256": env_hash,
            "protocol_sha256": sha256_file(protocol_path),
            "requested_model": model_id,
            "api": protocol["model"]["api"],
            "temperature": protocol["model"]["temperature"],
            "automatic_retries": protocol["model"]["automatic_retries"],
        },
    )
    selected = prepared
    if args.case_id:
        selected = [item for item in prepared if item["case_id"] == args.case_id]
        if not selected:
            raise SystemExit(f"Unknown case ID: {args.case_id}")

    for index, item in enumerate(selected, start=1):
        response_path = run_dir / "responses" / f"{item['blind_id']}.json"
        error_path = run_dir / "errors" / f"{item['blind_id']}.json"
        if response_path.exists():
            print(f"skip accepted response {item['blind_id']}", flush=True)
            continue
        if error_path.exists() and not args.retry_failed:
            print(f"skip recorded failure {item['blind_id']}", flush=True)
            continue
        prompt = read_json(run_dir / item["prompt"])
        attempt_number = 1
        previous_attempts = []
        attempts_path = run_dir / "attempts.jsonl"
        if attempts_path.exists():
            previous_attempts = load_jsonl(attempts_path)
            attempt_number += sum(
                attempt.get("event") == "request_started"
                and attempt["blind_id"] == item["blind_id"]
                for attempt in previous_attempts
            )
        request_record = {
            "event": "request_started",
            "timestamp": utc_now(),
            "case_id": item["case_id"],
            "blind_id": item["blind_id"],
            "attempt": attempt_number,
            "model": model_id,
            "prompt_sha256": item["prompt_sha256"],
        }
        append_jsonl(attempts_path, request_record)
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["user_prompt"]},
                ],
                temperature=protocol["model"]["temperature"],
            )
            elapsed = time.monotonic() - started
            normalized = normalize_chat_response(response)
            response_record = {
                "schema_version": 1,
                "experiment": protocol["experiment"],
                "run_id": args.run_id,
                "case_id": item["case_id"],
                "blind_id": item["blind_id"],
                "attempt": attempt_number,
                "requested_model": model_id,
                "reported_model": normalized["reported_model"],
                "created": normalized["created"],
                "response_id": normalized["response_id"],
                "finish_reason": normalized["finish_reason"],
                "content": normalized["content"],
                "usage": normalized["usage"],
                "wall_seconds": elapsed,
                "received_at": utc_now(),
                "raw_response_type": normalized["raw_type"],
                "raw_response": normalized["payload"],
            }
            write_json(response_path, response_record)
            if error_path.exists():
                error_path.unlink()
            append_jsonl(
                attempts_path,
                {
                    "event": "request_succeeded",
                    "timestamp": utc_now(),
                    "case_id": item["case_id"],
                    "blind_id": item["blind_id"],
                    "attempt": attempt_number,
                    "response_id": normalized["response_id"],
                    "wall_seconds": elapsed,
                },
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            error_record = {
                "event": "request_failed",
                "timestamp": utc_now(),
                "case_id": item["case_id"],
                "blind_id": item["blind_id"],
                "attempt": attempt_number,
                "model": model_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": elapsed,
            }
            write_json(error_path, error_record)
            append_jsonl(attempts_path, error_record)
            print(json.dumps(error_record, ensure_ascii=False), flush=True)
            if args.fail_fast:
                raise SystemExit(2)
        print(f"generation progress: {index}/{len(selected)}", flush=True)
    summarize_run(run_dir, protocol)


def summarize_run(run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    prepared = read_json(run_dir / "input_manifest.json")
    responses = []
    failures = []
    for item in prepared:
        response_path = run_dir / "responses" / f"{item['blind_id']}.json"
        error_path = run_dir / "errors" / f"{item['blind_id']}.json"
        if response_path.exists():
            responses.append(read_json(response_path))
        elif error_path.exists():
            failures.append(read_json(error_path))
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    usage_complete = True
    for response in responses:
        usage = response.get("usage")
        if not usage:
            usage_complete = False
            continue
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)
    blocker_path = run_dir / "blocker.json"
    blocker = read_json(blocker_path) if blocker_path.exists() else None
    if len(responses) == len(prepared):
        status = "complete"
    elif blocker:
        status = "blocked"
    else:
        status = "partial"
    wall_seconds_sum = sum(response["wall_seconds"] for response in responses)
    per_response_total_tokens = [
        response["usage"].get("total_tokens", 0)
        for response in responses
        if response.get("usage")
    ]
    metrics = {
        "run_id": protocol["run_id"],
        "status": status,
        "prepared_case_count": len(prepared),
        "accepted_response_count": len(responses),
        "failed_case_count": len(failures),
        "pending_case_count": len(prepared) - len(responses) - len(failures),
        "single_accepted_generation_per_case": (
            len({response["case_id"] for response in responses}) == len(responses)
            if responses
            else None
        ),
        "reported_models": sorted({response["reported_model"] for response in responses}),
        "usage_complete": usage_complete if responses else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_per_case_mean": input_tokens / len(responses) if responses else None,
        "output_tokens_per_case_mean": output_tokens / len(responses) if responses else None,
        "total_tokens_per_case_mean": total_tokens / len(responses) if responses else None,
        "total_tokens_per_case_min": min(per_response_total_tokens) if per_response_total_tokens else None,
        "total_tokens_per_case_max": max(per_response_total_tokens) if per_response_total_tokens else None,
        "wall_seconds_sum": wall_seconds_sum,
        "wall_seconds_per_case_mean": wall_seconds_sum / len(responses) if responses else None,
        "api_cost_usd": None,
        "cost_note": "Not inferred because the endpoint did not provide a frozen billing rate.",
        "semantic_correctness_status": "not_adjudicated",
        "blocker": blocker,
    }
    write_json(run_dir / "metrics.json", metrics)
    environment_path = run_dir / "environment.execute.json"
    if environment_path.exists() and metrics["status"] == "complete":
        environment = read_json(environment_path)
        environment["finished_at"] = max(response["received_at"] for response in responses)
        environment["accepted_response_count"] = len(responses)
        environment["final_status"] = "complete"
        write_json(environment_path, environment)
    summary = (
        f"# {protocol['experiment']} generation run\n\n"
        f"- Run ID: `{protocol['run_id']}`\n"
        f"- Status: `{metrics['status']}`\n"
        f"- Accepted responses: {metrics['accepted_response_count']}/{metrics['prepared_case_count']}\n"
        f"- Failed cases: {metrics['failed_case_count']}\n"
        f"- Requested model: `{protocol['model']['requested_id']}`\n"
        f"- Reported models: {', '.join(metrics['reported_models']) or 'none'}\n"
        f"- Total tokens: {metrics['total_tokens']}\n"
        "- Semantic correctness: not adjudicated\n"
        "- Claim update: neutral until blind author adjudication is complete.\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics


def summarize(args: argparse.Namespace) -> None:
    protocol, _ = load_protocol(args)
    run_dir = RUNS_DIR / args.run_id
    metrics = summarize_run(run_dir, protocol)
    print(json.dumps(metrics, indent=2))


def verify(args: argparse.Namespace) -> None:
    protocol, _ = load_protocol(args)
    paths = validate_frozen_inputs(protocol)
    run_dir = RUNS_DIR / args.run_id
    prepared = read_json(run_dir / "input_manifest.json")
    prompt_records = [read_json(run_dir / item["prompt"]) for item in prepared]
    checks = {
        "case_count_63": len(prepared) == 63,
        "unique_case_ids_63": len({item["case_id"] for item in prepared}) == 63,
        "unique_blind_ids_63": len({item["blind_id"] for item in prepared}) == 63,
        "unique_rules_63": len({item["rule"] for item in prepared}) == 63,
        "all_prompt_hashes_match": all(
            sha256_bytes(
                (record["system_prompt"] + "\n" + record["user_prompt"]).encode("utf-8")
            )
            == record["prompt_sha256"]
            == item["prompt_sha256"]
            for item, record in zip(prepared, prompt_records, strict=True)
        ),
        "all_retrievals_same_rule": all(
            record["rule"] == record["retrieval"]["rule"] for record in prompt_records
        ),
        "all_retrieval_scores_finite": all(
            math.isfinite(record["retrieval"]["score"]) for record in prompt_records
        ),
        "no_reference_field_in_prompts": all(
            record.get("benchmark_reference_included") is False
            and "reference_patch" not in json.dumps(record, ensure_ascii=False).lower()
            and "repaired_project" not in json.dumps(record, ensure_ascii=False).lower()
            for record in prompt_records
        ),
        "frozen_inputs_still_valid": bool(paths),
    }
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    if args.require_complete:
        responses = []
        parsed_dir = run_dir / "parsed_outputs"
        parsed_dir.mkdir(exist_ok=True)
        output_errors = []
        parse_modes = set()
        for item, prompt_record in zip(prepared, prompt_records, strict=True):
            response_path = run_dir / "responses" / f"{item['blind_id']}.json"
            if not response_path.exists():
                output_errors.append(f"missing response: {item['blind_id']}")
                continue
            response = read_json(response_path)
            responses.append(response)
            try:
                patch, parse_mode = parse_patch_object(response["content"])
                files, extra_paths = validate_patch_files(
                    patch, {record["path"] for record in prompt_record["target_files"]}
                )
                parse_modes.add(parse_mode)
                write_json(
                    parsed_dir / f"{item['blind_id']}.json",
                    {
                        "blind_id": item["blind_id"],
                        "rule": item["rule"],
                        "parse_mode": parse_mode,
                        "files": files,
                        "additional_paths": extra_paths,
                    },
                )
            except Exception as exc:
                output_errors.append(f"{item['blind_id']}: {type(exc).__name__}: {exc}")
        attempts = load_jsonl(run_dir / "attempts.jsonl")
        started_counts = {
            item["blind_id"]: sum(
                event.get("event") == "request_started"
                and event.get("blind_id") == item["blind_id"]
                for event in attempts
            )
            for item in prepared
        }
        succeeded_counts = {
            item["blind_id"]: sum(
                event.get("event") == "request_succeeded"
                and event.get("blind_id") == item["blind_id"]
                for event in attempts
            )
            for item in prepared
        }
        checks["all_63_responses_accepted"] = bool(
            metrics
            and metrics["status"] == "complete"
            and metrics["accepted_response_count"] == 63
            and metrics["failed_case_count"] == 0
            and metrics["pending_case_count"] == 0
        )
        checks["all_outputs_are_valid_patch_json"] = not output_errors
        checks["all_outputs_use_direct_json"] = parse_modes == {"direct_json"}
        checks["one_request_and_success_per_case"] = all(
            started_counts[item["blind_id"]] == 1
            and succeeded_counts[item["blind_id"]] == 1
            for item in prepared
        )
        checks["all_responses_report_frozen_model"] = all(
            response["requested_model"] == protocol["model"]["requested_id"]
            and response["reported_model"] == protocol["model"]["requested_id"]
            for response in responses
        )
        checks["all_finish_reasons_stop"] = all(
            response["finish_reason"] == "stop" for response in responses
        )
        checks["all_usage_present_and_positive"] = all(
            response.get("usage")
            and response["usage"].get("prompt_tokens", 0) > 0
            and response["usage"].get("completion_tokens", 0) > 0
            and response["usage"].get("total_tokens", 0) > 0
            for response in responses
        )
        checks["response_ids_unique"] = len(
            {response["response_id"] for response in responses}
        ) == len(responses)
        checks["metrics_reconcile_response_usage"] = bool(
            metrics
            and metrics["input_tokens"]
            == sum(response["usage"]["prompt_tokens"] for response in responses)
            and metrics["output_tokens"]
            == sum(response["usage"]["completion_tokens"] for response in responses)
            and metrics["total_tokens"]
            == sum(response["usage"]["total_tokens"] for response in responses)
        )
    result = {
        "run_id": args.run_id,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    if args.require_complete:
        result["output_errors"] = output_errors
    write_json(run_dir / "verification.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache" / "huggingface")
    )
    prepare_parser.add_argument("--resume-prepare", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--protocol")
    smoke_parser.add_argument("--smoke-id", required=True)
    smoke_parser.add_argument("--model")
    smoke_parser.set_defaults(func=smoke)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--protocol")
    execute_parser.add_argument("--run-id", required=True)
    execute_parser.add_argument("--model")
    execute_parser.add_argument("--case-id")
    execute_parser.add_argument("--retry-failed", action="store_true")
    execute_parser.add_argument("--fail-fast", action="store_true")
    execute_parser.set_defaults(func=execute)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--protocol")
    summarize_parser.add_argument("--run-id", required=True)
    summarize_parser.set_defaults(func=summarize)

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
