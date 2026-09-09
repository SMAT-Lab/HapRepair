#!/usr/bin/env python3
"""Run the EXP-AGENT-10 HapRepair condition on a pinned project copy."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from get_prompt import combine_repair_results, generate_fix_prompt  # type: ignore
from output_handler import (  # type: ignore
    ArkTSDeclarationFixer,
    remove_difflib_line,
)
from revision.code.fix_projects_codelinter import (  # type: ignore
    build_merged_blocks_for_file,
)

from build_gate import prepare_build_gate, run_build_gate
from public_api_guard import prepare_public_api_guard, run_public_api_guard
from run_agent_baseline import (
    compute_alert_metrics,
    copy_project,
    format_evaluator_feedback,
    restore_editable_files,
    sha256_file,
    snapshot_editable_files,
    source_diff,
    tree_manifest,
    verify_initial_findings,
)
from validation_gate import initialize_gate, load_state, run_scan, write_json


DEFAULT_SELECTION = SCRIPT_DIR / "selected_projects.json"
DEFAULT_PROTOCOL = SCRIPT_DIR / "protocol.json"
DEFAULT_CORPUS = SCRIPT_DIR.parent / "knowledge_base" / "rule_complete_383.jsonl"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "baseline_data" / "exp_agent_10" / "runs"
GENERATION_MODULE = (
    SCRIPT_DIR.parent / "independent_oracle" / "generation" / "run_generation.py"
)
ARKTS_VALIDATOR = SCRIPT_DIR / "validate_arkts_source.js"
CODE_BLOCK_RE = re.compile(
    r"```(?:arkts|javascript|js|ts|typescript)\s*\r?\n(.*?)\r?\n```",
    re.DOTALL | re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_fingerprint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_generation_helpers() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp_agent_10_generation_helpers", GENERATION_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import retrieval helpers: {GENERATION_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenRetriever:
    def __init__(
        self,
        corpus_path: Path,
        *,
        encoder_name: str,
        device: str,
        cache_dir: str,
    ) -> None:
        self.corpus_path = corpus_path.resolve()
        self.pairs = load_jsonl(self.corpus_path)
        self.by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in self.pairs:
            self.by_rule[pair["rule"]].append(pair)
        for pairs in self.by_rule.values():
            pairs.sort(key=lambda item: item["pair_id"])
        helpers = load_generation_helpers()
        self._embed_one = helpers.embed_one
        self._cosine = helpers.cosine
        self.model, self.tokenizer, self.encoder_revision = helpers.load_encoder(
            encoder_name, device, cache_dir
        )
        self.encoder_name = encoder_name
        self.device = device
        self._pair_vectors: dict[str, Any] = {}

    def retrieve(self, rule: str, context: str) -> tuple[dict[str, Any], float]:
        candidates = self.by_rule.get(rule, [])
        if not candidates:
            raise RuntimeError(f"No same-rule RAG example in frozen corpus: {rule}")
        query = self._embed_one(context, self.model, self.tokenizer)
        scored = []
        for pair in candidates:
            pair_id = pair["pair_id"]
            if pair_id not in self._pair_vectors:
                self._pair_vectors[pair_id] = self._embed_one(
                    pair["problem_code"], self.model, self.tokenizer
                )
            score = float(self._cosine(query, self._pair_vectors[pair_id]))
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite retrieval score for {pair_id}")
            scored.append((score, pair_id, pair))
        scored.sort(key=lambda item: (-item[0], item[1]))
        score, _, pair = scored[0]
        return pair, score


def render_rag_demo(pair: dict[str, Any], rank: int = 1) -> str:
    stored_diff = pair.get("difflib") or ""
    if not stored_diff.strip():
        stored_diff = "\n".join(
            difflib.unified_diff(
                pair["problem_code"].splitlines(),
                pair["repair_code"].splitlines(),
                lineterm="",
            )
        )
    return (
        f"Demo {rank}:\nRule Type:\n{pair['rule']}\n\n"
        f"Description:\n{pair.get('description', '')}\n\n"
        f"Problem Code:\n```arkts\n{pair['problem_code']}\n```\n\n"
        f"Fix Explanation:\n{pair.get('problem_explanation', '')}\n\n"
        f"Fixed Code:\n```arkts\n{pair['repair_code']}\n```\n\n"
        "Following is the difflib result of repairing the buggy code:\n\n"
        f"{stored_diff}\n\n"
    )


def load_client() -> tuple[Any, str, str]:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("openai and python-dotenv are required") from error
    env_path = WORKSPACE_ROOT / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    if not api_key or not api_base:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    if not api_base.rstrip("/").endswith("/v1"):
        api_base = f"{api_base.rstrip('/')}/v1"
    return (
        OpenAI(api_key=api_key, base_url=api_base, max_retries=0, timeout=600.0),
        endpoint_fingerprint(api_base),
        sha256_file(env_path),
    )


def normalize_response(response: Any) -> dict[str, Any]:
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported response type: {type(response).__name__}")
    text = getattr(response, "output_text", None) or payload.get("output_text")
    if not isinstance(text, str) or not text.strip():
        outputs = payload.get("output", [])
        parts = []
        for item in outputs if isinstance(outputs, list) else []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        text = "".join(parts)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Responses API returned no non-empty output text")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "text": text,
        "response_id": payload.get("id"),
        "reported_model": payload.get("model"),
        "usage": {
            key: int(value)
            for key, value in usage.items()
            if key in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(value, int)
        },
        "payload": payload,
    }


def validate_arkts_source(code: str, filename: str) -> dict[str, Any]:
    result = subprocess.run(
        ["node", str(ARKTS_VALIDATOR), filename],
        input=code,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ArkTS candidate validation failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def extract_validated_code(response_text: str, filename: str) -> tuple[str | None, dict[str, Any]]:
    blocks = CODE_BLOCK_RE.findall(response_text)
    source_kind = "single_fenced_code_block" if len(blocks) == 1 else "unfenced_response"
    if len(blocks) > 1:
        return None, {
            "status": "rejected",
            "reason": "multiple code blocks in full-file merge response",
            "code_block_count": len(blocks),
        }
    candidate = blocks[0] if blocks else response_text.strip()
    candidate = remove_difflib_line(candidate)
    declaration_fix = ArkTSDeclarationFixer().validate_and_fix(candidate)
    if declaration_fix:
        candidate = declaration_fix.fixed_code
    validation = validate_arkts_source(candidate, filename)
    if not validation["valid"]:
        return None, {
            "status": "rejected",
            "reason": "candidate is not syntactically valid ArkTS",
            "source_kind": source_kind,
            "diagnostics": validation["diagnostics"][:20],
        }
    return candidate, {
        "status": "accepted",
        "source_kind": source_kind,
        "diagnostics": [],
    }


def call_model(
    client: Any,
    prompt: str,
    call_dir: Path,
    *,
    model: str,
    effort: str,
    max_transport_attempts: int = 10,
) -> dict[str, Any]:
    call_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": effort},
    }
    write_json(call_dir / "request.json", request)
    try:
        from httpx import TransportError
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError as error:
        raise RuntimeError("openai is required") from error
    retryable = (
        APIConnectionError,
        APIError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        TransportError,
    )
    started_at = utc_now()
    started = time.monotonic()
    transport_attempts = []
    response = None
    for attempt_number in range(1, max_transport_attempts + 1):
        attempt_started = time.monotonic()
        try:
            # Stream long generations so the gateway observes progress instead of
            # terminating the request at its non-stream response deadline.
            with client.responses.stream(**request) as stream:
                for _event in stream:
                    pass
                response = stream.get_final_response()
            transport_attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "response_received",
                    "wall_seconds": time.monotonic() - attempt_started,
                }
            )
            break
        except retryable as error:
            transport_attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "transport_failed_before_response",
                    "wall_seconds": time.monotonic() - attempt_started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            write_json(call_dir / "transport_attempts.json", transport_attempts)
            if attempt_number == max_transport_attempts:
                raise
            time.sleep(min(2 ** (attempt_number - 1), 16))
    if response is None:
        raise RuntimeError("Model transport loop ended without a response")
    elapsed = time.monotonic() - started
    normalized = normalize_response(response)
    write_json(call_dir / "response.json", normalized["payload"])
    write_json(call_dir / "transport_attempts.json", transport_attempts)
    record = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "wall_seconds": elapsed,
        "requested_model": model,
        "reported_model": normalized["reported_model"],
        "response_id": normalized["response_id"],
        "usage": normalized["usage"],
        "transport_attempt_count": len(transport_attempts),
        "failed_transport_attempt_count": sum(
            item["status"] != "response_received" for item in transport_attempts
        ),
        "transport_attempts_path": str(call_dir / "transport_attempts.json"),
        "request_path": str(call_dir / "request.json"),
        "request_sha256": sha256_file(call_dir / "request.json"),
        "response_path": str(call_dir / "response.json"),
        "response_sha256": sha256_file(call_dir / "response.json"),
    }
    write_json(call_dir / "call.json", record)
    return {**record, "text": normalized["text"]}


def partition_pipeline_findings(
    workspace: Path, findings: list[dict[str, Any]]
) -> tuple[dict[Path, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    for finding in findings:
        rule = str(finding["rule"])
        if not rule.startswith("@") or "/" not in rule:
            raise RuntimeError(f"Unexpected HomeCheck rule identity: {rule}")
        category, rule_id = rule[1:].split("/", 1)
        path = (workspace / finding["relative_path"]).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError as error:
            raise RuntimeError(f"Finding escapes workspace: {path}") from error
        if not path.is_file():
            skipped.append(
                {
                    "finding": finding,
                    "reason": "localized path does not exist in the project",
                }
            )
            continue
        if path.suffix not in {".ets", ".ts"}:
            skipped.append(
                {
                    "finding": finding,
                    "reason": "HapRepair only processes existing .ets and .ts source files",
                }
            )
            continue
        grouped[path].append(
            {
                "file_path": str(path),
                "line": int(finding.get("line", 0)),
                "column": int(finding.get("column", 0)),
                "severity": finding.get("severity", ""),
                "category": category,
                "rule_id": rule_id,
                "message": finding.get("message", ""),
            }
        )
    return grouped, skipped


def repair_round(
    workspace: Path,
    findings: list[dict[str, Any]],
    retriever: FrozenRetriever,
    client: Any,
    round_dir: Path,
    *,
    model: str,
    effort: str,
    previous_feedback: str = "",
) -> dict[str, Any]:
    grouped, skipped_findings = partition_pipeline_findings(workspace, findings)
    round_dir.mkdir(parents=True, exist_ok=False)
    calls_dir = round_dir / "calls"
    calls_dir.mkdir()
    retrievals: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    call_index = 0

    for file_path in sorted(grouped, key=lambda path: path.as_posix()):
        if not file_path.is_file():
            raise RuntimeError(f"Localized file is missing: {file_path}")
        file_findings = grouped[file_path]
        original_text = file_path.read_text(encoding="utf-8")
        blocks, code_lines = build_merged_blocks_for_file(file_path, file_findings)
        code = "\n".join(code_lines)
        repair_results: list[tuple[str, str]] = []
        file_retrievals = []
        file_calls = []

        for block_index, block in enumerate(blocks, start=1):
            context = "\n\n".join(block["surrounding_context"])
            unique_rules = sorted({defect["rule"] for defect in block["defects"]})
            demos = []
            for rule in unique_rules:
                pair, score = retriever.retrieve(rule, context)
                demos.append(render_rag_demo(pair))
                retrieval = {
                    "relative_path": file_path.relative_to(workspace).as_posix(),
                    "block": block_index,
                    "rule": rule,
                    "rank": 1,
                    "pair_id": pair["pair_id"],
                    "score": score,
                }
                retrievals.append(retrieval)
                file_retrievals.append(retrieval)

            defect_description = "The code snippet containing the defect is as follows:\n"
            error_location = "The location of the defect is as follows:\n"
            for index, defect in enumerate(block["defects"], start=1):
                defect_description += f"Defect {index}:\n{defect['message']}\n"
                error_location += f"Defect {index}:\n{defect['code']}\n"
            prompt = generate_fix_prompt(
                "".join(demos),
                code,
                context,
                defect_description,
                error_location,
            )
            if previous_feedback:
                prompt += (
                    "\n\nEvaluator feedback from the previous rejected round:\n"
                    f"{previous_feedback}\n"
                    "Do not repeat the rejected change. Return only the requested repair output."
                )
            call_index += 1
            call = call_model(
                client,
                prompt,
                calls_dir / f"call_{call_index:04d}_block",
                model=model,
                effort=effort,
            )
            calls.append({key: value for key, value in call.items() if key != "text"})
            file_calls.append(call_index)
            repair_results.append((context, call["text"]))

        # A rejected merge must be a no-op. `code` is reconstructed by the
        # context extractor and may differ textually from the source on disk.
        fixed_code = original_text
        merge_output = {"status": "not_run", "reason": "no repair blocks"}
        if repair_results:
            merge_prompt = combine_repair_results(repair_results, code)
            if previous_feedback:
                merge_prompt += (
                    "\n\nThe previous round was rejected by the evaluator:\n"
                    f"{previous_feedback}\n"
                    "Return only one complete ArkTS source file. Do not include analysis, "
                    "tool calls, progress narration, or text outside the source code."
                )
            call_index += 1
            call = call_model(
                client,
                merge_prompt,
                calls_dir / f"call_{call_index:04d}_merge",
                model=model,
                effort=effort,
            )
            calls.append({key: value for key, value in call.items() if key != "text"})
            file_calls.append(call_index)
            candidate, merge_output = extract_validated_code(
                call["text"], file_path.name
            )
            if candidate is not None:
                fixed_code = candidate
        if not isinstance(fixed_code, str) or not fixed_code.strip():
            raise RuntimeError(f"HapRepair produced empty code for {file_path}")
        file_path.write_text(fixed_code, encoding="utf-8")
        files.append(
            {
                "relative_path": file_path.relative_to(workspace).as_posix(),
                "input_finding_count": len(file_findings),
                "block_count": len(blocks),
                "llm_call_indices": file_calls,
                "retrieval_count": len(file_retrievals),
                "changed": fixed_code != original_text,
                "merge_output": merge_output,
            }
        )

    write_json(round_dir / "retrievals.json", retrievals)
    write_json(round_dir / "calls.json", calls)
    write_json(round_dir / "files.json", files)
    write_json(round_dir / "skipped_findings.json", skipped_findings)
    usage: Counter[str] = Counter()
    for call in calls:
        usage.update(call["usage"])
    return {
        "input_alerts": len(findings),
        "repairable_alerts": len(findings) - len(skipped_findings),
        "skipped_finding_count": len(skipped_findings),
        "file_count": len(files),
        "retrieval_count": len(retrievals),
        "llm_call_count": len(calls),
        "usage": dict(usage),
        "retrievals_path": str(round_dir / "retrievals.json"),
        "calls_path": str(round_dir / "calls.json"),
        "files_path": str(round_dir / "files.json"),
        "skipped_findings_path": str(round_dir / "skipped_findings.json"),
    }


def load_scan_findings(scan: dict[str, Any]) -> list[dict[str, Any]]:
    if scan.get("status") != "scanned" or not scan.get("findings_path"):
        raise RuntimeError(f"HomeCheck scan failed: {scan.get('parse_error')}")
    return json.loads(Path(scan["findings_path"]).read_text(encoding="utf-8"))


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
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify input copy and initial scan without retrieval or model calls.",
    )
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    protocol_path = args.protocol.resolve()
    corpus_path = args.corpus.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    project_meta = select_project(selection, args.project)
    model = protocol["model"]
    max_rounds = int(protocol["homecheck"]["max_post_edit_validation_scans"])
    corpus = load_jsonl(corpus_path)
    if len(corpus) != 383 or len({item["rule"] for item in corpus}) != 63:
        raise SystemExit("Frozen HapRepair corpus is not the expected 383-pair/63-rule view")

    run_dir = args.run_root.resolve() / args.run_id / "haprepair" / args.project
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    gate_dir = run_dir / "validation_gate"
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir()
    source = Path(project_meta["source_path"]).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "condition": "haprepair",
        "run_id": args.run_id,
        "project": args.project,
        "status": "preparing",
        "started_at": utc_now(),
        "completed_at": None,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source": str(source),
        "commit": project_meta["commit"],
        "tree_oid": project_meta["tree_oid"],
        "model": model,
        "retrieval": {
            "corpus": str(corpus_path),
            "corpus_sha256": sha256_file(corpus_path),
            "pair_count": len(corpus),
            "rule_count": len({item["rule"] for item in corpus}),
            "filter": "same_rule",
            "top_k": 1,
            "encoder": "dunzhang/stella_en_1.5B_v5",
            "pooling": "unmasked mean of last_hidden_state",
            "similarity": "cosine",
            "tie_break": "pair_id ascending",
        },
        "rounds": [],
    }
    write_json(manifest_path, manifest)

    copy_started = time.monotonic()
    copy_project(source, workspace)
    manifest["copy_seconds"] = time.monotonic() - copy_started
    manifest["byte_identical_input"] = tree_manifest(source) == tree_manifest(workspace)
    if not manifest["byte_identical_input"]:
        manifest["status"] = "copy_verification_failed"
        write_json(manifest_path, manifest)
        raise SystemExit("Workspace copy differs from pinned source")

    initialize_gate(gate_dir, workspace, max_validation_scans=max_rounds)
    print(f"[initial] scanning {args.project}", flush=True)
    initial_scan = run_scan(gate_dir, "initial")
    initial_findings = load_scan_findings(initial_scan)
    frozen_findings = json.loads(Path(project_meta["findings_path"]).read_text(encoding="utf-8"))
    initial_match, initial_verification = verify_initial_findings(
        frozen_findings, initial_findings
    )
    manifest["initial_scan"] = initial_scan
    manifest["initial_verification"] = initial_verification
    if not initial_match:
        manifest["status"] = "initial_scan_mismatch"
        write_json(manifest_path, manifest)
        raise SystemExit(f"Initial scan mismatch: {initial_verification}")
    print(f"[initial] verified {len(initial_findings)} alerts", flush=True)
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

    client, endpoint_sha256, env_sha256 = load_client()
    manifest["endpoint_sha256"] = endpoint_sha256
    manifest["env_file_sha256"] = env_sha256
    retrieval_started = time.monotonic()
    retriever = FrozenRetriever(
        corpus_path,
        encoder_name=manifest["retrieval"]["encoder"],
        device=args.device,
        cache_dir=args.cache_dir,
    )
    manifest["retrieval"].update(
        {
            "device": args.device,
            "encoder_revision": retriever.encoder_revision,
            "initialization_seconds": time.monotonic() - retrieval_started,
        }
    )
    write_json(manifest_path, manifest)

    current_findings = initial_findings
    total_usage: Counter[str] = Counter()
    previous_feedback = ""
    experiment_started = time.monotonic()
    try:
        for round_number in range(1, max_rounds + 1):
            print(
                f"[haprepair] round {round_number}/{max_rounds}: "
                f"{len(current_findings)} localized alerts",
                flush=True,
            )
            snapshot = snapshot_editable_files(workspace)
            round_record = repair_round(
                workspace,
                current_findings,
                retriever,
                client,
                rounds_dir / f"round_{round_number:02d}",
                model=model["requested_id"],
                effort=model["reasoning_effort"],
                previous_feedback=previous_feedback,
            )
            total_usage.update(round_record["usage"])
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
            validation_scan = run_scan(gate_dir, "validation")
            current_findings = load_scan_findings(validation_scan)
            round_record["round"] = round_number
            round_record["public_api_guard"] = public_api_gate
            round_record["public_api_rollback_verification"] = public_api_rollback_verification
            round_record["evaluator_build_gate"] = evaluator_build
            round_record["rollback_reason"] = rollback_reason
            round_record["rollback_verification"] = rollback_verification
            round_record["validation_scan"] = validation_scan
            round_files = json.loads(Path(round_record["files_path"]).read_text(encoding="utf-8"))
            rejected_outputs = [
                item["relative_path"]
                for item in round_files
                if item.get("merge_output", {}).get("status") == "rejected"
            ]
            previous_feedback = format_evaluator_feedback(
                public_api_gate,
                evaluator_build,
                rejected_outputs=rejected_outputs,
            )
            round_record["feedback_for_next_round"] = previous_feedback
            manifest["rounds"].append(round_record)
            write_json(manifest_path, manifest)
            print(
                f"[validation] round {round_number}: {len(current_findings)} alerts remain",
                flush=True,
            )
            if not current_findings:
                break
    except Exception as error:
        manifest["status"] = "run_failed"
        manifest["completed_at"] = utc_now()
        manifest["failure"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise

    print("[final] evaluator-only HomeCheck scan", flush=True)
    final_scan = run_scan(gate_dir, "final")
    final_findings = load_scan_findings(final_scan)
    alert_metrics, alert_deltas = compute_alert_metrics(initial_findings, final_findings)
    deltas_path = run_dir / "alert_deltas.json"
    write_json(deltas_path, alert_deltas)
    gate_state = load_state(gate_dir)
    evaluator_builds = [
        record
        for round_record in manifest["rounds"]
        for record in (
            round_record["evaluator_build_gate"],
            round_record.get("rollback_verification"),
        )
        if record and record.get("build")
    ]
    manifest.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "wall_clock_seconds": time.monotonic() - experiment_started,
            "python": sys.version,
            "platform": platform.platform(),
            "final_scan": final_scan,
            "alert_metrics": alert_metrics,
            "alert_deltas_path": str(deltas_path),
            "validation_scan_count": gate_state["validation_attempts_consumed"],
            "build_status": (
                "passed" if build_gate_setup["available"] else "not_available"
            ),
            "test_status": "not_available",
            "build_count": len(evaluator_builds),
            "test_count": 0,
            "build_execution_seconds": sum(
                item["build"]["duration_seconds"] for item in evaluator_builds
            ),
            "test_execution_seconds": None,
            "evaluator_build_count": len(evaluator_builds),
            "evaluator_build_execution_seconds": sum(
                item["build"]["duration_seconds"] for item in evaluator_builds
            ),
            "validation_scope": build_gate_setup["validation_scope"],
            "input_tokens": total_usage.get("input_tokens", 0),
            "output_tokens": total_usage.get("output_tokens", 0),
            "total_tokens": total_usage.get("total_tokens", 0),
            "api_cost": None,
            "api_cost_note": "No traceable provider price or billing record was available.",
            "source_diff": source_diff(source, workspace, run_dir / "source_changes.patch"),
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
