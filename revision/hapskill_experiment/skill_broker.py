#!/usr/bin/env python3
"""Host-side operation broker that keeps HapRepair artifacts out of agent containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 32 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_requests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("retrieval requests must be a non-empty JSON list")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each retrieval request must be an object")
        rule = item.get("rule")
        context = item.get("context")
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError("each retrieval request requires a rule")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("each retrieval request requires non-empty agent context")
        top_k = int(item.get("top_k", 1))
        if top_k != 1:
            raise ValueError("the frozen experiment permits Top-1 retrieval only")
        normalized.append({"rule": rule, "context": context, "top_k": 1})
    return normalized


def run_skill(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"HapRepair operation returned invalid JSON (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(payload.get("error", f"operation exited {result.returncode}"))
    return payload


def handle_request(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    base = [sys.executable, str(args.skill_script)]
    if operation == "inspect-rule":
        rule = request.get("rule")
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError("inspect-rule requires a rule")
        return run_skill(
            base
            + [
                "inspect-rule",
                "--state-dir",
                str(args.state_dir),
                "--rule",
                rule,
            ]
        )
    if operation == "retrieve-repairs":
        requests = validate_requests(request.get("requests"))
        request_id = uuid.uuid4().hex
        request_path = args.request_dir / f"{request_id}.json"
        write_json(request_path, requests)
        return run_skill(
            base
            + [
                "retrieve-repairs",
                "--state-dir",
                str(args.state_dir),
                "--request-file",
                str(request_path),
                "--backend",
                "stella",
                "--device",
                args.device,
                "--model-cache",
                str(args.model_cache),
                "--embedding-cache",
                str(args.embedding_cache),
                "--top-k",
                "1",
            ]
        )
    raise ValueError(f"operation is not exposed to the agent: {operation!r}")


def receive_request(connection: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise ValueError("broker request exceeded the size limit")
        chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--skill-script", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.socket = args.socket.resolve()
    args.state_dir = args.state_dir.resolve()
    args.skill_script = args.skill_script.resolve()
    args.model_cache = args.model_cache.resolve()
    args.embedding_cache = args.embedding_cache.resolve()
    args.request_dir = args.state_dir / "broker_requests"
    args.request_dir.mkdir(parents=True, exist_ok=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise FileExistsError(f"broker socket already exists: {args.socket}")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(8)
        append_jsonl(args.log, {"event": "ready", "at": utc_now()})
        while True:
            connection, _ = server.accept()
            with connection:
                started = time.monotonic()
                request: dict[str, Any] = {}
                try:
                    request = receive_request(connection)
                    if request.get("operation") == "shutdown":
                        response = {"ok": True, "result": {"status": "stopping"}}
                        connection.sendall(json.dumps(response).encode("utf-8"))
                        append_jsonl(args.log, {"event": "shutdown", "at": utc_now()})
                        return
                    result = handle_request(args, request)
                    response = {"ok": True, "result": result}
                    status = result.get("status", "completed")
                except Exception as error:
                    response = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    status = "error"
                encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
                connection.sendall(encoded)
                request_summary = {
                    "operation": request.get("operation"),
                    "rule": request.get("rule"),
                    "request_count": len(request.get("requests", []))
                    if isinstance(request.get("requests"), list)
                    else None,
                }
                append_jsonl(
                    args.log,
                    {
                        "event": "request",
                        "at": utc_now(),
                        "elapsed_seconds": time.monotonic() - started,
                        "status": status,
                        "request_summary": request_summary,
                        "request_sha256": hashlib.sha256(
                            json.dumps(request, sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                        "response_sha256": hashlib.sha256(encoded).hexdigest(),
                    },
                )


if __name__ == "__main__":
    main()
