#!/usr/bin/env python3
"""Restricted agent-side client for the EXP-HAPSKILL-10 retrieval broker."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any


MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def exchange(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = client.recv(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("HapRepair broker response exceeded the size limit")
            chunks.append(chunk)
    response = json.loads(b"".join(chunks).decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "HapRepair broker request failed"))
    return response["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path("/hapskill-broker/broker.sock"),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    inspect = subparsers.add_parser("inspect-rule")
    inspect.add_argument("--rule", required=True)

    retrieve = subparsers.add_parser("retrieve-repairs")
    retrieve.add_argument("--request-file", type=Path, required=True)

    args = parser.parse_args()
    if args.operation == "inspect-rule":
        request = {"operation": "inspect-rule", "rule": args.rule}
    else:
        requests = json.loads(args.request_file.read_text(encoding="utf-8"))
        request = {"operation": "retrieve-repairs", "requests": requests}
    result = exchange(args.socket, request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
