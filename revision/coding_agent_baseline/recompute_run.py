#!/usr/bin/env python3
"""Recompute derived EXP-AGENT-10 fields from immutable raw run evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_agent_baseline import (
    command_status,
    extract_commands,
    parse_codex_events,
    provider_endpoint_fingerprint,
    source_diff,
)
from validation_gate import write_json


def measured_seconds(commands: list[dict[str, Any]]) -> float | None:
    durations = [item.get("duration_seconds") for item in commands]
    if not durations or not all(isinstance(value, (int, float)) for value in durations):
        return None
    return float(sum(durations))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_commands: list[dict[str, Any]] = []
    for round_record in manifest["rounds"]:
        trace_path = Path(round_record["turn"]["trace_path"])
        events = parse_codex_events(trace_path.read_text(encoding="utf-8"))
        commands = extract_commands(events)
        round_record["turn"]["commands"] = commands
        round_record["build_status"] = command_status(commands, "build")
        round_record["test_status"] = command_status(commands, "test")
        all_commands.extend(commands)

    build_commands = [item for item in all_commands if item["category"] == "build"]
    test_commands = [item for item in all_commands if item["category"] == "test"]
    manifest.update(
        {
            "build_status": command_status(all_commands, "build"),
            "test_status": command_status(all_commands, "test"),
            "build_count": len(build_commands),
            "test_count": len(test_commands),
            "build_execution_seconds": measured_seconds(build_commands),
            "test_execution_seconds": measured_seconds(test_commands),
            "build_test_commands": build_commands + test_commands,
            "restricted_accesses": [
                item for item in all_commands if item["restricted_path_reference"]
            ],
            "source_diff": source_diff(
                Path(manifest["source"]),
                run_dir / "workspace",
                run_dir / "source_changes.patch",
            ),
            "derived_fields_recomputed_at": datetime.now(timezone.utc).isoformat(),
            "derived_fields_recomputed_by": str(Path(__file__).resolve()),
        }
    )
    manifest["agent_runtime"]["model_endpoint_sha256"] = provider_endpoint_fingerprint(
        run_dir / "codex_home", manifest["model"]["provider"]
    )
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
