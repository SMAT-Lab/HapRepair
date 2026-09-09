#!/usr/bin/env python3
"""Evaluator-side public API compatibility guard for EXP-AGENT-10."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from validation_gate import sha256_file, write_json


EXTRACTOR = Path(__file__).with_name("extract_public_api.js")


def extract_public_api(workspace: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["node", str(EXTRACTOR), str(workspace.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Public API extraction failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def prepare_public_api_guard(workspace: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = output_dir / "baseline.json"
    write_json(baseline_path, extract_public_api(workspace))
    setup = {
        "schema_version": 1,
        "extractor": str(EXTRACTOR),
        "extractor_sha256": sha256_file(EXTRACTOR),
        "baseline": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
    }
    write_json(output_dir / "setup.json", setup)
    return setup


def run_public_api_guard(
    workspace: Path, output_dir: Path, setup: dict[str, Any], *, label: str
) -> dict[str, Any]:
    baseline = json.loads(Path(setup["baseline"]).read_text(encoding="utf-8"))
    observed = extract_public_api(workspace)
    baseline_api = baseline["api"]
    observed_api = observed["api"]
    changes = {
        "removed": sorted(set(baseline_api) - set(observed_api)),
        "added": sorted(set(observed_api) - set(baseline_api)),
        "changed": sorted(
            key
            for key in set(baseline_api) & set(observed_api)
            if baseline_api[key] != observed_api[key]
        ),
    }
    observed_path = output_dir / f"{label}_observed.json"
    write_json(observed_path, observed)
    record = {
        "label": label,
        "status": "passed" if not any(changes.values()) else "failed",
        "changes": changes,
        "observed": str(observed_path),
        "observed_sha256": sha256_file(observed_path),
    }
    write_json(output_dir / f"{label}.json", record)
    return record
