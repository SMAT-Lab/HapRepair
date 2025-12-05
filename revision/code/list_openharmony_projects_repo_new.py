#!/usr/bin/env python3
"""
Scan /home/LLMCodeRepair/repo_new (or a custom root) to discover
OpenHarmony/ArkTS project roots, using the same heuristic as
scripts/run_codelinter_projects.py, and save the list to a JSON file.

This is a small utility to prepare a stable project list for downstream
pipelines (e.g., fix_projects_codelinter).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = REPO_ROOT / "revision"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_run_codelinter_module():
    """
    Import scripts/run_codelinter_projects.py as a module without relying on the
    top-level 'scripts' package (which may conflict with site-packages).
    """
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "run_codelinter_projects.py"
    if not script_path.is_file():
        raise RuntimeError(f"run_codelinter_projects.py not found at {script_path}")

    spec = importlib.util.spec_from_file_location(
        "run_codelinter_projects_local",
        str(script_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "List OpenHarmony/ArkTS project roots under a given directory, "
            "using scripts/run_codelinter_projects.py::find_projects, and "
            "save the list to a JSON file."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/LLMCodeRepair/repo_new"),
        help="Root directory under which to search for Harmony projects "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REVISION_ROOT / "projects_repo_new_openharmony.json",
        help="Path to write JSON list of discovered projects "
        "(default: %(default)s).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")

    rcp = _load_run_codelinter_module()
    projects = rcp.find_projects(root, None)  # type: ignore[attr-defined]

    entries = [
        {
            "name": p.name,
            "path": str(p.resolve()),
        }
        for p in projects
    ]

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Discovered {len(entries)} project(s) under {root}")
    print(f"Project list written to: {out_path}")


if __name__ == "__main__":
    main()
