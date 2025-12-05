#!/usr/bin/env python3
"""
Lightweight driver to run one HapRepair round on a target project,
selected by index from revision/target_projects_haprepair.json.

Goal: minimal CLI parameters. You typically just pass an index:

  # 默认：第 0 个项目，round=1，模型 gpt-5-mini
  python revision/code/run_haprepair_round.py --index 0

This will:
  1) Load revision/target_projects_haprepair.json
  2) Pick the entry at position --index
  3) Call revision/code/fix_projects_codelinter.py with:
       --project-root <root_path from JSON>
       --round <round>
       --model-name <model>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
REVISION_ROOT = REPO_ROOT / "revision"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run one HapRepair round on a project selected from "
            "revision/target_projects_haprepair.json by index."
        )
    )
    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into target_projects_haprepair.json (default: 0). Ignored if --all is set.",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=1,
        help="Repair round index (default: 1).",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-mini",
        help="LLM model name for repair (default: gpt-5-mini).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="When set, run on all entries in target_projects_haprepair.json sequentially.",
    )
    ap.add_argument(
        "--max-project-workers",
        type=int,
        default=1,
        help=(
            "Maximum number of projects to run in parallel when --all is set "
            "(default: 1, meaning sequential)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    target_path = REVISION_ROOT / "target_projects_haprepair.json"
    if not target_path.is_file():
        raise SystemExit(f"target_projects_haprepair.json not found at {target_path}")

    data = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("target_projects_haprepair.json is empty or malformed (expected a list).")

    indices = range(len(data)) if args.all else [args.index]

    def run_one(idx: int) -> int:
        if idx < 0 or idx >= len(data):
            print(f"[warn] index {idx} out of range (0 <= index < {len(data)}); skipping.", file=sys.stderr)
            return 1

        proj = data[idx]
        name = proj.get("name", f"project_{idx}")
        root_path = proj.get("root_path")
        if not root_path:
            print(f"[warn] Entry {idx} missing 'root_path' field; skipping.", file=sys.stderr)
            return 1

        # Original project root from JSON (round 1 always从这里开始)
        original_root = Path(root_path).resolve()
        if not original_root.is_dir():
            print(f"[warn] Project root is not a directory: {original_root}; skipping.", file=sys.stderr)
            return 1

        # Compute snapshot root for this model/round/project, and skip if it already exists.
        model_dir = (
            args.model_name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        snapshot_root = (
            REVISION_ROOT
            / "fixed_projects"
            / model_dir
            / f"round_{args.round}"
            / original_root.name
        )
        if snapshot_root.exists():
            print(
                f"[info] Skipping index={idx}, name={name}: "
                f"snapshot already exists at {snapshot_root}",
                file=sys.stderr,
            )
            return 0

        # 从第二轮开始，默认基于上一轮的快照继续修复；
        # 如果上一轮快照不存在，则回退到最初的 original_root。
        if args.round <= 1:
            project_root = original_root
        else:
            prev_snapshot_root = (
                REVISION_ROOT
                / "fixed_projects"
                / model_dir
                / f"round_{args.round - 1}"
                / original_root.name
            )
            if prev_snapshot_root.is_dir():
                project_root = prev_snapshot_root
                print(
                    f"[info] Using previous round={args.round - 1} snapshot as input "
                    f"for round={args.round}: {project_root}",
                    file=sys.stderr,
                )
            else:
                project_root = original_root
                print(
                    f"[warn] Previous round={args.round - 1} snapshot not found for "
                    f"index={idx}, name={name}; falling back to original root: {project_root}",
                    file=sys.stderr,
                )

        if not project_root.is_dir():
            print(f"[warn] Selected project_root is not a directory: {project_root}; skipping.", file=sys.stderr)
            return 1

        print(
            f"[info] Selected index={idx}: "
            f"id={proj.get('id')}, name={name}, input_root={project_root}"
        )
        print(
            f"[info] Running round={args.round}, model={args.model_name} "
            f"via fix_projects_codelinter.py"
        )

        script = REVISION_ROOT / "code" / "fix_projects_codelinter.py"
        cmd = [
            sys.executable,
            str(script),
            "--project-root",
            str(project_root),
            "--round",
            str(args.round),
            "--model-name",
            args.model_name,
        ]

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(
                f"[warn] fix_projects_codelinter exited with {proc.returncode} for index={idx}, name={name}",
                file=sys.stderr,
            )
        return proc.returncode

    # When --all and max_project_workers > 1, run projects in parallel.
    if args.all and args.max_project_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_project_workers) as ex:
            future_to_idx = {ex.submit(run_one, idx): idx for idx in indices}
            for fut in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    _ = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"[warn] Unexpected error while running index={idx} in parallel: {exc}",
                        file=sys.stderr,
                    )
    else:
        for idx in indices:
            run_one(idx)


if __name__ == "__main__":
    main()
