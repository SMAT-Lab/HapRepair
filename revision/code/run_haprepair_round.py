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
import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
REVISION_ROOT = REPO_ROOT / "revision"
INPUTS_ROOT = REPO_ROOT / "repo_new" / "_haprepair_inputs"
DEFAULT_CODELINTER_CONFIG = REPO_ROOT / "revision" / "code-linter.json5"
DEFAULT_CODELINTER_LOG_ROOT = REPO_ROOT / "logs" / "codelinter_openharmony"


def _sanitize_model_dir(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _make_model_dir(model_name: str, run_tag: Optional[str]) -> str:
    base = _sanitize_model_dir(model_name)
    if not run_tag:
        return base
    tag = _sanitize_model_dir(run_tag)
    return f"{base}__{tag}"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw not in ("0", "false", "no", "off", "")


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


def _run_codelinter_snapshot(
    *,
    project_root: Path,
    log_dir: Path,
    config: Optional[Path],
) -> int:
    """
    Run CodeLinter on an existing snapshot and write its log under log_dir.
    Returns the CodeLinter exit code.
    """
    rcp = _load_run_codelinter_module()
    log_dir.mkdir(parents=True, exist_ok=True)
    # run_codelinter(target: Path, log_dir: Path, config: Path | None) -> int
    return int(rcp.run_codelinter(project_root, log_dir, config))  # type: ignore[attr-defined]


def _has_codelinter_project_markers(project_root: Path) -> bool:
    return (project_root / "build-profile.json5").is_file() or (project_root / "oh-package.json5").is_file()


def _hydrate_snapshot_with_project_metadata(
    *,
    snapshot_root: Path,
    original_root: Path,
    round_idx: int,
    model_dir: str,
) -> Path:
    """
    Some saved snapshots may omit non-.ts/.ets project files (e.g. module.json5 / resources / build-profile.json5),
    which causes CodeLinter to reject the inspection path in the next round. To make the next round stable, we:

      1) copy the full original project tree
      2) overlay the previous-round snapshot files on top (so repaired .ts/.ets remain)
    """
    hydrated_root = (INPUTS_ROOT / model_dir / f"round_{round_idx}" / snapshot_root.name).resolve()
    hydrated_root.parent.mkdir(parents=True, exist_ok=True)
    if hydrated_root.exists():
        shutil.rmtree(hydrated_root)
    shutil.copytree(original_root, hydrated_root)

    # Overlay snapshot contents onto the hydrated tree (do not delete original files).
    for src in snapshot_root.rglob("*"):
        rel = src.relative_to(snapshot_root)
        dst = hydrated_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return hydrated_root


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
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional run tag appended to output/log directories "
            "(keeps model-name for LLM calls)."
        ),
    )
    ap.add_argument(
        "--rag-type",
        type=str,
        default="difflib",
        help="RAG retriever type passed to get_rag_prompt (default: %(default)s).",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of RAG examples per rule (default: %(default)s).",
    )
    ap.add_argument(
        "--surrounding-context",
        dest="surrounding_context",
        action="store_true",
        default=True,
        help="Use surrounding context blocks (default: true).",
    )
    ap.add_argument(
        "--full-context",
        dest="surrounding_context",
        action="store_false",
        help="Use full file as context (disable surrounding blocks).",
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
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run repair even if the snapshot output directory already exists.",
    )
    ap.add_argument(
        "--fill-missing-after-logs",
        action="store_true",
        help=(
            "If a snapshot already exists but the CodeLinter after_round log "
            "is missing, run CodeLinter on the snapshot to backfill the log "
            "(no LLM repair)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Normalize model ids like 'qwen3=30b-a3b' -> 'qwen3-30b-a3b' to match
    # OpenAI-compatible /models listings used by some providers.
    if "=" in args.model_name:
        args.model_name = args.model_name.replace("=", "-")

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
        model_dir = _make_model_dir(args.model_name, args.run_tag)
        snapshot_root = (
            REVISION_ROOT
            / "fixed_projects"
            / model_dir
            / f"round_{args.round}"
            / original_root.name
        )
        force = args.force or _env_flag("HAPREPAIR_FORCE", default=False)

        # When snapshots already exist, optionally backfill missing after_round logs
        # so downstream summaries (e.g., *_remaining_defects.md) are complete.
        after_log_dir = (
            DEFAULT_CODELINTER_LOG_ROOT
            / model_dir
            / f"round_{args.round}_after_round{args.round}"
        )
        after_log_path = after_log_dir / f"{original_root.name}.log"
        fill_missing_after_logs = args.fill_missing_after_logs or _env_flag(
            "HAPREPAIR_FILL_MISSING_AFTER_LOGS", default=False
        )

        if snapshot_root.exists() and not force:
            if fill_missing_after_logs and not after_log_path.is_file():
                cfg = DEFAULT_CODELINTER_CONFIG if DEFAULT_CODELINTER_CONFIG.is_file() else None
                print(
                    f"[info] Snapshot exists but after_round log is missing; "
                    f"backfilling CodeLinter log for {original_root.name} -> {after_log_path}",
                    file=sys.stderr,
                )
                try:
                    project_root_for_backfill = snapshot_root
                    if (
                        not _has_codelinter_project_markers(snapshot_root)
                        and _has_codelinter_project_markers(original_root)
                    ):
                        hydrated = _hydrate_snapshot_with_project_metadata(
                            snapshot_root=snapshot_root,
                            original_root=original_root,
                            round_idx=args.round,
                            model_dir=model_dir,
                        )
                        print(
                            f"[warn] Snapshot missing Harmony project metadata; "
                            f"using hydrated copy for backfill: {hydrated}",
                            file=sys.stderr,
                        )
                        project_root_for_backfill = hydrated
                    return _run_codelinter_snapshot(
                        project_root=project_root_for_backfill,
                        log_dir=after_log_dir,
                        config=cfg,
                    )
                except Exception as exc:
                    # Best-effort fallback: if we have the round_<N> log, copy it.
                    before_log = (
                        DEFAULT_CODELINTER_LOG_ROOT
                        / model_dir
                        / f"round_{args.round}"
                        / f"{original_root.name}.log"
                    )
                    if before_log.is_file():
                        after_log_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(before_log, after_log_path)
                        print(
                            f"[warn] Failed to backfill via CodeLinter ({exc}); "
                            f"copied before-round log to {after_log_path}",
                            file=sys.stderr,
                        )
                        return 0
                    print(
                        f"[warn] Failed to backfill after_round log for {original_root.name}: {exc}",
                        file=sys.stderr,
                    )
                    return 1

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

                # If the previous snapshot is missing CodeLinter-required project markers,
                # but the original project has them, hydrate the snapshot before using it.
                if not _has_codelinter_project_markers(project_root) and _has_codelinter_project_markers(original_root):
                    hydrated = _hydrate_snapshot_with_project_metadata(
                        snapshot_root=project_root,
                        original_root=original_root,
                        round_idx=args.round,
                        model_dir=model_dir,
                    )
                    print(
                        f"[warn] Previous snapshot is missing Harmony project metadata "
                        f"(e.g. build-profile.json5/oh-package.json5). "
                        f"Using hydrated copy for round={args.round}: {hydrated}",
                        file=sys.stderr,
                    )
                    project_root = hydrated
                else:
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
            "--rag-type",
            args.rag_type,
            "--top-n",
            str(args.top_n),
        ]
        if args.run_tag:
            cmd.extend(["--run-tag", args.run_tag])
        if not args.surrounding_context:
            cmd.append("--full-context")

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
