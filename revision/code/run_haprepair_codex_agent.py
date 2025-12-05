#!/usr/bin/env python3
"""
Small driver script to invoke a Codex agent (with the `haprepair` MCP server)
on a real OpenHarmony project under /home/LLMCodeRepair/repo_new.

This script does NOT implement the repair logic itself. Instead, it:
  1) Uses the same project-root detection logic as scripts/run_codelinter_projects.py
     to pick a project under repo_new.
  2) Constructs a high-level instruction prompt for the Codex agent, telling it
     how to use the `haprepair` MCP tools (HomeCheck findings + context + RAG)
     to attempt repairs.
  3) Calls revision/agent/run_codex.py to actually talk to Codex.

You can adjust the project selection with --project-name or --root.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revision.agent.run_codex import run_codex  # type: ignore

# Base directory for this revision package (where we want to store traces
# and per-round workspaces)
REVISION_ROOT = Path(__file__).resolve().parent.parent

# Where to store per-round CodeLinter logs for HapRepair runs
HAPREPAIR_LOG_ROOT = REPO_ROOT / "logs" / "codelinter_openharmony_haprepair"


def _load_run_codelinter_module():
    """
    Import scripts/run_codelinter_projects.py as a module without relying on the
    top-level 'scripts' package (which may conflict with site-packages).
    """
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "run_codelinter_projects.py"
    if not script_path.is_file():
        raise SystemExit(f"run_codelinter_projects.py not found at {script_path}")

    spec = importlib.util.spec_from_file_location(
        "run_codelinter_projects_local", str(script_path)
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def pick_project(root: Path, project_name: str | None) -> Path:
    """Pick a project root under `root`, using run_codelinter_projects logic."""
    rcp = _load_run_codelinter_module()
    projects: List[Path] = rcp.find_projects(root, None)
    if not projects:
        raise SystemExit(f"No Harmony projects found under {root}")

    if project_name:
        for p in projects:
            if p.name == project_name:
                return p
        # If not found, fall back to first and warn
        print(
            f"[warn] Requested project_name={project_name!r} not found under {root}. "
            f"Using {projects[0]} instead.",
            file=sys.stderr,
        )
    return projects[0]


# --------------
# Log parsing
# --------------

FILE_HEADER_REGEX = re.compile(r"^(\/.+)\(\d+\)$")
SEVERITY_NORMALIZATION = {"warning": "warn"}
VALID_CATEGORIES = {"performance", "security"}


def parse_codelinter_log(log_path: Path) -> List[dict]:
    """Parse a CodeLinter log into a list of findings."""
    if not log_path.is_file():
        return []

    findings: List[dict] = []
    current_file: str | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            header = FILE_HEADER_REGEX.match(line.strip())
            if header:
                current_file = header.group(1)
                continue

            if not current_file:
                continue

            at_index = line.rfind("@")
            if at_index == -1:
                continue

            prefix = line[:at_index].strip()
            meta = line[at_index + 1 :].strip()

            m = re.match(r"^(\d+):(\d+)\s+(\w+)\s+(.*)$", prefix)
            if not m:
                continue
            line_num_str, col_num_str, severity_raw, message = m.groups()

            parts = meta.split("/", 1)
            if len(parts) != 2:
                continue
            category_raw, rule_raw = parts
            category = category_raw.strip().lower()
            if category not in VALID_CATEGORIES:
                continue

            sev_norm = SEVERITY_NORMALIZATION.get(severity_raw.lower(), severity_raw.lower())
            if sev_norm not in ("error", "warn", "suggestion"):
                continue

            findings.append(
                {
                    "file_path": current_file,
                    "line": int(line_num_str),
                    "column": int(col_num_str),
                    "severity": sev_norm,
                    "category": category,
                    "rule_id": rule_raw.strip(),
                    "message": message.strip(),
                }
            )

    return findings


def build_prompt(project_path: Path, log_dir: Path, max_rounds: int) -> str:
    """Construct a high-level instruction prompt for the Codex agent.

    Note: the caller should have already set the current working directory
    to `project_path` before invoking Codex. We mention this explicitly in
    the instructions so the agent can rely on relative paths.
    """
    proj_str = str(project_path)
    project_name = project_path.name
    log_path = log_dir / f"{project_name}.log"

    prompt = f"""
You are HapRepair, an ArkTS performance/security defect repair agent working on
real OpenHarmony projects.

Environment:
- The current working directory is the root of an OpenHarmony/ArkTS project:
  {proj_str}
- All relative paths you use should be interpreted from this directory.
- CodeLinter/HomeCheck has already been run on this project. Its textual log
  is stored under:
  {log_path}
- You have access to an MCP server named `haprepair` with the following tools:
  1) haprepair.list_codelinter_findings(project_name, log_dir?)
     - Parse the CodeLinter/HomeCheck log into structured findings with:
       file_path, line, column, severity, category, rule_id, message.
  2) haprepair.extract_file_context_blocks(project_root, file_path)
     - For a given ArkTS file, group related defects and return merged blocks:
       {{
         "defects": [{{rule, line, message, code}}, ...],
         "block_ranges": [[start_line, end_line], ...],
         "surrounding_context": ["code block 1", "code block 2", ...]
       }}
  3) haprepair.rag_retrieve_examples(rule_id, problem_code, rag_type, top_n)
     - Retrieve ArkTS repair examples (demo snippets) for a given rule, to be
       embedded into your repair prompt.

Your task (multi-round repair inside a single Codex run):

1) You operate on the **workspace copy** of the project (current directory),
   never on the original repo_new project. Treat this directory as mutable.

2) You may perform up to **{max_rounds} repair rounds**. Each round r works as:
   a) Call haprepair.run_codelinter_project with:
      - project_root = "{proj_str}"
      - log_dir      = "{log_dir}"
      to run HomeCheck/CodeLinter on the current workspace.
   b) Call haprepair.list_codelinter_findings with:
      - project_name = "{project_name}"
      - log_dir      = "{log_dir}"
      to obtain the current list of findings.
   c) If there are **no findings** remaining, STOP all further rounds.
   d) Otherwise, choose a manageable subset of files to repair in this round
      (prefer multi-defect files or important rules).

3) For each chosen file in the current round:
   a) Use haprepair.extract_file_context_blocks with:
      - project_root = "{proj_str}"
      - file_path    = the absolute file_path from the findings
      to obtain merged defect blocks and their surrounding_context.
   b) For each merged block:
      - Inspect the list of defects (rule_id + code, line, message).
      - Call haprepair.rag_retrieve_examples(rule_id, problem_code=context_snippet)
        to retrieve repair demos for that rule.
      - Read the original file content using your filesystem tools.
      - Construct a repair prompt that includes:
        * The surrounding_context from the merged block;
        * The HomeCheck findings (rule_id + message + line);
        * The RAG demos from haprepair.rag_retrieve_examples;
        * Clear instructions to produce a unified diff or the full fixed file.
      - Generate a patch (or full file) that fixes ALL defects in that block.
      - Apply the patch to the source file using your filesystem tools.

4) After you finish applying fixes for the current round, you MUST go back to
   step 2(a): re-run haprepair.run_codelinter_project on the workspace and
   re-evaluate the remaining findings. Repeat until either:
   - There are no findings left, or
   - You have completed {max_rounds} rounds.

5) At the end of your work (either because all defects are fixed or the maximum
   number of rounds is reached), produce a concise natural-language summary:
   - Which files you attempted to repair;
   - For each file: which rule_ids were targeted and how they were fixed;
   - Any blocks you chose to skip (and why).

Important constraints:
- Do NOT try to run external shell commands yourself; use only the MCP tools
  and filesystem tools available via Codex.
- When editing code, preserve the existing structure and formatting as much as
  possible, and focus strictly on fixing the reported defects.
"""
    return prompt.strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a HapRepair Codex agent on a repo_new OpenHarmony project."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/LLMCodeRepair/repo_new"),
        help="Root directory under which to search for Harmony projects "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project folder name to target (e.g., 'Gallery'). "
        "If omitted, the first discovered project is used.",
    )
    ap.add_argument(
        "--codex-arg",
        action="append",
        default=None,
        help="Extra argument(s) to pass through to `codex exec` "
        "(e.g., --codex-arg=--profile --codex-arg=haprepair).",
    )
    ap.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help=(
            "Optional path to write a JSON snapshot of the Codex run, updated "
            "incrementally while the agent is running (similar to piping "
            "`codex exec --json` through `tee`)."
        ),
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of repair rounds to run (default: %(default)s).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    project = pick_project(root, args.project_name)
    print(f"[info] Using project: {project}")

    # Decide trace output path BEFORE changing CWD, so that:
    # - default: revision/trace/<project_name>.json
    # - explicit --trace-out: respected as given, resolved relative to revision root.
    if args.trace_out is not None:
        trace_out = (REVISION_ROOT / args.trace_out).resolve()
    else:
        trace_dir = REVISION_ROOT / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_out = trace_dir / f"{project.name}.json"

    # Ensure parent dir exists (for explicit path case)
    trace_out.parent.mkdir(parents=True, exist_ok=True)

    extra_args = args.codex_arg or []

    # Single-copy workspace: we never modify the original repo_new project
    # directly. Instead, we copy it once to:
    #   revision/workspaces/<project_name>/<project_name>
    # and run all Codex multi-round logic inside that workspace.
    workspace_root = REVISION_ROOT / "workspaces" / project.name
    workspace_project = workspace_root / project.name

    if workspace_project.exists():
        print(f"[info] Reusing existing workspace: {workspace_project}")
    else:
        print(f"[info] Creating workspace from {project} -> {workspace_project}")
        workspace_root.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(project, workspace_project, dirs_exist_ok=True)

    # Prepare a dedicated log directory for this project; the agent will use
    # haprepair.run_codelinter_project with this log_dir for all internal rounds.
    project_log_dir = HAPREPAIR_LOG_ROOT / project.name
    project_log_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory to the workspace project root so that Codex
    # and its filesystem tools operate relative to this copy.
    os.chdir(str(workspace_project))

    prompt = build_prompt(workspace_project, project_log_dir, args.max_rounds)

    try:
        result = run_codex(
            prompt,
            extra_args=extra_args,
            output_path=str(trace_out),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    output = result.get("final_answer") or result.get("raw_output") or ""
    print("\n=== Codex agent output ===\n")
    sys.stdout.write(output)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
