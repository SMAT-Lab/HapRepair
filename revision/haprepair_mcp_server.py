#!/usr/bin/env python3
"""
MCP server exposing HapRepair-related tools for Codex:

- HomeCheck / CodeLinter integration on OpenHarmony projects
- RAG retrieval for ArkTS performance/security rules

This server is intended to be launched by Codex as an MCP server, e.g.:

  "mcpServers": {
    "haprepair": {
      "command": ["python", "revision/haprepair_mcp_server.py"]
    }
  }
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mcp.server.fastmcp import FastMCP


# Ensure we can import project-local modules (get_prompt, fix_projects, get_surrounding_context, etc.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Many helper modules (rules.json, pages/negative, etc.) assume the repo root
# as current working directory, so we normalize CWD here.
os.chdir(str(REPO_ROOT))

from get_prompt import get_rag_prompt  # type: ignore
from get_surrounding_context import (  # type: ignore
    get_single_file_surrounding_context,
    load_rules as load_context_rules,
)


server = FastMCP("haprepair")


# -------------------------------
# Globals for RAG (lazy-loaded)
# -------------------------------

_rag_model = None
_rag_tokenizer = None
_rag_index = None
_rules_cache: Optional[Dict[str, Any]] = None


def _ensure_rag_model() -> None:
    """Lazy-load RAG embedding model and Pinecone index.

    We import load_model_and_index lazily to avoid hard dependency failures
    (e.g., missing pinecone client) during MCP server startup.
    """
    global _rag_model, _rag_tokenizer, _rag_index
    if _rag_model is not None:
        return
    try:
        from fix_projects import load_model_and_index  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "RAG model is not available; failed to import load_model_and_index "
            "from fix_projects. Please ensure the VulRAG environment (with "
            "pinecone and related deps) is active."
        ) from exc
    _rag_model, _rag_tokenizer, _rag_index = load_model_and_index()


# -------------------------------
# Helpers for CodeLinter logs
# -------------------------------

FILE_HEADER_REGEX = re.compile(r"^(\/.+)\(\d+\)$")
VALID_CATEGORIES = {"performance", "security"}
SEVERITY_NORMALIZATION = {
    "warning": "warn",
}

# Project root detection (reused from scripts/run_codelinter_projects.py logic)
HVIGOR_FILES = ("hvigorfile.ts", "hvigorfile.js")


def _parse_codelinter_log(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a CodeLinter log file into a list of findings.

    Each finding is:
    {
      "file_path": str,
      "line": int,
      "column": int,
      "severity": "error" | "warn" | "suggestion",
      "category": "performance" | "security",
      "rule_id": str,
      "message": str,
    }
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    findings: List[Dict[str, Any]] = []
    current_file: Optional[str] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            # File header: "/abs/path/to/file.ets(123)"
            header = FILE_HEADER_REGEX.match(line.strip())
            if header:
                current_file = header.group(1)
                continue

            if not current_file:
                continue

            # Split message and "@category/rule"
            at_index = line.rfind("@")
            if at_index == -1:
                continue

            prefix = line[:at_index].strip()
            meta = line[at_index + 1 :].strip()

            # Example prefix: "12:34  warn  Something happened"
            m = re.match(r"^(\d+):(\d+)\s+(\w+)\s+(.*)$", prefix)
            if not m:
                continue

            line_num_str, col_num_str, severity_raw, message = m.groups()
            # Example meta: "performance/hp-arkui-remove-redundant-nest-container"
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


def _is_project_root(path: Path) -> bool:
    """Return True if the directory looks like a Harmony/ArkTS project root.

    Logic mirrors scripts/run_codelinter_projects.py:
    - hvigorfile.ts/js, or
    - build-profile.json5, or
    - oh-package.json5 present in the directory.
    """
    if any((path / name).is_file() for name in HVIGOR_FILES):
        return True
    if (path / "build-profile.json5").is_file():
        return True
    if (path / "oh-package.json5").is_file():
        return True
    return False


def _find_projects(root: Path, limit: int) -> List[Path]:
    """Recursively find Harmony project roots under a directory."""
    projects: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if _is_project_root(path):
            projects.append(path)
            dirnames[:] = []  # do not descend further under a project root
            if len(projects) >= limit:
                break
    return sorted(projects)


# -------------------------------
# MCP tools
# -------------------------------


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


def _run_codelinter_single(
    target: Path,
    log_dir: Path,
    config: Optional[Path],
) -> int:
    """
    Thin wrapper around scripts/run_codelinter_projects.py:run_codelinter.

    This ensures HapRepair uses the same CodeLinter invocation and log format
    as the standalone batch runner.
    """
    rcp = _load_run_codelinter_module()
    # run_codelinter(target: Path, log_dir: Path, config: Path | None) -> int
    return int(rcp.run_codelinter(target, log_dir, config))


@server.tool()
async def run_codelinter_project(
    project_root: str,
    config_path: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run CodeLinter/HomeCheck on a single OpenHarmony project.

    - project_root: absolute path to the project directory.
    - config_path: optional path to code-linter.json5; if omitted, defaults
      to '/home/LLMCodeRepair/revision/code-linter.json5' if present, or None.
    - log_dir: optional directory for logs; if omitted, defaults to
      '/home/LLMCodeRepair/logs/codelinter'.

    Returns a summary with log_path and exit_code.
    """
    target = Path(project_root).resolve()
    if not target.is_dir():
        raise ValueError(f"Project root is not a directory: {target}")

    # Determine config path
    config: Optional[Path]
    if config_path:
        config = Path(config_path).resolve()
    else:
        default_cfg = Path("/home/LLMCodeRepair/revision/code-linter.json5")
        config = default_cfg if default_cfg.is_file() else None

    # Determine log directory
    if log_dir:
        out_log_dir = Path(log_dir).resolve()
    else:
        out_log_dir = Path("/home/LLMCodeRepair/logs/codelinter")

    exit_code = _run_codelinter_single(target, out_log_dir, config)
    log_path = out_log_dir / f"{target.name}.log"

    return {
        "project_root": str(target),
        "log_path": str(log_path),
        "exit_code": int(exit_code),
        "log_exists": log_path.is_file(),
    }


@server.tool()
async def list_codelinter_findings(
    project_name: str,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List parsed CodeLinter findings for a given project.

    - project_name: typically the folder name; we look for <log_dir>/<project_name>.log.
    - log_dir: directory where CodeLinter logs are stored; default is
      '/home/LLMCodeRepair/logs/codelinter_openharmony'.

    Returns:
    {
      "project_name": ...,
      "log_path": ...,
      "findings": [ {file_path, line, column, severity, category, rule_id, message}, ... ]
    }
    """
    if log_dir:
        base = Path(log_dir).resolve()
    else:
        base = Path("/home/LLMCodeRepair/logs/codelinter_openharmony")

    log_path = base / f"{project_name}.log"
    findings = _parse_codelinter_log(log_path)
    return {
        "project_name": project_name,
        "log_path": str(log_path),
        "findings": findings,
    }


@server.tool()
async def list_openharmony_projects(
    root: str = "/home/LLMCodeRepair/repo_new",
    max_count: int = 50,
) -> Dict[str, Any]:
    """
    Discover OpenHarmony ArkTS project roots under a given directory, using
    the same heuristic as scripts/run_codelinter_projects.py:

    A directory is considered a project root if it contains at least one of:
    - hvigorfile.ts
    - hvigorfile.js
    - build-profile.json5
    - oh-package.json5

    Parameters:
    - root: base directory to search (default: /home/LLMCodeRepair/repo_new).
    - max_count: maximum number of projects to return (default: 50).

    Returns:
    {
      "root": "...",
      "projects": [
        {
          "name": "Gallery",
          "path": "/home/LLMCodeRepair/repo_new/OpenHarmony/app_samples/ETSUI/Gallery"
        },
        ...
      ]
    }
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"root is not a directory: {base}")

    projects = _find_projects(base, max_count)
    return {
        "root": str(base),
        "projects": [
            {"name": p.name, "path": str(p)}
            for p in projects
        ],
    }


@server.tool()
async def rag_retrieve_examples(
    rule_id: str,
    problem_code: str,
    rag_type: str = "gpt_diff",
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve ArkTS repair examples for a given rule using the existing RAG index.

    - rule_id: e.g. '@performance/hp-arkui-remove-redundant-nest-container'
    - problem_code: the code snippet or surrounding context containing the defect
    - rag_type: 'gpt_diff' (default) or 'difflib'
    - top_n: number of demos to retrieve (default 5)

    Returns:
    {
      "rule_id": ...,
      "rag_type": ...,
      "top_n": ...,
      "prompt": "<text to embed into a repair prompt>"
    }
    """
    _ensure_rag_model()

    repair_example = {
        "rule": rule_id,
        "problem_code": problem_code,
    }
    prompt_text = get_rag_prompt(
        repair_example,
        _rag_model,
        _rag_tokenizer,
        _rag_index,
        rag_type,
        number=top_n,
    )
    return {
        "rule_id": rule_id,
        "rag_type": rag_type,
        "top_n": top_n,
        "prompt": prompt_text,
    }


@server.tool()
async def extract_file_context_blocks(
    project_root: str,
    file_path: str,
) -> Dict[str, Any]:
    """
    Extract merged defect blocks and surrounding context for a single file.

    For legacy projects (deepseek_origin-style) we use the original
    get_single_file_surrounding_context logic, which reads CodeLinter
    result*.xlsx files.

    For repo_new-style projects (where HomeCheck/CodeLinter is run via
    scripts/run_codelinter_projects.py and logs are stored under
    /home/LLMCodeRepair/logs/codelinter_openharmony), we instead parse the
    textual log and synthesize surrounding_context directly from source.
    - file_path: absolute or relative path to the source file.

    Returns:
    {
      "project_root": ...,
      "file_path": ...,
      "merged_blocks": [
        {
          "defects": [
            {"rule": "...", "line": 123, "message": "...", "code": "..."},
            ...
          ],
          "block_ranges": [[start_line, end_line], ...],
          "surrounding_context": ["<code block 1>", "<code block 2>", ...]
        },
        ...
      ]
    }

    Note: code_lines are not returned to keep payload size reasonable; the
    agent can read the full file content separately via its filesystem tools.
    """
    proj_dir = Path(project_root).resolve()
    if not proj_dir.is_dir():
        raise ValueError(f"project_root is not a directory: {proj_dir}")

    # Accept either absolute or relative file_path
    file_path_obj = Path(file_path)
    if not file_path_obj.is_absolute():
        file_path_obj = (proj_dir / file_path_obj).resolve()

    if not file_path_obj.is_file():
        raise FileNotFoundError(f"file_path does not exist: {file_path_obj}")

    # Branch 1: legacy path using result*.xlsx (deepseek_origin-style projects).
    has_excel = any(proj_dir.rglob("result*.xlsx"))
    if has_excel:
        rules = load_context_rules()
        merged_blocks, _code_lines = get_single_file_surrounding_context(
            str(proj_dir),
            str(file_path_obj),
            rules,
        )
    else:
        # Branch 2: repo_new projects using CodeLinter logs.
        project_name = proj_dir.name
        log_path = Path("/home/LLMCodeRepair/logs/codelinter_openharmony") / f"{project_name}.log"
        findings = _parse_codelinter_log(log_path)
        abs_file_str = str(file_path_obj.resolve())
        file_findings = [
            f
            for f in findings
            if str(Path(f["file_path"]).resolve()) == abs_file_str
        ]

        code_text = file_path_obj.read_text(encoding="utf-8")
        code_lines = [""] + code_text.splitlines()
        max_idx = len(code_lines) - 1

        raw_blocks: List[Dict[str, Any]] = []
        for f in file_findings:
            line_no = int(f.get("line") or 0)
            if line_no <= 0 or line_no > max_idx:
                continue
            start = max(1, line_no - 5)
            end = min(max_idx, line_no + 5)
            defect = {
                "rule": f"@{f['category']}/{f['rule_id']}",
                "line": line_no,
                "message": f.get("message", ""),
                "code": code_lines[line_no].strip(),
            }
            raw_blocks.append(
                {
                    "range": [start, end],
                    "defects": [defect],
                }
            )

        merged_blocks = []
        if raw_blocks:
            raw_blocks.sort(key=lambda b: b["range"][0])
            current = raw_blocks[0]
            for blk in raw_blocks[1:]:
                cur_start, cur_end = current["range"]
                nxt_start, nxt_end = blk["range"]
                if nxt_start <= cur_end + 1:
                    current["range"][1] = max(cur_end, nxt_end)
                    current["defects"].extend(blk["defects"])
                else:
                    merged_blocks.append(current)
                    current = blk
            merged_blocks.append(current)

            merged_blocks = [
                {
                    "defects": blk["defects"],
                    "block_ranges": [blk["range"]],
                    "surrounding_context": [
                        "\n".join(code_lines[blk["range"][0] : blk["range"][1] + 1]).rstrip()
                    ],
                }
                for blk in merged_blocks
            ]

    return {
        "project_root": str(proj_dir),
        "file_path": str(file_path_obj),
        "merged_blocks": merged_blocks,
    }


if __name__ == "__main__":
    # FastMCP will detect stdio mode when launched as an MCP server.
    # Just call run() to start the event loop.
    server.run()
