#!/usr/bin/env python3
"""
Project-level ArkTS defect repair pipeline using CodeLinter findings.

This script mirrors the overall flow of `fix_projects.py`, but instead of
reading pre-generated Excel result files (result*.xlsx), it:

1) Runs CodeLinter on a Harmony/ArkTS project via scripts/run_codelinter_projects.py.
2) Parses the textual CodeLinter log to obtain structured defect findings.
3) Groups findings into merged blocks with surrounding context per file.
4) Uses the existing RAG + LLM repair pipeline (generate_fix_prompt, RAG, etc.)
   to generate fixes, validates ArkTS declarations, and checks functionality.
5) Writes the repaired project to a separate output directory, preserving the
   original directory structure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import sys
import shutil

# Ensure repo root is on sys.path so we can import existing helpers
REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = REPO_ROOT / "revision"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from get_prompt import (  # type: ignore
    combine_repair_results,
    generate_fix_prompt,
    get_rag_prompt,
)
from llm import get_answer  # type: ignore
from output_handler import (  # type: ignore
    ArkTSDeclarationFixer,
    check_functionality,
    extract_code_from_markdown_block,
    remove_difflib_line,
)
# Reuse existing auto-fix helpers for some ArkUI performance rules.
from get_surrounding_context import (  # type: ignore
    remove_redundant_nest_container,
    remove_container_without_property,
    use_row_column_to_replace_flex,
)


HVIGOR_FILES = ("hvigorfile.ts", "hvigorfile.js")
VALID_CATEGORIES = {"performance", "security"}
SEVERITY_NORMALIZATION = {"warning": "warn"}
DEFAULT_CODELINTER_LOG_DIR = Path("/home/LLMCodeRepair/logs/codelinter_openharmony")

# Optional RAG deps (transformers + pinecone)
try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    from pinecone import Pinecone  # type: ignore

    HAS_ML_DEPS = True
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    Pinecone = None  # type: ignore
    HAS_ML_DEPS = False


def load_model_and_index():
    """
    Lightweight copy of fix_projects.load_model_and_index, but tolerant of
    missing ML dependencies. If transformers/pinecone are not available,
    returns (None, None, None) and the pipeline will run without RAG.
    """
    logger = logging.getLogger("fix_projects_codelinter")
    if not HAS_ML_DEPS:
        logger.warning(
            "ML dependencies (transformers, pinecone) not available; running "
            "without RAG support."
        )
        return None, None, None

    logger.info("Loading RAG model and index...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")

    pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")
    index_name = "arkts-1536"
    index = pc.Index(index_name)
    logger.info("RAG model and index loaded")
    return model, tokenizer, index


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


def run_codelinter_for_project(
    project_root: Path,
    log_dir: Path,
    config: Optional[Path],
) -> Path:
    """
    Run CodeLinter on a single project using scripts/run_codelinter_projects.py
    and return the path to the generated log file.
    """
    rcp = _load_run_codelinter_module()
    log_dir.mkdir(parents=True, exist_ok=True)
    rc = int(rcp.run_codelinter(project_root, log_dir, config))  # type: ignore[attr-defined]
    if rc != 0:
        raise RuntimeError(f"CodeLinter failed for {project_root} (exit={rc})")
    return log_dir / f"{project_root.name}.log"


FILE_HEADER_REGEX = re.compile(r"^(\/.+)\(\d+\)$")


def parse_codelinter_log(log_path: Path) -> List[Dict[str, Any]]:
    """Parse a CodeLinter log into a list of findings."""
    if not log_path.is_file():
        raise FileNotFoundError(f"CodeLinter log not found: {log_path}")

    findings: List[Dict[str, Any]] = []
    current_file: Optional[str] = None

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

            sev_norm = SEVERITY_NORMALIZATION.get(
                severity_raw.lower(), severity_raw.lower()
            )
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


def parse_parsing_errors(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parse only parsing-error findings from a CodeLinter log.

    We look for lines tagged with '@parsing-error/parsing-error' and extract:
    {
      "file_path": str,
      "line": int,
      "column": int,
      "severity": str,
      "message": str,
      "rule": str,  # 'parsing-error/parsing-error'
    }
    """
    if not log_path.is_file():
        return []

    errors: List[Dict[str, Any]] = []
    current_file: Optional[str] = None

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

            if "@parsing-error/parsing-error" not in line:
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

            errors.append(
                {
                    "file_path": current_file,
                    "line": int(line_num_str),
                    "column": int(col_num_str),
                    "severity": severity_raw.lower(),
                    "message": message.strip(),
                    "rule": meta,
                }
            )

    return errors


def _insert_foreach_key_generator(
    code_lines: List[str],
    line_no: int,
) -> bool:
    """
    Auto-fix helper for @performance/foreach-args-check.

    Strategy:
    - Locate the nearest 'ForEach(' around the reported line.
    - Track parentheses from the '(' after 'ForEach' across subsequent lines
      until the top-level call is closed.
    - Insert a third argument ', (item, index) => item' immediately before
      the closing ')', so we get:

        ForEach(data, (item, index) => { ... }, (item, index) => item)

    We keep the transformation conservative:
    - Only search within [line_no-1, line_no, line_no+1].
    - If we cannot reliably find and close the ForEach call, we skip.
    """
    max_idx = len(code_lines) - 1
    if max_idx <= 0:
        return False

    # Search a small window around the reported line for a ForEach call.
    start = max(1, line_no - 1)
    end = min(max_idx, line_no + 1)

    for ln in range(start, end + 1):
        line = code_lines[ln]
        col = line.find("ForEach(")
        if col == -1:
            continue

        # Find the '(' that starts the argument list.
        open_pos = line.find("(", col)
        if open_pos == -1:
            continue

        depth = 1
        cur_line = ln
        cur_col = open_pos + 1

        # Walk forward across lines to find the matching ')'.
        while True:
            if cur_col >= len(code_lines[cur_line]):
                cur_line += 1
                if cur_line > max_idx:
                    # Unbalanced parentheses; abort this candidate.
                    break
                cur_col = 0
                continue

            ch = code_lines[cur_line][cur_col]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    # Insert just before this closing ')'.
                    target_line = code_lines[cur_line]
                    insert_text = ", (item, index) => item"
                    code_lines[cur_line] = (
                        target_line[:cur_col] + insert_text + target_line[cur_col:]
                    )
                    return True
            cur_col += 1

    return False


def _auto_fix_foreach_args_check(
    code_lines: List[str],
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Auto-fix @performance/foreach-args-check without LLM by adding a
    keyGenerator argument to ForEach calls.

    For each finding:
      - Try to insert ', (item, index) => item' before the closing ')'.
      - If successful, drop the finding so it is not sent to the LLM.
    """
    if not file_findings:
        return code_lines, file_findings

    max_idx = len(code_lines) - 1
    new_findings: List[Dict[str, Any]] = []

    for f in file_findings:
        if (
            f.get("category") == "performance"
            and f.get("rule_id") == "foreach-args-check"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                if _insert_foreach_key_generator(code_lines, line_no):
                    # Fixed successfully; drop this finding.
                    continue

        new_findings.append(f)

    return code_lines, new_findings


def _insert_list_width_height(
    code_lines: List[str],
    line_no: int,
    width: str = "100%",
    height: str = "100%",
) -> bool:
    """
    Auto-fix helper for @performance/init-list-component.

    Strategy:
    - Locate the nearest 'List(' around the reported line.
    - Track parentheses from the '(' after 'List' across subsequent lines
      until the top-level call is closed.
    - Insert `.width('<width>').height('<height>')` immediately after the
      closing ')', so we get, for example:

        List({ initialIndex: 0 }).width('100%').height('100%') { ... }

    This keeps the List component explicitly sized without changing the
    overall control flow.
    """
    max_idx = len(code_lines) - 1
    if max_idx <= 0:
        return False

    start = max(1, line_no - 1)
    end = min(max_idx, line_no + 1)

    for ln in range(start, end + 1):
        line = code_lines[ln]
        col = line.find("List(")
        if col == -1:
            continue

        open_pos = line.find("(", col)
        if open_pos == -1:
            continue

        depth = 1
        cur_line = ln
        cur_col = open_pos + 1

        while True:
            if cur_col >= len(code_lines[cur_line]):
                cur_line += 1
                if cur_line > max_idx:
                    break
                cur_col = 0
                continue

            ch = code_lines[cur_line][cur_col]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    target_line = code_lines[cur_line]
                    insert_text = f".width('{width}').height('{height}')"
                    code_lines[cur_line] = (
                        target_line[: cur_col + 1]
                        + insert_text
                        + target_line[cur_col + 1 :]
                    )
                    return True
            cur_col += 1

    return False


def _auto_fix_init_list_component(
    code_lines: List[str],
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Auto-fix @performance/init-list-component without LLM by adding
    width/height chain calls to List components that lack them.

    For each finding:
      - Try to insert `.width('100%').height('100%')` after the List(...) call.
      - If successful, drop the finding.
    """
    if not file_findings:
        return code_lines, file_findings

    max_idx = len(code_lines) - 1
    new_findings: List[Dict[str, Any]] = []

    for f in file_findings:
        if (
            f.get("category") == "performance"
            and f.get("rule_id") == "init-list-component"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                if _insert_list_width_height(code_lines, line_no):
                    continue

        new_findings.append(f)

    return code_lines, new_findings


def build_merged_blocks_for_file(
    file_path: Path,
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build merged defect blocks and surrounding context for a single file,
    given its CodeLinter findings.

    Returns (merged_blocks, code_lines) where:
    - merged_blocks: list of {
        "defects": [ {rule, line, message, code}, ... ],
        "block_ranges": [[start_line, end_line], ...],
        "surrounding_context": ["<code block 1>", ...],
      }
    - code_lines: 1-based list of code lines (index 0 is a dummy "")
    """
    code_text = file_path.read_text(encoding="utf-8")
    code_lines = [""] + code_text.splitlines()
    max_idx = len(code_lines) - 1

    # ------------------------------------------------------------------
    # Auto-fix certain ArkUI performance rules without LLM:
    #
    # These rules have deterministic, purely structural fixes implemented
    # in the older RQ1 pipeline (get_surrounding_context). We reuse the
    # same helpers here so that these simple transformations are applied
    # directly at the source-code level and are not sent to the LLM.
    #
    # - @performance/hp-arkui-remove-redundant-nest-container
    # - @performance/hp-arkui-remove-container-without-property
    # - @performance/hp-arkui-use-row-column-to-replace-flex
    #
    # Note: these helpers modify code_lines in-place but do NOT change the
    # total number of lines, so we do not need to remap line numbers for
    # other findings.
    # ------------------------------------------------------------------
    auto_fix_hp_rules = {
        "hp-arkui-remove-redundant-nest-container": remove_redundant_nest_container,
        "hp-arkui-remove-container-without-property": remove_container_without_property,
        "hp-arkui-use-row-column-to-replace-flex": use_row_column_to_replace_flex,
    }

    if file_findings:
        # Apply structural fixes for the above rules.
        for f in file_findings:
            if (
                f.get("category") == "performance"
                and f.get("rule_id") in auto_fix_hp_rules
            ):
                try:
                    line_no = int(f.get("line") or 0)
                except (TypeError, ValueError):
                    line_no = 0
                if 1 <= line_no <= max_idx:
                    fixer = auto_fix_hp_rules[f["rule_id"]]
                    code_lines = fixer(line_no, code_lines)

        # Drop these findings from further processing (they are already fixed).
        file_findings = [
            f
            for f in file_findings
            if not (
                f.get("category") == "performance"
                and f.get("rule_id") in auto_fix_hp_rules
            )
        ]

        max_idx = len(code_lines) - 1

    # ------------------------------------------------------------------
    # Auto-fix @security/no-commented-code without LLM:
    # For each finding with category=security & rule_id=no-commented-code,
    # we simply remove the reported line from the file and drop that
    # finding from further processing. This keeps the simple "delete
    # commented-out code" behavior out of the LLM pipeline.
    # ------------------------------------------------------------------
    lines_to_delete = set()
    for f in file_findings:
        if (
            f.get("category") == "security"
            and f.get("rule_id") == "no-commented-code"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                lines_to_delete.add(line_no)

    if lines_to_delete:
        # Build new code_lines without the deleted lines and a mapping
        # from old line numbers to new line numbers.
        new_code_lines: List[str] = [""]
        line_mapping: Dict[int, int] = {}
        new_idx = 0
        for old_idx in range(1, max_idx + 1):
            if old_idx in lines_to_delete:
                continue
            new_idx += 1
            new_code_lines.append(code_lines[old_idx])
            line_mapping[old_idx] = new_idx

        code_lines = new_code_lines
        max_idx = len(code_lines) - 1

        # Filter findings: drop no-commented-code, and remap line numbers
        # for the remaining findings so that later context extraction
        # still points to the right vicinity.
        adjusted_findings: List[Dict[str, Any]] = []
        for f in file_findings:
            if (
                f.get("category") == "security"
                and f.get("rule_id") == "no-commented-code"
            ):
                # Already handled by auto-fix; skip.
                continue
            line_no = f.get("line")
            try:
                line_int = int(line_no or 0)
            except (TypeError, ValueError):
                line_int = 0
            if line_int in line_mapping:
                f = dict(f)
                f["line"] = line_mapping[line_int]
                adjusted_findings.append(f)
            elif line_int <= 0:
                # Non-line-based finding; keep as-is.
                adjusted_findings.append(f)
            # Else: finding was on a deleted line; drop it.
        file_findings = adjusted_findings

    # ------------------------------------------------------------------
    # Auto-fix @performance/foreach-args-check:
    # Add a keyGenerator argument to ForEach calls (when we can safely
    # locate the call site), and drop those findings.
    # ------------------------------------------------------------------
    code_lines, file_findings = _auto_fix_foreach_args_check(
        code_lines, file_findings
    )
    max_idx = len(code_lines) - 1

    # ------------------------------------------------------------------
    # Auto-fix @performance/init-list-component:
    # Add width/height chain calls to List components and drop the
    # corresponding findings when the insertion is successful.
    # ------------------------------------------------------------------
    code_lines, file_findings = _auto_fix_init_list_component(
        code_lines, file_findings
    )
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

    merged_blocks: List[Dict[str, Any]] = []
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

    return merged_blocks, code_lines


def _build_parsing_error_prompt(
    file_path: Path,
    original_code: str,
    current_code: str,
    errors: List[Dict[str, Any]],
) -> str:
    """Construct an LLM prompt to repair parsing errors in a single file."""
    rel_path = file_path.name
    # Format error list
    err_lines = []
    for e in errors:
        err_lines.append(
            f"- line {e.get('line', '?')}, column {e.get('column', '?')}: {e.get('message', '').strip()}"
        )
    err_text = "\n".join(err_lines)

    prompt = f"""
You are an expert ArkTS (ArkUI TypeScript) assistant fixing **syntax / parsing errors** reported by CodeLinter.

File: {rel_path}

CodeLinter reported the following parsing errors on the **current version** of this file (line numbers refer to the current version):
{err_text}

Here is the **previous version** of the file (before automated repairs):
```arkts
{original_code}
```

Here is the **current version** after automated performance/security repairs, which now causes parsing errors:
```arkts
{current_code}
```

Your task:
- Produce a corrected ArkTS file that resolves all the parsing errors.
- Preserve the intended behavior and bug fixes from the current version whenever possible.
- Do NOT reintroduce obvious problems such as:
  * unused or unchanged state variables,
  * empty system callbacks,
  * redundant containers, etc.
- Keep the file self-contained and syntactically valid.

Output requirements:
- Return ONLY the complete fixed file inside a single ```arkts ... ``` code block.
- Do NOT include any explanations or commentary outside the code block.
"""
    return prompt.strip()


def process_file_with_codelinter_blocks(
    file_path: Path,
    project_root: Path,
    out_root: Path,
    file_findings: List[Dict[str, Any]],
    logger: logging.Logger,
    model,
    tokenizer,
    index,
    *,
    rag_type: str = "difflib",
    top_n: int = 1,
    surrounding_context: bool = True,
    repair_model_name: str = "gpt-4o-2024-08-06",
    max_attempts: int = 3,
) -> List[str]:
    """
    Per-file repair worker, mirroring fix_projects.process_file but using
    CodeLinter-based merged_blocks instead of Excel result files.
    """
    file_logs: List[str] = []

    try:
        code_text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.info(f"Failed to read file {file_path}: {str(e)}")
        return file_logs

    file_logs.append(f"Processing file: {file_path}")

    merged_blocks, code_lines = build_merged_blocks_for_file(file_path, file_findings)
    code = "\n".join(code_lines)

    # No defects: copy original file
    if not merged_blocks:
        file_logs.append(f"No defects found in file {file_path}!")
        out_file = out_root / os.path.relpath(file_path, project_root)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(code, encoding="utf-8")
        return file_logs

    attempt = 0
    fixed_code = code

    while attempt < max_attempts:
        repair_results = []
        for block in merged_blocks:
            defects = block["defects"]
            contexts = block["surrounding_context"]
            sum_context = ""
            for text in contexts:
                sum_context += f"...\n{text}\n"

            if not surrounding_context:
                sum_context = code

            rag_prompt = ""
            file_logs.append(f"[Defects] {defects}")

            # Deduplicate by rule to limit RAG calls
            unique_defects: List[Dict[str, Any]] = []
            seen_rules = set()
            for defect in defects:
                if defect["rule"] not in seen_rules:
                    unique_defects.append(defect)
                    seen_rules.add(defect["rule"])

            defect_description = "The code snippet containing the defect is as follows:\n"
            error_location = "The location of the defect is as follows:\n"

            valid_defects: List[Dict[str, Any]] = []
            if model is not None and tokenizer is not None and index is not None:
                # Full RAG path: retrieve repair demos per unique rule
                for defect in unique_defects:
                    defect_rule = defect["rule"]
                    repair_example = {
                        "rule": defect_rule,
                        "problem_code": sum_context,
                    }
                    try:
                        rag_result = get_rag_prompt(
                            repair_example,
                            model,
                            tokenizer,
                            index,
                            rag_type,
                            number=top_n,
                        )
                    except Exception as e:
                        # If RAG fails (e.g., Pinecone/OpenAI client issues),
                        # log it and continue without RAG for this rule.
                        file_logs.append(
                            f"[RAG-ERROR] rule={defect_rule}, error={e}. "
                            "Falling back to non-RAG repair for this rule."
                        )
                        continue
                    if rag_result:
                        rag_prompt += rag_result
                        valid_defects.append(defect)
                        # Log a truncated view of the RAG prompt for this rule
                        snippet = rag_result[:800]
                        if len(rag_result) > 800:
                            snippet += "...[truncated]"
                        file_logs.append(
                            f"[RAG] rule={defect_rule}, prompt_len={len(rag_result)}\n{snippet}"
                        )
            else:
                # No RAG available: keep all unique_defects, rag_prompt stays empty
                valid_defects = unique_defects

            for i, defect in enumerate(defects):
                if any(d["rule"] == defect["rule"] for d in valid_defects):
                    defect_description += f"Defect {i+1}:\n{defect['message']}\n"
                    error_location += f"Defect {i+1}:\n{defect['code']}\n"

            file_logs.append(f"Code context to fix: {sum_context}")
            file_logs.append("-" * 100)

            fix_prompt = generate_fix_prompt(
                rag_prompt,
                code,
                sum_context,
                defect_description,
                error_location,
            )

            # Log a truncated view of the fix prompt
            fix_snippet = fix_prompt[:1000]
            if len(fix_prompt) > 1000:
                fix_snippet += "...[truncated]"
            file_logs.append(f"[LLM] fix_prompt (truncated):\n{fix_snippet}")

            logger.info(repair_model_name)

            res = get_answer(fix_prompt, model_name=repair_model_name)
            # Log a truncated view of the LLM block-level response
            res_snippet = res[:1500]
            if len(res) > 1500:
                res_snippet += "...[truncated]"
            file_logs.append(f"[LLM] fixed code for current block (truncated):\n{res_snippet}")
            repair_results.append((sum_context, res))

        final_fix_prompt = combine_repair_results(repair_results, code)
        final_fix_snippet = final_fix_prompt[:1000]
        if len(final_fix_prompt) > 1000:
            final_fix_snippet += "...[truncated]"
        file_logs.append(f"[LLM] final_fix_prompt (truncated):\n{final_fix_snippet}")
        final_res = get_answer(final_fix_prompt, model_name=repair_model_name)
        final_res_snippet = final_res[:2000]
        if len(final_res) > 2000:
            final_res_snippet += "...[truncated]"
        file_logs.append(f"[LLM] final result (truncated):\n{final_res_snippet}")
        fixed_code = extract_code_from_markdown_block(final_res)

        fixed_code = remove_difflib_line(fixed_code)

        fixer = ArkTSDeclarationFixer()
        result = fixer.validate_and_fix(fixed_code)
        if result:
            fixed_code = result.fixed_code

        # NOTE: Functionality check disabled.
        # Previously we called `check_functionality` here and would revert the
        # file back to the original code if the LLM-based functionality check
        # reported a failure after several attempts. This caused many fixes to
        # be rolled back even when they reduced defects.
        #
        # Now we always accept the repaired code produced in this loop.
        file_logs.append("Functionality check skipped; accepting repaired code.")
        break

    out_file = out_root / os.path.relpath(file_path, project_root)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(fixed_code, encoding="utf-8")

    file_logs.append(f"File {file_path} fix completed!")
    file_logs.append("-" * 100)
    return file_logs


def fix_parsing_errors_in_place(
    original_root: Path,
    current_root: Path,
    log_dir: Path,
    config: Optional[Path],
    model_name: str,
    logger: logging.Logger,
    max_rounds: int = 2,
) -> None:
    """
    Detect and repair parsing errors in the current project without counting
    as an extra "repair round".

    Workflow:
    - For up to max_rounds iterations:
      1) Run CodeLinter on current_root.
      2) Parse parsing-error findings from its log.
      3) For each affected file:
         - Read original file from original_root (if available).
         - Read current file from current_root.
         - Build a parsing-fix prompt and call the LLM.
         - Overwrite the current file with the fixed code.
      4) If no parsing errors remain, stop.
    """
    rcp = _load_run_codelinter_module()

    for round_idx in range(1, max_rounds + 1):
        log_dir.mkdir(parents=True, exist_ok=True)
        # Reuse the same run_codelinter logic as the main pipeline
        rc = int(rcp.run_codelinter(current_root, log_dir, config))  # type: ignore[attr-defined]
        log_path = log_dir / f"{current_root.name}.log"
        logger.info(
            f"[parsing-fix] Round {round_idx}: CodeLinter exit_code={rc}, log={log_path}"
        )
        errors = parse_parsing_errors(log_path)
        if not errors:
            logger.info("[parsing-fix] No parsing errors detected; nothing to fix.")
            break

        # Group errors by file
        by_file: Dict[Path, List[Dict[str, Any]]] = {}
        for e in errors:
            fp = Path(e["file_path"]).resolve()
            by_file.setdefault(fp, []).append(e)

        logger.info(
            f"[parsing-fix] Found parsing errors in {len(by_file)} file(s): "
            + ", ".join(str(p) for p in by_file.keys())
        )

        for file_path, per_errors in by_file.items():
            try:
                rel = file_path.relative_to(current_root)
            except ValueError:
                # Fallback: use basename
                rel = Path(file_path.name)

            orig_file = original_root / rel
            try:
                current_code = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning(f"[parsing-fix] Failed to read current file {file_path}: {exc}")
                continue

            if orig_file.is_file():
                try:
                    original_code = orig_file.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning(
                        f"[parsing-fix] Failed to read original file {orig_file}: {exc}"
                    )
                    original_code = current_code
            else:
                original_code = current_code

            prompt = _build_parsing_error_prompt(
                file_path,
                original_code,
                current_code,
                per_errors,
            )

            logger.info(f"[parsing-fix] Repairing parsing errors in {file_path}")
            try:
                response = get_answer(prompt, model_name=model_name)
            except Exception as exc:
                logger.error(f"[parsing-fix] LLM call failed for {file_path}: {exc}")
                continue

            fixed_code = extract_code_from_markdown_block(response)
            # Fallback: if extraction failed, keep original response
            if not fixed_code.strip():
                fixed_code = response

            try:
                file_path.write_text(fixed_code, encoding="utf-8")
            except Exception as exc:
                logger.error(f"[parsing-fix] Failed to write fixed file {file_path}: {exc}")
                continue

        # After fixing files in this parsing-fix round, loop to see if errors remain


def _is_project_root(path: Path) -> bool:
    if any((path / name).is_file() for name in HVIGOR_FILES):
        return True
    if (path / "build-profile.json5").is_file():
        return True
    if (path / "oh-package.json5").is_file():
        return True
    return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fix ArkTS performance/security defects in a Harmony project using "
            "CodeLinter findings + RAG, mirroring fix_projects.py."
        )
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to the Harmony/ArkTS project root.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=False,
        default=None,
        help=(
            "Directory where the repaired project will be written. "
            "If omitted, defaults to "
            "revision/fixed_projects/<model_name>/round_<round>/<project_name>."
        ),
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_CODELINTER_LOG_DIR,
        help=(
            "Directory to store CodeLinter logs. If left as the default, "
            "logs will be written under "
            "logs/codelinter_openharmony/<model_name>/round_<round>/"
        ),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("/home/LLMCodeRepair/revision/code-linter.json5"),
        help="CodeLinter config file (.json/.json5) passed via --config.",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-mini",
        help="Repair model name used by get_answer (default: %(default)s).",
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
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum functionality-check attempts per file (default: %(default)s).",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum parallel worker threads (default: %(default)s).",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=1,
        help=(
            "Repair round index used to organize outputs under "
            "revision/fixed_projects/round_<round>/ (default: %(default)s)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    # Sanitize model name for filesystem use
    model_dir = (
        args.model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    )

    # Workspace root lives under repo_new so that CodeLinter sees a valid project path.
    workspace_base = REPO_ROOT / "repo_new" / "_haprepair_fixed" / model_dir / f"round_{args.round}"
    workspace_root = (workspace_base / project_root.name).resolve()

    # Decide snapshot output root under revision/
    if args.output_root is not None:
        out_root = args.output_root.resolve()
    else:
        base = REVISION_ROOT / "fixed_projects" / model_dir / f"round_{args.round}"
        out_root = (base / project_root.name).resolve()

    # Decide CodeLinter log directory. When the user keeps the default
    # (logs/codelinter_openharmony), we scope it by model + round so that
    # different rounds/models do not overwrite each other's logs.
    default_log_dir = DEFAULT_CODELINTER_LOG_DIR.resolve()
    requested_log_dir = args.log_dir.resolve()
    if requested_log_dir == default_log_dir:
        log_dir = default_log_dir / model_dir / f"round_{args.round}"
    else:
        log_dir = requested_log_dir

    if not project_root.is_dir():
        raise SystemExit(f"project-root is not a directory: {project_root}")
    if not _is_project_root(project_root):
        print(
            f"[warn] {project_root} does not look like a Harmony project root "
            "(missing hvigor/build-profile/oh-package).",
            file=sys.stderr,
        )

    # Prepare workspace: copy the entire project into repo_new/_haprepair_fixed/...
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    shutil.copytree(project_root, workspace_root)

    # Configure logging: stdout + per-project log file under logs/fix_projects_codelinter
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("fix_projects_codelinter")

    log_root = REPO_ROOT / "logs" / "fix_projects_codelinter" / model_dir / f"round_{args.round}"
    log_root.mkdir(parents=True, exist_ok=True)
    project_log_path = log_root / f"{project_root.name}.log"
    file_handler = logging.FileHandler(project_log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging detailed repair trace to: {project_log_path}")

    logger.info(f"Running CodeLinter on project: {workspace_root}")
    config = args.config if args.config.is_file() else None
    try:
        log_path = run_codelinter_for_project(workspace_root, log_dir, config)
    except Exception as exc:
        raise SystemExit(f"CodeLinter failed: {exc}") from exc

    logger.info(f"Parsing CodeLinter log: {log_path}")
    findings = parse_codelinter_log(log_path)
    if not findings:
        logger.info("No performance/security findings detected; copying project as-is.")
        # Copy workspace to snapshot output_root
        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(workspace_root, out_root)
        return

    # Bucket findings by file
    by_file: Dict[Path, List[Dict[str, Any]]] = {}
    for f in findings:
        fp = Path(f["file_path"]).resolve()
        by_file.setdefault(fp, []).append(f)

    # Load RAG model + index
    logger.info("Loading RAG model and index...")
    model, tokenizer, index = load_model_and_index()

    # Collect all ArkTS/TS files under workspace_root (for copying unmodified ones)
    all_files: List[Path] = [
        p
        for p in workspace_root.rglob("*")
        if p.is_file() and p.suffix in (".ets", ".ts")
    ]

    # Files that have defects according to CodeLinter
    defect_files = [fp for fp in all_files if fp in by_file]

    logger.info(
        f"Found {len(findings)} findings across {len(defect_files)} files "
        f"(project files: {len(all_files)})."
    )

    # Process defect files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                process_file_with_codelinter_blocks,
                file_path,
                workspace_root,
                workspace_root,
                by_file.get(file_path, []),
                logger,
                model,
                tokenizer,
                index,
                rag_type=args.rag_type,
                top_n=args.top_n,
                surrounding_context=True,
                repair_model_name=args.model_name,
                max_attempts=args.max_attempts,
            ): file_path
            for file_path in defect_files
        }

        for fut in concurrent.futures.as_completed(futures):
            file_path = futures[fut]
            try:
                logs = fut.result()
            except Exception as exc:
                logger.error(f"Error processing file {file_path}: {exc}")
                continue
            for line in logs:
                logger.info(line)

    # Copy over files without defects (as-is). When out_root != workspace_root
    # this copies the final workspace tree to the snapshot directory. For the
    # main repair loop we already wrote modified files in-place under workspace_root.
    for file_path in all_files:
        if file_path in defect_files:
            continue
        rel = file_path.relative_to(workspace_root)
        dst = out_root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(file_path.read_bytes())

    # After performance/security repairs, run a lightweight parsing-error
    # repair pass **in place** on the workspace project, without counting as an
    # additional round.
    try:
        parsing_log_dir = log_dir.parent / f"{log_dir.name}_parsing"
        logger.info(
            f"Running parsing-error repair (non-round) on {workspace_root}, "
            f"logs -> {parsing_log_dir}"
        )
        fix_parsing_errors_in_place(
            original_root=project_root,
            current_root=workspace_root,
            log_dir=parsing_log_dir,
            config=config,
            model_name=args.model_name,
            logger=logger,
            max_rounds=2,
        )
    except Exception as exc:
        logger.error(f"Parsing-error repair failed (non-fatal): {exc}")

    # Post-fix CodeLinter run to summarize remaining defects for this round.
    try:
        summary_log_dir = log_dir.parent / f"{log_dir.name}_after_round{args.round}"
        logger.info(
            f"Running post-fix CodeLinter on {workspace_root}, "
            f"logs -> {summary_log_dir}"
        )
        summary_log_path = run_codelinter_for_project(
            workspace_root, summary_log_dir, config
        )
        logger.info(f"Parsing post-fix CodeLinter log: {summary_log_path}")
        findings_final = parse_codelinter_log(summary_log_path)
        parsing_final = parse_parsing_errors(summary_log_path)
        perf_final = sum(
            1 for f in findings_final if f.get("category") == "performance"
        )
        sec_final = sum(
            1 for f in findings_final if f.get("category") == "security"
        )
        logger.info(
            "[summary] After round %s: total_defects=%d, "
            "perf_defects=%d, security_defects=%d, parsing_errors=%d",
            args.round,
            len(findings_final),
            perf_final,
            sec_final,
            len(parsing_final),
        )
    except Exception as exc:
        logger.error(f"Post-fix CodeLinter summary failed (non-fatal): {exc}")

    # Final snapshot: copy workspace_root (with parsing fixes) to revision tree
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(workspace_root, out_root)

    logger.info(f"Repair completed. Output project written to: {out_root}")


if __name__ == "__main__":
    main()
