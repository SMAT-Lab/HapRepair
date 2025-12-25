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
import contextlib
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
from code_repair import CodeContextExtractor  # type: ignore
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


def _rag_api_base() -> Optional[str]:
    base = os.getenv("RAG_API_BASE") or os.getenv("RAG_SERVICE_URL")
    if not base:
        return None
    base = base.rstrip("/")
    return base

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
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore

        model.eval()
        device_raw = os.environ.get("RAG_DEVICE", "cpu").strip().lower()
        dtype_raw = os.environ.get("RAG_DTYPE", "").strip().lower()

        if device_raw in ("gpu",):
            device_raw = "cuda"
        if device_raw in ("", "cpu"):
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(device_raw)
            except Exception:
                device = torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        if device.type == "cuda":
            if dtype_raw in ("bf16", "bfloat16"):
                model = model.to(dtype=torch.bfloat16)
            elif dtype_raw in ("fp16", "float16", "half"):
                model = model.to(dtype=torch.float16)

        model = model.to(device)

        # Prevent runaway CPU thread oversubscription when many projects run in parallel.
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except Exception:
        pass

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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw not in ("0", "false", "no", "off", "")


@contextlib.contextmanager
def _workspace_copy_lock() -> Iterable[None]:
    """
    Serialize workspace copy operations across processes to avoid huge memory spikes
    from many concurrent `copytree` runs (dirty page cache + metadata walk).

    Enabled by default. Disable with `HAPREPAIR_DISABLE_COPY_LOCK=1`.
    Override lock path with `HAPREPAIR_COPY_LOCK_FILE=/path/to/lock`.
    """
    if _env_flag("HAPREPAIR_DISABLE_COPY_LOCK", default=False):
        yield
        return

    lock_path = Path(os.environ.get("HAPREPAIR_COPY_LOCK_FILE", "/tmp/LLMCodeRepair_haprepair_copy.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl  # Linux/Unix only
    except Exception:
        yield
        return

    fh = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"[wait] Another project copy is in progress; waiting for global copy lock: {lock_path}",
                file=sys.stderr,
            )
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


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


def _count_braces(line: str) -> int:
    # Best-effort brace delta; good enough for our simple auto-fix heuristics.
    return line.count("{") - line.count("}")


def _auto_fix_use_id_in_get_resource_sync_api(
    code_lines: List[str],
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Auto-fix @performance/hp-arkui-use-id-in-get-resource-sync-api.

    Common pattern:
      resourceManager.getStringSync($r('app.string.xxx'))
    becomes:
      resourceManager.getStringSync($r('app.string.xxx').id)

    This keeps line count stable.
    """
    if not file_findings:
        return code_lines, file_findings

    max_idx = len(code_lines) - 1
    new_findings: List[Dict[str, Any]] = []

    pattern = re.compile(r"(getStringSync)\(\s*(\$r\([^)]*\))\s*\)")
    for f in file_findings:
        if (
            f.get("category") == "performance"
            and f.get("rule_id") == "hp-arkui-use-id-in-get-resource-sync-api"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                line = code_lines[line_no]
                # Skip if already uses .id
                if ".id" in line:
                    continue
                new_line = pattern.sub(r"\1(\2.id)", line)
                if new_line != line:
                    code_lines[line_no] = new_line
                    continue

        new_findings.append(f)

    return code_lines, new_findings


def _auto_fix_use_local_var_to_replace_state_var(
    code_lines: List[str],
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Auto-fix @performance/hp-arkui-use-local-var-to-replace-state-var.

    Heuristic:
    - Only targets `private` methods (skips `build()`).
    - For each @State field referenced repeatedly as `this.<state>` inside a method:
      - Prefix the first top-level statement with `let _local_<state> = this.<state>; `
      - Replace remaining `this.<state>` occurrences with `_local_<state>`
      - If the state is modified (e.g., `+=`, `=`), insert `this.<state> = _local_<state>`
        immediately before the method closing brace.
    """
    if not file_findings:
        return code_lines, file_findings

    # Only run if we have findings for this rule.
    if not any(
        f.get("category") == "performance"
        and f.get("rule_id") == "hp-arkui-use-local-var-to-replace-state-var"
        for f in file_findings
    ):
        return code_lines, file_findings

    max_idx = len(code_lines) - 1

    # Extract @State field names from struct scope.
    state_vars: List[str] = []
    state_re = re.compile(r"^\s*@State\s+([A-Za-z_$][\w$]*)\b")
    for ln in range(1, max_idx + 1):
        m = state_re.match(code_lines[ln])
        if m:
            state_vars.append(m.group(1))
    state_set = set(state_vars)
    if not state_set:
        return code_lines, file_findings

    # If the findings mention a subset of state vars, only target those.
    flagged_vars: set[str] = set()
    this_var_re = re.compile(r"\bthis\.([A-Za-z_$][\w$]*)\b")
    for f in file_findings:
        if (
            f.get("category") == "performance"
            and f.get("rule_id") == "hp-arkui-use-local-var-to-replace-state-var"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                for mm in this_var_re.finditer(code_lines[line_no]):
                    name = mm.group(1)
                    if name in state_set:
                        flagged_vars.add(name)
    if not flagged_vars:
        flagged_vars = set(state_vars)

    sig_re = re.compile(r"^\s*private\s+([A-Za-z_$][\w$]*)\s*\(")

    # Collect method ranges first (based on the original line indices), then
    # apply transformations bottom-up to keep indices stable.
    methods: List[Tuple[int, int, str]] = []
    i = 1
    while i <= max_idx:
        line = code_lines[i]
        m = sig_re.match(line)
        if not m:
            i += 1
            continue
        method_name = m.group(1)
        if method_name == "build":
            i += 1
            continue

        depth = _count_braces(line)
        if depth <= 0:
            i += 1
            continue
        start = i
        i += 1
        while i <= max_idx and depth > 0:
            depth += _count_braces(code_lines[i])
            i += 1
        end = i - 1
        if end > start:
            methods.append((start, end, method_name))

    insertion_points: List[int] = []

    for start, end, _method_name in reversed(methods):
        if end <= start + 1:
            continue

        # Identify the first top-level statement (depth==1) to prefix with init.
        depth_scan = 1
        first_stmt: Optional[int] = None
        for ln in range(start + 1, end):
            raw = code_lines[ln]
            stripped = raw.strip()
            if depth_scan == 1 and stripped and not stripped.startswith("}"):
                first_stmt = ln
                break
            depth_scan += _count_braces(raw)
        if first_stmt is None:
            continue

        body_lines = code_lines[start + 1 : end]
        body_text = "\n".join(body_lines)

        for var in sorted(flagged_vars):
            local = f"_local_{var}"
            needle = f"this.{var}"
            if local in body_text:
                continue
            if needle not in body_text:
                continue
            if body_text.count(needle) < 2:
                continue

            # Detect whether the state var is modified in this method.
            modifies = False
            assign_re = re.compile(rf"\bthis\.{re.escape(var)}\b\s*(\+\+|--|[+\-*/%]?=)")
            for ln in range(start + 1, end):
                if assign_re.search(code_lines[ln]):
                    modifies = True
                    break

            # Prefix the first statement with local init (same line).
            stmt_line = code_lines[first_stmt]
            indent = re.match(r"^\s*", stmt_line).group(0)  # type: ignore[union-attr]
            stmt_rest = stmt_line[len(indent) :]
            init_prefix = f"let {local} = {needle}; "
            if init_prefix.strip() not in stmt_line:
                code_lines[first_stmt] = f"{indent}{init_prefix}{stmt_rest}"

            # Replace uses in method body, but keep the init's `this.<var>` intact.
            placeholder = f"__KEEP_THIS_{var.upper()}__"
            init_line = code_lines[first_stmt]
            init_line = init_line.replace(needle, placeholder, 1)
            init_line = init_line.replace(needle, local)
            init_line = init_line.replace(placeholder, needle)
            code_lines[first_stmt] = init_line

            for ln in range(start + 1, end):
                if ln == first_stmt:
                    continue
                code_lines[ln] = code_lines[ln].replace(needle, local)

            # Write back before the method closing brace if needed.
            if modifies:
                # Insert immediately before the method's closing brace line.
                insert_at = end
                code_lines.insert(insert_at, f"{indent}{needle} = {local}")
                insertion_points.append(end)
                max_idx += 1
                end += 1
                # Keep body_text in sync for subsequent vars in same method.
                body_text = "\n".join(code_lines[start + 1 : end])

    # Drop findings handled by this auto-fix.
    adjusted_findings: List[Dict[str, Any]] = [
        f
        for f in file_findings
        if not (
            f.get("category") == "performance"
            and f.get("rule_id") == "hp-arkui-use-local-var-to-replace-state-var"
        )
    ]

    if insertion_points:
        insertion_points.sort()

        def shift_line(old_line: int) -> int:
            # Each insertion happens before its recorded original end line,
            # so any line >= insertion_point shifts by +1.
            import bisect

            return old_line + bisect.bisect_left(insertion_points, old_line)

        remapped: List[Dict[str, Any]] = []
        for f in adjusted_findings:
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if line_no > 0:
                f = dict(f)
                f["line"] = shift_line(line_no)
            remapped.append(f)
        adjusted_findings = remapped

    return code_lines, adjusted_findings


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


def _insert_cached_count_for_grid_or_lazyforeach(
    code_lines: List[str],
    line_no: int,
    cached_count: int = 4,
) -> bool:
    """
    Auto-fix helper for @performance/hp-arkui-set-cache-count-for-lazyforeach-grid.

    Strategy:
    - In a small window around the reported line, look for a Grid(...) or
      LazyForEach(...) call that starts a component.
    - Track parentheses from the '(' after the call keyword across subsequent
      lines until the top-level call is closed.
    - Insert `.cachedCount(<cached_count>)` immediately after this closing ')'
      so that, for example:

        Grid(this.scroller) {
          ...
        }

      becomes:

        Grid(this.scroller).cachedCount(4) {
          ...
        }

    We keep the transformation conservative and skip if we cannot reliably
    find and close the call, or if `.cachedCount(` already appears in the
    call expression line.
    """
    max_idx = len(code_lines) - 1
    if max_idx <= 0:
        return False

    start = max(1, line_no - 1)
    end = min(max_idx, line_no + 2)
    patterns = ["Grid(", "LazyForEach("]

    for ln in range(start, end + 1):
        line = code_lines[ln]
        for pattern in patterns:
            col = line.find(pattern)
            if col == -1:
                continue

            # If this line already contains a cachedCount call, skip.
            if ".cachedCount(" in line:
                return False

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
                        # Avoid inserting twice if cachedCount is already present.
                        if ".cachedCount(" in target_line:
                            return False
                        insert_text = f".cachedCount({cached_count})"
                        code_lines[cur_line] = (
                            target_line[: cur_col + 1]
                            + insert_text
                            + target_line[cur_col + 1 :]
                        )
                        return True
                cur_col += 1

    return False


def _auto_fix_set_cache_count_for_lazyforeach_grid(
    code_lines: List[str],
    file_findings: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Auto-fix @performance/hp-arkui-set-cache-count-for-lazyforeach-grid by
    inserting `.cachedCount(4)` on Grid/LazyForEach component calls used
    with LazyForEach in grids.

    For each finding:
      - Try to insert `.cachedCount(4)` after the Grid(...) or LazyForEach(...)
        call near the reported line.
      - If successful, drop the finding so it is not sent to the LLM.
    """
    if not file_findings:
        return code_lines, file_findings

    max_idx = len(code_lines) - 1
    new_findings: List[Dict[str, Any]] = []

    for f in file_findings:
        if (
            f.get("category") == "performance"
            and f.get("rule_id") == "hp-arkui-set-cache-count-for-lazyforeach-grid"
        ):
            try:
                line_no = int(f.get("line") or 0)
            except (TypeError, ValueError):
                line_no = 0
            if 1 <= line_no <= max_idx:
                if _insert_cached_count_for_grid_or_lazyforeach(code_lines, line_no):
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

    # ------------------------------------------------------------------
    # Auto-fix @performance/hp-arkui-set-cache-count-for-lazyforeach-grid:
    # Insert `.cachedCount(4)` on Grid/LazyForEach component calls used
    # with LazyForEach in grids, and drop the corresponding findings.
    # ------------------------------------------------------------------
    code_lines, file_findings = _auto_fix_set_cache_count_for_lazyforeach_grid(
        code_lines, file_findings
    )
    max_idx = len(code_lines) - 1

    # ------------------------------------------------------------------
    # Optional auto-fix @performance/hp-arkui-use-local-var-to-replace-state-var:
    # This transformation can be behavior-sensitive (state sync + reactivity),
    # so it is DISABLED by default. Enable with:
    #   HAPREPAIR_ENABLE_AUTOFIX_STATEVAR_LOCAL_CACHE=1
    # ------------------------------------------------------------------
    if os.environ.get("HAPREPAIR_ENABLE_AUTOFIX_STATEVAR_LOCAL_CACHE", "").strip() in ("1", "true", "yes", "on"):
        code_lines, file_findings = _auto_fix_use_local_var_to_replace_state_var(
            code_lines, file_findings
        )
        max_idx = len(code_lines) - 1

    # ------------------------------------------------------------------
    # Auto-fix @performance/hp-arkui-use-id-in-get-resource-sync-api:
    # Add `.id` when using $r(...) with getStringSync.
    # ------------------------------------------------------------------
    code_lines, file_findings = _auto_fix_use_id_in_get_resource_sync_api(
        code_lines, file_findings
    )
    max_idx = len(code_lines) - 1
    # Some performance rules need larger, structure-aware context (component/block scope)
    context_rules = {
        "avoid-overusing-custom-component-check",
        "dark-color-mode-check",
        "foreach-index-check",
        "hp-arkui-use-local-var-to-replace-state-var",
        "hp-arkui-use-onAnimationStart-for-swiper-preload",
        "hp-arkui-use-reusable-component",
        "waterflow-data-preload-check",
    }
    context_extractor: Optional[CodeContextExtractor] = None

    raw_blocks: List[Dict[str, Any]] = []
    for f in file_findings:
        line_no = int(f.get("line") or 0)
        if line_no <= 0 or line_no > max_idx:
            continue
        start = max(1, line_no - 5)
        end = min(max_idx, line_no + 5)
        rule_id = f.get("rule_id", "")
        if (
            f.get("category") == "performance"
            and rule_id in context_rules
        ):
            # Provide the entire file as context for these hard-to-fix rules.
            # These often need cross-section awareness (component structure,
            # preload wiring, dark mode resources, etc.).
            start, end = 1, max_idx
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
            # Provide the real code context directly, without artificial "..."
            # separators. The previous ellipsis markers could confuse the LLM
            # when mapping diffs back to the full file, leading to misplaced
            # edits and broken syntax in multi-block fixes.
            sum_context = "\n\n".join(contexts)

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
            use_rag = top_n > 0 and (
                _rag_api_base()
                or (model is not None and tokenizer is not None and index is not None)
            )
            if use_rag:
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
            if use_rag and not valid_defects:
                # If retrieval returns no matches, fall back to non-RAG defects
                # so we still include defect descriptions in the prompt.
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


def _sanitize_model_dir(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _make_model_dir(model_name: str, run_tag: Optional[str]) -> str:
    base = _sanitize_model_dir(model_name)
    if not run_tag:
        return base
    tag = _sanitize_model_dir(run_tag)
    return f"{base}__{tag}"


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
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional run tag appended to output/log directories "
            "(keeps model-name for LLM calls).",
        ),
    )
    ap.add_argument(
        "--rag-type",
        type=str,
        default="difflib",
        help=(
            "RAG retriever type passed to get_rag_prompt "
            "(gpt_diff/difflib/no_diff; default: %(default)s)."
        ),
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
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum functionality-check attempts per file (default: %(default)s).",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Maximum parallel worker threads within a project "
            "(default: %(default)s)."
        ),
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
    model_dir = _make_model_dir(args.model_name, args.run_tag)

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
    with _workspace_copy_lock():
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
    parsing_initial = parse_parsing_errors(log_path)
    perf_initial = sum(1 for f in findings if f.get("category") == "performance")
    sec_initial = sum(1 for f in findings if f.get("category") == "security")
    logger.info(
        "[summary] Before round %s: total_defects=%d, "
        "perf_defects=%d, security_defects=%d, parsing_errors=%d",
        args.round,
        len(findings),
        perf_initial,
        sec_initial,
        len(parsing_initial),
    )
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

    # Load RAG model + index only when needed (skip if using remote RAG API)
    rag_api_base = _rag_api_base()
    if args.top_n > 0 and not rag_api_base:
        logger.info("Loading RAG model and index...")
        model, tokenizer, index = load_model_and_index()
    else:
        if rag_api_base:
            logger.info(f"Using RAG API at {rag_api_base}; skipping local model/index load.")
        else:
            logger.info("RAG disabled (top_n <= 0); skipping RAG model/index load.")
        model = tokenizer = index = None

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
                surrounding_context=args.surrounding_context,
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
    summary_log_dir = log_dir.parent / f"{log_dir.name}_after_round{args.round}"
    findings_final: Optional[List[Dict[str, Any]]] = None
    parsing_final: Optional[List[Dict[str, Any]]] = None
    perf_final: Optional[int] = None
    sec_final: Optional[int] = None
    postfix_failed = False
    # Optional baseline re-check on the *input* snapshot. This guards against:
    # - nondeterminism / environment drift during long runs,
    # - cases where the initial log under-reports findings (e.g., partial analysis),
    # so that rollback decisions compare against a stable "input snapshot" reference.
    baseline_findings: Optional[List[Dict[str, Any]]] = None
    baseline_parsing: Optional[List[Dict[str, Any]]] = None
    try:
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
            "[summary] Round %s: before total_defects=%d (perf=%d, security=%d, parsing_errors=%d) "
            "-> after total_defects=%d (perf=%d, security=%d, parsing_errors=%d)",
            args.round,
            len(findings),
            perf_initial,
            sec_initial,
            len(parsing_initial),
            len(findings_final),
            perf_final,
            sec_final,
            len(parsing_final),
        )
    except Exception as exc:
        postfix_failed = True
        logger.error(f"Post-fix CodeLinter summary failed (will rollback): {exc}")

    def _should_rollback() -> bool:
        if postfix_failed:
            return True
        if findings_final is None or parsing_final is None:
            return True
        after_defects = len(findings_final)
        after_parsing = len(parsing_final)
        # Prefer comparing against the baseline defects of the input snapshot
        # if available; otherwise fall back to the initial "before" counts.
        before_defects = (
            len(baseline_findings) if baseline_findings is not None else len(findings)
        )
        before_parsing = (
            len(baseline_parsing) if baseline_parsing is not None else len(parsing_initial)
        )
        if after_defects > before_defects:
            return True
        if after_defects == before_defects and after_parsing > before_parsing:
            return True
        return False

    # If the post-fix results look worse than the initial counts, re-run CodeLinter
    # on the input snapshot (project_root) to get a stable baseline for rollback.
    if (
        not postfix_failed
        and findings_final is not None
        and parsing_final is not None
        and (
            len(findings_final) > len(findings)
            or (len(findings_final) == len(findings) and len(parsing_final) > len(parsing_initial))
        )
    ):
        baseline_log_dir = log_dir.parent / f"{log_dir.name}_baseline_input_round{args.round}"
        try:
            logger.info(
                f"[baseline] Re-running CodeLinter on input snapshot {project_root}, "
                f"logs -> {baseline_log_dir}"
            )
            baseline_log_path = run_codelinter_for_project(
                project_root, baseline_log_dir, config
            )
            baseline_findings = parse_codelinter_log(baseline_log_path)
            baseline_parsing = parse_parsing_errors(baseline_log_path)
            logger.info(
                "[baseline] Input snapshot defects=%d (parsing_errors=%d)",
                len(baseline_findings),
                len(baseline_parsing),
            )
        except Exception as exc:
            logger.warning(f"[baseline] Failed to compute input baseline; falling back to initial counts: {exc}")

    # Final snapshot: if the round makes things worse, keep the input snapshot.
    if _should_rollback():
        logger.warning(
            "[rollback] Round %s made results worse (or post-fix failed); keeping input snapshot %s",
            args.round,
            project_root,
        )
        # Overwrite the after_round log with the retained snapshot's results so
        # downstream summaries reflect what we actually keep.
        try:
            logger.info(
                f"[rollback] Running CodeLinter on retained input {project_root}, "
                f"logs -> {summary_log_dir}"
            )
            _ = run_codelinter_for_project(project_root, summary_log_dir, config)
        except Exception as exc:
            logger.error(f"[rollback] Failed to regenerate after_round log: {exc}")

        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(project_root, out_root)
        logger.info(f"Repair completed with rollback. Output project written to: {out_root}")
        return

    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(workspace_root, out_root)

    logger.info(f"Repair completed. Output project written to: {out_root}")


if __name__ == "__main__":
    main()
