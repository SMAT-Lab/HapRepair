#!/usr/bin/env python3
"""
Run Codex-based defect detection on the synthetic ArkTS dataset in
`revision/defects_location_detect` and evaluate against the JSON
ground truth `revision/defects_location_detect_gt.json`.

Usage:
  python detect_and_eval_codex.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
# Project under evaluation: copied OpenHarmony xts_acts module
PROJECT_ROOT = ROOT_DIR / "ace_ets_module_list02"
DATA_DIR = PROJECT_ROOT  # we will scan all .ets files recursively under this root
GT_PATH = ROOT_DIR / "ace_ets_module_list02_gt.json"
AGENT_SCRIPT = ROOT_DIR / "agent" / "run_codex.py"
RESULT_ROOT = ROOT_DIR / "ace_ets_module_list02_codex_eval"

# Try to import Codex helper directly for richer trace information
sys.path.insert(0, str(ROOT_DIR))
try:
    from agent.run_codex import run_codex as codex_run  # type: ignore
except Exception:
    codex_run = None


@dataclass
class FileResult:
    filename: str
    gt_rules: Set[str]
    gt_defects: Set[Tuple[str, int]]  # (rule, line)
    pred_rules: Set[str]
    pred_defects: Set[Tuple[str, int]]  # (rule, line)
    raw_output: str
    parse_ok: bool
    trace: List[Dict[str, Any]]


def load_ground_truth() -> Tuple[Dict[str, Set[str]], Dict[str, Set[Tuple[str, int]]]]:
    """Load per-file ground-truth rule sets and defect sets from JSON.

    JSON format (per file):
    {
      "relative/path/to/File.ets": {
        "defects": [
          {"line": 20, "severity": "...", "category": "performance", "rule": "@performance/...."},
          ...
        ],
        "rules": ["@performance/...", ...]
      },
      ...
    }
    """
    if not GT_PATH.is_file():
        raise FileNotFoundError(f"Ground truth JSON not found: {GT_PATH}")

    with GT_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)

    gt_rules: Dict[str, Set[str]] = {}
    gt_defects: Dict[str, Set[Tuple[str, int]]] = {}

    for fname, meta in raw.items():
        # rules
        rules = meta.get("rules") or []
        gt_rules[fname] = set(rules)

        # defects (rule+line)
        defect_items = meta.get("defects") or []
        defects_set: Set[Tuple[str, int]] = set()
        for item in defect_items:
            rule = item.get("rule")
            line = item.get("line")
            if isinstance(rule, str) and isinstance(line, int):
                defects_set.add((rule, line))
        gt_defects[fname] = defects_set

    return gt_rules, gt_defects


def build_prompt(code: str, filename: str, all_rules: List[str]) -> str:
    """Construct the detection prompt for Codex."""
    rules_text = "\n".join(f"- {r}" for r in all_rules)

    prompt = f"""You are an ArkTS performance defect detector.

You will receive a single ArkTS source file. Your task is to decide
which of the following performance rules are violated in this file.

Rules of interest (rule ids):
{rules_text}

Output format (MUST be valid JSON, nothing else):
- A JSON array of objects
- Each object MUST have:
  - "rule": the rule id string (one of the above)
  - "line": the 1-based line number where a violation of that rule occurs (integer)

Notes:
- If you believe multiple lines violate the same rule, output multiple objects
  with that rule and different line numbers.
- If none of the rules are violated, output [].
- Do NOT output explanations, comments, or markdown fences; ONLY the JSON array.

Now analyze this ArkTS file: {filename}

```arkts
{code}
```"""
    return prompt


def call_codex(prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Call Codex through run_codex helper and return (final_answer, trace).
    Falls back to subprocess if direct import is unavailable (trace will be empty).
    """
    # Preferred path: use Python API to get full trace JSON
    if codex_run is not None:
        result = codex_run(prompt, extra_args=None)
        raw_answer = (result.get("final_answer") or "").strip()
        trace = result.get("trace") or []
        if not isinstance(trace, list):
            trace = []
        return raw_answer, trace

    # Fallback: call the CLI wrapper as a subprocess (no detailed trace)
    if not AGENT_SCRIPT.is_file():
        raise FileNotFoundError(f"Agent script not found: {AGENT_SCRIPT}")

    proc = subprocess.run(
        [sys.executable, str(AGENT_SCRIPT)],
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(AGENT_SCRIPT.parent),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"run_codex.py failed (code {proc.returncode}).\n"
            f"stderr:\n{proc.stderr}"
        )

    return proc.stdout.strip(), []


def parse_prediction(raw: str) -> Tuple[Set[str], Set[Tuple[str, int]], bool]:
    """
    Parse Codex prediction JSON and return (rule_set, defect_set, parse_ok).
    Each defect is a (rule, line) pair. If parsing fails, returns empty sets.
    """
    raw = raw.strip()
    if not raw:
        return set(), set(), False

    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return set(), set(), False

    rules: Set[str] = set()
    defects: Set[Tuple[str, int]] = set()

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rule = item.get("rule")
                line = item.get("line")

                if isinstance(rule, str):
                    rules.add(rule)

                # allow both int and numeric string for line
                line_val: int | None = None
                if isinstance(line, int):
                    line_val = line
                elif isinstance(line, str) and line.isdigit():
                    line_val = int(line)

                if isinstance(rule, str) and line_val is not None:
                    defects.add((rule, line_val))

    return rules, defects, True


def evaluate() -> None:
    gt_rules_map, gt_defects_map = load_ground_truth()

    # Only evaluate files that appear in GT
    gt_files: List[str] = sorted(gt_rules_map.keys())

    # Collect all rules that appear in the ground truth
    all_rules: List[str] = sorted({r for rules in gt_rules_map.values() for r in rules})

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Prepare output directories for this run:
    # - timestamped snapshot directory
    # - stable "latest" directory used for resume/incremental writing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = RESULT_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    per_file_ndjson_path = latest_dir / "per_file_results.ndjson"

    # Load existing results from the stable "latest" directory for resume support
    existing_results: Dict[str, FileResult] = {}
    if latest_dir.is_dir() and per_file_ndjson_path.is_file():
        with per_file_ndjson_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fname = rec.get("filename")
                if not isinstance(fname, str):
                    continue

                # gt rules/defects: fall back to GT maps if not present
                gt_rules = gt_rules_map.get(fname, set())

                gt_def_list = rec.get("gt_defects") or []
                gt_defects: Set[Tuple[str, int]] = set()
                for d in gt_def_list:
                    if isinstance(d, dict):
                        rule = d.get("rule")
                        line_val = d.get("line")
                        if isinstance(rule, str) and isinstance(line_val, int):
                            gt_defects.add((rule, line_val))
                if not gt_defects and fname in gt_defects_map:
                    gt_defects = gt_defects_map[fname]

                pred_def_list = rec.get("pred_defects") or []
                pred_defects: Set[Tuple[str, int]] = set()
                for d in pred_def_list:
                    if isinstance(d, dict):
                        rule = d.get("rule")
                        line_val = d.get("line")
                        if isinstance(rule, str) and isinstance(line_val, int):
                            pred_defects.add((rule, line_val))

                pred_rules = {r for r, _ in pred_defects}

                parse_ok = bool(rec.get("parse_ok"))
                raw_output = rec.get("raw_output") or ""
                trace = rec.get("trace") or []
                if not isinstance(trace, list):
                    trace = []

                existing_results[fname] = FileResult(
                    filename=fname,
                    gt_rules=gt_rules,
                    gt_defects=gt_defects,
                    pred_rules=pred_rules,
                    pred_defects=pred_defects,
                    raw_output=raw_output,
                    parse_ok=parse_ok,
                    trace=trace,
                )

    completed_files: Set[str] = set(existing_results.keys())

    file_results: List[FileResult] = []
    pending_items: List[Tuple[str, Path]] = []

    # Separate already-completed files and pending files (GT files only)
    for rel_key in gt_files:
        if rel_key in completed_files:
            file_results.append(existing_results[rel_key])
            print(f"[resume] Skip already processed file: {rel_key}")
            continue

        path = DATA_DIR / rel_key
        if not path.is_file():
            print(f"[warn] GT file does not exist on disk, skip: {rel_key}")
            continue
        pending_items.append((rel_key, path))

    # Lock for safe concurrent append to NDJSON file
    file_write_lock = threading.Lock()

    def process_item(item: Tuple[str, Path]) -> FileResult:
        rel_key, path = item
        code = path.read_text(encoding="utf-8", errors="ignore")

        gt_rules = gt_rules_map.get(rel_key, set())
        gt_defects = gt_defects_map.get(rel_key, set())
        prompt = build_prompt(code, rel_key, all_rules)

        try:
            raw_output, trace = call_codex(prompt)
        except Exception as e:
            raw_output = f"ERROR: {e}"
            pred_rules: Set[str] = set()
            pred_defects: Set[Tuple[str, int]] = set()
            parse_ok = False
            trace = []
        else:
            pred_rules, pred_defects, parse_ok = parse_prediction(raw_output)

        res = FileResult(
            filename=rel_key,
            gt_rules=gt_rules,
            gt_defects=gt_defects,
            pred_rules=pred_rules,
            pred_defects=pred_defects,
            raw_output=raw_output,
            parse_ok=parse_ok,
            trace=trace,
        )

        # Incrementally persist this file's result (so progress is not lost).
        # We store defect-level info (rule + line); rule-level can be derived.
        record = {
            "filename": res.filename,
            "gt_defects": [
                {"rule": rule, "line": line} for rule, line in sorted(res.gt_defects)
            ],
            "pred_defects": [
                {"rule": rule, "line": line} for rule, line in sorted(res.pred_defects)
            ],
            "parse_ok": res.parse_ok,
            "raw_output": res.raw_output,
            "trace": res.trace,
        }
        with file_write_lock:
            with per_file_ndjson_path.open("a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        return res

    # Multi-threaded processing for pending files
    if pending_items:
        print(f"Processing {len(pending_items)} pending GT files with 8 threads...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_item = {executor.submit(process_item, it): it for it in pending_items}
            for future in as_completed(future_to_item):
                res = future.result()
                file_results.append(res)

    # Aggregate metrics
    # Rule-level counters
    tp_rules = fp_rules = fn_rules = 0
    # Defect-level counters (rule+line)
    tp_def = fp_def = fn_def = 0
    exact_match_files = 0
    parse_fail_files = 0

    for res in file_results:
        if not res.parse_ok:
            parse_fail_files += 1

        # Rule-level
        tp_rules_set = res.gt_rules & res.pred_rules
        fp_rules_set = res.pred_rules - res.gt_rules
        fn_rules_set = res.gt_rules - res.pred_rules

        tp_rules += len(tp_rules_set)
        fp_rules += len(fp_rules_set)
        fn_rules += len(fn_rules_set)

        # Defect-level (rule+line)
        tp_def_set = res.gt_defects & res.pred_defects
        fp_def_set = res.pred_defects - res.gt_defects
        fn_def_set = res.gt_defects - res.pred_defects

        tp_def += len(tp_def_set)
        fp_def += len(fp_def_set)
        fn_def += len(fn_def_set)

        # Exact match at defect-level
        if fn_def_set or fp_def_set:
            exact = False
        else:
            exact = True
        if exact:
            exact_match_files += 1

    total_files = len(file_results)
    total_gt_rules = sum(len(r.gt_rules) for r in file_results)
    total_gt_defects = sum(len(r.gt_defects) for r in file_results)

    print("=== Per-file results ===")
    for res in file_results:
        print(f"\nFile: {res.filename}")
        print(f"  GT rules   : {sorted(res.gt_rules) or '[]'}")
        print(f"  Pred rules : {sorted(res.pred_rules) or '[]'}")
        if not res.parse_ok:
            print("  NOTE: prediction JSON parse failed.")

    precision_rules = tp_rules / (tp_rules + fp_rules) if (tp_rules + fp_rules) > 0 else 0.0
    recall_rules = tp_rules / (tp_rules + fn_rules) if (tp_rules + fn_rules) > 0 else 0.0
    f1_rules = (
        2 * precision_rules * recall_rules / (precision_rules + recall_rules)
        if (precision_rules + recall_rules) > 0
        else 0.0
    )

    precision_def = tp_def / (tp_def + fp_def) if (tp_def + fp_def) > 0 else 0.0
    recall_def = tp_def / (tp_def + fn_def) if (tp_def + fn_def) > 0 else 0.0
    f1_def = (
        2 * precision_def * recall_def / (precision_def + recall_def)
        if (precision_def + recall_def) > 0
        else 0.0
    )

    print("\n=== Summary (defect-level only) ===")
    print(f"Total files          : {total_files}")
    print(f"Total GT defects     : {total_gt_defects}")
    print(f"Total predicted      : {tp_def + fp_def}")
    print(f"Correctly detected   : {tp_def}")
    print(f"Files with parse err.: {parse_fail_files}")
    print(f"Coverage (recall)    : {recall_def:.4f}")
    print(f"Precision            : {precision_def:.4f}")
    print(f"F1                   : {f1_def:.4f}")

    # Defect-level metrics split by single-defect vs multi-defect files
    def eval_group(items: List[FileResult]) -> Tuple[int, int, int, int, float, float, float]:
        tp_g = fp_g = fn_g = 0
        total_gt_g = 0
        for r in items:
            tp_set = r.gt_defects & r.pred_defects
            fp_set = r.pred_defects - r.gt_defects
            fn_set = r.gt_defects - r.pred_defects
            total_gt_g += len(r.gt_defects)
            tp_g += len(tp_set)
            fp_g += len(fp_set)
            fn_g += len(fn_set)
        prec = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
        rec = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return total_gt_g, tp_g, fp_g, fn_g, prec, rec, f1

    single_files = [r for r in file_results if len(r.gt_defects) == 1]
    multi_files = [r for r in file_results if len(r.gt_defects) > 1]

    total_gt_single, tp_s, fp_s, fn_s, prec_s, rec_s, f1_s = eval_group(single_files)
    total_gt_multi, tp_m, fp_m, fn_m, prec_m, rec_m, f1_m = eval_group(multi_files)

    print("\n=== By file type (defect-level) ===")
    print(
        f"Single-defect files: {len(single_files)}, "
        f"GT defects={total_gt_single}, "
        f"predicted={tp_s + fp_s}, "
        f"correct={tp_s}, "
        f"coverage={rec_s:.4f}, precision={prec_s:.4f}, F1={f1_s:.4f}"
    )
    print(
        f"Multi-defect files : {len(multi_files)}, "
        f"GT defects={total_gt_multi}, "
        f"predicted={tp_m + fp_m}, "
        f"correct={tp_m}, "
        f"coverage={rec_m:.4f}, precision={prec_m:.4f}, F1={f1_m:.4f}"
    )

    summary = {
        "total_files": total_files,
        "total_gt_rules": total_gt_rules,
        "total_gt_defects": total_gt_defects,
        "exact_match_files": exact_match_files,
        "parse_fail_files": parse_fail_files,
        "tp_rules": tp_rules,
        "fp_rules": fp_rules,
        "fn_rules": fn_rules,
        "precision_rules": precision_rules,
        "recall_rules": recall_rules,
        "f1_rules": f1_rules,
        "tp_defects": tp_def,
        "fp_defects": fp_def,
        "fn_defects": fn_def,
        "precision_defects": precision_def,
        "recall_defects": recall_def,
        "f1_defects": f1_def,
        "all_rules": all_rules,
        "timestamp": timestamp,
    }

    per_file_json = []
    for res in file_results:
        per_file_json.append(
            {
                "filename": res.filename,
                "gt_defects": [
                    {"rule": rule, "line": line} for rule, line in sorted(res.gt_defects)
                ],
                "pred_defects": [
                    {"rule": rule, "line": line} for rule, line in sorted(res.pred_defects)
                ],
                "parse_ok": res.parse_ok,
                "raw_output": res.raw_output,
                "trace": res.trace,
            }
        )

    # Write snapshot for this run (timestamped directory)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_dir / "per_file_results.json").open("w", encoding="utf-8") as f:
        json.dump(per_file_json, f, ensure_ascii=False, indent=2)

    # Also write a full JSON snapshot as NDJSON in the run directory
    run_ndjson = out_dir / "per_file_results.ndjson"
    with run_ndjson.open("w", encoding="utf-8") as f:
        for rec in per_file_json:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    # And keep a \"latest\" summary alongside the incremental NDJSON
    latest_dir = RESULT_ROOT / "latest"
    with (latest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (latest_dir / "per_file_results.json").open("w", encoding="utf-8") as f:
        json.dump(per_file_json, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_dir} (snapshot) and {latest_dir} (latest)")


if __name__ == "__main__":
    evaluate()
