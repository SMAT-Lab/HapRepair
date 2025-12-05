#!/usr/bin/env python3
"""
Evaluate Codex detection results against GT for ace_ets_module_list02
WITHOUT re-running Codex.

It reads:
- GT from /home/LLMCodeRepair/revision/ace_ets_module_list02_gt.json
- Latest Codex outputs from
  /home/LLMCodeRepair/revision/ace_ets_module_list02_codex_eval/latest/per_file_results.ndjson

Metrics are computed at defect-level (rule + line), and split into:
- overall
- single-defect files (GT has exactly 1 defect)
- multi-defect files (GT has more than 1 defect)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR / "ace_ets_module_list02"
GT_PATH = ROOT_DIR / "ace_ets_module_list02_gt.json"
RESULT_ROOT = ROOT_DIR / "ace_ets_module_list02_codex_eval"
LATEST_DIR = RESULT_ROOT / "latest"
LATEST_NDJSON = LATEST_DIR / "per_file_results.ndjson"


@dataclass
class EvalItem:
    filename: str
    gt_defects: Set[Tuple[str, int]]
    pred_defects: Set[Tuple[str, int]]


def load_ground_truth() -> Dict[str, Set[Tuple[str, int]]]:
    """Load GT defect sets (rule+line) per file."""
    if not GT_PATH.is_file():
        raise FileNotFoundError(f"GT JSON not found: {GT_PATH}")

    with GT_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)

    gt_def: Dict[str, Set[Tuple[str, int]]] = {}
    for fname, meta in raw.items():
        defect_items = meta.get("defects") or []
        s: Set[Tuple[str, int]] = set()
        for d in defect_items:
            rule = d.get("rule")
            line = d.get("line")
            if isinstance(rule, str) and isinstance(line, int):
                s.add((rule, line))
        gt_def[fname] = s
    return gt_def


def load_results() -> Dict[str, Set[Tuple[str, int]]]:
    """Load predicted defects (rule+line) per file from latest NDJSON results."""
    if not LATEST_NDJSON.is_file():
        raise FileNotFoundError(f"Latest NDJSON results not found: {LATEST_NDJSON}")

    results: Dict[str, Set[Tuple[str, int]]] = {}
    with LATEST_NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = rec.get("filename")
            if not isinstance(fname, str):
                continue
            pred_def_list = rec.get("pred_defects") or []
            s: Set[Tuple[str, int]] = set()
            for d in pred_def_list:
                if isinstance(d, dict):
                    rule = d.get("rule")
                    line_val = d.get("line")
                    if isinstance(rule, str) and isinstance(line_val, int):
                        s.add((rule, line_val))
            results[fname] = s
    return results


def eval_group(items: List[EvalItem]) -> Tuple[int, int, int, int, float, float, float]:
    """Return (total_gt, tp, fp, fn, precision, recall, f1) for a group."""
    tp = fp = fn = 0
    total_gt = 0
    for it in items:
        tp_set = it.gt_defects & it.pred_defects
        fp_set = it.pred_defects - it.gt_defects
        fn_set = it.gt_defects - it.pred_defects
        total_gt += len(it.gt_defects)
        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return total_gt, tp, fp, fn, prec, rec, f1


def main() -> None:
    gt_defects_map = load_ground_truth()
    pred_defects_map = load_results()

    items: List[EvalItem] = []
    for fname, gt_def in gt_defects_map.items():
        pred_def = pred_defects_map.get(fname, set())
        items.append(EvalItem(filename=fname, gt_defects=gt_def, pred_defects=pred_def))

    # Overall
    total_gt, tp, fp, fn, prec, rec, f1 = eval_group(items)

    # Split single vs multi defect files
    single_items = [it for it in items if len(it.gt_defects) == 1]
    multi_items = [it for it in items if len(it.gt_defects) > 1]

    total_gt_s, tp_s, fp_s, fn_s, prec_s, rec_s, f1_s = eval_group(single_items)
    total_gt_m, tp_m, fp_m, fn_m, prec_m, rec_m, f1_m = eval_group(multi_items)

    print("=== Defect-level evaluation (using latest Codex results) ===")
    print(f"Files (GT): {len(items)}  "
          f"Single-defect files: {len(single_items)},  "
          f"Multi-defect files: {len(multi_items)}")

    print("\n-- Overall --")
    print(f"GT defects     : {total_gt}")
    print(f"Predicted      : {tp + fp}")
    print(f"Correctly found: {tp}")
    print(f"Coverage       : {rec:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"F1             : {f1:.4f}")

    print("\n-- Single-defect files --")
    print(f"GT defects     : {total_gt_s}")
    print(f"Predicted      : {tp_s + fp_s}")
    print(f"Correctly found: {tp_s}")
    print(f"Coverage       : {rec_s:.4f}")
    print(f"Precision      : {prec_s:.4f}")
    print(f"F1             : {f1_s:.4f}")

    print("\n-- Multi-defect files --")
    print(f"GT defects     : {total_gt_m}")
    print(f"Predicted      : {tp_m + fp_m}")
    print(f"Correctly found: {tp_m}")
    print(f"Coverage       : {rec_m:.4f}")
    print(f"Precision      : {prec_m:.4f}")
    print(f"F1             : {f1_m:.4f}")


if __name__ == "__main__":
    main()
