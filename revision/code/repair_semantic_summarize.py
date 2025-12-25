#!/usr/bin/env python3
"""
Summarize manual semantic-eval labels for HapRepair repairs.

This script consumes:
  - candidates.jsonl (population strata sizes)
  - labels.csv (filled from labels_template.csv; one label per sample_id)

It outputs:
  - summary.json (counts + weighted correctness estimate + CI)
  - LaTeX table snippet (optional) that can be dropped into the paper.

Labels:
  - Correct
  - Suspicious
  - Incorrect

For correctness rate we treat:
  success = Correct
  failure = Suspicious or Incorrect
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * ((phat * (1 - phat) / n) + (z * z) / (4 * n * n)) ** 0.5) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}\\%"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize semantic-eval labels with stratified weighting + CI.")
    ap.add_argument("--candidates", type=Path, required=True, help="Path to candidates.jsonl.")
    ap.add_argument("--labels", type=Path, required=True, help="Path to filled labels CSV.")
    ap.add_argument("--seed", type=int, default=20251220, help="Bootstrap seed (default: 20251220).")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (default: 2000).")
    ap.add_argument("--out-json", type=Path, default=None, help="Write summary JSON to this path.")
    ap.add_argument("--latex-out", type=Path, default=None, help="Write LaTeX table snippet to this path.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    candidates = _read_jsonl(args.candidates)
    pop_by_stratum: Counter[str] = Counter()
    for c in candidates:
        sk = c.get("stratum_key")
        if isinstance(sk, str) and sk:
            pop_by_stratum[sk] += 1
    pop_total = sum(pop_by_stratum.values())
    if pop_total == 0:
        raise SystemExit("Population strata empty; candidates.jsonl missing stratum_key?")

    labels_by_id: Dict[str, str] = {}
    with args.labels.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("sample_id") or "").strip()
            lab = (row.get("label") or "").strip()
            if not sid or not lab:
                continue
            labels_by_id[sid] = lab

    # Join labels back to candidates (we assume sample_id == candidate_id)
    labeled: List[Dict[str, Any]] = []
    for c in candidates:
        cid = c.get("candidate_id")
        if not isinstance(cid, str):
            continue
        lab = labels_by_id.get(cid)
        if not lab:
            continue
        row = dict(c)
        row["label"] = lab
        labeled.append(row)

    if not labeled:
        raise SystemExit("No labeled rows matched candidates. Did you fill labels_template.csv and keep sample_id?")

    label_counts = Counter(r["label"] for r in labeled)
    n = len(labeled)
    k_correct = sum(1 for r in labeled if r["label"].lower() == "correct")

    # Unweighted (sample) Wilson CI for reference
    sample_lo, sample_hi = _wilson_interval(k_correct, n)

    # Weighted correctness estimate: sum(pop_w * rate_stratum)
    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in labeled:
        by_stratum[str(r["stratum_key"])].append(r)

    # For strata with no labeled items, we cannot estimate rate; we drop them
    # and renormalize weights (report coverage).
    covered_pop = sum(pop_by_stratum[s] for s in by_stratum.keys() if s in pop_by_stratum)
    covered_weight = covered_pop / pop_total

    def weighted_estimate(sampled_by_stratum: Dict[str, List[Dict[str, Any]]]) -> float:
        num = 0.0
        denom = 0.0
        for s, items in sampled_by_stratum.items():
            pop = pop_by_stratum.get(s, 0)
            if pop <= 0 or not items:
                continue
            denom += pop
            rate = sum(1 for r in items if r["label"].lower() == "correct") / len(items)
            num += pop * rate
        return num / denom if denom > 0 else 0.0

    weighted = weighted_estimate(by_stratum)

    # Stratified bootstrap CI (within-stratum resampling)
    boot: List[float] = []
    strata_keys = [s for s in by_stratum.keys() if s in pop_by_stratum]
    for _ in range(args.bootstrap):
        resampled: Dict[str, List[Dict[str, Any]]] = {}
        for s in strata_keys:
            items = by_stratum[s]
            if not items:
                continue
            resampled[s] = [rng.choice(items) for _ in range(len(items))]
        boot.append(weighted_estimate(resampled))
    boot.sort()
    lo = boot[int(0.025 * len(boot))] if boot else 0.0
    hi = boot[int(0.975 * len(boot))] if boot else 0.0

    summary = {
        "n_labeled": n,
        "label_counts": dict(label_counts),
        "sample_correct": k_correct,
        "sample_correct_rate": k_correct / n,
        "sample_correct_wilson_95": [sample_lo, sample_hi],
        "weighted_correct_rate": weighted,
        "weighted_bootstrap_95": [lo, hi],
        "population_total_candidates": pop_total,
        "population_covered_weight": covered_weight,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.latex_out:
        correct = label_counts.get("Correct", 0)
        suspicious = label_counts.get("Suspicious", 0)
        incorrect = label_counts.get("Incorrect", 0)
        # Be tolerant to casing
        if correct == 0 and any(k.lower() == "correct" for k in label_counts):
            correct = sum(v for k, v in label_counts.items() if k.lower() == "correct")
        if suspicious == 0 and any(k.lower() == "suspicious" for k in label_counts):
            suspicious = sum(v for k, v in label_counts.items() if k.lower() == "suspicious")
        if incorrect == 0 and any(k.lower() == "incorrect" for k in label_counts):
            incorrect = sum(v for k, v in label_counts.items() if k.lower() == "incorrect")

        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        table = "\n".join(
            [
                "% Auto-generated by repair_semantic_summarize.py",
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{\\refine{Manual semantic validation of repairs on a stratified sample.}}",
                "\\resizebox{\\columnwidth}{!}{",
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "\\textbf{Label} & \\textbf{Count} & \\textbf{Share} & \\textbf{Lower} & \\textbf{Upper} \\\\",
                "\\midrule",
                f"Correct & {correct} & {_fmt_pct(correct/n)} & -- & -- \\\\",
                f"Suspicious & {suspicious} & {_fmt_pct(suspicious/n)} & -- & -- \\\\",
                f"Incorrect & {incorrect} & {_fmt_pct(incorrect/n)} & -- & -- \\\\",
                "\\midrule",
                (
                    "Correctness (weighted) & "
                    f"{correct}/{n} & {_fmt_pct(summary['weighted_correct_rate'])} & "
                    f"{_fmt_pct(summary['weighted_bootstrap_95'][0])} & {_fmt_pct(summary['weighted_bootstrap_95'][1])} \\\\"
                ),
                "\\bottomrule",
                "\\end{tabular}}",
                "\\label{tab:repair_semantic_validation}",
                "\\end{table}",
                "",
            ]
        )
        args.latex_out.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()

