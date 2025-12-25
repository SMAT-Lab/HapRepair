#!/usr/bin/env python3
"""
Summarize an RQ1 sampling/label JSON file and print paper-ready stats.

Input format: the JSON produced by scripts/sample_homecheck_findings.py
or revision/rq1_homecheck_samples_expert.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def clopper_pearson_lower_bound_all_successes(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return (alpha / 2.0) ** (1.0 / n)


@dataclass(frozen=True)
class Summary:
    n: int
    tp: int
    fp: int
    precision: Optional[float]
    cp95_lower_bound: Optional[float]
    by_category: Counter
    by_severity: Counter
    by_rule: Counter


def load_samples(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise ValueError("JSON missing 'samples' list")
    return samples


def summarize(samples: List[Dict[str, Any]]) -> Summary:
    n = len(samples)
    tp = 0
    fp = 0
    by_category: Counter = Counter()
    by_severity: Counter = Counter()
    by_rule: Counter = Counter()

    for s in samples:
        by_category[(s.get("category") or "unknown").strip()] += 1
        by_severity[(s.get("severity") or "unknown").strip()] += 1
        by_rule[(s.get("rule_id") or "unknown").strip()] += 1

        final = s.get("final_is_true_defect")
        if final is True:
            tp += 1
        elif final is False:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    cp95_lb = None
    if fp == 0 and tp == n and n > 0:
        cp95_lb = clopper_pearson_lower_bound_all_successes(n, alpha=0.05)

    return Summary(
        n=n,
        tp=tp,
        fp=fp,
        precision=precision,
        cp95_lower_bound=cp95_lb,
        by_category=by_category,
        by_severity=by_severity,
        by_rule=by_rule,
    )


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "json_path",
        type=Path,
        help="Path to rq1 sample JSON (e.g., revision/rq1_homecheck_samples_expert.json)",
    )
    ap.add_argument("--top-k-rules", type=int, default=10)
    args = ap.parse_args()

    samples = load_samples(args.json_path)
    s = summarize(samples)

    print(f"n={s.n} tp={s.tp} fp={s.fp} precision={format_pct(s.precision)}")
    if s.cp95_lower_bound is not None:
        print(f"CP95_lower_bound={format_pct(s.cp95_lower_bound)}")
    print("by_category:", dict(s.by_category))
    print("by_severity:", dict(s.by_severity))
    print(f"n_rules={len(s.by_rule)} top_rules:")
    for rid, c in s.by_rule.most_common(args.top_k_rules):
        print(f"  {c}\t{rid}")


if __name__ == "__main__":
    main()

