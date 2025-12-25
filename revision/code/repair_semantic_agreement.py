#!/usr/bin/env python3
"""
Compute inter-judge agreement metrics for LLM semantic evaluation runs.

Input: a semantic-eval run directory containing `llm_judgments.jsonl`
Output: prints a JSON report and optionally writes it to `agreement.json`
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LABELS = ("Correct", "Suspicious", "Incorrect")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cohen_kappa(a: List[str], b: List[str], labels: Tuple[str, ...]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    ca = Counter(a)
    cb = Counter(b)
    pe = 0.0
    for lab in labels:
        pe += (ca.get(lab, 0) / n) * (cb.get(lab, 0) / n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)

def _pabak_k(po: float, k: int) -> Optional[float]:
    if k <= 1:
        return None
    # Generalized prevalence-adjusted bias-adjusted kappa for k categories.
    # Reduces to 2*Po-1 for k=2.
    return (po - 1.0 / k) / (1.0 - 1.0 / k)


def _gwet_ac1(a: List[str], b: List[str], labels: Tuple[str, ...]) -> Optional[float]:
    """
    Gwet's AC1 for nominal ratings.
    This is often more stable than Cohen's kappa under strong label prevalence.
    """
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n

    ca = Counter(a)
    cb = Counter(b)
    # Average marginal proportion across raters for each label.
    p = {lab: 0.5 * (ca.get(lab, 0) / n + cb.get(lab, 0) / n) for lab in labels}
    k = len(labels)
    if k <= 1:
        return None
    pe = sum(p[lab] * (1.0 - p[lab]) for lab in labels) / (k - 1)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def _binarize(label: str) -> str:
    return "Correct" if label == "Correct" else "NotCorrect"


def _risk_bucket(label: str) -> str:
    return "Incorrect" if label == "Incorrect" else "NotIncorrect"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute agreement metrics between two LLM judges.")
    ap.add_argument("--run-dir", type=Path, required=True, help="semantic_eval run directory.")
    ap.add_argument("--out", type=Path, default=None, help="If set, write JSON report to this file.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    judgments_path = run_dir / "llm_judgments.jsonl"
    if not judgments_path.is_file():
        raise SystemExit(f"llm_judgments.jsonl not found: {judgments_path}")

    rows = _read_jsonl(judgments_path)
    a_labels: List[str] = []
    b_labels: List[str] = []
    missing = 0

    for r in rows:
        a = (r.get("judge_a") or {}).get("label")
        b = (r.get("judge_b") or {}).get("label")
        if a not in LABELS or b not in LABELS:
            missing += 1
            continue
        a_labels.append(a)
        b_labels.append(b)

    n_total = len(rows)
    n_used = len(a_labels)
    agreement = sum(1 for x, y in zip(a_labels, b_labels) if x == y)
    agreement_rate = (agreement / n_used) if n_used else None

    # Confusion matrix: A (rows) x B (cols)
    cm: Dict[str, Dict[str, int]] = {la: {lb: 0 for lb in LABELS} for la in LABELS}
    for x, y in zip(a_labels, b_labels):
        cm[x][y] += 1

    kappa_3 = _cohen_kappa(a_labels, b_labels, labels=LABELS)
    pabak_3 = _pabak_k(agreement_rate, k=len(LABELS)) if agreement_rate is not None else None
    ac1_3 = _gwet_ac1(a_labels, b_labels, labels=LABELS)
    kappa_correct = _cohen_kappa(
        [_binarize(x) for x in a_labels],
        [_binarize(y) for y in b_labels],
        labels=("Correct", "NotCorrect"),
    )
    kappa_incorrect = _cohen_kappa(
        [_risk_bucket(x) for x in a_labels],
        [_risk_bucket(y) for y in b_labels],
        labels=("Incorrect", "NotIncorrect"),
    )

    report = {
        "run_dir": str(run_dir),
        "n_total": n_total,
        "n_used": n_used,
        "n_missing_or_invalid": missing,
        "labels": list(LABELS),
        "agreement_rate": agreement_rate,
        "cohen_kappa_3way": kappa_3,
        "pabak_3way": pabak_3,
        "gwet_ac1_3way": ac1_3,
        "cohen_kappa_correct_vs_not": kappa_correct,
        "cohen_kappa_incorrect_vs_not": kappa_incorrect,
        "judge_a_counts": dict(Counter(a_labels)),
        "judge_b_counts": dict(Counter(b_labels)),
        "confusion_matrix": cm,
    }

    out_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(out_text)
    if args.out is not None:
        out_path = args.out
    else:
        out_path = run_dir / "agreement.json"
    out_path.write_text(out_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
