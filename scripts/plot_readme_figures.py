"""Regenerate README figures from summary/gpt-5.1_remaining_defects.md.

Outputs:
  fig/project_defects_bar.png - per-project initial defect counts
  fig/results.png             - total remaining defects across rounds
  fig/rule_top10_bar.png      - top-10 rules by initial defect count
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "summary" / "gpt-5.1_remaining_defects.md"
FIG = ROOT / "fig"


def parse_table(lines: list[str], header_key: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for line in lines:
        if header_key in line and line.lstrip().startswith("|"):
            in_table = True
            continue
        if in_table:
            if not line.lstrip().startswith("|"):
                if line.strip() == "":
                    continue
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("- "):
                continue
            rows.append(cells)
    return rows


def to_int(x: str) -> int:
    return int(x) if x.isdigit() else 0


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()

    project_rows = parse_table(lines, "Project")
    rule_rows = parse_table(lines, "Rule")

    projects = [(r[0], to_int(r[1])) for r in project_rows if r[0] != "合计"]
    projects.sort(key=lambda x: x[1], reverse=True)

    total_row = next(r for r in project_rows if r[0] == "合计")
    baseline_total = to_int(total_row[1])
    round_totals = [to_int(total_row[i]) for i in range(3, 9)]
    round_totals = [v for v in round_totals if v or round_totals.index(v) < 5]
    round_labels = ["Baseline"] + [f"Round {i+1}" for i in range(len(round_totals))]
    round_values = [baseline_total] + round_totals

    rules = [(r[0], to_int(r[1])) for r in rule_rows]
    rules.sort(key=lambda x: x[1], reverse=True)
    top_rules = rules[:10]

    # --- figure 1: per-project defect bar ---
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [p[0] for p in projects]
    values = [p[1] for p in projects]
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Initial defects")
    ax.set_title(f"Per-project initial defects (N={len(projects)}, total={baseline_total})")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "project_defects_bar.png", dpi=150)
    plt.close(fig)

    # --- figure 2: rounds progression ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(round_labels, round_values, marker="o", color="#E45756", linewidth=2)
    for x, y in zip(round_labels, round_values):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_ylabel("Remaining defects")
    ax.set_title("HapRepair (gpt-5.1): remaining defects across repair rounds")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    final = round_values[-1]
    fixed_pct = 100.0 * (baseline_total - final) / baseline_total
    ax.text(
        0.98,
        0.95,
        f"Fixed: {baseline_total - final}/{baseline_total} ({fixed_pct:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F4F4", edgecolor="#BBBBBB"),
    )
    fig.tight_layout()
    fig.savefig(FIG / "results.png", dpi=150)
    plt.close(fig)

    # --- figure 3: top-10 rules ---
    fig, ax = plt.subplots(figsize=(10, 6))
    rule_names = [r[0].replace("performance/", "perf/").replace("security/", "sec/") for r in top_rules]
    rule_values = [r[1] for r in top_rules]
    ax.barh(range(len(rule_values))[::-1], rule_values, color="#72B7B2")
    ax.set_yticks(range(len(rule_values))[::-1])
    ax.set_yticklabels(rule_names, fontsize=9)
    ax.set_xlabel("Initial defect count")
    ax.set_title("Top-10 rules by initial defect count")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(rule_values):
        ax.text(v, len(rule_values) - 1 - i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "rule_top10_bar.png", dpi=150)
    plt.close(fig)

    print("Wrote:")
    for name in ("project_defects_bar.png", "results.png", "rule_top10_bar.png"):
        print(f"  {FIG / name}")


if __name__ == "__main__":
    main()
