#!/usr/bin/env python3
"""
Delta Check: summarize code-change magnitudes for oracle-resolved findings.

Goal:
  Provide a simple, reproducible check that the compliance-resolution gains are
  not dominated by "gaming" behaviors such as massive code deletion.

Method:
  1) Count findings in a baseline CodeLinter scan (default: deepseek-chat round_1).
  2) Count remaining findings in the final scan after repair (default: gpt-5.1 round_5_after_round5).
  3) Attribute each "fixed finding" to (project, rel_path) by counting per-file per-rule decreases.
  4) Compute diff statistics between the original project file and the final repaired snapshot file.
  5) Aggregate patch-size/deletion statistics, weighted by the number of fixed findings attributed to the file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import difflib


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
DEFAULT_TARGETS = REPO_ROOT / "revision" / "target_projects_haprepair.json"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "codelinter_openharmony"
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "revision" / "fixed_projects"

VALID_CATEGORIES = {"performance", "security"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
FILE_HEADER_REGEX = re.compile(r"^(\/.+)\(\d+\)$")


@dataclass(frozen=True)
class FindingGroupKey:
    project: str
    rel_path: str
    category: str
    rule_id: str


@dataclass
class FileDiffStats:
    changed: int
    added: int
    deleted: int

    @property
    def total(self) -> int:
        return int(self.changed + self.added + self.deleted)

    @property
    def net(self) -> int:
        return int(self.added - self.deleted)

    @property
    def add_del_total(self) -> int:
        return int(self.added + self.deleted)

    def deleted_share_of_add_del(self) -> float:
        denom = self.add_del_total
        return (self.deleted / denom) if denom > 0 else 0.0


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _rel_path_from_log_file_path(file_path: str, project_name: str) -> Optional[str]:
    needle = f"/{project_name}/"
    idx = file_path.rfind(needle)
    if idx == -1:
        return None
    return file_path[idx + len(needle) :].lstrip("/")


def _diff_stats(before: Sequence[str], after: Sequence[str]) -> FileDiffStats:
    sm = difflib.SequenceMatcher(a=before, b=after)
    added = deleted = changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added += (j2 - j1)
        elif tag == "delete":
            deleted += (i2 - i1)
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)
    return FileDiffStats(changed=int(changed), added=int(added), deleted=int(deleted))


def parse_codelinter_counts(log_path: Path, project: str) -> Counter[FindingGroupKey]:
    if not log_path.is_file():
        raise FileNotFoundError(f"CodeLinter log not found: {log_path}")

    counts: Counter[FindingGroupKey] = Counter()
    current_file: Optional[str] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = ANSI_ESCAPE_RE.sub("", raw_line).rstrip("\n")
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

            meta = line[at_index + 1 :].strip()
            if "/" not in meta:
                continue

            category, rule_id = meta.split("/", 1)
            category = category.strip()
            rule_id = rule_id.strip()
            if category not in VALID_CATEGORIES or not rule_id:
                continue

            rel = _rel_path_from_log_file_path(current_file, project)
            if rel is None:
                continue

            counts[FindingGroupKey(project=project, rel_path=rel, category=category, rule_id=rule_id)] += 1

    return counts


def _pct(x: float) -> float:
    return 100.0 * x


def _quantile(sorted_vals: List[int], q: float) -> int:
    if not sorted_vals:
        return 0
    q = max(0.0, min(1.0, q))
    idx = int(math.ceil(q * len(sorted_vals)) - 1)
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return int(sorted_vals[idx])


def _weighted_expand(values_and_weights: Iterable[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    for v, w in values_and_weights:
        if w <= 0:
            continue
        out.extend([int(v)] * int(w))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Delta Check summarization for oracle-resolved findings.")
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="Path to target_projects_haprepair.json.")
    ap.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="logs/codelinter_openharmony root.")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT, help="revision/fixed_projects root.")

    ap.add_argument("--baseline-model-dir", type=str, default="deepseek-chat", help="Model dir used for baseline scan logs.")
    ap.add_argument("--baseline-round", type=int, default=1, help="Baseline round index (default: 1).")

    ap.add_argument("--final-model-dir", type=str, default="gpt-5.1", help="Model dir used for final (repaired) scan logs.")
    ap.add_argument("--final-round", type=int, default=5, help="Final round index (default: 5).")
    ap.add_argument(
        "--allow-missing-final-logs",
        action="store_true",
        help=(
            "When set, if round_<final>_after_round<final>/<project>.log is missing, fall back to the latest "
            "available round_r_after_roundr/<project>.log for r<=final_round (common when a project reaches 0 "
            "defects early and the driver stops writing later logs)."
        ),
    )

    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "summary" / "gpt-5.1_delta_check.json")
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=REPO_ROOT / "-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps" / "delta_check_table.tex",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    if not isinstance(targets, list) or not targets:
        raise SystemExit(f"Malformed targets list: {args.targets}")

    baseline_config_total = sum(int(p.get("total_defects", 0) or 0) for p in targets)

    baseline_dir = args.log_root / args.baseline_model_dir / f"round_{args.baseline_round}"
    final_model_root = args.log_root / args.final_model_dir

    def resolve_final_log(project: str) -> Path:
        preferred = final_model_root / f"round_{args.final_round}_after_round{args.final_round}" / f"{project}.log"
        if preferred.is_file():
            return preferred
        if not args.allow_missing_final_logs:
            return preferred
        for r in range(args.final_round, 0, -1):
            cand = final_model_root / f"round_{r}_after_round{r}" / f"{project}.log"
            if cand.is_file():
                return cand
        return preferred

    baseline_counts: Counter[FindingGroupKey] = Counter()
    final_counts: Counter[FindingGroupKey] = Counter()

    for proj in targets:
        project = proj.get("name")
        root_path = proj.get("root_path")
        if not isinstance(project, str) or not isinstance(root_path, str):
            continue

        base_log = baseline_dir / f"{project}.log"
        fin_log = resolve_final_log(project)
        baseline_counts.update(parse_codelinter_counts(base_log, project))
        if fin_log.is_file():
            final_counts.update(parse_codelinter_counts(fin_log, project))
        else:
            if not args.allow_missing_final_logs:
                raise FileNotFoundError(f"CodeLinter log not found: {fin_log}")

    baseline_total = int(sum(baseline_counts.values()))
    final_total = int(sum(final_counts.values()))
    net_fixed_total = int(baseline_total - final_total)
    if net_fixed_total < 0:
        raise SystemExit(f"Computed negative net_fixed_total={net_fixed_total}; check baseline/final log inputs.")

    resolved_total = 0
    introduced_total = 0
    resolved_by_file: Dict[Tuple[str, str], int] = defaultdict(int)
    resolved_by_category: Counter[str] = Counter()
    introduced_by_category: Counter[str] = Counter()

    all_keys = set(baseline_counts.keys()) | set(final_counts.keys())
    for k in all_keys:
        before_n = int(baseline_counts.get(k, 0))
        after_n = int(final_counts.get(k, 0))
        d = int(before_n - after_n)
        if d > 0:
            resolved_total += d
            resolved_by_file[(k.project, k.rel_path)] += d
            resolved_by_category[k.category] += d
        elif d < 0:
            introduced_total += (-d)
            introduced_by_category[k.category] += (-d)

    if (resolved_total - introduced_total) != net_fixed_total:
        raise SystemExit(
            "Inconsistent accounting: resolved_total - introduced_total != net_fixed_total "
            f"({resolved_total} - {introduced_total} != {net_fixed_total})."
        )

    final_snapshot_root = args.snapshot_root / args.final_model_dir / f"round_{args.final_round}"

    per_file_rows: List[Dict[str, object]] = []
    missing_after_files: List[Tuple[str, str]] = []

    for proj in targets:
        project = proj.get("name")
        root_path = proj.get("root_path")
        if not isinstance(project, str) or not isinstance(root_path, str):
            continue

        original_root = Path(root_path).resolve()
        output_root = (final_snapshot_root / project).resolve()

        rel_paths = [rp for (p, rp), n in resolved_by_file.items() if p == project and n > 0]
        for rel_path in rel_paths:
            weight = int(resolved_by_file[(project, rel_path)])
            before_file = (original_root / rel_path).resolve()
            after_file = (output_root / rel_path).resolve()

            before_text = _read_text(before_file) if before_file.is_file() else ""
            after_text = _read_text(after_file) if after_file.is_file() else ""
            if not after_file.is_file():
                missing_after_files.append((project, rel_path))

            before_lines = before_text.splitlines()
            after_lines = after_text.splitlines()

            stats = _diff_stats(before_lines, after_lines)

            lines_before = int(len(before_lines))
            lines_after = int(len(after_lines))
            deleted_ratio = (stats.deleted / lines_before) if lines_before > 0 else 0.0

            per_file_rows.append(
                {
                    "project": project,
                    "rel_path": rel_path,
                    "fixed_findings": weight,
                    "lines_before": lines_before,
                    "lines_after": lines_after,
                    "changed": stats.changed,
                    "added": stats.added,
                    "deleted": stats.deleted,
                    "total_edit": stats.total,
                    "net": stats.net,
                    "deleted_share_add_del": stats.deleted_share_of_add_del(),
                    "deleted_share_of_file": deleted_ratio,
                    "file_missing_after": (not after_file.is_file()),
                }
            )

    if not per_file_rows:
        raise SystemExit("No per-file rows produced; unexpected (fixed_total > 0 but no attributed files).")

    file_level_total = len(per_file_rows)
    file_level_resolved_weight = sum(int(r["fixed_findings"]) for r in per_file_rows)
    if file_level_resolved_weight != resolved_total:
        raise SystemExit("Internal error: file-level weighted total does not match resolved_total.")

    def count_weight(pred) -> int:
        return sum(int(r["fixed_findings"]) for r in per_file_rows if pred(r))

    def count_files(pred) -> int:
        return sum(1 for r in per_file_rows if pred(r))

    thresholds = {
        "deleted_ge_50": lambda r: int(r["deleted"]) >= 50,
        "deleted_ge_100": lambda r: int(r["deleted"]) >= 100,
        "deleted_ge_200": lambda r: int(r["deleted"]) >= 200,
        "delete_dominant_ge_50": lambda r: (int(r["deleted"]) >= 50 and float(r["deleted_share_add_del"]) >= 0.9),
        "file_deleted_or_missing": lambda r: bool(r["file_missing_after"]),
        "deleted_ge_50_and_ge_30pct_file": lambda r: (
            int(r["deleted"]) >= 50 and float(r["deleted_share_of_file"]) >= 0.30
        ),
        "total_edit_ge_200": lambda r: int(r["total_edit"]) >= 200,
    }

    by_case_counts = {k: int(count_weight(v)) for k, v in thresholds.items()}
    by_file_counts = {k: int(count_files(v)) for k, v in thresholds.items()}

    def expand_metric(metric: str) -> List[int]:
        return _weighted_expand((int(r[metric]), int(r["fixed_findings"])) for r in per_file_rows)

    total_edit_cases = sorted(expand_metric("total_edit"))
    deleted_cases = sorted(expand_metric("deleted"))
    net_cases = sorted(expand_metric("net"))

    def summarise_cases(vals: List[int]) -> Dict[str, int]:
        if not vals:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        return {
            "p50": _quantile(vals, 0.50),
            "p90": _quantile(vals, 0.90),
            "p95": _quantile(vals, 0.95),
            "p99": _quantile(vals, 0.99),
            "max": int(vals[-1]),
        }

    summary = {
        "baseline": {
            "model_dir": args.baseline_model_dir,
            "round": args.baseline_round,
            "total_findings": baseline_total,
        },
        "final": {
            "model_dir": args.final_model_dir,
            "round": args.final_round,
            "total_findings": final_total,
        },
        "baseline_config": {
            "targets_path": str(args.targets),
            "total_findings": int(baseline_config_total),
        },
        "net_effect": {
            "net_fixed_findings": net_fixed_total,
            "resolved_findings": resolved_total,
            "introduced_findings": introduced_total,
            "resolved_by_category": dict(resolved_by_category),
            "introduced_by_category": dict(introduced_by_category),
        },
        "attribution": {
            "file_units_with_fixes": file_level_total,
            "missing_after_files": len(missing_after_files),
        },
        "delta_check": {
            "case_quantiles_total_edit_lines": summarise_cases(total_edit_cases),
            "case_quantiles_deleted_lines": summarise_cases(deleted_cases),
            "case_quantiles_net_lines": summarise_cases(net_cases),
            "case_mean_total_edit_lines": float(statistics.mean(total_edit_cases)) if total_edit_cases else 0.0,
            "case_mean_deleted_lines": float(statistics.mean(deleted_cases)) if deleted_cases else 0.0,
            "threshold_counts_by_case": by_case_counts,
            "threshold_counts_by_file": by_file_counts,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    resolved_total_f = float(resolved_total) if resolved_total > 0 else 1.0
    def fmt_pct(count: int) -> str:
        return f"{_pct(count / resolved_total_f):.2f}\\%"

    suspicious_cases = max(
        by_case_counts.get("file_deleted_or_missing", 0),
        by_case_counts.get("deleted_ge_200", 0),
        by_case_counts.get("delete_dominant_ge_50", 0),
        by_case_counts.get("deleted_ge_50_and_ge_30pct_file", 0),
    )
    conservative_net_fixed = max(0, net_fixed_total - suspicious_cases)
    denom = int(baseline_config_total) if baseline_config_total else baseline_total
    conservative_rate = conservative_net_fixed / denom if denom else 0.0

    tex_lines = [
        "% Auto-generated by revision/code/delta_check_summarize.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\refine{Delta check on oracle-resolved findings (to rule out deletion-based gaming).}}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Metric (finding-weighted)} & \\textbf{Value} & \\textbf{Share} \\\\",
        "\\midrule",
        f"Net fixed findings (baseline minus final) & {net_fixed_total} & -- \\\\",
        f"Resolved findings (decreases; used for weighting) & {resolved_total} & -- \\\\",
        f"Introduced findings (increases) & {introduced_total} & -- \\\\",
        f"Median deleted lines per resolved finding's file & {summary['delta_check']['case_quantiles_deleted_lines']['p50']} & -- \\\\",
        f"95th percentile deleted lines & {summary['delta_check']['case_quantiles_deleted_lines']['p95']} & -- \\\\",
        f"Deleted \\(\\ge 100\\) lines & {by_case_counts['deleted_ge_100']} & {fmt_pct(by_case_counts['deleted_ge_100'])} \\\\",
        f"Files with deleted \\(\\ge 100\\) lines & {by_file_counts['deleted_ge_100']}/{file_level_total} & -- \\\\",
        f"Deleted \\(\\ge 200\\) lines & {by_case_counts['deleted_ge_200']} & {fmt_pct(by_case_counts['deleted_ge_200'])} \\\\",
        f"Deletion-dominant (deleted \\(\\ge 50\\) and \\(\\ge 90\\%\\) of add/del) & {by_case_counts['delete_dominant_ge_50']} & {fmt_pct(by_case_counts['delete_dominant_ge_50'])} \\\\",
        "\\midrule",
        (
            "Conservative resolution rate after dropping all suspicious cases & "
            f"{conservative_net_fixed}/{denom} & {_pct(conservative_rate):.2f}\\% \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}}",
        "\\label{tab:delta_check}",
        "\\end{table}",
        "",
    ]

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(tex_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
