#!/usr/bin/env python3
"""
Prepare a stratified sample for LLM-judged semantic evaluation at *file* granularity.

Rationale:
  - Finding-level validation can over-count correlated edits within a file.
  - File-level validation shows the LLM the full before/after file along with
    the set of static findings that disappeared in that file for a given round.

We define a "fixed finding" as present in the round_r CodeLinter log but absent
in the round_r_after_roundr log, matched by (category, rule_id, message) with
multiset subtraction per file (line/column are *not* used for matching to avoid
false "fixes" caused by line-shifts, e.g., inserting/removing blank lines).

Outputs under revision/semantic_eval/<run_id>/:
  - candidates.jsonl: eligible file-level candidates with strata metadata
  - sample.jsonl: sampled files incl. full before/after text + fixed findings list
  - labels_template.csv: CSV template for labeling (optional; LLM labeling uses a separate script)
  - sampling_meta.json: seed/quotas/skip statistics for reproducibility
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import difflib
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
DEFAULT_TARGETS = REPO_ROOT / "revision" / "target_projects_haprepair.json"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "codelinter_openharmony"
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "revision" / "fixed_projects"
DEFAULT_OUT_BASE = REPO_ROOT / "revision" / "semantic_eval"


SEVERITY_NORMALIZATION = {
    "error": "error",
    "warn": "warn",
    "warning": "warn",
    "suggestion": "suggestion",
}
VALID_CATEGORIES = {"performance", "security"}

FILE_HEADER_REGEX = re.compile(r"^(\/.+)\(\d+\)$")


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rel_path_from_log_file_path(file_path: str, project_name: str) -> Optional[str]:
    needle = f"/{project_name}/"
    idx = file_path.rfind(needle)
    if idx == -1:
        return None
    return file_path[idx + len(needle) :].lstrip("/")


def parse_codelinter_log(log_path: Path) -> List[Dict[str, Any]]:
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


@dataclass(frozen=True)
class FindingKey:
    rel_path: str
    line: int
    column: int
    category: str
    rule_id: str


FindingSig = Tuple[str, str, str]  # (category, rule_id, message)


@dataclass
class FileDiffStats:
    changed: int
    added: int
    deleted: int
    shape: str


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
    total = added + deleted + changed
    if total == 0:
        shape = "modify"
    elif added > 0 and deleted == 0 and changed == 0:
        shape = "add"
    elif deleted > 0 and added == 0 and changed == 0:
        shape = "delete"
    else:
        shape = "modify"
    return FileDiffStats(changed=changed, added=added, deleted=deleted, shape=shape)


def _unified_diff_text(
    before_lines: Sequence[str],
    after_lines: Sequence[str],
    *,
    rel_path: str,
    max_lines: int = 600,
) -> str:
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"before/{rel_path}",
            tofile=f"after/{rel_path}",
            lineterm="",
            n=3,
        )
    )
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... (diff truncated, total_lines={len(diff)})"]
    return "\n".join(diff)

def _bin_patch_size(total_changed_lines: int) -> str:
    return "single" if total_changed_lines <= 1 else "multi"


def _bin_stage(round_fixed: int) -> str:
    return "early" if round_fixed <= 2 else "late"


def _bin_context(file_defects_before: int) -> str:
    return "single_defect_file" if file_defects_before <= 1 else "multi_defect_file"


def _category_bin(perf: int, sec: int) -> str:
    if perf > 0 and sec > 0:
        return "both"
    if sec > 0:
        return "security_only"
    return "performance_only"


def _make_stratum_key(cat_bin: str, stage: str, patch_size: str, context: str) -> str:
    return f"{cat_bin}|{stage}|{patch_size}|{context}"


def _allocate_counts(total: int, groups: Dict[Tuple[str, ...], List[Any]], min_each: int = 0) -> Dict[Tuple[str, ...], int]:
    sizes = {k: len(v) for k, v in groups.items() if v}
    if not sizes:
        return {}
    grand = sum(sizes.values())
    targets: Dict[Tuple[str, ...], int] = {}
    for k, n in sizes.items():
        t = int(round(total * (n / grand)))
        if min_each and t < min_each:
            t = min_each
        targets[k] = min(t, n)

    while sum(targets.values()) > total:
        k = max(targets.keys(), key=lambda kk: targets[kk])
        if targets[k] <= 0:
            break
        targets[k] -= 1
    while sum(targets.values()) < total:
        k = max(sizes.keys(), key=lambda kk: sizes[kk] - targets.get(kk, 0))
        if targets.get(k, 0) >= sizes[k]:
            break
        targets[k] = targets.get(k, 0) + 1
    return targets


@dataclass
class FileCandidate:
    candidate_id: str
    model_dir: str
    round_fixed: int
    project: str
    rel_path: str
    stage_bin: str
    context_bin: str
    patch_shape: str
    patch_size_bin: str
    file_defects_before: int
    fixed_findings_count: int
    fixed_perf_count: int
    fixed_sec_count: int
    fixed_category_bin: str
    file_lines_before: int
    file_lines_after: int
    file_chars_before: int
    file_chars_after: int
    stratum_key: str
    input_root: str
    output_root: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare file-level semantic-eval samples for HapRepair repairs.")
    ap.add_argument("--model-dir", type=str, default="gpt-5.1", help="Model directory name under logs/snapshots.")
    ap.add_argument("--max-round", type=int, default=5, help="Maximum round index to include (default: 5).")
    ap.add_argument("--n-sample", type=int, default=50, help="Total number of files to sample (default: 50).")
    ap.add_argument("--seed", type=int, default=20251220, help="Random seed for sampling (default: 20251220).")
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="Path to target_projects_haprepair.json.")
    ap.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root of codelinter logs.")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT, help="Root of fixed snapshots.")
    ap.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Output base directory.")
    ap.add_argument("--max-lines", type=int, default=600, help="Skip files with > this many lines (default: 600).")
    ap.add_argument("--max-chars", type=int, default=60000, help="Skip files with > this many chars (default: 60000).")
    ap.add_argument("--write-md", action="store_true", help="Also write a human-friendly sample.md sheet.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    targets_raw = json.loads(_read_text(args.targets))
    if not isinstance(targets_raw, list) or not targets_raw:
        raise SystemExit(f"Malformed targets file (expected non-empty list): {args.targets}")

    run_id = f"{args.model_dir}_files_n{args.n_sample}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = (args.out_base / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[FileCandidate] = []
    skipped: List[Dict[str, Any]] = []

    for proj in targets_raw:
        project = proj.get("name")
        root_path = proj.get("root_path")
        if not isinstance(project, str) or not isinstance(root_path, str):
            continue
        original_root = Path(root_path).resolve()

        for r in range(1, args.max_round + 1):
            before_log = args.log_root / args.model_dir / f"round_{r}" / f"{project}.log"
            after_log = args.log_root / args.model_dir / f"round_{r}_after_round{r}" / f"{project}.log"
            if not before_log.is_file() or not after_log.is_file():
                skipped.append(
                    {
                        "project": project,
                        "round": r,
                        "reason": "missing_log",
                        "before_log": str(before_log),
                        "after_log": str(after_log),
                    }
                )
                continue

            before_findings = parse_codelinter_log(before_log)
            after_findings = parse_codelinter_log(after_log)

            before_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            after_counts_by_file: Dict[str, Counter[FindingSig]] = defaultdict(Counter)

            for f in before_findings:
                rel = _rel_path_from_log_file_path(str(f["file_path"]), project)
                if rel is None:
                    continue
                before_by_file[rel].append(f)

            for f in after_findings:
                rel = _rel_path_from_log_file_path(str(f["file_path"]), project)
                if rel is None:
                    continue
                sig: FindingSig = (str(f["category"]), str(f["rule_id"]), str(f.get("message", "")))
                after_counts_by_file[rel][sig] += 1

            if r == 1:
                input_root = original_root
            else:
                input_root = (args.snapshot_root / args.model_dir / f"round_{r-1}" / project).resolve()
            output_root = (args.snapshot_root / args.model_dir / f"round_{r}" / project).resolve()

            for rel_path, fs in before_by_file.items():
                fixed_findings: List[Dict[str, Any]] = []
                after_counts = Counter(after_counts_by_file.get(rel_path, Counter()))
                for f in fs:
                    sig: FindingSig = (str(f["category"]), str(f["rule_id"]), str(f.get("message", "")))
                    if after_counts.get(sig, 0) > 0:
                        after_counts[sig] -= 1
                        continue
                    fixed_findings.append(
                        {
                            "category": str(f["category"]),
                            "rule_id": str(f["rule_id"]),
                            "severity": str(f.get("severity", "")),
                            "message": str(f.get("message", "")),
                            "line": int(f["line"]),
                            "column": int(f["column"]),
                        }
                    )

                if not fixed_findings:
                    continue

                before_file = (input_root / rel_path).resolve()
                after_file = (output_root / rel_path).resolve()
                if not before_file.is_file() or not after_file.is_file():
                    skipped.append(
                        {
                            "project": project,
                            "round": r,
                            "rel_path": rel_path,
                            "reason": "missing_file_in_snapshot",
                            "before_file": str(before_file),
                            "after_file": str(after_file),
                        }
                    )
                    continue

                before_text = _read_text(before_file)
                after_text = _read_text(after_file)
                before_lines = before_text.splitlines()
                after_lines = after_text.splitlines()
                if (
                    len(before_lines) > args.max_lines
                    or len(after_lines) > args.max_lines
                    or len(before_text) > args.max_chars
                    or len(after_text) > args.max_chars
                ):
                    skipped.append(
                        {
                            "project": project,
                            "round": r,
                            "rel_path": rel_path,
                            "reason": "file_too_large",
                            "before_lines": len(before_lines),
                            "after_lines": len(after_lines),
                            "before_chars": len(before_text),
                            "after_chars": len(after_text),
                        }
                    )
                    continue

                stats = _diff_stats(before_lines, after_lines)
                total_changed = stats.changed + stats.added + stats.deleted
                patch_size_bin = _bin_patch_size(total_changed)
                stage_bin = _bin_stage(r)
                context_bin = _bin_context(len(fs))
                fixed_perf = sum(1 for it in fixed_findings if it["category"] == "performance")
                fixed_sec = sum(1 for it in fixed_findings if it["category"] == "security")
                cat_bin = _category_bin(fixed_perf, fixed_sec)
                stratum_key = _make_stratum_key(cat_bin, stage_bin, patch_size_bin, context_bin)

                candidate_id = _stable_id(args.model_dir, project, str(r), rel_path)
                candidates.append(
                    FileCandidate(
                        candidate_id=candidate_id,
                        model_dir=args.model_dir,
                        round_fixed=r,
                        project=project,
                        rel_path=rel_path,
                        stage_bin=stage_bin,
                        context_bin=context_bin,
                        patch_shape=stats.shape,
                        patch_size_bin=patch_size_bin,
                        file_defects_before=len(fs),
                        fixed_findings_count=len(fixed_findings),
                        fixed_perf_count=fixed_perf,
                        fixed_sec_count=fixed_sec,
                        fixed_category_bin=cat_bin,
                        file_lines_before=len(before_lines),
                        file_lines_after=len(after_lines),
                        file_chars_before=len(before_text),
                        file_chars_after=len(after_text),
                        stratum_key=stratum_key,
                        input_root=str(input_root),
                        output_root=str(output_root),
                    )
                )

    if not candidates:
        raise SystemExit("No eligible file-level candidates found (check max-lines/max-chars or paths).")

    # Sampling: primary strata = (fixed_category_bin, stage)
    prim_groups: Dict[Tuple[str, str], List[FileCandidate]] = defaultdict(list)
    for c in candidates:
        prim_groups[(c.fixed_category_bin, c.stage_bin)].append(c)
    prim_targets = _allocate_counts(args.n_sample, prim_groups, min_each=2)

    selected: List[FileCandidate] = []
    selected_ids: set[str] = set()

    for prim_key, prim_k in prim_targets.items():
        prim_pool = prim_groups[prim_key][:]
        rng.shuffle(prim_pool)
        for c in prim_pool[:prim_k]:
            if c.candidate_id in selected_ids:
                continue
            selected.append(c)
            selected_ids.add(c.candidate_id)

    if len(selected) < args.n_sample:
        remaining = [c for c in candidates if c.candidate_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.n_sample - len(selected)])

    selected = selected[: args.n_sample]

    # Write population (eligible candidates)
    _jsonl_write(
        out_dir / "candidates.jsonl",
        (dataclasses.asdict(c) for c in sorted(candidates, key=lambda x: (x.project, x.round_fixed, x.rel_path))),
    )

    # Render sample.jsonl with full before/after file + list of fixed findings
    sample_rows: List[Dict[str, Any]] = []
    for c in sorted(selected, key=lambda x: (x.stage_bin, x.fixed_category_bin, x.project, x.round_fixed, x.rel_path)):
        before_file = Path(c.input_root) / c.rel_path
        after_file = Path(c.output_root) / c.rel_path
        before_text = _read_text(before_file)
        after_text = _read_text(after_file)

        # Recompute fixed findings list from logs for this (project, round, rel_path) to keep sample self-contained.
        # This avoids persisting a potentially huge per-file finding map in memory.
        before_log = args.log_root / args.model_dir / f"round_{c.round_fixed}" / f"{c.project}.log"
        after_log = args.log_root / args.model_dir / f"round_{c.round_fixed}_after_round{c.round_fixed}" / f"{c.project}.log"
        before_findings = parse_codelinter_log(before_log)
        after_findings = parse_codelinter_log(after_log)

        after_counts: Counter[FindingSig] = Counter()
        for f in after_findings:
            rel = _rel_path_from_log_file_path(str(f["file_path"]), c.project)
            if rel != c.rel_path:
                continue
            sig: FindingSig = (str(f["category"]), str(f["rule_id"]), str(f.get("message", "")))
            after_counts[sig] += 1

        fixed_findings: List[Dict[str, Any]] = []
        for f in before_findings:
            rel = _rel_path_from_log_file_path(str(f["file_path"]), c.project)
            if rel != c.rel_path:
                continue
            sig: FindingSig = (str(f["category"]), str(f["rule_id"]), str(f.get("message", "")))
            if after_counts.get(sig, 0) > 0:
                after_counts[sig] -= 1
                continue
            fixed_findings.append(
                {
                    "category": str(f["category"]),
                    "rule_id": str(f["rule_id"]),
                    "severity": str(f.get("severity", "")),
                    "message": str(f.get("message", "")),
                    "line": int(f["line"]),
                    "column": int(f["column"]),
                }
            )

        row = dataclasses.asdict(c)
        row.update(
            {
                "before_file_text": before_text,
                "after_file_text": after_text,
                "file_diff": _unified_diff_text(before_text.splitlines(), after_text.splitlines(), rel_path=c.rel_path),
                "fixed_findings": fixed_findings,
            }
        )
        sample_rows.append(row)

    _jsonl_write(out_dir / "sample.jsonl", sample_rows)

    # Label template CSV (optional; useful for audits)
    label_path = out_dir / "labels_template.csv"
    with label_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "label",
                "annotator",
                "notes",
                "project",
                "round_fixed",
                "rel_path",
            ],
        )
        w.writeheader()
        for row in sample_rows:
            w.writerow(
                {
                    "sample_id": row["candidate_id"],
                    "label": "",
                    "annotator": "",
                    "notes": "",
                    "project": row["project"],
                    "round_fixed": row["round_fixed"],
                    "rel_path": row["rel_path"],
                }
            )

    if args.write_md:
        md_path = out_dir / "sample.md"
        md_lines: List[str] = []
        md_lines.append(f"# File-level repair semantic-eval sample ({args.model_dir})")
        md_lines.append("")
        md_lines.append(f"- n_files={len(sample_rows)}, seed={args.seed}, max_round={args.max_round}")
        md_lines.append(f"- max_lines={args.max_lines}, max_chars={args.max_chars}")
        md_lines.append("")
        for row in sample_rows:
            md_lines.append(f"## {row['candidate_id']}")
            md_lines.append(f"- project: `{row['project']}`")
            md_lines.append(f"- round_fixed: `{row['round_fixed']}` ({row['stage_bin']})")
            md_lines.append(f"- file: `{row['rel_path']}`")
            md_lines.append(f"- fixed_findings: `{len(row.get('fixed_findings', []))}` ({row['fixed_category_bin']})")
            md_lines.append("")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "unit": "file",
        "model_dir": args.model_dir,
        "max_round": args.max_round,
        "n_sample": args.n_sample,
        "seed": args.seed,
        "candidate_files_eligible": len(candidates),
        "sample_files": len(sample_rows),
        "max_lines": args.max_lines,
        "max_chars": args.max_chars,
        "primary_targets": {f"{k[0]}|{k[1]}": v for k, v in prim_targets.items()},
        "skip_counts": dict(Counter(s.get("reason") for s in skipped)),
        "skipped_sample": skipped[:200],
        "notes": (
            "Eligible population excludes files exceeding max_lines/max_chars so that full before/after text can be shown to judges."
        ),
    }
    (out_dir / "sampling_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] Wrote candidates: {out_dir / 'candidates.jsonl'}")
    print(f"[ok] Wrote sample: {out_dir / 'sample.jsonl'}")
    print(f"[ok] Wrote label template: {label_path}")
    if args.write_md:
        print(f"[ok] Wrote review sheet: {out_dir / 'sample.md'}")


if __name__ == "__main__":
    main()
