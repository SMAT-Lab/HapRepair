#!/usr/bin/env python3
"""
Prepare a stratified sample for manual semantic evaluation of HapRepair fixes.

We treat "repair success" as a finding that disappears between the CodeLinter
before/after logs of a given round (proxy metric). Because "pass static rules"
is not equivalent to semantic correctness, this script creates an independent,
human-in-the-loop validation set by sampling across risk-relevant strata.

Inputs (defaults match our revision pipeline):
  - revision/target_projects_haprepair.json
  - logs/codelinter_openharmony/<model_dir>/round_<r>/<project>.log
  - logs/codelinter_openharmony/<model_dir>/round_<r>_after_round<r>/<project>.log
  - revision/fixed_projects/<model_dir>/round_<r>/<project>/... (snapshots)

Outputs (under revision/semantic_eval/<run_id>/):
  - candidates.jsonl            All resolved findings with metadata + strata.
  - sample.jsonl                The sampled subset with local context + diff.
  - labels_template.csv         CSV template to fill (Correct/Suspicious/Incorrect).
  - sample.md                   Human-friendly review sheet (optional).
  - sampling_meta.json          Reproducibility metadata (seed, quotas, etc.).
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
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


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


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _rel_path_from_log_file_path(file_path: str, project_name: str) -> Optional[str]:
    """
    CodeLinter logs contain absolute paths under various workspace roots.
    Normalize to the path relative to the project root by finding the last
    occurrence of "/<project_name>/".
    """
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


@dataclass
class Candidate:
    candidate_id: str
    model_dir: str
    round_fixed: int
    project: str
    category: str
    rule_id: str
    severity: str
    message: str
    rel_path: str
    line: int
    column: int
    file_defects_before: int
    file_changed_lines: int
    file_added_lines: int
    file_deleted_lines: int
    patch_shape: str  # add|delete|modify
    patch_size_bin: str  # single|multi
    context_bin: str  # single_defect_file|multi_defect_file
    stage_bin: str  # early|late
    rule_freq_bin: str  # high|low (computed after counting)
    stratum_key: str

    # Optional paths (for later rendering)
    input_root: str
    output_root: str


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


def _context_window(lines: Sequence[str], line_1based: int, radius: int = 6) -> str:
    if not lines:
        return ""
    idx = max(0, min(len(lines) - 1, line_1based - 1))
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    out: List[str] = []
    for i in range(lo, hi):
        out.append(f"{i+1:6d}  {lines[i]}")
    return "\n".join(out)


def _local_unified_diff(
    before_lines: Sequence[str],
    after_lines: Sequence[str],
    *,
    fromfile: str,
    tofile: str,
    around_line_1based: int,
    radius: int = 12,
    max_lines: int = 200,
) -> str:
    if not before_lines and not after_lines:
        return ""
    idx = max(0, around_line_1based - 1)
    lo = max(0, idx - radius)
    hi = idx + radius + 1
    before_slice = list(before_lines[lo:hi])
    after_slice = list(after_lines[lo:hi])
    diff = list(
        difflib.unified_diff(
            before_slice,
            after_slice,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
            n=3,
        )
    )
    if len(diff) > max_lines:
        diff = diff[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff)


def _bin_patch_size(total_changed_lines: int) -> str:
    return "single" if total_changed_lines <= 1 else "multi"


def _bin_stage(round_fixed: int) -> str:
    return "early" if round_fixed <= 2 else "late"


def _bin_context(file_defects_before: int) -> str:
    return "single_defect_file" if file_defects_before <= 1 else "multi_defect_file"


def _make_stratum_key(category: str, stage: str, patch_size: str, context: str) -> str:
    return f"{category}|{stage}|{patch_size}|{context}"


def _weighted_pick(
    rng: random.Random,
    items: List[Candidate],
    k: int,
    *,
    shape_counts: Counter[str],
    freq_counts: Counter[str],
) -> List[Candidate]:
    picked: List[Candidate] = []
    pool = items[:]
    for _ in range(min(k, len(pool))):
        weights: List[float] = []
        for c in pool:
            w_shape = 1.0 / (1.0 + shape_counts[c.patch_shape])
            w_freq = 1.0 / (1.0 + freq_counts[c.rule_freq_bin])
            weights.append(w_shape * w_freq)
        choice = rng.choices(pool, weights=weights, k=1)[0]
        pool.remove(choice)
        picked.append(choice)
        shape_counts[choice.patch_shape] += 1
        freq_counts[choice.rule_freq_bin] += 1
    return picked


def _allocate_counts(total: int, groups: Dict[Tuple[str, ...], List[Candidate]], min_each: int = 0) -> Dict[Tuple[str, ...], int]:
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

    # Fix rounding drift
    while sum(targets.values()) > total:
        k = max(targets.keys(), key=lambda kk: targets[kk])
        if targets[k] <= 0:
            break
        targets[k] -= 1
    while sum(targets.values()) < total:
        # Add to the largest remaining capacity group
        k = max(sizes.keys(), key=lambda kk: sizes[kk] - targets.get(kk, 0))
        if targets.get(k, 0) >= sizes[k]:
            break
        targets[k] = targets.get(k, 0) + 1
    return targets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare stratified manual semantic-eval samples for HapRepair repairs.")
    ap.add_argument("--model-dir", type=str, default="gpt-5.1", help="Model directory name under logs/snapshots.")
    ap.add_argument("--max-round", type=int, default=5, help="Maximum round index to include (default: 5).")
    ap.add_argument("--n-sample", type=int, default=150, help="Total sample size (default: 150).")
    ap.add_argument("--seed", type=int, default=20251220, help="Random seed for sampling (default: 20251220).")
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="Path to target_projects_haprepair.json.")
    ap.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root of codelinter logs.")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT, help="Root of fixed snapshots.")
    ap.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Output base directory.")
    ap.add_argument("--write-md", action="store_true", help="Also write a human-friendly sample.md sheet.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    targets_raw = json.loads(_read_text(args.targets))
    if not isinstance(targets_raw, list) or not targets_raw:
        raise SystemExit(f"Malformed targets file (expected non-empty list): {args.targets}")

    run_id = f"{args.model_dir}_n{args.n_sample}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = (args.out_base / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_cache: Dict[Tuple[int, str, str], FileDiffStats] = {}

    candidates: List[Candidate] = []
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

            # Normalize findings to stable keys (relative to project root)
            before_keys: Dict[FindingKey, Dict[str, Any]] = {}
            by_file_before: Counter[str] = Counter()
            for f in before_findings:
                rel = _rel_path_from_log_file_path(str(f["file_path"]), project)
                if rel is None:
                    continue
                by_file_before[rel] += 1
                key = FindingKey(
                    rel_path=rel,
                    line=int(f["line"]),
                    column=int(f["column"]),
                    category=str(f["category"]),
                    rule_id=str(f["rule_id"]),
                )
                # Keep the first occurrence (duplicates are rare)
                before_keys.setdefault(key, f)

            after_set: set[FindingKey] = set()
            for f in after_findings:
                rel = _rel_path_from_log_file_path(str(f["file_path"]), project)
                if rel is None:
                    continue
                after_set.add(
                    FindingKey(
                        rel_path=rel,
                        line=int(f["line"]),
                        column=int(f["column"]),
                        category=str(f["category"]),
                        rule_id=str(f["rule_id"]),
                    )
                )

            fixed_keys = [k for k in before_keys.keys() if k not in after_set]
            if not fixed_keys:
                continue

            if r == 1:
                input_root = original_root
            else:
                input_root = (args.snapshot_root / args.model_dir / f"round_{r-1}" / project).resolve()
            output_root = (args.snapshot_root / args.model_dir / f"round_{r}" / project).resolve()

            for k in fixed_keys:
                meta = before_keys[k]
                rel_path = k.rel_path

                diff_key = (r, project, rel_path)
                stats = diff_cache.get(diff_key)
                if stats is None:
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
                    before_lines = _read_text(before_file).splitlines()
                    after_lines = _read_text(after_file).splitlines()
                    stats = _diff_stats(before_lines, after_lines)
                    diff_cache[diff_key] = stats

                total_changed = stats.changed + stats.added + stats.deleted
                patch_size_bin = _bin_patch_size(total_changed)
                stage_bin = _bin_stage(r)
                context_bin = _bin_context(int(by_file_before.get(rel_path, 0)))
                stratum_key = _make_stratum_key(k.category, stage_bin, patch_size_bin, context_bin)

                candidate_id = _stable_id(
                    args.model_dir,
                    project,
                    str(r),
                    rel_path,
                    str(k.line),
                    str(k.column),
                    k.category,
                    k.rule_id,
                )

                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        model_dir=args.model_dir,
                        round_fixed=r,
                        project=project,
                        category=k.category,
                        rule_id=k.rule_id,
                        severity=str(meta.get("severity", "")),
                        message=str(meta.get("message", "")),
                        rel_path=rel_path,
                        line=int(k.line),
                        column=int(k.column),
                        file_defects_before=int(by_file_before.get(rel_path, 0)),
                        file_changed_lines=int(total_changed),
                        file_added_lines=int(stats.added),
                        file_deleted_lines=int(stats.deleted),
                        patch_shape=str(stats.shape),
                        patch_size_bin=patch_size_bin,
                        context_bin=context_bin,
                        stage_bin=stage_bin,
                        rule_freq_bin="TBD",
                        stratum_key=stratum_key,
                        input_root=str(input_root),
                        output_root=str(output_root),
                    )
                )

    if not candidates:
        raise SystemExit("No candidates found. Check model-dir/log-root/snapshot-root paths.")

    # Compute rule frequency bins on resolved findings (high vs low).
    rule_counts = Counter(c.rule_id for c in candidates)
    # Define "high frequency" as top-20 rules by count (stable and easy to explain).
    high_rules = {r for r, _ in rule_counts.most_common(20)}
    for c in candidates:
        c.rule_freq_bin = "high" if c.rule_id in high_rules else "low"

    # Sampling: primary = (category, stage), secondary = (patch_size_bin, context_bin)
    prim_groups: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
    for c in candidates:
        prim_groups[(c.category, c.stage_bin)].append(c)

    prim_targets = _allocate_counts(args.n_sample, prim_groups, min_each=5)

    selected: List[Candidate] = []
    selected_ids: set[str] = set()
    shape_counts: Counter[str] = Counter()
    freq_counts: Counter[str] = Counter()

    for prim_key, prim_k in prim_targets.items():
        prim_pool = prim_groups[prim_key]
        sec_groups: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
        for c in prim_pool:
            sec_groups[(c.patch_size_bin, c.context_bin)].append(c)
        sec_targets = _allocate_counts(prim_k, sec_groups, min_each=1)
        for sec_key, sec_k in sec_targets.items():
            sec_pool = [c for c in sec_groups[sec_key] if c.candidate_id not in selected_ids]
            rng.shuffle(sec_pool)
            picked = _weighted_pick(rng, sec_pool, sec_k, shape_counts=shape_counts, freq_counts=freq_counts)
            for p in picked:
                if p.candidate_id in selected_ids:
                    continue
                selected.append(p)
                selected_ids.add(p.candidate_id)

    # Fill any remaining slots uniformly (still biased by underrepresented bins).
    if len(selected) < args.n_sample:
        remaining = [c for c in candidates if c.candidate_id not in selected_ids]
        rng.shuffle(remaining)
        extra = _weighted_pick(
            rng,
            remaining,
            args.n_sample - len(selected),
            shape_counts=shape_counts,
            freq_counts=freq_counts,
        )
        for p in extra:
            if p.candidate_id in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(p.candidate_id)

    selected = selected[: args.n_sample]

    # Write candidates.jsonl (metadata only)
    _jsonl_write(
        out_dir / "candidates.jsonl",
        (
            dataclasses.asdict(c)
            for c in sorted(candidates, key=lambda x: (x.project, x.round_fixed, x.rel_path, x.line, x.rule_id))
        ),
    )

    # Render sample with local context + diff
    sample_rows: List[Dict[str, Any]] = []
    for c in sorted(selected, key=lambda x: (x.stage_bin, x.category, x.project, x.round_fixed, x.rel_path, x.line)):
        before_file = Path(c.input_root) / c.rel_path
        after_file = Path(c.output_root) / c.rel_path
        before_lines = _read_text(before_file).splitlines() if before_file.is_file() else []
        after_lines = _read_text(after_file).splitlines() if after_file.is_file() else []
        before_ctx = _context_window(before_lines, c.line, radius=6)
        after_ctx = _context_window(after_lines, c.line, radius=6)
        local_diff = _local_unified_diff(
            before_lines,
            after_lines,
            fromfile=f"before/{c.rel_path}",
            tofile=f"after/{c.rel_path}",
            around_line_1based=c.line,
            radius=12,
            max_lines=200,
        )
        row = dataclasses.asdict(c)
        row.update(
            {
                "before_context": before_ctx,
                "after_context": after_ctx,
                "local_diff": local_diff,
            }
        )
        sample_rows.append(row)

    _jsonl_write(out_dir / "sample.jsonl", sample_rows)

    # Label template CSV
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
                "category",
                "rule_id",
                "rel_path",
                "line",
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
                    "category": row["category"],
                    "rule_id": row["rule_id"],
                    "rel_path": row["rel_path"],
                    "line": row["line"],
                }
            )

    # Optional markdown sheet
    if args.write_md:
        md_path = out_dir / "sample.md"
        md_lines: List[str] = []
        md_lines.append(f"# Repair semantic-eval sample ({args.model_dir})")
        md_lines.append("")
        md_lines.append(f"- n={len(sample_rows)}, seed={args.seed}, max_round={args.max_round}")
        md_lines.append("")
        for row in sample_rows:
            md_lines.append(f"## {row['candidate_id']}")
            md_lines.append(f"- project: `{row['project']}`")
            md_lines.append(f"- round_fixed: `{row['round_fixed']}` ({row['stage_bin']})")
            md_lines.append(f"- rule: `{row['category']}/{row['rule_id']}` ({row['severity']})")
            md_lines.append(f"- file: `{row['rel_path']}`:{row['line']}:{row['column']}")
            md_lines.append(f"- patch: shape=`{row['patch_shape']}`, changed_lines={row['file_changed_lines']}")
            md_lines.append("")
            md_lines.append("### Before (local context)")
            md_lines.append("```")
            md_lines.append(row.get("before_context", ""))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("### After (local context)")
            md_lines.append("```")
            md_lines.append(row.get("after_context", ""))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("### Local diff")
            md_lines.append("```diff")
            md_lines.append(row.get("local_diff", ""))
            md_lines.append("```")
            md_lines.append("")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "model_dir": args.model_dir,
        "max_round": args.max_round,
        "n_sample": args.n_sample,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "sample_count": len(sample_rows),
        "high_rules_top_k": 20,
        "primary_targets": {f"{k[0]}|{k[1]}": v for k, v in prim_targets.items()},
        "notes": (
            "A resolved finding is defined as present in round_r log but absent in round_r_after_roundr "
            "log, keyed by (rel_path,line,column,category,rule_id). This is a static proxy for repair success."
        ),
        "skipped": skipped[:200],  # keep meta bounded
    }
    (out_dir / "sampling_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] Wrote candidates: {out_dir / 'candidates.jsonl'}")
    print(f"[ok] Wrote sample: {out_dir / 'sample.jsonl'}")
    print(f"[ok] Wrote label template: {label_path}")
    if args.write_md:
        print(f"[ok] Wrote review sheet: {out_dir / 'sample.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
