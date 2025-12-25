#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from code_repair import CodeContextExtractor


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FILE_HEADER_RE = re.compile(r"^(\/.+)\(\d+\)$")
LINE_META_RE = re.compile(r"^(\d+):(\d+)\s+(\w+)\s+(.*)$")
VALID_CATEGORIES = {"performance", "security"}
SEVERITY_NORMALIZATION = {"warning": "warn"}


@dataclass(frozen=True)
class Defect:
    project: str
    file_path: str
    rel_path: str
    line: int
    column: int
    category: str
    rule: str  # category/rule
    severity: str
    message: str


@dataclass(frozen=True)
class DefectFeatures:
    context_lines: int
    context_chars: int
    primary_block_lines: int
    brace_depth: int
    block_ranges_count: int
    needs_more_context: bool
    merged_group_size: int
    cross_function: bool
    bucket: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def relative_path_in_project(file_path: str, project: str) -> str:
    needle = f"/{project}/"
    idx = file_path.find(needle)
    if idx == -1:
        return file_path
    return file_path[idx + len(needle) :]


def parse_defects_in_log(path: Path, project: str) -> List[Defect]:
    defects: List[Defect] = []
    if not path.is_file():
        return defects

    current_file: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = strip_ansi(raw_line).rstrip("\n")
            if not line:
                continue

            header = FILE_HEADER_RE.match(line.strip())
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

            m = LINE_META_RE.match(prefix)
            if not m:
                continue
            line_num_raw, col_raw, severity_raw, message = m.groups()

            parts = meta.split("/", 1)
            if len(parts) != 2:
                continue
            category_raw, rule_raw = parts
            category = category_raw.strip().lower()
            if category not in VALID_CATEGORIES:
                continue

            severity = SEVERITY_NORMALIZATION.get(
                severity_raw.strip().lower(), severity_raw.strip().lower()
            )
            if severity not in ("error", "warn", "suggestion"):
                continue

            try:
                line_num = int(line_num_raw)
                col = int(col_raw)
            except ValueError:
                continue

            rule_id = rule_raw.strip()
            full_rule = f"{category}/{rule_id}"

            defects.append(
                Defect(
                    project=project,
                    file_path=current_file,
                    rel_path=relative_path_in_project(current_file, project),
                    line=line_num,
                    column=col,
                    category=category,
                    rule=full_rule,
                    severity=severity,
                    message=message.strip(),
                )
            )

    return defects


def parse_defects_in_dir(dir_path: Path) -> List[Defect]:
    out: List[Defect] = []
    if not dir_path.is_dir():
        return out
    for log_path in sorted(dir_path.glob("*.log")):
        out.extend(parse_defects_in_log(log_path, project=log_path.stem))
    return out


def ranges_overlap(r1: List[Tuple[int, int]], r2: List[Tuple[int, int]]) -> bool:
    for s1, e1 in r1:
        for s2, e2 in r2:
            if not (e2 + 1 < s1 or s2 - 1 > e1):
                return True
    return False


@lru_cache(maxsize=512)
def load_code_lines(file_path: str) -> List[str]:
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return [""]
    return [""] + text.splitlines()


@lru_cache(maxsize=512)
def compute_brace_depths(file_path: str) -> Tuple[int, ...]:
    lines = load_code_lines(file_path)
    depths = [0] * len(lines)
    depth = 0
    for i in range(1, len(lines)):
        depths[i] = depth
        depth += lines[i].count("{") - lines[i].count("}")
        if depth < 0:
            depth = 0
    return tuple(depths)


def load_rules_needs_more_context(rules_path: Path) -> Dict[str, bool]:
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    out: Dict[str, bool] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule") or "")
        needs = bool(item.get("needsMoreContext"))
        if rule.startswith("@"):
            rule = rule[1:]
        if rule:
            out[rule] = needs
    return out


def extract_context_for_defect(
    extractor: CodeContextExtractor,
    code_lines: List[str],
    line_num: int,
    needs_more_context: bool,
) -> Tuple[List[Tuple[int, int]], str, int, int]:
    start_idx, end_idx, surrounding_context = extractor._extract_blocks(code_lines, line_num)
    block_ranges: List[Tuple[int, int]] = [(start_idx, end_idx)]

    if not needs_more_context:
        return block_ranges, surrounding_context or "", start_idx, end_idx

    context = extractor.extract_arkts_context(code_lines, line_num)
    code_blocks: List[Tuple[int, int, str]] = []
    if surrounding_context:
        code_blocks.append((start_idx, end_idx, surrounding_context))

    def maybe_add_point(line_idx: int, content: str) -> None:
        for b_start, b_end, _ in code_blocks:
            if b_start <= line_idx <= b_end:
                return
        block_ranges.append((line_idx, line_idx))
        code_blocks.append((line_idx, line_idx, content))

    for def_start, content in context.get("definition", []):
        if isinstance(def_start, int) and isinstance(content, str):
            maybe_add_point(def_start, content)
    for use_start, content in context.get("usage", []):
        if isinstance(use_start, int) and isinstance(content, str):
            maybe_add_point(use_start, content)

    code_blocks.sort(key=lambda x: x[0])
    surrounding_context = "\n".join(content for _, _, content in code_blocks)
    block_ranges.sort(key=lambda x: x[0])
    return block_ranges, surrounding_context, start_idx, end_idx


def bucket_defect(
    *,
    group_size: int,
    needs_more_context: bool,
    block_ranges_count: int,
    context_lines: int,
    brace_depth: int,
    trivial_max_lines: int,
    trivial_max_depth: int,
) -> str:
    cross_function = block_ranges_count > 1
    # Multi-defect *and* cross-function context: this approximates the hardest
    # "merged prompt" cases where fixes require navigating multiple code regions.
    if group_size >= 2 and cross_function:
        return "multi-defect merged context"

    # Small local context with shallow nesting.
    if (not cross_function) and context_lines <= trivial_max_lines and brace_depth <= trivial_max_depth:
        return "trivial local fixes"

    return "context-sensitive fixes"


def classify_defects(
    defects: List[Defect],
    rules_needs_more_context: Dict[str, bool],
    *,
    trivial_max_lines: int,
    trivial_max_depth: int,
) -> Dict[Defect, DefectFeatures]:
    extractor = CodeContextExtractor()
    by_file: Dict[str, List[Defect]] = defaultdict(list)
    for d in defects:
        by_file[d.file_path].append(d)

    out: Dict[Defect, DefectFeatures] = {}

    for file_path, file_defects in by_file.items():
        code_lines = load_code_lines(file_path)
        brace_depths = compute_brace_depths(file_path)

        ranges_by_idx: List[List[Tuple[int, int]]] = []
        features_by_idx: List[Tuple[int, int, int, int, int, bool]] = []
        # (context_lines, context_chars, primary_block_lines, brace_depth, block_ranges_count, needs_more_context)

        for d in file_defects:
            needs_more_context = rules_needs_more_context.get(d.rule, True)
            if d.line <= 0 or d.line >= len(code_lines):
                block_ranges = [(d.line, d.line)]
                context_text = ""
                primary_start = primary_end = d.line
            else:
                block_ranges, context_text, primary_start, primary_end = extract_context_for_defect(
                    extractor, code_lines, d.line, needs_more_context
                )
            ctx_lines = 0 if not context_text else len(context_text.splitlines())
            ctx_chars = len(context_text)
            primary_block_lines = max(0, primary_end - primary_start + 1)
            brace_depth = brace_depths[d.line] if 0 <= d.line < len(brace_depths) else 0
            block_ranges_count = len(block_ranges)

            ranges_by_idx.append(block_ranges)
            features_by_idx.append(
                (
                    ctx_lines,
                    ctx_chars,
                    primary_block_lines,
                    brace_depth,
                    block_ranges_count,
                    needs_more_context,
                )
            )

        uf = UnionFind(len(file_defects))
        for i in range(len(file_defects)):
            for j in range(i + 1, len(file_defects)):
                if ranges_overlap(ranges_by_idx[i], ranges_by_idx[j]):
                    uf.union(i, j)

        group_sizes: Counter[int] = Counter(uf.find(i) for i in range(len(file_defects)))

        for i, d in enumerate(file_defects):
            root = uf.find(i)
            group_size = group_sizes[root]
            (
                ctx_lines,
                ctx_chars,
                primary_block_lines,
                brace_depth,
                block_ranges_count,
                needs_more_context,
            ) = features_by_idx[i]

            cross_function = block_ranges_count > 1
            bucket = bucket_defect(
                group_size=group_size,
                needs_more_context=needs_more_context,
                block_ranges_count=block_ranges_count,
                context_lines=ctx_lines,
                brace_depth=brace_depth,
                trivial_max_lines=trivial_max_lines,
                trivial_max_depth=trivial_max_depth,
            )

            out[d] = DefectFeatures(
                context_lines=ctx_lines,
                context_chars=ctx_chars,
                primary_block_lines=primary_block_lines,
                brace_depth=brace_depth,
                block_ranges_count=block_ranges_count,
                needs_more_context=needs_more_context,
                merged_group_size=group_size,
                cross_function=cross_function,
                bucket=bucket,
            )

    return out


def summarize_buckets(defect_features: Iterable[DefectFeatures]) -> Dict[str, dict]:
    by_bucket: Dict[str, List[DefectFeatures]] = defaultdict(list)
    for feat in defect_features:
        by_bucket[feat.bucket].append(feat)

    out: Dict[str, dict] = {}
    for bucket, feats in by_bucket.items():
        context_lines = [f.context_lines for f in feats]
        brace_depths = [f.brace_depth for f in feats]
        cross = sum(1 for f in feats if f.cross_function)
        multi = sum(1 for f in feats if f.merged_group_size >= 2)

        out[bucket] = {
            "count": len(feats),
            "median_context_lines": int(statistics.median(context_lines)) if context_lines else 0,
            "median_brace_depth": int(statistics.median(brace_depths)) if brace_depths else 0,
            "cross_function_share": (cross / len(feats)) if feats else 0.0,
            "multi_defect_share": (multi / len(feats)) if feats else 0.0,
        }
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def prompt_model_choice(base_dir: Path, default_model: str = "gpt-5.1") -> str:
    models = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    if not models:
        raise SystemExit(f"在 {base_dir} 下没有找到可用的模型目录。")
    default = default_model if default_model in models else models[0]
    print("可用模型：")
    for idx, name in enumerate(models, start=1):
        default_mark = " (默认)" if name == default else ""
        print(f"[{idx}] {name}{default_mark}")
    prompt = f"请选择模型（输入编号或名称，直接回车使用默认 {default}）："
    while True:
        choice = input(prompt).strip()
        if not choice:
            return default
        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(models):
                return models[i]
        if choice in models:
            return choice
        print("无效输入，请重新选择。")


def discover_round_dirs(model_dir: Path) -> List[Tuple[str, Path]]:
    candidates = [
        ("Initial (round_1)", model_dir / "round_1"),
        ("After Round1", model_dir / "round_1_after_round1"),
        ("After Round2", model_dir / "round_2_after_round2"),
        ("After Round3", model_dir / "round_3_after_round3"),
        ("After Round4", model_dir / "round_4_after_round4"),
        ("After Round5", model_dir / "round_5_after_round5"),
        ("After Round6", model_dir / "round_6_after_round6"),
    ]
    out: List[Tuple[str, Path]] = []
    for label, p in candidates:
        if p.is_dir() and any(p.glob("*.log")):
            out.append((label, p))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bucket defects by difficulty proxies (context size / depth / merged contexts)."
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (default: summary/<model>_difficulty_buckets.md).",
    )
    parser.add_argument("--trivial-max-lines", type=int, default=5)
    parser.add_argument("--trivial-max-depth", type=int, default=2)
    parser.add_argument(
        "--only-initial-and-final",
        action="store_true",
        help="Only analyze Initial and latest After-Round directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path("/home/LLMCodeRepair")
    base_dir = repo_root / "logs" / "codelinter_openharmony"
    model = args.model if args.model else prompt_model_choice(base_dir)
    model_dir = base_dir / model

    rules_needs_more_context = load_rules_needs_more_context(repo_root / "rules.json")
    round_dirs = discover_round_dirs(model_dir)
    if not round_dirs:
        raise SystemExit(f"未找到任何可用日志目录：{model_dir}")

    if args.only_initial_and_final and len(round_dirs) >= 2:
        round_dirs = [round_dirs[0], round_dirs[-1]]

    output_path = (
        args.output
        if args.output is not None
        else (repo_root / "summary" / f"{model}_difficulty_buckets.md")
    )

    all_round_bucket_counts: Dict[str, Counter[str]] = {}
    all_round_bucket_summaries: Dict[str, Dict[str, dict]] = {}
    initial_rule_counts: Counter[str] = Counter()
    final_rule_counts: Counter[str] = Counter()
    final_rule_bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for label, dir_path in round_dirs:
        defects = parse_defects_in_dir(dir_path)
        rule_counts = Counter(d.rule for d in defects)
        if label == round_dirs[0][0]:
            initial_rule_counts = rule_counts
        if label == round_dirs[-1][0]:
            final_rule_counts = rule_counts
        feat_map = classify_defects(
            defects,
            rules_needs_more_context,
            trivial_max_lines=args.trivial_max_lines,
            trivial_max_depth=args.trivial_max_depth,
        )
        if label == round_dirs[-1][0]:
            for d, feat in feat_map.items():
                final_rule_bucket_counts[feat.bucket][d.rule] += 1
        bucket_counts = Counter(f.bucket for f in feat_map.values())
        all_round_bucket_counts[label] = bucket_counts
        all_round_bucket_summaries[label] = summarize_buckets(feat_map.values())

    initial_label = round_dirs[0][0]
    final_label = round_dirs[-1][0]
    initial_counts = all_round_bucket_counts[initial_label]
    final_counts = all_round_bucket_counts[final_label]

    buckets = [
        "trivial local fixes",
        "context-sensitive fixes",
        "multi-defect merged context",
    ]

    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            f"{model} 缺陷难度分桶统计（基于上下文/深度/合并上下文的代理指标）\n"
            "====================================================================\n\n"
            f"日志来源：`logs/codelinter_openharmony/{model}`（按 CodeLinter 逐条缺陷行解析，过滤 performance/security）。\n\n"
            "分桶规则（可用脚本参数调整）：\n"
            f"- `multi-defect merged context`：同一文件内，按上下文范围重叠合并后，一个上下文组包含 ≥2 条缺陷，且该缺陷需要跨函数/跨代码片段上下文（`block_ranges>1`）；\n"
            f"- `trivial local fixes`：不跨函数/跨片段（`block_ranges=1`），且 `context_lines≤{args.trivial_max_lines}` 且 `brace_depth≤{args.trivial_max_depth}`；\n"
            "- `context-sensitive fixes`：其余缺陷。\n\n"
        )

        f.write("**Bucketed Fix Rate（Initial → Final）**\n")
        f.write("| Bucket | Initial | Final | Fixed | Fix Rate |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for b in buckets:
            init = int(initial_counts.get(b, 0))
            fin = int(final_counts.get(b, 0))
            fixed = max(0, init - fin)
            rate = (fixed / init) if init else 0.0
            f.write(f"| {b} | {init} | {fin} | {fixed} | {fmt_pct(rate)} |\n")

        f.write("\n**Bucket Diagnostics（Median context / depth / cross-function share）**\n")
        f.write(f"- Initial：{initial_label}\n")
        init_summary = all_round_bucket_summaries[initial_label]
        for b in buckets:
            s = init_summary.get(b)
            if not s:
                continue
            f.write(
                f"  - {b}: median_context_lines={s['median_context_lines']}, "
                f"median_brace_depth={s['median_brace_depth']}, "
                f"cross_function={fmt_pct(s['cross_function_share'])}, "
                f"multi_defect={fmt_pct(s['multi_defect_share'])}\n"
            )

        if final_rule_counts:
            f.write("\n**Final Remaining Rules (Top 15)**\n")
            f.write(f"- Final：{final_label}\n")
            f.write("| Rule | Bucket | Final | Initial |\n")
            f.write("| --- | --- | --- | --- |\n")
            # Sort by final remaining count desc.
            for rule, fin in final_rule_counts.most_common(15):
                bucket = "unknown"
                best = 0
                for b in buckets:
                    cnt = int(final_rule_bucket_counts.get(b, {}).get(rule, 0))
                    if cnt > best:
                        best = cnt
                        bucket = b
                init = int(initial_rule_counts.get(rule, 0))
                f.write(f"| {rule} | {bucket} | {fin} | {init} |\n")

        if len(round_dirs) > 2 or not args.only_initial_and_final:
            f.write("\n**Remaining Defects By Round (Buckets)**\n")
            f.write("| Bucket | " + " | ".join(label for label, _ in round_dirs) + " |\n")
            f.write("| --- | " + " | ".join("---" for _ in round_dirs) + " |\n")
            for b in buckets:
                row = [str(int(all_round_bucket_counts[label].get(b, 0))) for label, _ in round_dirs]
                f.write(f"| {b} | " + " | ".join(row) + " |\n")

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
