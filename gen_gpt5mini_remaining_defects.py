#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional


DEFECTS_RE = re.compile(r"-Defects:\s*(\d+)")
FILE_HEADER_RE = re.compile(r"^(\/.+)\(\d+\)$")
LINE_META_RE = re.compile(r"^(\d+):(\d+)\s+(\w+)\s+(.*)$")
VALID_CATEGORIES = {"performance", "security"}
SEVERITY_NORMALIZATION = {"warning": "warn"}


def parse_defects_from_summary(path: Path) -> int:
    """
    Parse defect count from the trailing '-Defects:' line in a CodeLinter log.

    This is a coarse fallback when structured parsing fails.
    """
    defects = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = DEFECTS_RE.search(line)
                if m:
                    defects = int(m.group(1))
    except FileNotFoundError:
        return 0
    # If no Defects line, treat as 0 defects for that round.
    return defects if defects is not None else 0


def parse_defects_structured(path: Path) -> Optional[int]:
    """
    Parse defect count using the same rules as `fix_projects_codelinter.py`:
    - Only performance/security categories are counted.
    - Entries without a rule id (missing '@category/rule') are ignored.
    """
    if not path.is_file():
        return None

    current_file: Optional[str] = None
    count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
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
            _, _, severity_raw, _ = m.groups()

            parts = meta.split("/", 1)
            if len(parts) != 2:
                continue
            category_raw, _ = parts
            category = category_raw.strip().lower()
            if category not in VALID_CATEGORIES:
                continue

            severity = SEVERITY_NORMALIZATION.get(
                severity_raw.lower(), severity_raw.lower()
            )
            if severity not in ("error", "warn", "suggestion"):
                continue

            count += 1

    return count


def count_defects(path: Path) -> int:
    """
    Prefer structured parsing (aligned with the repair pipeline summary),
    fall back to the '-Defects:' summary line if parsing fails.
    """
    structured = parse_defects_structured(path)
    if structured is not None:
        return structured
    return parse_defects_from_summary(path)


def parse_rule_counts(path: Path) -> Counter[str]:
    """
    Count defects per rule (category/rule) in a CodeLinter log.
    Only performance/security entries are counted; malformed lines are skipped.
    """
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts

    current_file: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
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
            _, _, severity_raw, _ = m.groups()

            parts = meta.split("/", 1)
            if len(parts) != 2:
                continue
            category_raw, rule_raw = parts
            category = category_raw.strip().lower()
            if category not in VALID_CATEGORIES:
                continue

            severity = SEVERITY_NORMALIZATION.get(
                severity_raw.lower(), severity_raw.lower()
            )
            if severity not in ("error", "warn", "suggestion"):
                continue

            counts[f"{category}/{rule_raw.strip()}"] += 1

    return counts


def collect_stats(
    base_dir: Path,
    model_name: str = "gpt-5-mini",
    project_names: Optional[List[str]] = None,
    initial_from_target: Optional[Dict[str, int]] = None,
) -> List[
    Tuple[
        str,
        int | None,  # init_baseline
        int | None,  # init_scan (round_1 log)
        int | None,  # r1
        int | None,  # r2
        int | None,  # r3
        int | None,  # r4
        int | None,  # r5
        int | None,  # r6
    ]
]:
    """
    Collect per-project defect stats:
    returns list of (project, init_baseline, init_scan, r1, r2, r3, r4, r5, r6).
    """
    model_dir = base_dir / model_name

    # Initial defects from round_1
    initial_dir = model_dir / "round_1"

    # Per-round "before repair" logs (used for inferring 0 when the pipeline
    # early-exits without writing an after_round log).
    before_dirs = {
        "Round1": model_dir / "round_1",
        "Round2": model_dir / "round_2",
        "Round3": model_dir / "round_3",
        "Round4": model_dir / "round_4",
        "Round5": model_dir / "round_5",
        "Round6": model_dir / "round_6",
    }

    # Remaining defects after each repair round
    round_dirs = [
        ("Round1", model_dir / "round_1_after_round1"),
        ("Round2", model_dir / "round_2_after_round2"),
        ("Round3", model_dir / "round_3_after_round3"),
        ("Round4", model_dir / "round_4_after_round4"),
        ("Round5", model_dir / "round_5_after_round5"),
        ("Round6", model_dir / "round_6_after_round6"),
    ]

    # 1) Initial defects
    initial: Dict[str, int] = {}
    if initial_dir.is_dir():
        for log_path in sorted(initial_dir.glob("*.log")):
            proj = log_path.stem
            initial[proj] = count_defects(log_path)

    # 2) Remaining defects for each round
    remaining: Dict[str, Dict[str, int]] = {}
    for label, d in round_dirs:
        if not d.is_dir():
            continue
        for log_path in sorted(d.glob("*.log")):
            proj = log_path.stem
            defects = count_defects(log_path)
            remaining.setdefault(proj, {})[label] = defects

    # 3) Final project list
    if project_names:
        # Preserve the order in target_projects_haprepair.json; any project that
        # has never been run for a given round will show '-' for that cell.
        projects = project_names
    else:
        # Fallback: discover all projects that have at least one log.
        projects = sorted(set(initial.keys()) | set(remaining.keys()))
    rows: List[
        Tuple[
            str,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
        ]
    ] = []
    for proj in projects:
        # 初始缺陷分两种口径：
        # - init_baseline：revision/target_projects_haprepair.json 中的基线统计（稳定、与模型无关）
        # - init_scan：本次运行 round_1 的 CodeLinter 扫描结果（可能受工具版本/环境影响）
        init_baseline = (
            initial_from_target.get(proj) if initial_from_target is not None else None
        )
        init_scan = initial.get(proj)
        # 每一轮剩余缺陷：只有在对应 *_after_roundX 目录下有日志时才有值。
        r1 = remaining.get(proj, {}).get("Round1")
        r2 = remaining.get(proj, {}).get("Round2")
        r3 = remaining.get(proj, {}).get("Round3")
        r4 = remaining.get(proj, {}).get("Round4")
        r5 = remaining.get(proj, {}).get("Round5")
        r6 = remaining.get(proj, {}).get("Round6")

        # If an after_round log is missing but the corresponding "before" log
        # exists and already has 0 defects, treat the remaining defects as 0.
        # This typically happens when the repair pipeline early-exits on 0
        # findings and therefore doesn't write an after_round log.
        def maybe_infer_zero(label: str, current: int | None) -> int | None:
            if current is not None:
                return current
            before_dir = before_dirs.get(label)
            if not before_dir or not before_dir.is_dir():
                return None
            before_log = before_dir / f"{proj}.log"
            if not before_log.is_file():
                return None
            try:
                before_defects = count_defects(before_log)
            except Exception:
                return None
            return 0 if before_defects == 0 else None

        r1 = maybe_infer_zero("Round1", r1)
        r2 = maybe_infer_zero("Round2", r2)
        r3 = maybe_infer_zero("Round3", r3)
        r4 = maybe_infer_zero("Round4", r4)
        r5 = maybe_infer_zero("Round5", r5)
        r6 = maybe_infer_zero("Round6", r6)

        rows.append((proj, init_baseline, init_scan, r1, r2, r3, r4, r5, r6))

    return rows


def collect_rule_counts_by_round(
    base_dir: Path,
    model_name: str = "gpt-5-mini",
) -> Dict[str, Counter[str]]:
    """
    Aggregate remaining defects per rule for each round directory.
    Keys: Initial, Round1, Round2, Round3, Round4, Round5, Round6
    """
    model_dir = base_dir / model_name
    round_dirs = {
        "Initial": model_dir / "round_1",
        "Round1": model_dir / "round_1_after_round1",
        "Round2": model_dir / "round_2_after_round2",
        "Round3": model_dir / "round_3_after_round3",
        "Round4": model_dir / "round_4_after_round4",
        "Round5": model_dir / "round_5_after_round5",
        "Round6": model_dir / "round_6_after_round6",
    }

    out: Dict[str, Counter[str]] = {}
    for label, d in round_dirs.items():
        if not d.is_dir():
            continue
        agg: Counter[str] = Counter()
        for log_path in d.glob("*.log"):
            agg.update(parse_rule_counts(log_path))
        out[label] = agg
    return out


def write_markdown(
    output_path: Path,
    rows: List[
        Tuple[
            str,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
        ]
    ],
    model_name: str,
    rule_counts: Optional[Dict[str, Counter[str]]] = None,
) -> None:
    """Write markdown table summarizing defect stats."""
    # 合计行中，对于“未检测”的轮次按 0 缺陷统计。
    total_init_baseline = total_init_scan = 0
    total_r1 = total_r2 = total_r3 = total_r4 = total_r5 = total_r6 = 0
    for _, init_baseline, init_scan, r1, r2, r3, r4, r5, r6 in rows:
        if init_baseline is not None:
            total_init_baseline += init_baseline
        if init_scan is not None:
            total_init_scan += init_scan
        if r1 is not None:
            total_r1 += r1
        if r2 is not None:
            total_r2 += r2
        if r3 is not None:
            total_r3 += r3
        if r4 is not None:
            total_r4 += r4
        if r5 is not None:
            total_r5 += r5
        if r6 is not None:
            total_r6 += r6

    def fmt_cell(v: int | None) -> str:
        # None 表示该轮没有对应日志（未检测），用 '-' 与真实 0 区分
        return "-" if v is None else str(v)

    header = (
        f"{model_name} 六轮修复各项目缺陷统计\n"
        "========================================\n\n"
        f"数据来源：`logs/codelinter_openharmony/{model_name}` 下 "
        "`round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3`、"
        "`round_4_after_round4`、`round_5_after_round5`、`round_6_after_round6` "
        "的 CodeLinter 日志：优先按日志中的实际性能/安全条目数（带 @category/rule 的行）统计，"
        "若结构化解析失败再退回 `-Defects` 汇总行；\n"
        "“初始缺陷(基线)”取自 `revision/target_projects_haprepair.json` 中的性能/安全缺陷数（实验配置基线，稳定且与模型无关）；\n"
        "“初始缺陷(扫描)”取自本次运行 `round_1/<project>.log` 的实际扫描结果（可能受工具版本/环境影响）；\n"
        "表中数值 0 表示该轮缺陷数为 0（包括：有 after_round 日志，或该轮 round_N 日志已为 0 且修复脚本提前退出未写 after_round 日志）；"
        "'-' 表示该轮没有对应日志且无法推断（通常是未检测或中途失败）。\n\n"
        "| Project | 初始缺陷(基线) | 初始缺陷(扫描) | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 | Round4剩余缺陷 | Round5剩余缺陷 | Round6剩余缺陷 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for proj, init_baseline, init_scan, r1, r2, r3, r4, r5, r6 in rows:
            f.write(
                f"| {proj} | {fmt_cell(init_baseline)} | {fmt_cell(init_scan)} | {fmt_cell(r1)} | "
                f"{fmt_cell(r2)} | {fmt_cell(r3)} | {fmt_cell(r4)} | {fmt_cell(r5)} | {fmt_cell(r6)} |\n"
            )
        f.write(
            f"| 合计 | {total_init_baseline} | {total_init_scan} | {total_r1} | {total_r2} | {total_r3} | {total_r4} | {total_r5} | {total_r6} |\n"
        )

        # Append per-rule counts if provided
        if rule_counts:
            # Collect all rules across rounds
            all_rules = set()
            for c in rule_counts.values():
                all_rules.update(c.keys())
            all_rules = sorted(all_rules)

            def get_count(label: str, rule: str) -> Optional[int]:
                if label not in rule_counts:
                    return None
                return rule_counts[label].get(rule)

            f.write(
                "\n按规则统计（各轮剩余缺陷数，性能/安全条目）\n"
                "------------------------------------------\n\n"
                "| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 | Round6 |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            )
            for rule in all_rules:
                f.write(
                    f"| {rule} | "
                    f"{fmt_cell(get_count('Initial', rule))} | "
                    f"{fmt_cell(get_count('Round1', rule))} | "
                    f"{fmt_cell(get_count('Round2', rule))} | "
                    f"{fmt_cell(get_count('Round3', rule))} | "
                    f"{fmt_cell(get_count('Round4', rule))} | "
                    f"{fmt_cell(get_count('Round5', rule))} | "
                    f"{fmt_cell(get_count('Round6', rule))} |\n"
                )


def prompt_model_choice(base_dir: Path, default_model: str = "gpt-5-mini") -> str:
    """
    Let the user pick a model directory under base_dir.
    Accept either the index in the printed list or the directory name.
    """
    if not base_dir.is_dir():
        raise SystemExit(f"模型日志目录不存在：{base_dir}")

    models = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    if not models:
        raise SystemExit(f"在 {base_dir} 下没有找到可用的模型目录。")

    # 默认模型存在则采用，否则取第一个。
    default = default_model if default_model in models else models[0]

    print("可用模型：")
    for idx, name in enumerate(models, start=1):
        default_mark = " (默认)" if name == default else ""
        print(f"[{idx}] {name}{default_mark}")

    prompt = (
        f"请选择模型（输入编号或名称，直接回车使用默认 {default}）："
    )
    while True:
        choice = input(prompt).strip()
        if not choice:
            return default

        # 数字编号
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]

        # 目录名
        if choice in models:
            return choice

        print("无效输入，请重新选择。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate remaining-defects markdown from CodeLinter logs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model directory under logs/codelinter_openharmony (skip prompt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write markdown output (default: ./summary).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path("/home/LLMCodeRepair")
    base_dir = repo_root / "logs" / "codelinter_openharmony"
    model = args.model if args.model else prompt_model_choice(base_dir)

    # Load project list from revision/target_projects_haprepair.json so that
    # the summary strictly follows the HapRepair experiment configuration.
    target_path = repo_root / "revision" / "target_projects_haprepair.json"
    project_names: Optional[List[str]] = None
    initial_from_target: Optional[Dict[str, int]] = None
    if target_path.is_file():
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                project_names = []
                initial_from_target = {}
                for item in data:
                    if not isinstance(item, dict) or "name" not in item:
                        continue
                    name = str(item.get("name"))
                    project_names.append(name)
                    if "perf_defects" in item or "security_defects" in item:
                        perf = int(item.get("perf_defects") or 0)
                        sec = int(item.get("security_defects") or 0)
                        initial_from_target[name] = perf + sec
                    elif "total_defects" in item:
                        initial_from_target[name] = int(item.get("total_defects") or 0)
        except Exception:
            # If parsing fails, fall back to auto-discovery based on logs.
            project_names = None
            initial_from_target = None

    rows = collect_stats(
        base_dir=base_dir,
        model_name=model,
        project_names=project_names,
        initial_from_target=initial_from_target,
    )
    rule_counts = collect_rule_counts_by_round(base_dir=base_dir, model_name=model)
    summary_dir = args.output_dir if args.output_dir else (repo_root / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / f"{model}_remaining_defects.md"
    write_markdown(out_path, rows, model, rule_counts=rule_counts)

    print(f"Wrote {out_path} with {len(rows)} projects.")


if __name__ == "__main__":
    main()
