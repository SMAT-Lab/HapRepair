#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Dict, Tuple, List


DEFECTS_RE = re.compile(r"-Defects:\s*(\d+)")


def parse_defects_from_log(path: Path) -> int:
    """
    Parse Defects count from a single CodeLinter log.

    Note: this function is only called when the log file exists. If the log
    does not contain a `-Defects:` summary line we treat it as 0 defects for
    that round.
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


def collect_stats(
    base_dir: Path, model_name: str = "gpt-5-mini"
) -> List[Tuple[str, int | None, int | None, int | None, int | None]]:
    """
    Collect per-project defect stats:
    returns list of (project, initial, r1, r2, r3).
    """
    model_dir = base_dir / model_name

    # Initial defects from round_1
    initial_dir = model_dir / "round_1"

    # Remaining defects after each repair round
    round_dirs = [
        ("Round1", model_dir / "round_1_after_round1"),
        ("Round2", model_dir / "round_2_after_round2"),
        ("Round3", model_dir / "round_3_after_round3"),
    ]

    # 1) Initial defects
    initial: Dict[str, int] = {}
    if initial_dir.is_dir():
        for log_path in sorted(initial_dir.glob("*.log")):
            proj = log_path.stem
            initial[proj] = parse_defects_from_log(log_path)

    # 2) Remaining defects for each round
    remaining: Dict[str, Dict[str, int]] = {}
    for label, d in round_dirs:
        if not d.is_dir():
            continue
        for log_path in sorted(d.glob("*.log")):
            proj = log_path.stem
            defects = parse_defects_from_log(log_path)
            remaining.setdefault(proj, {})[label] = defects

    # 3) Union of projects
    projects = sorted(set(initial.keys()) | set(remaining.keys()))
    rows: List[Tuple[str, int | None, int | None, int | None, int | None]] = []
    for proj in projects:
        # 初始缺陷：只有在 round_1 有日志时才有值；否则视为“未检测”
        init = initial.get(proj)
        # 每一轮剩余缺陷：只有在对应 *_after_roundX 目录下有日志时才有值。
        r1 = remaining.get(proj, {}).get("Round1")
        r2 = remaining.get(proj, {}).get("Round2")
        r3 = remaining.get(proj, {}).get("Round3")
        rows.append((proj, init, r1, r2, r3))

    return rows


def write_markdown(
    output_path: Path,
    rows: List[Tuple[str, int | None, int | None, int | None, int | None]],
    model_name: str,
) -> None:
    """Write markdown table summarizing defect stats."""
    # 合计行中，对于“未检测”的轮次按 0 缺陷统计。
    total_init = sum(v for _, v, _, _, _ in rows if v is not None)
    total_r1 = sum(v for _, _, v, _, _ in rows if v is not None)
    total_r2 = sum(v for _, _, _, v, _ in rows if v is not None)
    total_r3 = sum(v for _, _, _, _, v in rows if v is not None)

    def fmt_cell(v: int | None) -> str:
        # None 表示这一轮没有对应的日志（未检测），用 '-' 标记与真实的 0 缺陷区分开。
        return "-" if v is None else str(v)

    header = (
        f"{model_name} 三轮修复前三轮各项目缺陷统计\n"
        "========================================\n\n"
        f"数据来源：`logs/codelinter_openharmony/{model_name}` 下的 "
        "`round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3` "
        "日志中的 `Defects` 汇总；\n"
        "表中数值 0 表示该轮已运行 CodeLinter 且缺陷数为 0，'-' 表示该轮没有对应日志（未检测）。\n\n"
        "| Project | 初始缺陷 | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 |\n"
        "| --- | --- | --- | --- | --- |\n"
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for proj, init, r1, r2, r3 in rows:
            f.write(
                f"| {proj} | {fmt_cell(init)} | {fmt_cell(r1)} | "
                f"{fmt_cell(r2)} | {fmt_cell(r3)} |\n"
            )
        f.write(f"| 合计 | {total_init} | {total_r1} | {total_r2} | {total_r3} |\n")


def main() -> None:
    repo_root = Path("/home/LLMCodeRepair")
    base_dir = repo_root / "logs" / "codelinter_openharmony"
    model = "gpt-5-mini"

    rows = collect_stats(base_dir=base_dir, model_name=model)
    out_path = repo_root / f"{model}_remaining_defects.md"
    write_markdown(out_path, rows, model)

    print(f"Wrote {out_path} with {len(rows)} projects.")


if __name__ == "__main__":
    main()
