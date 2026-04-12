#!/usr/bin/env python3
"""
Compute the share of triggered *performance* rule types that are context-dependent
vs template-sufficient under our RQ2 prompt-construction policy.

Operationalization (mirrors the repair pipeline):
  - A rule type is "context-dependent" if it is flagged by rule metadata
    `needsMoreContext=true` in `rules.json`, OR it is in the hardcoded
    full-file-context allowlist used by `revision/code/fix_projects_codelinter.py`
    for high-freedom refactorings.
  - Triggered rule types are extracted from CodeLinter logs under:
      logs/codelinter_openharmony/<model_dir>/**.log
    using `@performance/<rule-id>` tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


RULE_TOKEN_RE = re.compile(r"@(?P<cat>[A-Za-z0-9_-]+)/(?P<rid>[A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class ContextPolicy:
    needs_more_context_true: Set[str]
    full_file_context_rules: Set[str]

    def is_context_dependent(self, perf_rule: str) -> bool:
        """
        perf_rule: "performance/<rule-id>"
        """
        if perf_rule in self.needs_more_context_true:
            return True
        _, rid = perf_rule.split("/", 1)
        return rid in self.full_file_context_rules


def load_needs_more_context_true(rules_json: Path) -> Set[str]:
    data = json.loads(rules_json.read_text(encoding="utf-8"))
    out: Set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("needsMoreContext")):
            continue
        rule = str(item.get("rule") or "")
        if rule.startswith("@"):
            rule = rule[1:]
        if rule.startswith("performance/"):
            out.add(rule)
    return out


def load_full_file_context_rules(fix_projects_codelinter_py: Path) -> Set[str]:
    """
    Parse the `context_rules = {...}` set literal in fix_projects_codelinter.py.
    The list is expected to contain bare rule ids (without category prefix).
    """
    text = fix_projects_codelinter_py.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"context_rules\s*=\s*\{(?P<body>[^}]+)\}", text, re.S)
    if not m:
        raise ValueError("Failed to locate context_rules set in fix_projects_codelinter.py")
    body = m.group("body")
    out: Set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.rstrip(",")
        line = line.strip().strip('"').strip("'")
        if line:
            out.add(line)
    return out


def iter_rule_tokens(text: str) -> Iterable[Tuple[str, str]]:
    for m in RULE_TOKEN_RE.finditer(text):
        yield m.group("cat"), m.group("rid")


def extract_triggered_perf_rules(log_root: Path) -> Set[str]:
    out: Set[str] = set()
    for log_path in sorted(log_root.rglob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for cat, rid in iter_rule_tokens(text):
            if cat == "performance":
                out.add(f"{cat}/{rid}")
    return out


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def render_markdown(
    *,
    model_dir: str,
    triggered_perf: List[str],
    context_dependent: List[str],
    template_sufficient: List[str],
) -> str:
    total = len(triggered_perf)
    ctx = len(context_dependent)
    local = len(template_sufficient)
    ratio = (ctx / total) if total else 0.0
    lines: List[str] = []
    lines.append(f"RQ2 triggered performance rule types: `{model_dir}`")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"- Triggered performance rule types: {total}")
    lines.append(f"- Context-dependent (needsMoreContext/full-file): {ctx} ({fmt_pct(ratio)})")
    lines.append(f"- Local/template-sufficient: {local} ({fmt_pct(1.0 - ratio)})")
    lines.append("")
    lines.append("## Context-dependent rules")
    for r in context_dependent:
        lines.append(f"- `{r}`")
    lines.append("")
    lines.append("## Local/template-sufficient rules")
    for r in template_sufficient:
        lines.append(f"- `{r}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="gpt-5.1", help="logs/codelinter_openharmony/<model-dir>")
    ap.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/codelinter_openharmony"),
        help="Root of CodeLinter logs (default: logs/codelinter_openharmony).",
    )
    ap.add_argument(
        "--rules-json",
        type=Path,
        default=Path("rules.json"),
        help="Path to rules.json containing needsMoreContext annotations.",
    )
    ap.add_argument(
        "--fixer-py",
        type=Path,
        default=Path("revision/code/fix_projects_codelinter.py"),
        help="Path to fix_projects_codelinter.py (source of full-file context allowlist).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path (prints to stdout if omitted).",
    )
    args = ap.parse_args()

    log_dir = args.log_root / args.model_dir
    if not log_dir.is_dir():
        raise SystemExit(f"Log dir not found: {log_dir}")

    policy = ContextPolicy(
        needs_more_context_true=load_needs_more_context_true(args.rules_json),
        full_file_context_rules=load_full_file_context_rules(args.fixer_py),
    )

    triggered_perf = sorted(extract_triggered_perf_rules(log_dir))
    ctx = sorted([r for r in triggered_perf if policy.is_context_dependent(r)])
    local = sorted([r for r in triggered_perf if r not in set(ctx)])

    md = render_markdown(
        model_dir=args.model_dir,
        triggered_perf=triggered_perf,
        context_dependent=ctx,
        template_sufficient=local,
    )

    if args.output is None:
        print(md)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
