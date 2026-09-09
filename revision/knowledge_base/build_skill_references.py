#!/usr/bin/env python3
"""Build deterministic per-rule repair references for the HapRepair Skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def slug(rule: str) -> str:
    value = rule.removeprefix("@").replace("/", "--")
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") + ".md"


def fenced(value: str) -> str:
    fence = "````" if "```" in value else "```"
    return f"{fence}arkts\n{value.rstrip()}\n{fence}"


def render_rule(rule: str, pairs: list[dict[str, Any]]) -> str:
    lines = [
        f"# {rule}",
        "",
        f"Static repair references: {len(pairs)}. These are examples, not patches to copy.",
        "Inspect repository context and checker evidence before adapting an example.",
        "",
    ]
    for index, pair in enumerate(pairs, 1):
        lines.extend(
            [
                f"## Example {index}: `{pair['pair_id']}`",
                "",
                str(pair.get("description") or "No additional rule description."),
                "",
                "### Triggering pattern",
                "",
                fenced(str(pair.get("problem_code") or "")),
                "",
                "### Repair pattern",
                "",
                fenced(str(pair.get("repair_code") or "")),
                "",
            ]
        )
        explanation = str(pair.get("problem_explanation") or "").strip()
        if explanation and explanation != str(pair.get("description") or "").strip():
            lines.extend(["### Rationale", "", explanation, ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    output = args.output.resolve()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pair = json.loads(line)
            grouped[str(pair["rule"])].append(pair)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    index = []
    for rule in sorted(grouped):
        pairs = sorted(grouped[rule], key=lambda item: str(item["pair_id"]))
        filename = slug(rule)
        content = render_rule(rule, pairs)
        (output / filename).write_text(content, encoding="utf-8")
        index.append(
            {
                "rule": rule,
                "file": filename,
                "example_count": len(pairs),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "source_corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "pair_count": sum(item["example_count"] for item in index),
        "rule_count": len(index),
        "rules": index,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = [
        "# Repair Reference Index",
        "",
        "Open the exact per-rule file before repairing an observed rule.",
        "The examples are static author-written repair guidance, not retrieval results.",
        "",
        "| Rule | Examples | Reference |",
        "| --- | ---: | --- |",
    ]
    table.extend(
        f"| `{item['rule']}` | {item['example_count']} | [{item['file']}]({item['file']}) |"
        for item in index
    )
    (output / "index.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
