#!/usr/bin/env python3
"""
Lightweight rule-based fixes for simple, non-semantic CodeLinter violations.

Current focus: add missing keyGenerator for ForEach (foreeach-index-check).
This is intentionally conservative: we only touch ForEach calls with exactly
two arguments (data source + item lambda) and append a stable key generator
using the first lambda parameter name.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def split_top_level_args(arg_str: str) -> List[str]:
    """Split argument string by top-level commas (paren/bracket/brace aware)."""
    args: List[str] = []
    buf: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    escape = False

    for ch in arg_str:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            continue
        if in_double:
            buf.append(ch)
            if ch == '"':
                in_double = False
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            continue

        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1

        if ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def add_foreach_key_generator(content: str) -> tuple[str, bool]:
    """
    Find ForEach calls with exactly two arguments and append a key generator:
    ForEach(data, (item, index) => { ... })
    -> ForEach(data, (item, index) => { ... }, (item) => item)
    """
    out_parts: List[str] = []
    idx = 0
    needle = "ForEach("
    changed = False

    while True:
        start = content.find(needle, idx)
        if start == -1:
            out_parts.append(content[idx:])
            break
        out_parts.append(content[idx:start])
        # Find matching closing paren for this call
        i = start + len(needle)
        depth = 1
        in_single = False
        in_double = False
        escape = False
        while i < len(content):
            ch = content[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_single:
                if ch == "'":
                    in_single = False
            elif in_double:
                if ch == '"':
                    in_double = False
            else:
                if ch == "'":
                    in_single = True
                elif ch == '"':
                    in_double = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        if depth != 0:
            # Unbalanced; give up on this occurrence.
            out_parts.append(content[start:])
            break

        call_str = content[start + len(needle) : i]
        args = split_top_level_args(call_str)
        if len(args) == 2:
            lambda_src = args[1]
            item_name = "item"
            # Try to extract first parameter name from lambda signature
            # e.g., "(foo, index) =>" -> foo
            for token in lambda_src.split():
                if token.startswith("("):
                    token = token[1:]
                if token and (token[0].isalpha() or token.startswith("_")):
                    # Strip trailing punctuation/colon
                    name = ""
                    for ch in token:
                        if ch.isalnum() or ch == "_":
                            name += ch
                        else:
                            break
                    if name:
                        item_name = name
                    break
            key_gen = f"({item_name}) => {item_name}"
            new_call = f"{args[0]}, {args[1]}, {key_gen}"
            out_parts.append(f"{needle}{new_call})")
            changed = True
        else:
            # Leave untouched
            out_parts.append(f"{needle}{call_str})")
        idx = i + 1

    return "".join(out_parts), changed


def process_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    new_src, changed = add_foreach_key_generator(src)
    if changed and new_src != src:
        path.write_text(new_src, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply simple rule-based fixes (non-semantic)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to scan (*.ets files).",
    )
    args = parser.parse_args()

    ets_files = list(args.root.rglob("*.ets"))
    total_changed = 0
    for path in ets_files:
        if process_file(path):
            total_changed += 1
    print(f"Processed {len(ets_files)} files; modified {total_changed} files.")


if __name__ == "__main__":
    main()
