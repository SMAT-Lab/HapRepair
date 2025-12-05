#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional


def _ensure_json_flag(extra_args: Optional[Iterable[str]]) -> List[str]:
    """Return a list of extra args, ensuring '--json' is present."""
    args: List[str] = list(extra_args) if extra_args is not None else []
    if "--json" not in args:
        args.append("--json")
    return args


def run_codex(
    prompt: str,
    extra_args: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Codex CLI with the given prompt and return a structured result:

    {
        "trace": [event_dict, ...],   # parsed JSONL events from Codex
        "usage": {...} or None,       # last 'usage' object seen in events
        "final_answer": str or None,  # text of the last agent_message/item
        "raw_output": str,            # raw stdout from Codex (JSONL text)
    }

    - prompt: text passed as the first positional argument to `codex exec`
    - extra_args: any extra CLI args to pass through to Codex
      (the '--json' flag will be added automatically if missing).
    - output_path: optional path to a JSON file. If provided, a snapshot of
      the result is written after each new event is read (trace/usage/final).

    Raises RuntimeError if Codex CLI is not found or exits with non-zero code.
    """
    if not shutil.which("codex"):
        raise RuntimeError("Codex CLI not found. Please install and authenticate 'codex'.")

    args: List[str] = ["codex", "exec", prompt]
    args.extend(_ensure_json_flag(extra_args))

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    events: List[Dict[str, Any]] = []
    usage: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    raw_output_parts: List[str] = []

    stderr_buf: List[str] = []

    def _pump_stderr() -> None:
        for line in proc.stderr or []:
            stderr_buf.append(line)

    t_err = threading.Thread(target=_pump_stderr, daemon=True)
    t_err.start()

    def _write_snapshot(intermediate: bool = False) -> None:
        """Write current snapshot to output_path, if provided.

        When intermediate=True and we have no structured events yet, we avoid
        prematurely treating raw_output as final_answer.
        """
        if not output_path:
            return

        raw_output = "".join(raw_output_parts)
        snapshot_final = final_answer
        snapshot_usage = usage

        # In early stages with no events parsed, do not guess final_answer yet.
        if intermediate and not events:
            snapshot_final = None

        snapshot = {
            "trace": events,
            "usage": snapshot_usage,
            "final_answer": snapshot_final,
            "raw_output": raw_output,
        }

        dir_name = os.path.dirname(os.path.abspath(output_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    # Stream Codex stdout line by line, updating trace/usage/final_answer
    for line in proc.stdout or []:
        raw_output_parts.append(line)
        stripped = line.strip()
        if stripped:
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                ev = None

            if isinstance(ev, dict):
                events.append(ev)

                # Usage usually appears on e.g. turn.completed
                if "usage" in ev and isinstance(ev["usage"], dict):
                    usage = ev["usage"]

                # Final answer: last completed agent/message item with text/content
                if ev.get("type") == "item.completed":
                    item = ev.get("item") or {}
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type in ("agent_message", "message", "assistant_message"):
                            text_val: Optional[str] = None
                            if isinstance(item.get("text"), str):
                                text_val = item["text"]
                            else:
                                content = item.get("content")
                                # content may be a string or list of segments
                                if isinstance(content, str):
                                    text_val = content
                                elif isinstance(content, list):
                                    parts: List[str] = []
                                    for seg in content:
                                        if (
                                            isinstance(seg, dict)
                                            and isinstance(seg.get("text"), str)
                                        ):
                                            parts.append(seg["text"])
                                    if parts:
                                        text_val = "".join(parts)

                            if text_val is not None:
                                final_answer = text_val

        # After processing each line, write an intermediate snapshot
        _write_snapshot(intermediate=True)

    proc.wait()
    t_err.join(timeout=0.5)

    stdout = "".join(raw_output_parts)

    if proc.returncode != 0:
        err_text = "".join(stderr_buf)
        raise RuntimeError(f"codex exec failed (code {proc.returncode}):\n{err_text}")

    # If no events parsed, treat full stdout as final answer
    if not events and stdout:
        final_answer = stdout

    # Final snapshot (ensures final_answer is populated if possible)
    _write_snapshot(intermediate=False)

    return {
        "trace": events,
        "usage": usage,
        "final_answer": final_answer,
        "raw_output": stdout,
    }


def main() -> None:
    """
    CLI wrapper: read prompt from stdin, forward any extra args,
    print only the final answer (if available), otherwise raw Codex output.
    """
    prompt = sys.stdin.read()
    if not prompt:
        return

    extra_args = sys.argv[1:]
    try:
        result = run_codex(prompt, extra_args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    output = result.get("final_answer") or result.get("raw_output") or ""
    sys.stdout.write(output)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
