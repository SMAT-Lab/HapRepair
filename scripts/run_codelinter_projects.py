#!/usr/bin/env python3
"""
Batch runner for CodeLinter across multiple OpenHarmony projects.

It scans the target root directory, finds child folders that look like
valid Harmony projects (presence of hvigor or build-profile), and
runs `codelinter <project_path>` for each of them.  The stdout/stderr
of every run is saved to an individual log file for later review.
"""
from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import os
import time
import errno
from pathlib import Path
from typing import IO, Iterable, List, Tuple
import re


HVIGOR_FILES = ("hvigorfile.ts", "hvigorfile.js")
SEVERITIES = ("error", "warn", "suggestion")
RULE_PATTERN = re.compile(r"^\s*\d+:\d+\s+(\w+)\s+.*@(performance|security)/\S+")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw not in ("0", "false", "no", "off", "")


@contextlib.contextmanager
def _codelinter_global_lock() -> Iterable[None]:
    """
    Serialize CodeLinter executions across processes to avoid CPU overload.

    Enabled by default. Disable with `CODELINTER_DISABLE_LOCK=1`.
    Override lock path with `CODELINTER_LOCK_FILE=/path/to/lock`.
    """
    if _env_flag("CODELINTER_DISABLE_LOCK", default=False):
        yield
        return

    lock_path = Path(os.environ.get("CODELINTER_LOCK_FILE", "/tmp/LLMCodeRepair_codelinter.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # flock works across processes; keep the fd open for the duration of the run.
    try:
        import fcntl  # Linux/Unix only
    except Exception:
        # If flock isn't available, best-effort: run without cross-process locking.
        yield
        return

    fh: IO[str] = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"[wait] Another CodeLinter run is in progress; waiting for global lock: {lock_path}",
                file=sys.stderr,
            )
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch CodeLinter runner for multiple projects.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/LLMCodeRepair/repo"),
        help="Directory that contains candidate project folders (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/LLMCodeRepair/logs/codelinter"),
        help="Directory used to store CodeLinter logs for each project.",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="Optional specific project folder names (relative to --root). "
        "If omitted, the script recursively searches for project roots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/LLMCodeRepair/revision/code-linter.json5"),
        help="Optional CodeLinter config file (.json/.json5) passed via -c/--config.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="After linting, summarize each project's performance/security issues in Markdown.",
    )
    return parser.parse_args()


def find_projects(root: Path, names: Iterable[str] | None = None) -> List[Path]:
    projects: List[Path] = []
    if names:
        for raw in names:
            path = (root / Path(raw)).resolve()
            if not path.exists():
                print(f"[skip] {path} does not exist.", file=sys.stderr)
                continue
            if is_project_root(path):
                projects.append(path)
            else:
                print(f"[skip] {path} is missing hvigor/build-profile files.", file=sys.stderr)
        return projects

    discovered: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if is_project_root(path):
            discovered.append(path)
            dirnames[:] = []
    return sorted(discovered)


def is_project_root(path: Path) -> bool:
    if any((path / name).is_file() for name in HVIGOR_FILES):
        return True
    if (path / "build-profile.json5").is_file():
        return True
    if (path / "oh-package.json5").is_file():
        return True
    return False


def run_codelinter(target: Path, log_dir: Path, config: Path | None) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{target.name}.log"
    cmd = ["codelinter"]
    if config:
        cmd += ["--config", str(config)]
    cmd.append(str(target))

    def write_log_with_retries(content: str) -> None:
        # Some environments intermittently throw EIO while writing large logs.
        # Retry a few times to avoid losing the after_round log.
        retries = int(os.environ.get("CODELINTER_LOG_WRITE_RETRIES", "5"))
        delay = float(os.environ.get("CODELINTER_LOG_WRITE_RETRY_DELAY", "0.2"))
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                log_path.write_text(content, encoding="utf-8", errors="ignore")
                return
            except OSError as exc:
                last_exc = exc
                if getattr(exc, "errno", None) not in (errno.EIO,):
                    raise
                if attempt == retries - 1:
                    break
                time.sleep(delay * (attempt + 1))
        if last_exc:
            raise last_exc

    with _codelinter_global_lock():
        print(f"[run ] {' '.join(cmd)}")
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            stdout, stderr = proc.communicate()
            content = stdout + ("\n" if stdout and not stdout.endswith("\n") else "") + stderr
            write_log_with_retries(content)
            if proc.returncode == 0:
                print(f"[ ok ] {target} (see {log_path})")
            else:
                print(f"[fail] {target} (exit={proc.returncode}, see {log_path})", file=sys.stderr)
            return proc.returncode or 0


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Root directory {root} does not exist or is not a directory.", file=sys.stderr)
        return 1

    config = args.config.resolve() if args.config else None
    if config and not config.is_file():
        print(f"Config file {config} does not exist.", file=sys.stderr)
        return 1

    projects = find_projects(root, args.project)
    if not projects:
        print("No valid Harmony projects were found.", file=sys.stderr)
        return 1

    failures = 0
    processed: List[Tuple[Path, int]] = []
    for project in projects:
        rc = run_codelinter(project, args.log_dir, config)
        processed.append((project, rc))
        if rc != 0:
            failures += 1

    print(f"Completed CodeLinter runs for {len(projects)} project(s). Failures: {failures}.")

    if args.summary and processed:
        print()
        print("Markdown summary of CodeLinter findings:")
        print(generate_markdown_summary(args.log_dir, [proj for proj, _ in processed]))

    return 0 if failures == 0 else 2


def generate_markdown_summary(log_dir: Path, projects: List[Path]) -> str:
    table = ["| Project | Performance issues | Security issues |", "|---|---:|---:|"]
    severity_totals = {
        "performance": {sev: 0 for sev in SEVERITIES},
        "security": {sev: 0 for sev in SEVERITIES},
    }

    for project in projects:
        perf, sec = summarize_project_log(log_dir / f"{project.name}.log", severity_totals)
        table.append(f"| {project.name} | {perf} | {sec} |")

    severity_lines = []
    for category in ("performance", "security"):
        parts = ", ".join(f"{sev} {severity_totals[category][sev]}" for sev in SEVERITIES)
        severity_lines.append(f"- {category}: {parts}")

    return "\n".join(table + ["", "Severity totals:", *severity_lines])


def summarize_project_log(log_path: Path, totals: dict) -> Tuple[int, int]:
    perf = sec = 0
    if not log_path.is_file():
        return perf, sec
    with log_path.open() as fh:
        for line in fh:
            match = RULE_PATTERN.search(line)
            if not match:
                continue
            severity = match.group(1).lower()
            if severity == "warning":
                severity = "warn"
            if severity not in SEVERITIES:
                continue
            category = match.group(2)
            if category == "performance":
                perf += 1
            else:
                sec += 1
            totals[category][severity] += 1
    return perf, sec


if __name__ == "__main__":
    raise SystemExit(main())
