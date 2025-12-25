#!/usr/bin/env bash
# Parallel CodeLinter runner for repo_new projects.

set -uo pipefail

ROOT="${1:-/home/LLMCodeRepair/repo_new}"
LOG_DIR="${2:-/home/LLMCodeRepair/logs/codelinter_repo_new}"
CONFIG="${3:-/home/LLMCodeRepair/revision/code-linter.json5}"

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_LIST="$(mktemp)"

cleanup() {
  rm -f "$PROJECT_LIST"
}
trap cleanup EXIT

if [[ ! -d "$ROOT" ]]; then
  echo "Root directory not found: $ROOT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

python3 - "$ROOT" "$PROJECT_LIST" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2])
script_dir = Path(sys.argv[3])
sys.path.append(str(script_dir))

from run_codelinter_projects import find_projects  # noqa: E402

projects = find_projects(root)
if not projects:
    sys.exit("No valid Harmony projects were found.")

with out_path.open("w", encoding="utf-8") as fh:
    for proj in projects:
        fh.write(str(proj) + "\n")
PY

mkdir -p "$LOG_DIR"

lint_cmd_status=0
export LOG_DIR CONFIG

if command -v parallel >/dev/null 2>&1; then
  parallel -j "$(nproc)" --bar '
    out="${LOG_DIR}/{/}.log"
    echo "[run] {} -> ${out}"
    codelinter --config "${CONFIG}" "{}" >"${out}" 2>&1
  ' :::: "$PROJECT_LIST" || lint_cmd_status=$?
else
  echo "GNU parallel not found; falling back to xargs (no progress bar)." >&2
  xargs -P "$(nproc)" -I{} bash -lc '
    out="${LOG_DIR}/$(basename "{}").log"
    echo "[run] {} -> ${out}"
    codelinter --config "${CONFIG}" "{}" >"${out}" 2>&1
  ' :::: "$PROJECT_LIST" || lint_cmd_status=$?
fi

python3 - "$LOG_DIR" "$PROJECT_LIST" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
project_list = Path(sys.argv[2])
script_dir = Path(sys.argv[3])
sys.path.append(str(script_dir))

from run_codelinter_projects import generate_markdown_summary  # noqa: E402

projects = [Path(line.strip()) for line in project_list.read_text(encoding="utf-8").splitlines() if line.strip()]
if not projects:
    sys.exit("No projects to summarize.")

print()
print("Markdown summary of CodeLinter findings:")
print(generate_markdown_summary(log_dir, projects))
PY

exit "$lint_cmd_status"
