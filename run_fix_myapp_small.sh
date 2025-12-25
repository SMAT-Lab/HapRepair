#!/usr/bin/env bash
# Run fix_projects_codelinter on the reduced MyApplication2_small sample set.
# Outputs go to /home/LLMCodeRepair/tmp/fixed_samples_small and logs to
# /home/LLMCodeRepair/tmp/codelinter_myapp2_small.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load API_KEY / API_BASE from .env
set -a
source "$ROOT/.env"
export API_KEY="${API_KEY:-$OPENAI_API_KEY}"
export API_BASE="${API_BASE:-$OPENAI_API_BASE}"
set +a

PROJECT_ROOT="$ROOT/tmp/MyApplication2_small"
OUT_ROOT="$ROOT/tmp/fixed_samples_small"
LOG_DIR="$ROOT/tmp/codelinter_myapp2_small"
CONFIG="$PROJECT_ROOT/code-linter.json5"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

nohup python "$ROOT/revision/code/fix_projects_codelinter.py" \
  --project-root "$PROJECT_ROOT" \
  --output-root "$OUT_ROOT" \
  --log-dir "$LOG_DIR" \
  --config "$CONFIG" \
  --max-workers 2 \
  > "$ROOT/tmp/fix_myapp_small.nohup.log" 2>&1 &

echo "Started fix_projects_codelinter in background. PID=$!; tail -f $ROOT/tmp/fix_myapp_small.nohup.log"
