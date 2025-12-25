#!/usr/bin/env bash
#
# Detect projects whose HapRepair run for a given model/round did not finish.
# Heuristics:
# - fix_projects_codelinter log exists under logs/fix_projects_codelinter/<model>/round_<round>/<project>.log
# - Log contains the marker "Repair completed"
# - Snapshot exists under revision/fixed_projects/<model>/round_<round>/<project>
#
# Usage: bash scripts/find_incomplete_runs.sh [model] [round]
# Defaults: model=gpt-5-mini, round=1

set -euo pipefail

MODEL="${1:-gpt-5-mini}"
ROUND="${2:-1}"

LOG_ROOT="logs/fix_projects_codelinter/${MODEL}/round_${ROUND}"
REV_ROOT="revision/fixed_projects/${MODEL}/round_${ROUND}"

if [ ! -d "$LOG_ROOT" ]; then
  echo "[error] Log directory not found: ${LOG_ROOT}" >&2
  exit 1
fi

if [ ! -d "$REV_ROOT" ]; then
  echo "[warn] Snapshot directory not found: ${REV_ROOT}" >&2
fi

incomplete=()
complete=()

shopt -s nullglob
for log in "${LOG_ROOT}"/*.log; do
  project="$(basename "${log%.log}")"
  snapshot="${REV_ROOT}/${project}"

  has_marker=0
  if grep -q "Repair completed" "$log"; then
    has_marker=1
  fi

  has_snapshot=0
  if [ -d "$snapshot" ]; then
    has_snapshot=1
  fi

  if [ "$has_marker" -eq 1 ] && [ "$has_snapshot" -eq 1 ]; then
    complete+=("$project")
  else
    incomplete+=("$project")
  fi
done
shopt -u nullglob

echo "Model=${MODEL}, round=${ROUND}"
echo "Log dir: ${LOG_ROOT}"
echo "Snapshot dir: ${REV_ROOT}"
echo

echo "Incomplete projects:"
if [ "${#incomplete[@]}" -eq 0 ]; then
  echo "  (none)"
else
  for p in "${incomplete[@]}"; do
    bash scripts/delete_run.sh "$p" "$ROUND"  "$MODEL" 
    echo "  - $p"
  done
fi

echo
echo "Completed projects:"
if [ "${#complete[@]}" -eq 0 ]; then
  echo "  (none)"
else
  for p in "${complete[@]}"; do
    echo "  - $p"
  done
fi
