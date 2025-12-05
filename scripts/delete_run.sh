#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <project_name> <start_round> <model_name>" >&2
  echo "Example: $0 audio_suite 1 gpt-5-mini" >&2
  exit 1
fi

PROJECT="$1"
START_ROUND="$2"
MODEL="$3"

if ! [[ "$START_ROUND" =~ ^[0-9]+$ ]]; then
  echo "ERROR: <start_round> must be a positive integer, got '$START_ROUND'" >&2
  exit 1
fi

echo "Deleting artifacts for project='${PROJECT}', model='${MODEL}', rounds >= ${START_ROUND}"

LOG_MODEL_DIR="logs/codelinter_openharmony/${MODEL}"
HAP_BASE="repo_new/_haprepair_fixed/${MODEL}"
REV_BASE="revision/fixed_projects/${MODEL}"

delete_round_dirs() {
  local base_dir="$1"      # e.g., logs/codelinter_openharmony/gpt-5-mini
  local kind="$2"          # description for logging
  local is_logs="$3"       # "yes" -> delete *.log; "no" -> delete project dir

  if [ ! -d "$base_dir" ]; then
    return 0
  }

  for d in "${base_dir}"/round_*; do
    [ -d "$d" ] || continue
    local name
    name="$(basename "$d")"
    # Extract the leading round number from names like:
    #   round_1
    #   round_1_after_round1
    #   round_1_parsing
    local round_num
    round_num="$(printf '%s\n' "$name" | sed -n 's/^round_\([0-9][0-9]*\).*/\1/p')"
    [ -n "$round_num" ] || continue
    if [ "$round_num" -lt "$START_ROUND" ]; then
      continue
    fi

    if [ "$is_logs" = "yes" ]; then
      local path="${d}/${PROJECT}.log"
      if [ -f "$path" ]; then
        echo "[delete] ${kind} log: $path"
        rm -f "$path"
      fi
    else
      local proj_dir="${d}/${PROJECT}"
      if [ -d "$proj_dir" ]; then
        echo "[delete] ${kind} dir: $proj_dir"
        rm -rf "$proj_dir"
      fi
    fi
  done
}

# 1) Delete CodeLinter logs under logs/codelinter_openharmony/<MODEL>/round_*.
delete_round_dirs "$LOG_MODEL_DIR" "codelinter" "yes"

# 2) Delete HapRepair workspace snapshots under repo_new/_haprepair_fixed/<MODEL>/round_*.
delete_round_dirs "$HAP_BASE" "workspace snapshot" "no"

# 3) Delete revision snapshots under revision/fixed_projects/<MODEL>/round_*.
delete_round_dirs "$REV_BASE" "revision snapshot" "no"

echo "Done."

