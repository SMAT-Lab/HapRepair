#!/usr/bin/env bash
set -euo pipefail

# Re-run the "context_full" ablation only for projects whose previous run logs
# indicate that no RAG examples were injected (i.e., missing "[RAG] rule=" lines).
#
# Usage:
#   bash revision/code/run_haprepair_context_full_rerun_missing_rag.sh [ROUND] [MODEL]
#
# Notes:
# - Uses the same run-tag as the original full-context ablation:
#     ${ABLATION_TAG_PREFIX:-ablation}_context_full
#   so it overwrites those per-project logs/snapshots when --force is used.
# - Regenerates the remaining-defects markdown under ${ABLATION_SUMMARY_DIR:-...}.
#
# Env overrides:
#   ABLATION_BASE_RAG_TYPE="difflib"
#   ABLATION_BASE_TOP_N="1"
#   ABLATION_TAG_PREFIX="ablation"
#   ABLATION_SUMMARY_DIR="/home/LLMCodeRepair/summary/ablation"
#   ABLATION_START_RAG="1"
#   ABLATION_RAG_HOST="127.0.0.1"
#   ABLATION_RAG_PORT="8000"
#   ABLATION_RAG_LOG="/home/LLMCodeRepair/logs/rag_service.log"

ROUND="${1:-1}"
MODEL="${2:-gpt-5.1}"

if [[ "${MODEL}" == *"="* ]]; then
  MODEL="${MODEL//=/-}"
fi

if [ -f "/home/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/miniconda3/etc/profile.d/conda.sh"
fi
conda activate VulRAG

if [ -f "/home/LLMCodeRepair/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  . "/home/LLMCodeRepair/.env"
  set +a
fi

if [ -n "${ZHIZENGZENG_API_KEY:-}" ]; then
  export API_KEY="${ZHIZENGZENG_API_KEY}"
fi
if [ -n "${ZHIZENGZENG_API_BASE:-}" ]; then
  export API_BASE="${ZHIZENGZENG_API_BASE}"
fi

TAG_PREFIX="${ABLATION_TAG_PREFIX:-ablation}"
BASE_RAG_TYPE="${ABLATION_BASE_RAG_TYPE:-difflib}"
BASE_TOP_N="${ABLATION_BASE_TOP_N:-1}"
SUMMARY_DIR="${ABLATION_SUMMARY_DIR:-/home/LLMCodeRepair/summary/ablation}"
RAG_START="${ABLATION_START_RAG:-1}"
RAG_HOST="${ABLATION_RAG_HOST:-127.0.0.1}"
RAG_PORT="${ABLATION_RAG_PORT:-8000}"
RAG_LOG="${ABLATION_RAG_LOG:-/home/LLMCodeRepair/logs/rag_service.log}"
RAG_TIMEOUT="${ABLATION_RAG_TIMEOUT:-120}"

rag_api_base() {
  local base="${RAG_API_BASE:-http://${RAG_HOST}:${RAG_PORT}}"
  base="${base%/}"
  if [[ "${base}" == */api/v1 ]]; then
    printf '%s' "$base"
  else
    printf '%s' "${base}/api/v1"
  fi
}

export RAG_API_BASE
RAG_API_BASE="$(rag_api_base)"
export RAG_API_TIMEOUT="${RAG_TIMEOUT}"

RAG_PID=""
cleanup_rag() {
  if [[ -n "${RAG_PID}" ]]; then
    echo "[context-full-rerun] stopping RAG service (pid=${RAG_PID})"
    kill "${RAG_PID}" >/dev/null 2>&1 || true
  fi
}

ensure_rag_service() {
  local status_url="${RAG_API_BASE}/rag/status"
  if curl -fsS "${status_url}" >/dev/null 2>&1; then
    if curl -fsS "${status_url}" | python -c "import json,sys; d=json.load(sys.stdin); ok=bool(d.get('success')) and bool(d.get('rag_available')) and bool(d.get('vector_db_connected')) and bool(d.get('embedding_model_loaded')); raise SystemExit(0 if ok else 1)"; then
      echo "[context-full-rerun] RAG service already running and ready at ${RAG_API_BASE}"
      return 0
    fi
    echo "[context-full-rerun] RAG service is running but not ready; will try to restart"
  fi

  if [[ ! "${RAG_START}" =~ ^(1|true|yes|on)$ ]]; then
    echo "[context-full-rerun] RAG service not running; ABLATION_START_RAG=0, skipping startup"
    return 0
  fi

  echo "[context-full-rerun] starting RAG service at http://${RAG_HOST}:${RAG_PORT}"
  mkdir -p "$(dirname "${RAG_LOG}")"
  pkill -f "uvicorn main:app.*--host ${RAG_HOST}.*--port ${RAG_PORT}" >/dev/null 2>&1 || true
  nohup uvicorn main:app --app-dir /home/LLMCodeRepair/server \
    --host "${RAG_HOST}" --port "${RAG_PORT}" \
    >"${RAG_LOG}" 2>&1 &
  RAG_PID=$!
  trap cleanup_rag EXIT

  local retries=60
  for _ in $(seq 1 "${retries}"); do
    if curl -fsS "${status_url}" >/dev/null 2>&1; then
      if curl -fsS "${status_url}" | python -c "import json,sys; d=json.load(sys.stdin); ok=bool(d.get('success')) and bool(d.get('rag_available')) and bool(d.get('vector_db_connected')) and bool(d.get('embedding_model_loaded')); raise SystemExit(0 if ok else 1)"; then
        echo "[context-full-rerun] RAG service is ready"
        return 0
      fi
    fi
    sleep 1
  done

  echo "[context-full-rerun] RAG service failed to start or is not ready; check ${RAG_LOG}"
  return 1
}

ensure_rag_service

run_tag="${TAG_PREFIX}_context_full"
sanitize_dir() {
  local s="$1"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s// /_}"
  printf '%s' "$s"
}

model_dir="$(sanitize_dir "${MODEL}")__$(sanitize_dir "${run_tag}")"

echo "[context-full-rerun] run_tag=${run_tag} rag_type=${BASE_RAG_TYPE} top_n=${BASE_TOP_N} context=full"
echo "[context-full-rerun] indices to re-run (previously missing RAG examples): 0 2 3 4 6 7 9 16 20 26 27 28 30 31 33"

indices=(0 2 3 4 6 7 9 16 20 26 27 28 30 31 33)

echo "[context-full-rerun] deleting old artifacts first (snapshots + logs) for the selected indices..."
for idx in "${indices[@]}"; do
  proj_name="$(python - <<PY
import json
from pathlib import Path
data=json.loads(Path('/home/LLMCodeRepair/revision/target_projects_haprepair.json').read_text(encoding='utf-8'))
root=data[int(${idx})]['root_path']
print(Path(root).name)
PY
)"

  rm -rf "/home/LLMCodeRepair/revision/fixed_projects/${model_dir}/round_${ROUND}/${proj_name}" || true
  rm -rf "/home/LLMCodeRepair/repo_new/_haprepair_fixed/${model_dir}/round_${ROUND}/${proj_name}" || true

  rm -f "/home/LLMCodeRepair/logs/fix_projects_codelinter/${model_dir}/round_${ROUND}/${proj_name}.log" || true

  rm -f "/home/LLMCodeRepair/logs/codelinter_openharmony/${model_dir}/round_${ROUND}/${proj_name}.log" || true
  rm -f "/home/LLMCodeRepair/logs/codelinter_openharmony/${model_dir}/round_${ROUND}_after_round${ROUND}/${proj_name}.log" || true
  rm -f "/home/LLMCodeRepair/logs/codelinter_openharmony/${model_dir}/round_${ROUND}_parsing/${proj_name}.log" || true
done

for idx in "${indices[@]}"; do
  echo "[context-full-rerun] index=${idx} $(date -Iseconds)"
  python /home/LLMCodeRepair/revision/code/run_haprepair_round.py \
    --index "${idx}" \
    --round "${ROUND}" \
    --model-name "${MODEL}" \
    --rag-type "${BASE_RAG_TYPE}" \
    --top-n "${BASE_TOP_N}" \
    --run-tag "${run_tag}" \
    --full-context \
    --force
done

python /home/LLMCodeRepair/gen_gpt5mini_remaining_defects.py \
  --model "${model_dir}" \
  --output-dir "${SUMMARY_DIR}"

echo "[context-full-rerun] done: ${SUMMARY_DIR}/${model_dir}_remaining_defects.md"
