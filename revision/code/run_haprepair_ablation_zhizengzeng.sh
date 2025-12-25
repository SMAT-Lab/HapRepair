#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash revision/code/run_haprepair_ablation_zhizengzeng.sh [ROUND] [MODEL] [MAX_PROJECT_WORKERS]
#
# Defaults:
#   ROUND = 1              (repair round number)
#   MODEL = gpt-5.1        (LLM model name passed to Python)
#   MAX_PROJECT_WORKERS=1  (不同项目之间的最大并行数，1 表示顺序执行)
#
# This script runs all ablation experiments:
#   1) RAG on/off + top-k
#   2) Diff type (gpt_diff / difflib / no_diff)
#   3) Context strategy (surrounding / full file)
#
# Customize with env vars (space-separated lists):
#   ABLATION_TOP_NS="1 3 5"
#   ABLATION_DIFF_TYPES="gpt_diff difflib no_diff"
#   ABLATION_CONTEXTS="surrounding full"
#   ABLATION_BASE_RAG_TYPE="difflib"
#   ABLATION_BASE_TOP_N="1"
#   ABLATION_BASE_CONTEXT="surrounding"
#   ABLATION_TAG_PREFIX="ablation"
#   ABLATION_SUMMARY_DIR="/home/LLMCodeRepair/summary/ablation"
#   ABLATION_SKIP_BASELINE="1"  # skip baseline even if not present
#   ABLATION_START_RAG="1"      # start RAG service locally if not running
#   ABLATION_RAG_HOST="127.0.0.1"
#   ABLATION_RAG_PORT="8000"
#   ABLATION_RAG_LOG="/home/LLMCodeRepair/logs/rag_service.log"

ROUND="${1:-1}"
MODEL="${2:-gpt-5.1}"
MAX_PROJECT_WORKERS="${3:-1}"

# Normalize model ids like qwen3=30b-a3b -> qwen3-30b-a3b
if [[ "${MODEL}" == *"="* ]]; then
  MODEL="${MODEL//=/-}"
fi

# 激活 VulRAG 环境
if [ -f "/home/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/miniconda3/etc/profile.d/conda.sh"
fi
conda activate VulRAG

# 如果根目录下有 .env，则先加载（将其中的变量 export 出来）
if [ -f "/home/LLMCodeRepair/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  . "/home/LLMCodeRepair/.env"
  set +a
fi

# 将 ZHIZENGZENG_* 映射到通用 API_*，供 llm.py 使用
if [ -n "${ZHIZENGZENG_API_KEY:-}" ]; then
  export API_KEY="${ZHIZENGZENG_API_KEY}"
fi
if [ -n "${ZHIZENGZENG_API_BASE:-}" ]; then
  export API_BASE="${ZHIZENGZENG_API_BASE}"
fi

TAG_PREFIX="${ABLATION_TAG_PREFIX:-ablation}"
BASE_RAG_TYPE="${ABLATION_BASE_RAG_TYPE:-difflib}"
BASE_TOP_N="${ABLATION_BASE_TOP_N:-1}"
BASE_CONTEXT="${ABLATION_BASE_CONTEXT:-surrounding}"
SUMMARY_DIR="${ABLATION_SUMMARY_DIR:-/home/LLMCodeRepair/summary/ablation}"
RAG_START="${ABLATION_START_RAG:-1}"
RAG_HOST="${ABLATION_RAG_HOST:-127.0.0.1}"
RAG_PORT="${ABLATION_RAG_PORT:-8000}"
RAG_LOG="${ABLATION_RAG_LOG:-/home/LLMCodeRepair/logs/rag_service.log}"
RAG_TIMEOUT="${ABLATION_RAG_TIMEOUT:-120}"

read -r -a RAG_TOP_NS <<< "${ABLATION_TOP_NS:-1 3 5}"
read -r -a DIFF_TYPES <<< "${ABLATION_DIFF_TYPES:-gpt_diff difflib no_diff}"
read -r -a CONTEXTS <<< "${ABLATION_CONTEXTS:-surrounding full}"

sanitize_dir() {
  local s="$1"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s// /_}"
  printf '%s' "$s"
}

ensure_full_context() {
  local has_full=0
  for ctx in "${CONTEXTS[@]}"; do
    if [[ "${ctx}" == "full" ]]; then
      has_full=1
      break
    fi
  done
  if [[ "${has_full}" -eq 0 ]]; then
    CONTEXTS+=("full")
    echo "[ablation] adding missing context setting: full"
  fi
}

rag_api_base() {
  local base="${RAG_API_BASE:-http://${RAG_HOST}:${RAG_PORT}}"
  base="${base%/}"
  if [[ "${base}" == */api/v1 ]]; then
    printf '%s' "$base"
  else
    printf '%s' "${base}/api/v1"
  fi
}

RAG_API_BASE="$(rag_api_base)"
export RAG_API_BASE
export RAG_API_TIMEOUT="${RAG_TIMEOUT}"

RAG_PID=""
cleanup_rag() {
  if [[ -n "${RAG_PID}" ]]; then
    echo "[ablation] stopping RAG service (pid=${RAG_PID})"
    kill "${RAG_PID}" >/dev/null 2>&1 || true
  fi
}

ensure_rag_service() {
  local status_url="${RAG_API_BASE}/rag/status"
  if curl -fsS "${status_url}" >/dev/null 2>&1; then
    # Verify the service is actually usable (connected + embedding loaded)
    if curl -fsS "${status_url}" | python -c "import json,sys; d=json.load(sys.stdin); ok=bool(d.get('success')) and bool(d.get('rag_available')) and bool(d.get('vector_db_connected')) and bool(d.get('embedding_model_loaded')); raise SystemExit(0 if ok else 1)"; then
      echo "[ablation] RAG service already running and ready at ${RAG_API_BASE}"
      return 0
    fi
    echo "[ablation] RAG service is running but not ready (vector DB/embedding not available); will try to restart"
  fi

  if [[ ! "${RAG_START}" =~ ^(1|true|yes|on)$ ]]; then
    echo "[ablation] RAG service not running; ABLATION_START_RAG=0, skipping startup"
    return 0
  fi

  echo "[ablation] starting RAG service at http://${RAG_HOST}:${RAG_PORT}"
  mkdir -p "$(dirname "${RAG_LOG}")"
  # Best-effort cleanup if something is already bound to the port.
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
        echo "[ablation] RAG service is ready"
        return 0
      fi
    fi
    sleep 1
  done

  echo "[ablation] RAG service failed to start; check ${RAG_LOG}"
  return 1
}

ensure_full_context

model_dir_for_tag() {
  local tag="$1"
  local model_dir
  model_dir="$(sanitize_dir "${MODEL}")__$(sanitize_dir "${TAG_PREFIX}_${tag}")"
  printf '%s' "$model_dir"
}

should_skip_baseline() {
  local flag="${ABLATION_SKIP_BASELINE:-}"
  if [[ "${flag}" =~ ^(1|true|yes|on)$ ]]; then
    return 0
  fi
  local model_dir
  model_dir="$(model_dir_for_tag "baseline")"
  if [[ -d "/home/LLMCodeRepair/logs/codelinter_openharmony/${model_dir}/round_${ROUND}_after_round${ROUND}" ]]; then
    return 0
  fi
  if [[ -d "/home/LLMCodeRepair/revision/fixed_projects/${model_dir}/round_${ROUND}" ]]; then
    return 0
  fi
  return 1
}

run_one() {
  local tag="$1"
  local top_n="$2"
  local rag_type="$3"
  local context="$4"
  local ctx_flag=()

  if [[ "${context}" == "full" ]]; then
    ctx_flag=(--full-context)
  fi

  tag="${tag//\//_}"
  tag="${tag// /_}"

  echo "[ablation] run_tag=${TAG_PREFIX}_${tag} rag_type=${rag_type} top_n=${top_n} context=${context}"

  python /home/LLMCodeRepair/revision/code/run_haprepair_round.py \
    --all \
    --round "${ROUND}" \
    --model-name "${MODEL}" \
    --max-project-workers "${MAX_PROJECT_WORKERS}" \
    --rag-type "${rag_type}" \
    --top-n "${top_n}" \
    --run-tag "${TAG_PREFIX}_${tag}" \
    "${ctx_flag[@]}"

  local model_dir
  model_dir="$(sanitize_dir "${MODEL}")__$(sanitize_dir "${TAG_PREFIX}_${tag}")"

  python /home/LLMCodeRepair/gen_gpt5mini_remaining_defects.py \
    --model "${model_dir}" \
    --output-dir "${SUMMARY_DIR}"
}

# Ensure RAG service is up before running ablation
ensure_rag_service

# Baseline
if should_skip_baseline; then
  echo "[ablation] baseline already exists; skipping"
else
  run_one "baseline" "${BASE_TOP_N}" "${BASE_RAG_TYPE}" "${BASE_CONTEXT}"
fi

# No RAG (top-n = 0)
run_one "no_rag" "0" "${BASE_RAG_TYPE}" "${BASE_CONTEXT}"

# RAG top-k ablation (skip baseline top-n)
for top_n in "${RAG_TOP_NS[@]}"; do
  if [[ "${top_n}" == "${BASE_TOP_N}" ]]; then
    continue
  fi
  run_one "rag_top${top_n}" "${top_n}" "${BASE_RAG_TYPE}" "${BASE_CONTEXT}"
done

# Diff type ablation (skip baseline diff type)
for rag_type in "${DIFF_TYPES[@]}"; do
  if [[ "${rag_type}" == "${BASE_RAG_TYPE}" ]]; then
    continue
  fi
  run_one "diff_${rag_type}" "${BASE_TOP_N}" "${rag_type}" "${BASE_CONTEXT}"
done

# Context strategy ablation (skip baseline context)
for ctx in "${CONTEXTS[@]}"; do
  if [[ "${ctx}" == "${BASE_CONTEXT}" ]]; then
    continue
  fi
  run_one "context_${ctx}" "${BASE_TOP_N}" "${BASE_RAG_TYPE}" "${ctx}"
done
