#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash revision/code/run_haprepair_context_full_zhizengzeng.sh [ROUND] [MODEL] [MAX_PROJECT_WORKERS]
#
# Defaults:
#   ROUND = 1              (repair round number)
#   MODEL = gpt-5.1        (LLM model name passed to Python)
#   MAX_PROJECT_WORKERS=1  (不同项目之间的最大并行数，1 表示顺序执行)
#
# This script runs ONLY the "full context" (entire code) setting.
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
SUMMARY_DIR="${ABLATION_SUMMARY_DIR:-/home/LLMCodeRepair/summary/ablation}"
RAG_START="${ABLATION_START_RAG:-1}"
RAG_HOST="${ABLATION_RAG_HOST:-127.0.0.1}"
RAG_PORT="${ABLATION_RAG_PORT:-8000}"
RAG_LOG="${ABLATION_RAG_LOG:-/home/LLMCodeRepair/logs/rag_service.log}"
RAG_TIMEOUT="${ABLATION_RAG_TIMEOUT:-120}"

sanitize_dir() {
  local s="$1"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s// /_}"
  printf '%s' "$s"
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
    echo "[context-full] stopping RAG service (pid=${RAG_PID})"
    kill "${RAG_PID}" >/dev/null 2>&1 || true
  fi
}

ensure_rag_service() {
  local status_url="${RAG_API_BASE}/rag/status"
  if curl -fsS "${status_url}" >/dev/null 2>&1; then
    if curl -fsS "${status_url}" | python -c "import json,sys; d=json.load(sys.stdin); ok=bool(d.get('success')) and bool(d.get('rag_available')) and bool(d.get('vector_db_connected')) and bool(d.get('embedding_model_loaded')); raise SystemExit(0 if ok else 1)"; then
      echo "[context-full] RAG service already running and ready at ${RAG_API_BASE}"
      return 0
    fi
    echo "[context-full] RAG service is running but not ready (vector DB/embedding not available); will try to restart"
  fi

  if [[ ! "${RAG_START}" =~ ^(1|true|yes|on)$ ]]; then
    echo "[context-full] RAG service not running; ABLATION_START_RAG=0, skipping startup"
    return 0
  fi

  echo "[context-full] starting RAG service at http://${RAG_HOST}:${RAG_PORT}"
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
        echo "[context-full] RAG service is ready"
        return 0
      fi
    fi
    sleep 1
  done

  echo "[context-full] RAG service failed to start; check ${RAG_LOG}"
  return 1
}

ensure_rag_service

run_tag="${TAG_PREFIX}_context_full"
model_dir="$(sanitize_dir "${MODEL}")__$(sanitize_dir "${run_tag}")"

echo "[context-full] run_tag=${run_tag} rag_type=${BASE_RAG_TYPE} top_n=${BASE_TOP_N} context=full"

python /home/LLMCodeRepair/revision/code/run_haprepair_round.py \
  --all \
  --round "${ROUND}" \
  --model-name "${MODEL}" \
  --max-project-workers "${MAX_PROJECT_WORKERS}" \
  --rag-type "${BASE_RAG_TYPE}" \
  --top-n "${BASE_TOP_N}" \
  --run-tag "${run_tag}" \
  --full-context

python /home/LLMCodeRepair/gen_gpt5mini_remaining_defects.py \
  --model "${model_dir}" \
  --output-dir "${SUMMARY_DIR}"
