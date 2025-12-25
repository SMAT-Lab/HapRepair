#!/usr/bin/env bash
set -euo pipefail

# End-to-end semantic evaluation pipeline (LLM-judged):
#  1) Prepare stratified sample from CodeLinter before/after logs + snapshots
#  2) Label each sample with two judge LLMs (gpt-5.1 via Packy, deepseek-chat via zhizengzeng)
#  3) Summarize labels to JSON
#  4) (Optional) write LaTeX table + rebuild paper PDF
#
# Usage:
#   bash /home/LLMCodeRepair/revision/code/run_repair_semantic_eval_llm.sh [MODEL_DIR] [MAX_ROUND] [N_SAMPLE] [SEED] [PAPER_DIR]
#
# Defaults:
#   MODEL_DIR = gpt-5.1
#   MAX_ROUND = 5
#   N_SAMPLE  = 150
#   SEED      = 20251220
#   PAPER_DIR = /home/LLMCodeRepair/-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps
#
# Optional env vars:
#   EVAL_UNIT=file                    # file|finding
#   JUDGE_A=gpt-5.1
#   JUDGE_B=deepseek-chat
#   DISAGREEMENT_POLICY=confidence   # suspicious|incorrect_if_any|confidence
#   UNCLEAR_POLICY=likely_correct    # suspicious|likely_correct
#   SLEEP_SEC=0
#   LABEL_WORKERS=64                 # parallel labeling workers
#   WRITE_MD=1
#   BUILD_LATEX=1
#   WRITE_LATEX_TABLE=0              # when 1, overwrite semantic_eval_table.tex
#   DRY_RUN=0
#   MAX_ITEMS=0                      # if >0, only label first N items

MODEL_DIR="${1:-gpt-5.1}"
MAX_ROUND="${2:-5}"
N_SAMPLE="${3:-150}"
SEED="${4:-20251220}"
PAPER_DIR="${5:-/home/LLMCodeRepair/-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps}"

EVAL_UNIT="${EVAL_UNIT:-file}"
JUDGE_A="${JUDGE_A:-gpt-5.1}"
JUDGE_B="${JUDGE_B:-deepseek-chat}"
DISAGREEMENT_POLICY="${DISAGREEMENT_POLICY:-confidence}"
UNCLEAR_POLICY="${UNCLEAR_POLICY:-likely_correct}"
SLEEP_SEC="${SLEEP_SEC:-0}"
LABEL_WORKERS="${LABEL_WORKERS:-64}"
WRITE_MD="${WRITE_MD:-1}"
BUILD_LATEX="${BUILD_LATEX:-1}"
WRITE_LATEX_TABLE="${WRITE_LATEX_TABLE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_ITEMS="${MAX_ITEMS:-0}"

BASE_DIR="/home/LLMCodeRepair"
OUT_BASE="${BASE_DIR}/revision/semantic_eval"

if [ -f "/home/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate VulRAG >/dev/null 2>&1 || true
fi

if [ -f "${BASE_DIR}/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  . "${BASE_DIR}/.env"
  set +a
fi

mkdir -p "${OUT_BASE}"

PREP_SCRIPT="${BASE_DIR}/revision/code/repair_semantic_prepare_files.py"
if [[ "${EVAL_UNIT}" == "finding" ]]; then
  PREP_SCRIPT="${BASE_DIR}/revision/code/repair_semantic_prepare.py"
fi

PREP_ARGS=(python "${PREP_SCRIPT}" --model-dir "${MODEL_DIR}" --max-round "${MAX_ROUND}" --n-sample "${N_SAMPLE}" --seed "${SEED}")
if [[ "${WRITE_MD}" =~ ^(1|true|yes|on)$ ]]; then
  PREP_ARGS+=(--write-md)
fi

echo "[step] prepare sample: unit=${EVAL_UNIT} model_dir=${MODEL_DIR} max_round=${MAX_ROUND} n=${N_SAMPLE} seed=${SEED}"
"${PREP_ARGS[@]}"

PATTERN="${MODEL_DIR}_n${N_SAMPLE}_seed${SEED}_"
if [[ "${EVAL_UNIT}" == "file" ]]; then
  PATTERN="${MODEL_DIR}_files_n${N_SAMPLE}_seed${SEED}_"
fi
RUN_DIR="$(ls -1dt "${OUT_BASE}/${PATTERN}"* 2>/dev/null | head -n 1 || true)"
if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
  echo "[error] failed to locate run_dir under ${OUT_BASE} with prefix ${PATTERN}" >&2
  exit 1
fi
echo "[info] run_dir=${RUN_DIR}"

LABEL_ARGS=(
  python "${BASE_DIR}/revision/code/repair_semantic_label_llm.py"
  --run-dir "${RUN_DIR}"
  --judge-a "${JUDGE_A}"
  --judge-b "${JUDGE_B}"
  --disagreement-policy "${DISAGREEMENT_POLICY}"
  --unclear-policy "${UNCLEAR_POLICY}"
  --sleep-sec "${SLEEP_SEC}"
  --max-workers "${LABEL_WORKERS}"
)
if [[ "${DRY_RUN}" =~ ^(1|true|yes|on)$ ]]; then
  LABEL_ARGS+=(--dry-run)
fi
if [ "${MAX_ITEMS}" -gt 0 ] 2>/dev/null; then
  LABEL_ARGS+=(--max-items "${MAX_ITEMS}")
fi

echo "[step] label with two judges: ${JUDGE_A} + ${JUDGE_B} (policy=${DISAGREEMENT_POLICY})"
"${LABEL_ARGS[@]}"

echo "[step] compute judge agreement metrics"
python "${BASE_DIR}/revision/code/repair_semantic_agreement.py" --run-dir "${RUN_DIR}" >/dev/null
echo "[ok] agreement_json=${RUN_DIR}/agreement.json"

SUMMARY_JSON="${RUN_DIR}/summary.json"
echo "[step] summarize labels -> ${SUMMARY_JSON}"
python "${BASE_DIR}/revision/code/repair_semantic_summarize.py" \
  --candidates "${RUN_DIR}/candidates.jsonl" \
  --labels "${RUN_DIR}/labels_llm.csv" \
  --out-json "${SUMMARY_JSON}"

echo "[ok] summary_json=${SUMMARY_JSON}"

if [[ "${WRITE_LATEX_TABLE}" =~ ^(1|true|yes|on)$ ]]; then
  TABLE_TEX="${PAPER_DIR}/semantic_eval_table.tex"
  echo "[step] write LaTeX table: ${TABLE_TEX}"
  python "${BASE_DIR}/revision/code/repair_semantic_summarize.py" \
    --candidates "${RUN_DIR}/candidates.jsonl" \
    --labels "${RUN_DIR}/labels_llm.csv" \
    --latex-out "${TABLE_TEX}"
  echo "[ok] latex_table=${TABLE_TEX}"
fi

if [[ "${BUILD_LATEX}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[step] build paper PDF"
  (cd "${PAPER_DIR}" && latexmk -pdf -interaction=nonstopmode main.tex)
  echo "[ok] built: ${PAPER_DIR}/main.pdf"
fi

echo "[done] semantic eval pipeline completed"
