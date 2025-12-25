#!/usr/bin/env bash
set -euo pipefail

source /home/miniconda3/etc/profile.d/conda.sh
conda activate VulRAG

if [ -f /home/LLMCodeRepair/.env ]; then
  set -a
  . /home/LLMCodeRepair/.env
  set +a
fi

export RAG_API_BASE=http://127.0.0.1:8000/api/v1
export RAG_API_TIMEOUT=120
export PYTHONUNBUFFERED=1

indices=(15 16 17 18 20 21 24 25 29 32)

printf '%s\n' "${indices[@]}" | xargs -n1 -P4 -I{} bash -lc "echo [rerun-parallel] index={} \$(date -Iseconds); python /home/LLMCodeRepair/revision/code/run_haprepair_round.py --index {} --round 1 --model-name gpt-5.1 --run-tag ablation_rag_top5 --rag-type difflib --top-n 5 --force"
