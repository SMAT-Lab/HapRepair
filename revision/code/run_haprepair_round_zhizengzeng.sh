#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash revision/code/run_haprepair_round_zhizengzeng.sh [ROUND] [MODEL] [MAX_PROJECT_WORKERS]
#
# Defaults:
#   ROUND = 1              (repair round number)
#   MODEL = gpt-5-mini     (LLM model name passed到 Python)
#   MAX_PROJECT_WORKERS=1  (不同项目之间的最大并行数，1 表示顺序执行)
#
# 你可以在外面提前 export ZHIZENGZENG_API_KEY / ZHIZENGZENG_API_BASE，
# 这里会把它们映射到 API_KEY / API_BASE，供 llm.py 使用。

ROUND="${1:-1}"
MODEL="${2:-gpt-5-mini}"
MAX_PROJECT_WORKERS="${3:-1}"

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

# if [ -n "${OPENAI_API_KEY:-}" ]; then
#   export API_KEY="${OPENAI_API_KEY}"
# fi
# if [ -n "${OPENAI_API_BASE:-}" ]; then
#   export API_BASE="${OPENAI_API_BASE}"
# fi

if [ -n "${ZHIZENGZENG_API_KEY:-}" ]; then
  export API_KEY="${ZHIZENGZENG_API_KEY}"
fi
if [ -n "${ZHIZENGZENG_API_BASE:-}" ]; then
  export API_BASE="${ZHIZENGZENG_API_BASE}"
fi

# printenv API_KEY
# printenv API_BASE

python /home/LLMCodeRepair/revision/code/run_haprepair_round.py \
  --all \
  --round "${ROUND}" \
  --model-name "${MODEL}" \
  --max-project-workers "${MAX_PROJECT_WORKERS}"
