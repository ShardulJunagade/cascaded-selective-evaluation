#!/usr/bin/env bash
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CLUSTER_DIR}/.." && pwd)"
ENV_FILE="${DELTA_ENV_FILE:-${CLUSTER_DIR}/delta.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy delta.env.example to delta.env and set ACCOUNT." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"
: "${ACCOUNT:?Set ACCOUNT in cluster/delta.env; run 'accounts' to list valid projects}"

cd "${REPO_ROOT}"
mkdir -p logs
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: bash cluster/setup_delta.sh" >&2
  exit 2
fi

dependency=()
data_job="skipped"
if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  data_job="$(sbatch --parsable --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION:-cpu}" cluster/download_data.sbatch)"
  dependency=(--dependency="afterok:${data_job}")
else
  .venv/bin/python cluster/verify_full_data.py
fi

score_3b="$(sbatch --parsable --account="${ACCOUNT}" \
  --partition="${GPU_PARTITION:-gpuA100x4}" "${dependency[@]}" \
  --job-name=cse-qwen3b cluster/score.sbatch qwen2.5-vl-3b-instruct)"
score_7b="$(sbatch --parsable --account="${ACCOUNT}" \
  --partition="${GPU_PARTITION:-gpuA100x4}" "${dependency[@]}" \
  --job-name=cse-qwen7b cluster/score.sbatch qwen2.5-vl-7b-instruct)"
evaluate_job="$(sbatch --parsable --account="${ACCOUNT}" \
  --partition="${CPU_PARTITION:-cpu}" \
  --dependency="afterok:${score_3b}:${score_7b}" cluster/evaluate.sbatch)"

cat > cluster/last_submission.txt <<EOF
data=${data_job}
score_3b=${score_3b}
score_7b=${score_7b}
evaluate=${evaluate_job}
EOF

cat cluster/last_submission.txt
echo
echo "Monitor: squeue -u ${USER}"
echo "Logs:    tail -f logs/score-cse-qwen3b-${score_3b}.out"
