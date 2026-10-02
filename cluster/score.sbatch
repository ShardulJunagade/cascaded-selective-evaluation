#!/usr/bin/env bash
#SBATCH --job-name=cse-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --constraint=scratch
#SBATCH --output=logs/score-%x-%j.out
#SBATCH --error=logs/score-%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail

JUDGE="${1:?Usage: sbatch cluster/score.sbatch JUDGE_NAME}"
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

child_pid=""
requeue_job() {
  echo "Received a scheduler signal; stopping at the last completed JSONL chunk."
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_job USR1 TERM

python cluster/verify_full_data.py
python cluster/repair_jsonl.py "result/vlm_v2_maj/${JUDGE}.eval_pool.jsonl"
nvidia-smi
srun --kill-on-bad-exit=1 python -u -m open_cascade score \
  configs/vlm_v2_guarantee.yaml --judge "${JUDGE}" &
child_pid=$!
wait "${child_pid}"
child_pid=""
trap - USR1 TERM

python cluster/verify_scored.py "${JUDGE}"
