#!/usr/bin/env bash
#SBATCH --job-name=cse-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --constraint=scratch
#SBATCH --output=logs/evaluate-%j.out
#SBATCH --error=logs/evaluate-%j.err
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

python cluster/verify_full_data.py
python cluster/verify_scored.py qwen2.5-vl-3b-instruct
python cluster/verify_scored.py qwen2.5-vl-7b-instruct
python -u -m open_cascade evaluate configs/vlm_v2_guarantee.yaml

echo "Results: outputs/vlm_v2_guarantee/baseline_comparison.json"
