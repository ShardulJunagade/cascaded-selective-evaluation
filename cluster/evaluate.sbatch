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
for judge in granite-vision-3.3-2b qwen2.5-vl-3b-instruct \
             phi-3.5-vision-instruct internvl3-8b; do
  python cluster/verify_scored.py "${judge}"
done
python -u -m open_cascade evaluate configs/vlm_v2_3judge.yaml
python -u -m open_cascade evaluate configs/vlm_v2_4judge.yaml

echo "Results: outputs/vlm_v2_{3judge,4judge}/baseline_comparison.json"
