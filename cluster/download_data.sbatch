#!/usr/bin/env bash
#SBATCH --job-name=cse-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --constraint=scratch
#SBATCH --output=logs/data-%j.out
#SBATCH --error=logs/data-%j.err
#SBATCH --requeue
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"
bash cluster/configure_storage.sh

if [[ "${ALLOW_PARTIAL_DATA:-0}" != "1" ]]; then
  python - <<'PY'
from huggingface_hub import get_token
if not get_token():
    raise SystemExit(
        "No Hugging Face token found. Accept the VisionArena-Battle terms, then run "
        "`hf auth login` with the archive's HF_HOME before submitting.")
PY
fi

python -u get_data.py --legacy-split 0
python -m open_cascade prepare-data configs/vlm_v2_guarantee.yaml
python cluster/verify_full_data.py
