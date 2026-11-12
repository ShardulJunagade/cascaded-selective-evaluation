#!/usr/bin/env bash
set -euo pipefail

# Run the complete benchmark inside an already allocated `srun --pty bash` shell.
# Re-running is safe: downloads resume and completed judgment rows are skipped.
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"
bash cluster/configure_storage.sh

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  python - <<'PY'
from huggingface_hub import get_token
if not get_token():
    raise SystemExit(
        "No Hugging Face token found. Accept the VisionArena-Battle terms and run "
        "`source cluster/common.sh && hf auth login` first.")
PY
  python -u get_data.py --legacy-split 0
  python -m open_cascade prepare-data configs/vlm_v2_4judge.yaml
fi
python cluster/verify_full_data.py

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null; then
  echo "No NVIDIA GPU is visible. Run this inside the allocated GPU shell." >&2
  exit 2
fi
nvidia-smi

judges=(granite-vision-3.3-2b qwen2.5-vl-3b-instruct \
        phi-3.5-vision-instruct internvl3-8b)

echo "Running one-pair compatibility checks for all four model families."
for judge in "${judges[@]}"; do
  python -u cluster/smoke_judge.py "${judge}"
done

for judge in "${judges[@]}"; do
  cache="result/vlm_v2_multifamily/${judge}.eval_pool.jsonl"
  python cluster/repair_jsonl.py "${cache}"
  python -u -m open_cascade score configs/vlm_v2_4judge.yaml --judge "${judge}"
  python cluster/verify_scored.py "${judge}"
done

python -u -m open_cascade evaluate configs/vlm_v2_3judge.yaml
python -u -m open_cascade evaluate configs/vlm_v2_4judge.yaml
echo "Complete:"
echo "  outputs/vlm_v2_3judge/baseline_comparison.json"
echo "  outputs/vlm_v2_4judge/baseline_comparison.json"
