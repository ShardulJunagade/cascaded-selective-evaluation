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

module reset
module load "${PYTHON_MODULE:-pytorch-conda/2.8}"
cd "${REPO_ROOT}"

if [[ ! -d .venv ]]; then
  python -m venv --system-site-packages .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-open.txt

bash cluster/configure_storage.sh
python -m pytest -q
python - <<'PY'
import torch
import transformers
import vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY

echo
echo "Environment ready. Submit the benchmark with: bash cluster/submit_all.sh"
