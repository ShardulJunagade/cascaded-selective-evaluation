#!/usr/bin/env bash
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CLUSTER_DIR}/.." && pwd)"
DELTA_ENV_FILE="${DELTA_ENV_FILE:-${CLUSTER_DIR}/delta.env}"

if [[ ! -f "${DELTA_ENV_FILE}" ]]; then
  echo "Missing ${DELTA_ENV_FILE}. Copy cluster/delta.env.example to cluster/delta.env." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${DELTA_ENV_FILE}"

module reset
module load "${PYTHON_MODULE:-pytorch-conda/2.8}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run: bash cluster/setup_delta.sh" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"
DEFAULT_BULK_ROOT="${SCRATCH:-${REPO_ROOT}/.cluster-storage}/cascaded-selective-evaluation"
export CSE_BULK_ROOT="${BULK_ROOT:-${DEFAULT_BULK_ROOT}}"
export HF_HOME="${HF_HOME_OVERRIDE:-${CSE_BULK_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

mkdir -p "${HF_HOME}" logs outputs result
