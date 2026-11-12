#!/usr/bin/env bash
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${CLUSTER_DIR}/common.sh"

link_directory() {
  local link_path="$1"
  local target_path="$2"
  mkdir -p "${target_path}" "$(dirname "${link_path}")"
  if [[ -L "${link_path}" ]]; then
    return
  fi
  if [[ -e "${link_path}" ]]; then
    echo "Refusing to replace existing ${link_path}. Move it or set BULK_ROOT accordingly." >&2
    exit 2
  fi
  ln -s "${target_path}" "${link_path}"
}

link_directory "${REPO_ROOT}/data/raw" "${CSE_BULK_ROOT}/raw"
link_directory "${REPO_ROOT}/data/vlm_v2/media" "${CSE_BULK_ROOT}/media"

echo "Bulk data: ${CSE_BULK_ROOT}"
echo "HF cache:  ${HF_HOME}"
