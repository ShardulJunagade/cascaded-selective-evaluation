#!/usr/bin/env bash
# Shared helpers for scripts/api_round/*.sh. Sourced by the other scripts, not run directly.
#
# Every script tees its output to logs/<script>-<timestamp>.log and rewrites
# logs/progress.log after each step, so `bash scripts/api_round/status.sh` (or
# `cat logs/progress.log`) shows what is running from any other terminal.

set -Eeuo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"   # image paths in the data are relative to the repo root

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PYTHON="${PYTHON:-python}"
FAST_EXPORT=data/vlm_v2_fast_export
RESULT_DIR=result/vlm_v2_fast_1k_n3_k1
EXPECTED_ROWS=1000
API_CONFIG=configs/vlm_v2_fast_api.yaml
OPEN_JUDGES=(granite-vision-3.3-2b qwen2.5-vl-3b-instruct phi-3.5-vision-instruct internvl3-8b)
API_JUDGES=(kimi-k2.5 gemini-flash)
# API keys live outside the repo (chmod 600) so they never reach git.
KEY_FILE="${CASCADE_KEY_FILE:-${HOME}/.config/cascade/api_keys.env}"
PROGRESS_FILE="${REPO_ROOT}/logs/progress.log"   # *.log is gitignored

mkdir -p logs outputs result

STEP=0
TOTAL_STEPS=0
STEP_NAME=""
STEP_START=0
FAILED=0
HISTORY=()
WARNINGS=()

fmt_dur() {
  local s=$1
  printf '%dh%02dm%02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

write_progress() {  # $1 = RUNNING | DONE | FAILED
  local now
  now=$(date +%s)
  {
    echo "script:   ${SCRIPT_NAME}   [$1]"
    echo "updated:  $(date '+%Y-%m-%d %H:%M:%S')   (total elapsed $(fmt_dur $((now - SCRIPT_START))))"
    if [[ "$1" == RUNNING && -n "${STEP_NAME}" ]]; then
      echo "now:      step ${STEP}/${TOTAL_STEPS}: ${STEP_NAME}"
      echo "          started $(date -d "@${STEP_START}" '+%H:%M:%S' 2>/dev/null || echo "${STEP_START}")"
    fi
    echo "log:      ${LOG_FILE}"
    echo
    if ((${#HISTORY[@]})); then printf '%s\n' "${HISTORY[@]}"; fi
    if ((${#WARNINGS[@]})); then
      echo
      printf 'WARNING: %s\n' "${WARNINGS[@]}"
    fi
  } > "${PROGRESS_FILE}.tmp"
  mv "${PROGRESS_FILE}.tmp" "${PROGRESS_FILE}"
}

start_script() {  # $1 = short name, $2 = number of steps
  SCRIPT_NAME="$1"
  TOTAL_STEPS="$2"
  SCRIPT_START=$(date +%s)
  LOG_FILE="${REPO_ROOT}/logs/${SCRIPT_NAME}-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  trap 'on_error ${LINENO}' ERR
  trap on_exit EXIT
  echo "${SCRIPT_NAME} started $(date '+%Y-%m-%d %H:%M:%S') in ${REPO_ROOT}"
  echo "log: ${LOG_FILE}"
  echo "progress: bash scripts/api_round/status.sh   (from another terminal)"
  write_progress RUNNING
}

finish_step() {
  if [[ ${STEP} -gt 0 && -n "${STEP_NAME}" ]]; then
    HISTORY+=("  [done]    ${STEP}/${TOTAL_STEPS} ${STEP_NAME}  ($(fmt_dur $(( $(date +%s) - STEP_START ))))")
    STEP_NAME=""
  fi
}

step() {  # $1 = description of the step that starts now
  finish_step
  STEP=$((STEP + 1))
  STEP_NAME="$1"
  STEP_START=$(date +%s)
  echo
  echo "======================================================================"
  echo "[$(date '+%H:%M:%S')] STEP ${STEP}/${TOTAL_STEPS}: ${STEP_NAME}"
  echo "======================================================================"
  write_progress RUNNING
}

skip_step() {  # $1 = reason
  echo "  -> skipped: $1"
  HISTORY+=("  [skipped] ${STEP}/${TOTAL_STEPS} ${STEP_NAME}  ($1)")
  STEP_NAME=""
  write_progress RUNNING
}

warn() {
  echo "WARNING: $*"
  WARNINGS+=("$*")
  write_progress RUNNING
}

die() {
  echo
  echo "ERROR: $*"
  HISTORY+=("  [FAILED]  ${STEP}/${TOTAL_STEPS} ${STEP_NAME:-?}  ($*)")
  FAILED=1
  write_progress FAILED
  exit 1
}

on_error() {
  local code=$?
  [[ ${FAILED} -eq 1 ]] && return
  echo
  echo "FAILED (exit ${code}) in step ${STEP}/${TOTAL_STEPS}: ${STEP_NAME:-?} (line $1). See ${LOG_FILE}"
  HISTORY+=("  [FAILED]  ${STEP}/${TOTAL_STEPS} ${STEP_NAME:-?}  (exit ${code}; see log)")
  FAILED=1
  write_progress FAILED
}

on_exit() {
  local code=$?
  if [[ ${FAILED} -eq 0 ]]; then
    if [[ ${code} -eq 0 ]]; then
      finish_step
      write_progress DONE
    else
      HISTORY+=("  [stopped] ${STEP}/${TOTAL_STEPS} ${STEP_NAME:-?}  (exit ${code})")
      write_progress FAILED
    fi
  fi
  echo
  echo "${SCRIPT_NAME} finished with exit ${code} after $(fmt_dur $(( $(date +%s) - SCRIPT_START )))"
  if ((${#WARNINGS[@]})); then
    echo "Warnings:"
    printf '  - %s\n' "${WARNINGS[@]}"
  fi
}

rows() {  # non-empty lines in a file, 0 if missing
  if [[ -f "$1" ]]; then grep -c . "$1" || true; else echo 0; fi
}

cache_file() { echo "${RESULT_DIR}/$1.eval_pool.jsonl"; }

cache_complete() { [[ $(rows "$(cache_file "$1")") -ge ${EXPECTED_ROWS} ]]; }

load_api_keys() {
  if [[ -f "${KEY_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${KEY_FILE}"
    set +a
  fi
}

has_openrouter_key() { [[ -n "${OPENROUTER_API_KEY:-}" ]]; }
has_gemini_key() { [[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; }

have_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
    && "${PYTHON}" -c "import vllm" >/dev/null 2>&1
}
