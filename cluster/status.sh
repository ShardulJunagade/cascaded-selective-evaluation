#!/usr/bin/env bash
set -euo pipefail

echo "Jobs"
squeue -u "${USER}"
echo
echo "Cached rows"
for judge in granite-vision-3.3-2b qwen2.5-vl-3b-instruct \
             phi-3.5-vision-instruct internvl3-8b; do
  file="result/vlm_v2_multifamily/${judge}.eval_pool.jsonl"
  if [[ -f "${file}" ]]; then
    printf '%-32s %8s / 25308\n' "${judge}" "$(wc -l < "${file}")"
  else
    printf '%-32s %s\n' "${judge}" "not started"
  fi
done
echo
for chain in vlm_v2_3judge vlm_v2_4judge; do
  file="outputs/${chain}/baseline_comparison.json"
  [[ -f "${file}" ]] && echo "Complete: ${file}" || echo "Pending:  ${file}"
done
