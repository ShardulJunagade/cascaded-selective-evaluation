#!/usr/bin/env bash
set -euo pipefail

echo "Jobs"
squeue -u "${USER}"
echo
echo "Cached rows"
for judge in qwen2.5-vl-3b-instruct qwen2.5-vl-7b-instruct; do
  file="result/vlm_v2_maj/${judge}.eval_pool.jsonl"
  if [[ -f "${file}" ]]; then
    printf '%-32s %8s / 25308\n' "${judge}" "$(wc -l < "${file}")"
  else
    printf '%-32s %s\n' "${judge}" "not started"
  fi
done
echo
if [[ -f outputs/vlm_v2_guarantee/baseline_comparison.json ]]; then
  echo "Evaluation complete: outputs/vlm_v2_guarantee/baseline_comparison.json"
else
  echo "Evaluation pending"
fi
