#!/usr/bin/env bash
# Show what the api_round scripts are doing. Run from any terminal, any time:
#
#   bash scripts/api_round/status.sh            # once
#   watch -n 30 bash scripts/api_round/status.sh  # refresh every 30 s (Ctrl-C to stop)

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
RESULT_DIR=result/vlm_v2_fast_1k_n3_k1

echo "=== progress (logs/progress.log)"
if [[ -f logs/progress.log ]]; then cat logs/progress.log; else echo "no script has run yet"; fi

echo
echo "=== scored pairs per judge (${RESULT_DIR})"
for judge in granite-vision-3.3-2b qwen2.5-vl-3b-instruct phi-3.5-vision-instruct \
             internvl3-8b kimi-k2.5 gemini-flash; do
  file="${RESULT_DIR}/${judge}.eval_pool.jsonl"
  n=0
  [[ -f "${file}" ]] && n=$(grep -c . "${file}")
  printf '  %-26s %5s / 1000\n' "${judge}" "${n}"
done

echo
echo "=== experiment outputs"
for name in vlm_v2_fast_4judge fast_api_kimi fast_api_gemini fast_api_kimi_gemini \
            fast_api_gemini_kimi fast_mixed_6judge fast_mixed_5judge_kimi; do
  if [[ -f "outputs/${name}/global_router_comparison.json" ]]; then
    echo "  done     outputs/${name}/"
  elif [[ -f "outputs/${name}/baseline_comparison.json" ]]; then
    echo "  partial  outputs/${name}/ (baseline done, router pending)"
  else
    echo "  pending  outputs/${name}/"
  fi
done

latest="$(ls -t logs/setup_env-*.log logs/build_data-*.log logs/run_experiments-*.log 2>/dev/null | head -1)"
if [[ -n "${latest}" ]]; then
  echo
  echo "=== last lines of ${latest}"
  tail -c 4000 "${latest}" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 8
fi
