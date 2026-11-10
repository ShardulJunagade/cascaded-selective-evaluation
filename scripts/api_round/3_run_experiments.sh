#!/usr/bin/env bash
# This round's experiments on the 1,000-pair fast benchmark (same splits as Assignment 2).
#
#   bash scripts/api_round/3_run_experiments.sh [options]
#
#   E1  Kimi K2.5 alone                         outputs/fast_api_kimi/
#   E2  Gemini Flash alone                      outputs/fast_api_gemini/
#   E3  Kimi -> Gemini cascade                  outputs/fast_api_kimi_gemini/
#   E3b Gemini -> Kimi (escalation-order check) outputs/fast_api_gemini_kimi/
#   E0  the 4 open judges, re-run               outputs/vlm_v2_fast_4judge/
#   E4  4 open judges + Kimi + Gemini           outputs/fast_mixed_6judge/
#   E5  4 open judges + Kimi                    outputs/fast_mixed_5judge_kimi/
#
# Each run writes baseline_comparison.json (Trust or Escalate and the baselines) and
# global_router_comparison.json (the end-to-end router). A summary table is printed at the
# end and saved to outputs/api_round_summary.txt.
#
# Only API scoring costs money (6,000 requests per API judge). A short smoke test runs
# first and prints the estimated cost; the script then asks before full scoring.
# Evaluations are CPU only. E0/E4/E5 need the open-judge caches: if they are missing and a
# GPU with vLLM is visible, run_fast_benchmark.py scores them (hours); otherwise those
# steps are skipped. Re-running is safe: scoring resumes, finished evaluations are skipped.
#
# Options:
#   --yes            do not ask before paid scoring (needed when running without a terminal)
#   --quick          3 splits instead of 1,000 for every evaluation (test the pipeline);
#                    results go to outputs/<name>_quick/
#   --skip-gpu       never start GPU scoring of the open judges
#   --force-eval     re-run evaluations even if their output already exists
#   --smoke-pairs N  pairs per smoke test (default 3)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

YES=0
QUICK=0
SKIP_GPU=0
FORCE_EVAL=0
SMOKE_PAIRS=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --quick) QUICK=1 ;;
    --skip-gpu) SKIP_GPU=1 ;;
    --force-eval) FORCE_EVAL=1 ;;
    --smoke-pairs) SMOKE_PAIRS="$2"; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

SUFFIX=""
EVAL_EXTRA=()
if [[ ${QUICK} -eq 1 ]]; then
  SUFFIX="_quick"
  EVAL_EXTRA=(--set evaluate.n_splits=3)
fi
KIMI_OK=0
GEMINI_OK=0

start_script run_experiments 14

smoke() {  # $1 = judge
  mkdir -p outputs/smoke
  "${PYTHON}" test_api_judge.py --judge "$1" --n "${SMOKE_PAIRS}" --annotators 3 --k 1 \
      --split "${FAST_EXPORT}/eval_pool.jsonl" --fewshot "${FAST_EXPORT}/fewshot.jsonl" \
      --out "outputs/smoke/$1.jsonl"
}

score_api() {  # $1 = judge; scoring is resumable, so a second pass retries failed pairs
  local judge="$1" file
  file="$(cache_file "${judge}")"
  echo "cache: ${file}  ($(rows "${file}")/${EXPECTED_ROWS} rows so far)"
  "${PYTHON}" -u -m open_cascade score "${API_CONFIG}" --judge "${judge}"
  if ! cache_complete "${judge}"; then
    echo "only $(rows "${file}")/${EXPECTED_ROWS} rows; one more pass to retry failed pairs"
    "${PYTHON}" -u -m open_cascade score "${API_CONFIG}" --judge "${judge}"
  fi
  if cache_complete "${judge}"; then
    echo "${judge}: ${EXPECTED_ROWS}/${EXPECTED_ROWS} rows scored"
  else
    warn "${judge} has $(rows "${file}")/${EXPECTED_ROWS} rows; re-run this script to resume"
  fi
}

run_eval() {  # $1 = run name, $2 = config, $3... = judges in escalation order
  local name="$1${SUFFIX}" config="$2" judge list
  shift 2
  local judges=("$@") missing=()
  if [[ -f "outputs/${name}/global_router_comparison.json" && ${FORCE_EVAL} -eq 0 ]]; then
    skip_step "already done: outputs/${name}/ (use --force-eval to redo)"
    return 0
  fi
  for judge in "${judges[@]}"; do
    cache_complete "${judge}" || missing+=("${judge}")
  done
  if ((${#missing[@]})); then
    skip_step "not scored yet: ${missing[*]}"
    return 0
  fi
  list="$(IFS=,; echo "${judges[*]}")"
  echo "judges: ${judges[*]}"
  echo "output: outputs/${name}/"
  "${PYTHON}" -u -m open_cascade evaluate "${config}" \
      --set "judges=[${list}]" --set "name=${name}" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}
}

# ----------------------------------------------------------------------------------------
step "Preflight: benchmark data, packages, keys, caches"
[[ $(rows "${FAST_EXPORT}/eval_pool.jsonl") -eq ${EXPECTED_ROWS} ]] \
  || die "${FAST_EXPORT}/eval_pool.jsonl does not have ${EXPECTED_ROWS} rows. Run: bash scripts/api_round/2_build_data.sh"
"${PYTHON}" -c "import httpx, open_cascade.judges.api" \
  || die "packages missing. Run: bash scripts/api_round/1_setup_env.sh"
load_api_keys
if has_openrouter_key; then echo "  OPENROUTER_API_KEY: set"; else warn "OPENROUTER_API_KEY not set: Kimi steps will be skipped (bash scripts/api_round/1_setup_env.sh saves it)"; fi
if has_gemini_key; then echo "  GEMINI_API_KEY: set"; else warn "GEMINI_API_KEY not set: Gemini steps will be skipped (bash scripts/api_round/1_setup_env.sh saves it)"; fi
echo "  scored caches in ${RESULT_DIR}:"
for judge in "${OPEN_JUDGES[@]}" "${API_JUDGES[@]}"; do
  printf '    %-26s %5s/%s\n' "${judge}" "$(rows "$(cache_file "${judge}")")" "${EXPECTED_ROWS}"
done
if have_gpu; then echo "  GPU + vLLM: available"; else echo "  GPU + vLLM: not available (open judges cannot be scored here)"; fi
if [[ ${QUICK} -eq 1 ]]; then echo "  QUICK MODE: 3 splits per evaluation, outputs suffixed _quick"; fi

# ----------------------------------------------------------------------------------------
step "Smoke test: Kimi K2.5 (${SMOKE_PAIRS} pairs, prints cost estimate)"
if cache_complete kimi-k2.5; then
  KIMI_OK=1
  skip_step "Kimi already fully scored"
elif ! has_openrouter_key; then
  skip_step "no OPENROUTER_API_KEY"
else
  smoke kimi-k2.5 || die "Kimi smoke test failed; fix it before paying for full scoring"
  KIMI_OK=1
fi

# ----------------------------------------------------------------------------------------
step "Smoke test: Gemini Flash (${SMOKE_PAIRS} pairs, first ever run of this judge)"
if cache_complete gemini-flash; then
  GEMINI_OK=1
  skip_step "Gemini already fully scored"
elif ! has_gemini_key; then
  skip_step "no GEMINI_API_KEY"
elif smoke gemini-flash; then
  GEMINI_OK=1
else
  warn "Gemini smoke test failed; Gemini steps skipped. Check the key/model, then re-run."
fi

# ----------------------------------------------------------------------------------------
step "Score Kimi K2.5 on 1,000 pairs (6,000 requests, paid)"
todo=()
if [[ ${KIMI_OK} -eq 1 ]] && ! cache_complete kimi-k2.5; then todo+=(kimi-k2.5); fi
if [[ ${GEMINI_OK} -eq 1 ]] && ! cache_complete gemini-flash; then todo+=(gemini-flash); fi
if ((${#todo[@]})) && [[ ${YES} -eq 0 ]]; then
  echo
  echo "About to score: ${todo[*]} (6,000 requests each). Check the cost lines printed by"
  echo "the smoke tests above ('1,000 samples ~ \$...')."
  if [[ -t 0 ]]; then
    read -r -p "Start paid scoring? [y/N] " answer
    if [[ ! "${answer}" =~ ^[Yy] ]]; then
      warn "paid scoring declined; API experiments skipped"
      KIMI_OK=0
      GEMINI_OK=0
    fi
  else
    die "no terminal to confirm paid scoring; re-run with --yes"
  fi
fi
if cache_complete kimi-k2.5; then
  skip_step "already scored"
elif [[ ${KIMI_OK} -eq 1 ]]; then
  score_api kimi-k2.5
else
  skip_step "Kimi not available"
fi

# ----------------------------------------------------------------------------------------
step "Score Gemini Flash on 1,000 pairs (6,000 requests, paid)"
if cache_complete gemini-flash; then
  skip_step "already scored"
elif [[ ${GEMINI_OK} -eq 1 ]]; then
  score_api gemini-flash || warn "Gemini scoring failed; re-run this script to resume"
else
  skip_step "Gemini not available"
fi

# ----------------------------------------------------------------------------------------
step "E1: Kimi alone"
run_eval fast_api_kimi "${API_CONFIG}" kimi-k2.5

step "E2: Gemini alone"
run_eval fast_api_gemini "${API_CONFIG}" gemini-flash

step "E3: Kimi -> Gemini cascade"
run_eval fast_api_kimi_gemini "${API_CONFIG}" kimi-k2.5 gemini-flash

step "E3b: Gemini -> Kimi cascade (escalation-order check)"
run_eval fast_api_gemini_kimi "${API_CONFIG}" gemini-flash kimi-k2.5

# ----------------------------------------------------------------------------------------
step "Score the 4 open judges on GPU (run_fast_benchmark.py, hours)"
open_missing=()
for judge in "${OPEN_JUDGES[@]}"; do
  cache_complete "${judge}" || open_missing+=("${judge}")
done
if ((${#open_missing[@]} == 0)); then
  skip_step "all 4 open-judge caches complete"
elif [[ ${SKIP_GPU} -eq 1 ]]; then
  skip_step "--skip-gpu"
elif ! have_gpu; then
  skip_step "no GPU with vLLM visible"
  warn "open judges not scored (${open_missing[*]}). For E0/E4/E5, re-run this script inside a GPU allocation after INSTALL_VLLM=1 bash scripts/api_round/1_setup_env.sh"
else
  nvidia-smi -L
  "${PYTHON}" -u run_fast_benchmark.py
fi

# ----------------------------------------------------------------------------------------
step "E0: the 4 open judges, re-run on the same splits"
run_eval vlm_v2_fast_4judge configs/vlm_v2_fast_4judge.yaml "${OPEN_JUDGES[@]}"

step "E4: 4 open judges + Kimi + Gemini (6 stages)"
run_eval fast_mixed_6judge "${API_CONFIG}" "${OPEN_JUDGES[@]}" kimi-k2.5 gemini-flash

step "E5: 4 open judges + Kimi (5 stages)"
run_eval fast_mixed_5judge_kimi "${API_CONFIG}" "${OPEN_JUDGES[@]}" kimi-k2.5

# ----------------------------------------------------------------------------------------
step "Summary"
"${PYTHON}" - "${SUFFIX}" <<'PY' | tee outputs/api_round_summary.txt
import json
import sys
from pathlib import Path

suffix = sys.argv[1]
runs = [("E0", "vlm_v2_fast_4judge", "4 open judges"),
        ("E1", "fast_api_kimi", "Kimi"),
        ("E2", "fast_api_gemini", "Gemini"),
        ("E3", "fast_api_kimi_gemini", "Kimi -> Gemini"),
        ("E3b", "fast_api_gemini_kimi", "Gemini -> Kimi"),
        ("E4", "fast_mixed_6judge", "4 open + Kimi + Gemini"),
        ("E5", "fast_mixed_5judge_kimi", "4 open + Kimi")]


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def pct(value):
    return f"{100 * value:6.2f}" if value is not None else "   n/a"


def row(label, s, calls=None):
    agreement = (s.get("agreement") or {}).get("mean")
    coverage = (s.get("coverage") or {}).get("mean")
    calls = f"{calls:5.2f}" if calls is not None else "    -"
    print(f"    {label:<34} {pct(agreement)}  {pct(coverage)}  "
          f"{pct(s.get('guarantee_success_rate'))}  {calls}")


print("Target: agreement >= 85% (alpha 0.15) with probability 0.9 (delta 0.1).")
print("Numbers are means over splits, in %. 'calls' = mean judges queried per pair.")
for tag, name, label in runs:
    base = load(Path("outputs") / f"{name}{suffix}" / "baseline_comparison.json")
    glob = load(Path("outputs") / f"{name}{suffix}" / "global_router_comparison.json")
    print(f"\n{tag:<4}{label}   (outputs/{name}{suffix}/)")
    if not base and not glob:
        print("    not run")
        continue
    print(f"    {'method':<34} {'agree':>6}  {'cover':>6}  {'guar.':>6}  calls")
    if base:
        policies = base["policies"]
        for key, text in [("cascaded_selective", "Trust or Escalate (per-stage)"),
                          ("cascaded_heuristic", "heuristic cascade (no guarantee)")]:
            if key in policies:
                row(text, policies[key])
        singles = {k: v for k, v in policies.items() if k.startswith("selective:")}
        if singles:
            best = max(singles, key=lambda k: (singles[k]["coverage"]["mean"] or 0))
            row(f"best certified single: {best.split(':', 1)[1]}"[:34], singles[best])
    if glob:
        for key, text in [("global_raw_confidence", "end-to-end router: raw conf."),
                          ("global_learned_router", "end-to-end router: learned")]:
            s = glob["methods"].get(key)
            if s:
                row(text, s, (s.get("mean_judges_called") or {}).get("mean"))
PY

finish_step
echo
echo "All steps finished. Summary saved to outputs/api_round_summary.txt"
