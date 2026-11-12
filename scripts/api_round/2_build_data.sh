#!/usr/bin/env bash
# Setup part 2: rebuild the 1,000-pair fast benchmark used in Assignment 2 (CPU only).
#
#   bash scripts/api_round/2_build_data.sh
#
# Downloads at most 1,000 raw examples per source into data/vlm_v2_small, exports the
# balanced-small corpus, then draws the deterministic 1,000-pair profile (seed 42, N=3,
# K=1) into data/vlm_v2_fast_export. The download is the slow part and resumes if
# interrupted; re-running skips anything already built.
#
# Options: FORCE_REBUILD=1 re-runs the download/build even if the corpus exists.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_script build_data 6

small_ready() {
  "${PYTHON}" -c "from run_small_benchmark import prepared_rows; import sys; sys.exit(0 if prepared_rows() else 1)" 2>/dev/null
}

# ----------------------------------------------------------------------------------------
step "Preflight: packages and Hugging Face token"
"${PYTHON}" -c "import open_cascade, huggingface_hub, imagehash, pandas, pyarrow, PIL" \
  || die "packages missing. Run: bash scripts/api_round/1_setup_env.sh"
has_local_hf_token \
  || die "no HF_TOKEN in ${KEY_FILE}. Run: SKIP_INSTALL=1 bash scripts/api_round/1_setup_env.sh"
echo "  Hugging Face account in use: $(hf_account)   (token from ${KEY_FILE})"
df -h "${REPO_ROOT}" | sed 's/^/  /'

# ----------------------------------------------------------------------------------------
step "Download sources and build the balanced-small corpus (slow: network bound)"
if small_ready && [[ "${FORCE_REBUILD:-0}" != 1 ]]; then
  skip_step "data/vlm_v2_small_export already built ($(rows data/vlm_v2_small_export/eval_pool.jsonl) rows)"
else
  "${PYTHON}" -c "from run_cluster_benchmark import require_visionarena_access as check; check()"
  "${PYTHON}" -u get_data.py --limit 1000 --legacy-split 0 \
      --out-dir data/vlm_v2_small --export-dir data/vlm_v2_small_export
fi

# ----------------------------------------------------------------------------------------
step "Verify the balanced-small corpus"
"${PYTHON}" cluster/verify_small_data.py

# ----------------------------------------------------------------------------------------
step "Draw the 1,000-pair fast profile (seed 42, N=3, K=1)"
"${PYTHON}" cluster/build_fast_profile.py

# ----------------------------------------------------------------------------------------
step "Prepare data for the experiment config"
"${PYTHON}" -m open_cascade prepare-data "${API_CONFIG}"

# ----------------------------------------------------------------------------------------
step "Check the benchmark files"
"${PYTHON}" - "${FAST_EXPORT}" "${EXPECTED_ROWS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

export, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for line in (export / "eval_pool.jsonl").open(encoding="utf-8")
        if line.strip()]
manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
fewshot = [line for line in (export / "fewshot.jsonl").open(encoding="utf-8") if line.strip()]
missing_images = [r["image_path"] for r in rows if not Path(r.get("image_path", "")).is_file()]

print(f"  pairs:            {len(rows)} (expected {expected})")
print(f"  annotators/k:     N={manifest['n_annotators']}, K={manifest['k_shot']}, "
      f"few-shot pools={len(fewshot)}")
print(f"  requests/judge:   {manifest['requests_per_judge']}")
print(f"  selection digest: {manifest['selection_digest'][:16]}...  (compare across machines)")
for source, count in sorted(Counter((r.get("source") or {}).get("dataset", "unknown") for r in rows).items()):
    print(f"    {source:<18} {count}")
print(f"  images on disk:   {len(rows) - len(missing_images)}/{len(rows)}")

problems = []
if len(rows) != expected:
    problems.append(f"expected {expected} pairs, found {len(rows)}")
if (manifest["n_annotators"], manifest["k_shot"]) != (3, 1) or len(fewshot) != 3:
    problems.append("expected N=3, K=1 with 3 few-shot pools")
if missing_images:
    problems.append(f"{len(missing_images)} image files missing, e.g. {missing_images[0]}")
if problems:
    raise SystemExit("Benchmark check failed: " + "; ".join(problems))
print("  benchmark OK")
PY

finish_step
echo
echo "Data ready. Next: bash scripts/api_round/3_run_experiments.sh"
