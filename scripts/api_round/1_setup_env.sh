#!/usr/bin/env bash
# Setup part 1: Python packages, offline tests, network, Hugging Face access, API keys.
#
#   bash scripts/api_round/1_setup_env.sh
#
# Run it with your environment active (conda/venv). Safe to re-run; it only installs what
# is missing. It asks for your Hugging Face token and API keys if they are not set yet and
# saves them to this folder's .env (gitignored, chmod 600). They apply only to this folder:
# HF_TOKEN overrides a global `hf auth login`, which is left untouched.
#
# Options (environment variables):
#   SKIP_INSTALL=1   skip pip
#   RESET_HF_TOKEN=1 ask for a new Hugging Face token even if .env already has one
#   INSTALL_VLLM=1   also install vLLM (only needed to re-score the open judges on a GPU)
#   REQ_FILE=...     install from this requirements file instead of requirements-open.txt
#   PYTHON=...       interpreter to use (default: python)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_script setup_env 8

# ----------------------------------------------------------------------------------------
step "Check repo and branch"
branch="$(git rev-parse --abbrev-ref HEAD)"
echo "repo:   ${REPO_ROOT}"
echo "branch: ${branch}   commit: $(git log -1 --format='%h %s')"
if [[ "${branch}" != api-judges ]]; then
  warn "on branch '${branch}', not api-judges. Run: git checkout api-judges && git pull"
fi
if git fetch -q origin 2>/dev/null; then
  behind="$(git rev-list --count HEAD..origin/api-judges 2>/dev/null || echo 0)"
  if [[ "${behind}" -gt 0 ]]; then
    warn "${behind} new commit(s) on origin/api-judges. Run: git pull"
  else
    echo "up to date with origin/api-judges"
  fi
else
  warn "git fetch failed; could not check for new commits"
fi

# ----------------------------------------------------------------------------------------
step "Check Python environment"
command -v "${PYTHON}" >/dev/null || die "'${PYTHON}' not found. Activate your environment first."
"${PYTHON}" - <<'PY'
import sys
print("python:", sys.executable, sys.version.split()[0])
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9+ is required")
PY
if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  warn "no virtualenv/conda environment is active; packages go into $(command -v "${PYTHON}")"
fi

# ----------------------------------------------------------------------------------------
step "Install Python packages"
if [[ "${SKIP_INSTALL:-0}" == 1 ]]; then
  skip_step "SKIP_INSTALL=1"
else
  req="${REQ_FILE:-logs/requirements-cpu.txt}"
  if [[ -z "${REQ_FILE:-}" ]]; then
    # requirements-open.txt without vLLM: vLLM is large and only needed for local GPU scoring.
    grep -v '^vllm' requirements-open.txt > "${req}"
  fi
  pip_args=(-r "${req}")
  if [[ -d wheels ]]; then
    pip_args=(--find-links wheels "${pip_args[@]}")
    echo "using local wheels/ where possible"
  fi
  echo "installing from ${req}"
  "${PYTHON}" -m pip install "${pip_args[@]}"
  echo "installing httpx (used by the API judges; not in requirements-open.txt)"
  "${PYTHON}" -m pip install httpx
  if [[ "${INSTALL_VLLM:-0}" == 1 ]]; then
    vllm_pin="$(grep '^vllm' requirements-open.txt || echo vllm)"
    echo "installing ${vllm_pin}"
    "${PYTHON}" -m pip install "${vllm_pin}"
  fi
fi

# ----------------------------------------------------------------------------------------
step "Check imports"
"${PYTHON}" - <<'PY'
import importlib

required = ["numpy", "scipy", "torch", "yaml", "tqdm", "datasets", "PIL", "huggingface_hub",
            "imagehash", "pandas", "pyarrow", "httpx", "open_cascade",
            "open_cascade.judges.api", "open_cascade.experiments.global_router"]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"  OK       {name:<40} {getattr(module, '__version__', '')}")
    except Exception as error:
        print(f"  MISSING  {name:<40} {type(error).__name__}: {error}")
        missing.append(name)
try:
    import vllm
    print(f"  OK       {'vllm (optional, GPU scoring)':<40} {vllm.__version__}")
except Exception:
    print(f"  absent   {'vllm (optional, GPU scoring)':<40} only needed to re-score open judges")
if missing:
    raise SystemExit(f"Missing packages: {missing}")
PY

# ----------------------------------------------------------------------------------------
step "Run the offline API-judge tests (mocked, no network, no cost)"
"${PYTHON}" -m pytest -q tests/test_api_judge.py

# ----------------------------------------------------------------------------------------
step "Check network access and disk space"
check_url() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import httpx
try:
    status = httpx.get(sys.argv[1], timeout=15, follow_redirects=True).status_code
    print(f"reachable (HTTP {status})")
except Exception as error:
    print(f"UNREACHABLE ({type(error).__name__})")
PY
}
for target in "Hugging Face|https://huggingface.co" \
              "OpenRouter (Kimi)|https://openrouter.ai/api/v1/models" \
              "Google AI (Gemini)|https://generativelanguage.googleapis.com"; do
  label="${target%%|*}"
  result="$(check_url "${target#*|}")"
  printf '  %-20s %s\n' "${label}" "${result}"
  if [[ "${result}" == UNREACHABLE* ]]; then
    warn "${label} is unreachable from this machine (proxy/firewall?)"
  fi
done
df -h "${REPO_ROOT}" | sed 's/^/  /'
avail_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
if [[ "${avail_kb}" -lt $((15 * 1024 * 1024)) ]]; then
  warn "less than 15 GB free in ${REPO_ROOT}; the data download may not fit"
fi

# ----------------------------------------------------------------------------------------
save_key() {  # $1 = variable name, $2 = value
  mkdir -p "$(dirname "${KEY_FILE}")"
  (
    umask 077
    touch "${KEY_FILE}"
    grep -v "^$1=" "${KEY_FILE}" > "${KEY_FILE}.tmp" || true
    printf '%s=%q\n' "$1" "$2" >> "${KEY_FILE}.tmp"
    mv "${KEY_FILE}.tmp" "${KEY_FILE}"
  )
  chmod 600 "${KEY_FILE}"
}

# ----------------------------------------------------------------------------------------
step "Hugging Face token (this folder only) and VisionArena access"
# The token is kept in this folder's .env as HF_TOKEN, which overrides any global
# `hf auth login` token, so the global login is neither used nor changed.
if [[ "${RESET_HF_TOKEN:-0}" == 1 ]]; then
  unset HF_TOKEN
fi
if ! has_local_hf_token; then
  if [[ -t 0 ]]; then
    echo "  No HF_TOKEN in ${KEY_FILE} yet."
    echo "  Create a Read token for YOUR account at https://huggingface.co/settings/tokens"
    read -r -s -p "  Paste your Hugging Face token (input hidden): " value
    echo
    [[ -n "${value}" ]] || die "no Hugging Face token entered"
    export HF_TOKEN="${value}"
    [[ "$(hf_account)" != "?" ]] || die "Hugging Face rejected that token; check it was copied in full"
    save_key HF_TOKEN "${value}"
    echo "  saved HF_TOKEN to ${KEY_FILE} (used only by this folder)"
  else
    die "no HF_TOKEN in ${KEY_FILE}; run this script in a terminal to enter it"
  fi
fi
account="$(hf_account)"
echo "  Hugging Face account in use: ${account}   (token from ${KEY_FILE})"
if ! "${PYTHON}" -c "from run_cluster_benchmark import require_visionarena_access as check; check()"; then
  die "account '${account}' cannot read VisionArena yet. Open https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle signed in as '${account}', accept the terms / request access, wait until the page says access is granted, then re-run. To enter a different token: RESET_HF_TOKEN=1 SKIP_INSTALL=1 bash scripts/api_round/1_setup_env.sh"
fi
echo "  VisionArena (gated) access OK"

# ----------------------------------------------------------------------------------------
step "API keys"
for var in OPENROUTER_API_KEY GEMINI_API_KEY; do
  if [[ -n "${!var:-}" ]]; then
    echo "  ${var}: set"
    if ! grep -q "^${var}=" "${KEY_FILE}" 2>/dev/null; then
      save_key "${var}" "${!var}"
      echo "  saved ${var} to ${KEY_FILE} for the experiment script"
    fi
  elif [[ "${var}" == GEMINI_API_KEY && -n "${GOOGLE_API_KEY:-}" ]]; then
    echo "  GEMINI_API_KEY: not set, but GOOGLE_API_KEY is (the Gemini judge accepts either)"
  elif [[ -t 0 ]]; then
    read -r -s -p "  Enter ${var} (input hidden; leave blank to skip): " value
    echo
    if [[ -n "${value}" ]]; then
      save_key "${var}" "${value}"
      echo "  saved ${var} to ${KEY_FILE}"
    else
      warn "${var} not set; experiments that need it will be skipped"
    fi
  else
    warn "${var} not set; experiments that need it will be skipped"
  fi
done

finish_step
echo
echo "Setup complete. Next: bash scripts/api_round/2_build_data.sh"
