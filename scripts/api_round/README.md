# API-judge round: run scripts

These scripts set up the environment, rebuild the Assignment 2 fast benchmark (1,000 pairs, N=3, K=1), and run this round's experiments: the API judges alone, the API judges in cascades, and mixed open + API cascades. Run them from the repo root on the `api-judges` branch, inside `tmux` so they survive a dropped SSH session.

```bash
bash scripts/api_round/1_setup_env.sh        # packages, tests, network, HF login, API keys
bash scripts/api_round/2_build_data.sh       # download + rebuild data/vlm_v2_fast_export (CPU)
bash scripts/api_round/3_run_experiments.sh  # smoke tests, paid API scoring, all evaluations
bash scripts/api_round/status.sh             # progress, from any terminal
python scripts/api_round/tables_md.py        # all finished tables -> outputs/api_round_tables.md
```

`tables_md.py` can run at any time, including while experiments are still going. It includes every finished run, plus a **Kept** column (Success + (100 − Answered)). Kept counts splits where the method abstained on everything as safe, so compare that column with the 90% target.

Each script writes a full log to `logs/<script>-<time>.log`. It also rewrites `logs/progress.log` after every step with the step that is running, the steps done or skipped and how long each took, and any warnings. `status.sh` adds the scored-pair count for each judge and which outputs exist.

All scripts are safe to re-run. The download and scoring resume where they stopped, and evaluations that already finished are skipped.

Run `3_run_experiments.sh --help` for its options: `--yes`, `--quick`, `--skip-gpu`, `--force-eval` and `--smoke-pairs`.

Your Hugging Face token and API keys are saved to this folder's `.env`, which is gitignored and set to chmod 600. They apply only to this checkout. `HF_TOKEN` takes priority over any global `hf auth login`, and that global login is left unchanged. To enter a different Hugging Face token, run `RESET_HF_TOKEN=1 SKIP_INSTALL=1 bash scripts/api_round/1_setup_env.sh`. Never commit `.env`.

| Run | Judges | Output |
|---|---|---|
| E0 | 4 open judges (re-run) | `outputs/vlm_v2_fast_4judge/` |
| E1 | Kimi K2.5 | `outputs/fast_api_kimi/` |
| E2 | Gemini Flash | `outputs/fast_api_gemini/` |
| E3 | Kimi → Gemini | `outputs/fast_api_kimi_gemini/` |
| E3b | Gemini → Kimi | `outputs/fast_api_gemini_kimi/` |
| E4 | 4 open + Kimi + Gemini | `outputs/fast_mixed_6judge/` |
| E5 | 4 open + Kimi | `outputs/fast_mixed_5judge_kimi/` |

E0, E4 and E5 need the four open-judge caches in `result/vlm_v2_fast_1k_n3_k1/`. If the caches are missing, script 3 scores them with `run_fast_benchmark.py` when a GPU with vLLM is visible. That takes hours, and vLLM is installed with `INSTALL_VLLM=1 bash scripts/api_round/1_setup_env.sh`. Without a GPU, script 3 skips those steps. The summary table is saved to `outputs/api_round_summary.txt`.
