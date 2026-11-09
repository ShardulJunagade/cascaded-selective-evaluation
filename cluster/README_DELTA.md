# Run the full VLM benchmark on NCSA Delta

This archive contains the complete code, exact few-shot pools, configs, tests, and Slurm
workflow. Large public data and model weights are downloaded on Delta. The runner builds
the full five-source corpus, scores four model families once, then evaluates three-judge and
four-judge cascades plus all model, setup, heuristic, point-estimate, and calibrated-cascade
baselines over 1,000 leakage-safe splits.

The tested chains are:

```text
3 judges: IBM Granite Vision 2B -> Qwen2.5-VL 3B -> Microsoft Phi-3.5 Vision 4.2B
4 judges: IBM Granite Vision 2B -> Qwen2.5-VL 3B -> Microsoft Phi-3.5 Vision 4.2B
          -> InternVL3 8B
```

## 1. Upload and unpack

From your local machine:

```bash
scp cascaded-selective-evaluation-delta.zip \
  rpatel25@login.delta.ncsa.illinois.edu:~/
ssh rpatel25@login.delta.ncsa.illinois.edu
unzip cascaded-selective-evaluation-delta.zip
cd cascaded-selective-evaluation
```

## 2. Configure Delta

```bash
cp cluster/delta.env.example cluster/delta.env
accounts
```

Edit `cluster/delta.env` and set `ACCOUNT` to a project printed by `accounts`. Defaults use
the production `cpu` and `gpuA100x4` partitions. Each scoring job requests one A100.

## 3. Install once

```bash
bash cluster/setup_delta.sh
```

This creates `.venv`, installs the requirements, configures bulk data under `$SCRATCH`, and
runs the CPU test suite. If Delta changes its current PyTorch module, use `module spider
pytorch-conda` and update `PYTHON_MODULE` in `cluster/delta.env`.

The full corpus needs access to gated VisionArena data. Accept the terms at
<https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle>, then authenticate into the
same cache location used by the jobs:

```bash
source cluster/common.sh
hf auth login
```

Do not put a Hugging Face token in `delta.env`; it is intentionally excluded from the ZIP.

## 4. Run inside your existing GPU allocation

If you already opened a GPU shell with `srun --pty bash`, do not submit another Slurm job
from inside it. Run:

```bash
python run_cluster_benchmark.py
```

No Delta environment file is required. The Python runner uses the active virtual environment
and visible GPU. It first runs a real one-pair compatibility check for all four model
families, then runs data preparation, all four judges, and both evaluations sequentially.
It writes its own timestamped log and `outputs/cluster-run-state.json`.
Re-running the command is safe: source downloads resume, scoring skips completed JSONL rows,
and an incomplete last line caused by allocation termination is repaired before resuming.

Monitor cached progress from another shell with:

```bash
python check_cluster_benchmark.py
```

The allocation must remain active for scoring. If it expires, request another GPU shell,
return to this directory, and run `python run_cluster_benchmark.py` again.

## Optional: submit the complete dependency chain

Use this only when you want the scheduler to manage separate CPU and GPU jobs:

```bash
bash cluster/submit_all.sh
```

The launcher prints and records six job IDs:

1. CPU download, corpus build, and verification.
2. Granite Vision 2B scoring after the data job succeeds.
3. Qwen2.5-VL-3B scoring after the data job succeeds.
4. Phi-3.5 Vision 4.2B scoring after the data job succeeds.
5. InternVL3 8B scoring after the data job succeeds.
6. CPU evaluation of the three- and four-judge chains after all scoring jobs succeed.

Scoring writes a checkpoint after every chunk. The batch jobs request requeue and restart
from cached JSONL rows after a node failure, preemption, or the five-minute wall-time signal.
To inspect progress:

```bash
bash cluster/status.sh
tail -f logs/score-cse-qwen3b-<JOBID>.out
```

If a job stops without requeueing, submit the workflow again with `SKIP_DATA="1"` in
`cluster/delta.env`. Completed data and judgment rows are detected and reused.

## 5. Collect results

The final artifact is:

```text
outputs/vlm_v2_3judge/baseline_comparison.json
outputs/vlm_v2_4judge/baseline_comparison.json
```

It includes the common pool inventory, every split, pairwise and macro-source accuracy,
coverage, guarantee success rate, per-source results, and the judge-setup baselines. Copy it
back with:

```bash
scp rpatel25@login.delta.ncsa.illinois.edu:\
~/cascaded-selective-evaluation/outputs/vlm_v2_4judge/baseline_comparison.json .
```

## Useful recovery commands

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
python -m prepare_data.verify_export
python cluster/verify_scored.py granite-vision-3.3-2b
python cluster/verify_scored.py qwen2.5-vl-3b-instruct
python cluster/verify_scored.py phi-3.5-vision-instruct
python cluster/verify_scored.py internvl3-8b
```

Raw data is about 43 GB, prepared media about 4 GB, and model caches require additional
space. The default keeps these under `$SCRATCH/cascaded-selective-evaluation`; code, compact
exports, cached judgments, logs, and final results stay beside the unzipped repository.
