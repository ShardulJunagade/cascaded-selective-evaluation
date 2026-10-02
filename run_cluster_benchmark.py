#!/usr/bin/env python3
"""Run the complete 3-judge and 4-judge VLM benchmark on the current GPU.

Run this from the repository root inside an existing GPU allocation:

    python run_cluster_benchmark.py

The script is restart-safe. Downloads resume, existing scored rows are skipped, and an
incomplete final JSONL line is repaired before each judge resumes. Bulk data and Hugging
Face caches are placed under $SCRATCH when Delta provides it. Compact judgments, logs, and
final results remain in the repository.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parent
JUDGES = [
    "granite-vision-3.3-2b",
    "qwen2.5-vl-3b-instruct",
    "phi-3.5-vision-instruct",
    "internvl3-8b",
]
FULL_CONFIG = "configs/vlm_v2_4judge.yaml"
EXPECTED_ROWS = 25_308


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true",
                        help="require and reuse an already prepared full corpus")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip the one-pair compatibility check")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="verify existing caches and only run both evaluations")
    parser.add_argument("--judges", nargs="+", choices=JUDGES, default=JUDGES,
                        help="score only these judges; evaluation still requires all four")
    return parser.parse_args()


class RunLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")

    def command(self, argv: list[str]) -> None:
        self.write("\n$ " + " ".join(argv))
        process = subprocess.Popen(
            argv, cwd=ROOT, env=os.environ.copy(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            self.handle.write(line)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, argv)


def safe_link(link: Path, target: Path, log: RunLog) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(target, target_is_directory=True)
    log.write(f"Linked {link.relative_to(ROOT)} -> {target}")


def configure_storage(log: RunLog) -> None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        log.write("$SCRATCH is not set; storing downloads and model cache in the repository/home.")
        return
    bulk = Path(scratch) / "cascaded-selective-evaluation"
    safe_link(ROOT / "data/raw", bulk / "raw", log)
    safe_link(ROOT / "data/vlm_v2/media", bulk / "media", log)

    # Preserve a token found in the default cache before redirecting HF_HOME to scratch.
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        token = None
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HF_HOME", str(bulk / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(bulk / "huggingface/hub"))
    # Retain the deprecated alias for older Transformers/vLLM releases.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("HF_DATASETS_CACHE", str(bulk / "huggingface/datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(bulk / "huggingface/xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(bulk / "huggingface/transformers"))
    log.write(f"Bulk storage: {bulk}")
    log.write(f"HF_HOME: {os.environ['HF_HOME']}")
    log.write(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")


def full_data_exists() -> bool:
    path = ROOT / "data/vlm_v2_export/eval_pool.jsonl"
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) == EXPECTED_ROWS


def require_visionarena_access() -> None:
    """Fail before the large downloads unless the token can read the gated corpus."""
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    token = get_token()
    if not token:
        raise SystemExit(
            "The full corpus needs gated VisionArena access. Accept its Hugging Face terms, "
            "run `hf auth login`, then rerun this command.")
    try:
        # dataset_info is public even before the gate is accepted. Probe a real
        # parquet shard with HEAD so this verifies file access without downloading
        # another ~500 MB merely to perform the preflight.
        shard_url = hf_hub_url(
            "lmarena-ai/VisionArena-Battle",
            "data/train-00000-of-00031.parquet",
            repo_type="dataset",
        )
        get_hf_file_metadata(shard_url, token=token)
    except Exception as error:  # Hub exception classes differ across installed versions.
        raise SystemExit(
            "The saved Hugging Face token cannot read lmarena-ai/VisionArena-Battle. "
            "Open https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle while "
            "signed into the token's account, accept/request access, ensure a fine-grained "
            "token allows public gated repositories, then run `hf auth login` again. "
            f"Hugging Face returned: {type(error).__name__}: {error}"
        ) from error


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log = RunLog(ROOT / f"logs/interactive-{stamp}.log")
    state_path = ROOT / "outputs/cluster-run-state.json"
    state = {"started_at": datetime.now().astimezone().isoformat(),
             "status": "running", "log": str(log.path.relative_to(ROOT))}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        configure_storage(log)
        if not full_data_exists():
            if args.skip_data or args.evaluate_only:
                raise SystemExit("The complete 25,308-row corpus is not present.")
            require_visionarena_access()
            log.command([sys.executable, "-u", "get_data.py", "--legacy-split", "0"])
        log.command([sys.executable, "-m", "open_cascade", "prepare-data", FULL_CONFIG])
        log.command([sys.executable, "cluster/verify_full_data.py"])

        if not args.evaluate_only:
            import torch
            if not torch.cuda.is_available():
                raise SystemExit("No CUDA GPU is visible. Run inside the allocated GPU shell.")
            log.write(f"GPU: {torch.cuda.get_device_name(0)}")

            if not args.skip_smoke:
                for judge in args.judges:
                    log.command([sys.executable, "-u", "cluster/smoke_judge.py", judge])

            for judge in args.judges:
                cache = f"result/vlm_v2_multifamily/{judge}.eval_pool.jsonl"
                log.command([sys.executable, "cluster/repair_jsonl.py", cache])
                log.command([sys.executable, "-u", "-m", "open_cascade", "score",
                             FULL_CONFIG, "--judge", judge])
                log.command([sys.executable, "cluster/verify_scored.py", judge])

        for judge in JUDGES:
            log.command([sys.executable, "cluster/verify_scored.py", judge])
        for config in ("configs/vlm_v2_3judge.yaml", "configs/vlm_v2_4judge.yaml"):
            log.command([sys.executable, "-u", "-m", "open_cascade", "evaluate", config])

        state.update({"status": "complete",
                      "finished_at": datetime.now().astimezone().isoformat(),
                      "results": [
                          "outputs/vlm_v2_3judge/baseline_comparison.json",
                          "outputs/vlm_v2_4judge/baseline_comparison.json",
                      ]})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.write("\nBenchmark complete. Results saved under outputs/vlm_v2_{3judge,4judge}/")
        return 0
    except BaseException as error:
        state.update({"status": "failed", "error": str(error),
                      "finished_at": datetime.now().astimezone().isoformat()})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.write(f"\nFAILED: {error}")
        raise
    finally:
        log.handle.close()


if __name__ == "__main__":
    sys.exit(main())
