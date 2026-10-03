#!/usr/bin/env python3
"""Run the defendable balanced 1,000-per-source benchmark on the current GPU.

The capped Hugging Face loaders stream a seeded shuffle and stop at 1,000 raw rows,
avoiding full-dataset Arrow generation. Outputs and judge caches are separate from the
full benchmark, and every stage is restart-safe.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from run_cluster_benchmark import (JUDGES, RunLog, configure_storage,
                                   require_visionarena_access)


ROOT = Path(__file__).resolve().parent
LIMIT = 1_000
CONFIG_3 = "configs/vlm_v2_small_3judge.yaml"
CONFIG_4 = "configs/vlm_v2_small_4judge.yaml"
RESULT_DIR = "result/vlm_v2_balanced_1k"
EXPORT_DIR = ROOT / "data/vlm_v2_small_export"
MANIFEST = ROOT / "data/vlm_v2_small/manifest.json"


def prepared_rows() -> int:
    pool = EXPORT_DIR / "eval_pool.jsonl"
    if not pool.is_file() or not MANIFEST.is_file():
        return 0
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if manifest.get("config", {}).get("limit") != LIMIT:
        return 0
    with pool.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def cache_rows(judge: str) -> int:
    path = ROOT / RESULT_DIR / f"{judge}.eval_pool.jsonl"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log = RunLog(ROOT / f"logs/small-{stamp}.log")
    state_path = ROOT / "outputs/small-cluster-run-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"started_at": datetime.now().astimezone().isoformat(),
             "status": "running", "profile": "balanced-1000-per-source",
             "log": str(log.path.relative_to(ROOT))}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        configure_storage(log)
        rows = prepared_rows()
        if not rows:
            require_visionarena_access()
            log.command([sys.executable, "-u", "get_data.py", "--limit", str(LIMIT),
                         "--legacy-split", "0", "--out-dir", "data/vlm_v2_small",
                         "--export-dir", "data/vlm_v2_small_export"])
            rows = prepared_rows()
        log.command([sys.executable, "cluster/verify_small_data.py"])
        log.command([sys.executable, "-m", "open_cascade", "prepare-data", CONFIG_4])

        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU is visible. Run inside the allocated GPU shell.")
        log.write(f"GPU: {torch.cuda.get_device_name(0)}")

        for judge in JUDGES:
            cache = f"{RESULT_DIR}/{judge}.eval_pool.jsonl"
            if cache_rows(judge) == rows:
                log.write(f"Reusing complete cache: {judge} ({rows} rows)")
            else:
                log.command([sys.executable, "-u", "cluster/smoke_judge.py", judge,
                             "--config", CONFIG_4])
                log.command([sys.executable, "cluster/repair_jsonl.py", cache])
                log.command([sys.executable, "-u", "-m", "open_cascade", "score",
                             CONFIG_4, "--judge", judge])
            log.command([sys.executable, "cluster/verify_scored.py", judge,
                         "--result-dir", RESULT_DIR, "--expected-rows", str(rows)])

        for config in (CONFIG_3, CONFIG_4):
            log.command([sys.executable, "-u", "-m", "open_cascade", "evaluate", config])

        results = ["outputs/vlm_v2_small_3judge/baseline_comparison.json",
                   "outputs/vlm_v2_small_4judge/baseline_comparison.json"]
        state.update({"status": "complete", "eval_rows": rows,
                      "finished_at": datetime.now().astimezone().isoformat(),
                      "results": results})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.write("\nBalanced-small benchmark complete.")
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
