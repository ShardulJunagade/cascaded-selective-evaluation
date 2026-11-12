#!/usr/bin/env python3
"""Run the fast 1,000-row, N=3, K=1 benchmark on the current GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from run_cluster_benchmark import (JUDGES, RunLog, configure_storage,
                                   require_visionarena_access)
from run_small_benchmark import LIMIT, prepared_rows


ROOT = Path(__file__).resolve().parent
CONFIG_3 = "configs/vlm_v2_fast_3judge.yaml"
CONFIG_4 = "configs/vlm_v2_fast_4judge.yaml"
RESULT_DIR = "result/vlm_v2_fast_1k_n3_k1"
EXPORT_DIR = ROOT / "data/vlm_v2_fast_export"
TARGET_ROWS = 1_000


def fast_rows() -> int:
    manifest_path = EXPORT_DIR / "manifest.json"
    pool_path = EXPORT_DIR / "eval_pool.jsonl"
    if not manifest_path.is_file() or not pool_path.is_file():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("target_rows") != TARGET_ROWS
            or manifest.get("n_annotators") != 3
            or manifest.get("k_shot") != 1):
        return 0
    with pool_path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return rows if rows == manifest.get("selected_rows") else 0


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
    log = RunLog(ROOT / f"logs/fast-{stamp}.log")
    state_path = ROOT / "outputs/fast-cluster-run-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"started_at": datetime.now().astimezone().isoformat(),
             "status": "running", "profile": "fast-1000-n3-k1-order-swap",
             "log": str(log.path.relative_to(ROOT))}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        configure_storage(log)
        if not prepared_rows():
            require_visionarena_access()
            log.command([sys.executable, "-u", "get_data.py", "--limit", str(LIMIT),
                         "--legacy-split", "0", "--out-dir", "data/vlm_v2_small",
                         "--export-dir", "data/vlm_v2_small_export"])
        log.command([sys.executable, "cluster/verify_small_data.py"])
        log.command([sys.executable, "cluster/build_fast_profile.py"])
        rows = fast_rows()
        if not rows:
            raise RuntimeError("Fast profile was not built correctly")
        log.command([sys.executable, "-m", "open_cascade", "prepare-data", CONFIG_4])

        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU is visible. Run inside the allocated GPU shell.")
        log.write(f"GPU: {torch.cuda.get_device_name(0)}")
        log.command([sys.executable, "cluster/check_vllm_runtime.py"])

        for judge in JUDGES:
            cache = f"{RESULT_DIR}/{judge}.eval_pool.jsonl"
            if cache_rows(judge) == rows:
                log.write(f"Reusing complete cache: {judge} ({rows} rows)")
            else:
                # The first real chunk is the compatibility check. Avoid loading every
                # model twice through a separate smoke run in this time-focused profile.
                log.command([sys.executable, "cluster/repair_jsonl.py", cache])
                log.command([sys.executable, "-u", "-m", "open_cascade", "score",
                             CONFIG_4, "--judge", judge])
            log.command([sys.executable, "cluster/verify_scored.py", judge,
                         "--result-dir", RESULT_DIR, "--expected-rows", str(rows)])

        for config in (CONFIG_3, CONFIG_4):
            log.command([sys.executable, "-u", "-m", "open_cascade", "evaluate", config])

        results = ["outputs/vlm_v2_fast_3judge/baseline_comparison.json",
                   "outputs/vlm_v2_fast_4judge/baseline_comparison.json"]
        state.update({"status": "complete", "eval_rows": rows,
                      "requests_per_judge": rows * 3 * 2,
                      "finished_at": datetime.now().astimezone().isoformat(),
                      "results": results})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.write("\nFast benchmark complete.")
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
    raise SystemExit(main())
