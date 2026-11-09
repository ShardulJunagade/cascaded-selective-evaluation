#!/usr/bin/env python3
"""Verify the balanced 1,000-per-source export and print its composition."""
import json
from collections import Counter
from pathlib import Path

LIMIT = 1_000
MIN_EVAL_ROWS = 1_500
SOURCES = {"rlhf_v", "visit_bench", "visionarena", "mm_rlhf", "judge_anything"}
export = Path("data/vlm_v2_small_export/eval_pool.jsonl")
manifest_path = Path("data/vlm_v2_small/manifest.json")

if not export.is_file() or not manifest_path.is_file():
    raise SystemExit("Balanced-small export is missing; run run_small_benchmark.py")

counts = Counter()
rows = 0
with export.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        counts[row.get("source", {}).get("dataset", "unknown")] += 1
        rows += 1

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("config", {}).get("limit") != LIMIT:
    raise SystemExit("Manifest was not built with limit=1000")
if rows != manifest.get("counts", {}).get("eval_pool"):
    raise SystemExit("Manifest and balanced-small eval_pool row count disagree")
if rows < MIN_EVAL_ROWS:
    raise SystemExit(f"Balanced-small pool is unexpectedly small: {rows} rows")
missing = SOURCES - set(counts)
if missing:
    raise SystemExit(f"Balanced-small pool is missing sources: {sorted(missing)}")

print(f"Balanced-small corpus verified: {rows} eval rows")
for source in sorted(counts):
    print(f"  {source:<18} {counts[source]}")
