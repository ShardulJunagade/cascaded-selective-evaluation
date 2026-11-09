#!/usr/bin/env python3
"""Fail unless the exact full corpus needed by the benchmark is present."""
import json
from pathlib import Path

EXPECTED_EVAL_ROWS = 25_308
export = Path("data/vlm_v2_export/eval_pool.jsonl")
manifest_path = Path("data/vlm_v2/manifest.json")

if not export.is_file():
    raise SystemExit(f"Missing {export}")
with export.open(encoding="utf-8") as handle:
    count = sum(1 for line in handle if line.strip())
if count != EXPECTED_EVAL_ROWS:
    raise SystemExit(f"Expected {EXPECTED_EVAL_ROWS} eval rows, found {count}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
failed = {name: info.get("status") for name, info in manifest["downloads"].items()
          if info.get("status") != "ok"}
if failed:
    raise SystemExit(f"Not every source downloaded successfully: {failed}")
if manifest["counts"]["eval_pool"] != EXPECTED_EVAL_ROWS:
    raise SystemExit("Manifest and eval_pool row count disagree")

print(f"Full corpus verified: {count} eval_pool rows from five sources")
