#!/usr/bin/env python3
"""Validate one completed judge cache before downstream evaluation."""
import argparse
import json
from pathlib import Path

EXPECTED_EVAL_ROWS = 25_308

parser = argparse.ArgumentParser()
parser.add_argument("judge")
args = parser.parse_args()
path = Path("result/vlm_v2_maj") / f"{args.judge}.eval_pool.jsonl"

count = 0
with path.open(encoding="utf-8") as handle:
    for number, line in enumerate(handle, 1):
        row = json.loads(line)
        if len(row.get("probs", [])) != 2:
            raise SystemExit(f"{path}:{number}: invalid aggregate probabilities")
        simulations = (row.get("judge_details") or {}).get("simulations", [])
        if not simulations:
            raise SystemExit(f"{path}:{number}: missing per-run judge details")
        count += 1
if count != EXPECTED_EVAL_ROWS:
    raise SystemExit(f"{path}: expected {EXPECTED_EVAL_ROWS} rows, found {count}")
print(f"Scored cache verified: {args.judge}, {count} rows")
