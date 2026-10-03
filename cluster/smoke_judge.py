#!/usr/bin/env python3
"""Run one real pair through a judge before committing to its full scoring pass."""
import argparse
from pathlib import Path
import sys

# Direct execution sets sys.path[0] to cluster/, not the repository root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_cascade.config import load_config
from open_cascade.data import load_fewshot, read_jsonl
from open_cascade.judges import build_judge

parser = argparse.ArgumentParser()
parser.add_argument("judge")
parser.add_argument("--config", default="configs/vlm_v2_4judge.yaml")
args = parser.parse_args()
config = load_config(args.config)
sample = read_jsonl(config.data.split_file("eval_pool"))[0]
fewshot = load_fewshot(config.data.fewshot_file)

judge = build_judge(args.judge, enable_prefix_caching=True,
                    max_fewshot_examples=config.scoring.max_fewshot_examples)
try:
    probabilities, details = judge.simulate_annotators_batch_with_details([sample], fewshot)
    if len(probabilities) != 1 or len(probabilities[0]) != 2:
        raise SystemExit(f"Invalid smoke result for {args.judge}: {probabilities}")
    if not details[0].get("simulations"):
        raise SystemExit(f"No per-run details for {args.judge}")
    print(f"Smoke passed: {args.judge}: {probabilities[0]}")
finally:
    judge.shutdown()
