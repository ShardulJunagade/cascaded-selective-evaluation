#!/usr/bin/env python3
"""Print saved progress for the cluster benchmark without loading a model."""
import json
from pathlib import Path

judges = ["granite-vision-3.3-2b", "qwen2.5-vl-3b-instruct",
          "phi-3.5-vision-instruct", "internvl3-8b"]
print("Scored rows")
for judge in judges:
    path = Path("result/vlm_v2_multifamily") / f"{judge}.eval_pool.jsonl"
    count = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0
    print(f"  {judge:<30} {count:>6} / 25308")

print("\nResults")
for chain in ("vlm_v2_3judge", "vlm_v2_4judge"):
    path = Path("outputs") / chain / "baseline_comparison.json"
    print(f"  {chain:<20} {'complete' if path.exists() else 'pending'}")

state = Path("outputs/cluster-run-state.json")
if state.exists():
    print("\nLast run")
    print(json.dumps(json.loads(state.read_text(encoding="utf-8")), indent=2))
