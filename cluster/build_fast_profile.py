#!/usr/bin/env python3
"""Build a deterministic, source-stratified fast profile from the small corpus."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_cascade.data import read_jsonl, sample_key, write_jsonl
from open_cascade.experiments.multi_split import _components


def source_name(row: dict) -> str:
    return row.get("source", {}).get("dataset", "unknown")


def select_component_stratified(rows: list[dict], target: int, seed: int) -> list[dict]:
    """Choose whole leakage components while tracking the source proportions."""
    if target <= 0 or target > len(rows):
        raise ValueError(f"target must be in [1, {len(rows)}], got {target}")

    by_key = {sample_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Input eval_pool contains duplicate sample identities")
    components = [[by_key[key] for key in keys] for keys in _components(rows)]
    source_totals = Counter(source_name(row) for row in rows)
    proportions = {source: count / len(rows) for source, count in source_totals.items()}

    def stable_rank(component: list[dict]) -> str:
        keys = sorted(sample_key(row) for row in component)
        return hashlib.sha256(f"{seed}\n".encode() + "\n".join(keys).encode()).hexdigest()

    remaining = [(component, stable_rank(component)) for component in components]
    selected: list[dict] = []
    selected_counts: Counter = Counter()

    while len(selected) < target:
        room = target - len(selected)
        candidates = [(component, rank) for component, rank in remaining
                      if len(component) <= room]
        if not candidates:
            break

        def score(item):
            component, rank = item
            counts = selected_counts + Counter(source_name(row) for row in component)
            size = len(selected) + len(component)
            error = sum((counts[source] / size - share) ** 2
                        for source, share in proportions.items())
            return error, rank

        component, rank = min(candidates, key=score)
        selected.extend(component)
        selected_counts.update(source_name(row) for row in component)
        remaining.remove((component, rank))

    # The whole original component must be either entirely selected or entirely absent.
    selected_keys = {sample_key(row) for row in selected}
    for component in components:
        overlap = sum(sample_key(row) in selected_keys for row in component)
        if overlap not in (0, len(component)):
            raise AssertionError("Fast selection split a leakage component")

    return selected


def build(source: Path, output: Path, target: int, seed: int,
          annotators: int, k_shot: int) -> dict:
    eval_rows = read_jsonl(source / "eval_pool.jsonl")
    fewshot = read_jsonl(source / "fewshot.jsonl")
    if len(fewshot) < annotators:
        raise ValueError(f"Need {annotators} few-shot pools, found {len(fewshot)}")

    selected = select_component_stratified(eval_rows, target, seed)
    fast_fewshot = []
    for pool in fewshot[:annotators]:
        examples = pool.get("evaluated_samples", [])
        if len(examples) < k_shot:
            raise ValueError(f"Few-shot pool has {len(examples)} examples; need {k_shot}")
        fast_fewshot.append({**pool, "evaluated_samples": examples[:k_shot]})

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected, output / "eval_pool.jsonl")
    write_jsonl(fast_fewshot, output / "fewshot.jsonl")

    counts = Counter(source_name(row) for row in selected)
    digest = hashlib.sha256(
        "\n".join(sorted(sample_key(row) for row in selected)).encode()
    ).hexdigest()
    manifest = {
        "profile": "fast-stratified",
        "seed": seed,
        "target_rows": target,
        "selected_rows": len(selected),
        "source_rows": dict(sorted(counts.items())),
        "n_annotators": annotators,
        "k_shot": k_shot,
        "response_orderings": 2,
        "requests_per_judge": len(selected) * annotators * 2,
        "selection_digest": digest,
        "selection_unit": "complete prompt/image leakage component",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/vlm_v2_small_export"))
    parser.add_argument("--output", type=Path, default=Path("data/vlm_v2_fast_export"))
    parser.add_argument("--target", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--k-shot", type=int, default=1)
    args = parser.parse_args()
    manifest = build(args.source, args.output, args.target, args.seed,
                     args.annotators, args.k_shot)
    print(f"Fast profile ready: {manifest['selected_rows']} rows, "
          f"{manifest['requests_per_judge']} requests/judge")
    for source, count in manifest["source_rows"].items():
        print(f"  {source:<18} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
