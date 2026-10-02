"""Check the exported splits against the repo's own loaders before scoring.

A judge run costs GPU hours, so the failure modes worth catching cheaply are
the ones that would only surface mid-run or, worse, not at all:

* an `image_path` the judge cannot open,
* a `preferences` majority that disagrees with the label the pipeline computed,
* a label that is constant, which is the defect that invalidated the previous
  VLM numbers,
* few-shot examples leaking into the evaluation pool,
* `sample_key` collisions, which would make `merge_data` silently drop rows.

    python -m prepare_data.verify_export
    python -m prepare_data.verify_export --dir ./data/vlm_v2_export
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from prepare_data.config import EXPORT_ROOT, REPO_ROOT

LABEL_TO_INDEX = {"A": 1, "B": 2}


def majority(preferences: Dict[str, int]) -> int:
    """The label `cascaded_evaluation.util.prepare_data` would derive."""
    return Counter(preferences.values()).most_common(1)[0][0]


def check_split(path: Path, name: str) -> Tuple[List[str], Dict]:
    from open_cascade.data import read_jsonl, sample_key

    problems: List[str] = []
    rows = read_jsonl(path)
    if not rows:
        return [f"{name}: empty"], {}

    labels = Counter()
    keys = Counter()
    missing_images = 0

    for i, row in enumerate(rows):
        for field in ("instruction", "outputs", "preferences"):
            if field not in row:
                problems.append(f"{name}[{i}]: missing {field}")
        if len(row.get("outputs", [])) != 2:
            problems.append(f"{name}[{i}]: outputs is not a pair")

        image = row.get("image_path")
        if not image:
            problems.append(f"{name}[{i}]: no image_path")
        elif not (REPO_ROOT / image).exists() and not Path(image).exists():
            missing_images += 1

        winner = majority(row["preferences"])
        labels[winner] += 1
        keys[sample_key(row)] += 1

        declared = row.get("source", {}).get("label")
        if declared and LABEL_TO_INDEX.get(declared) != winner:
            problems.append(
                f"{name}[{i}]: preferences majority {winner} disagrees with "
                f"source.label {declared}")

    if missing_images:
        problems.append(f"{name}: {missing_images}/{len(rows)} image_path values "
                        "do not resolve from the repo root")

    duplicates = sum(count - 1 for count in keys.values() if count > 1)
    if duplicates:
        problems.append(f"{name}: {duplicates} sample_key collisions; merge_data "
                        "would silently drop these")

    share_a = labels.get(1, 0) / len(rows)
    if not 0.35 < share_a < 0.65:
        problems.append(
            f"{name}: label balance is {share_a:.1%} A. A near-constant label makes "
            "any agreement number meaningless - check position randomization.")

    return problems, {"rows": len(rows), "share_a": round(share_a, 4),
                      "unique_keys": len(keys)}


def check_fewshot(path: Path, eval_keys: set) -> Tuple[List[str], Dict]:
    from open_cascade.data import load_fewshot, read_jsonl, sample_key

    problems: List[str] = []
    pools = load_fewshot(path)
    if not pools:
        return [f"{path.name}: no few-shot pools"], {}

    sizes = {len(pool) for pool in pools}
    if len(sizes) > 1:
        problems.append(f"{path.name}: pools have differing K {sorted(sizes)}")

    seen = set()
    for pool in pools:
        for example in pool:
            key = sample_key(example)
            if key in eval_keys:
                problems.append(f"{path.name}: example also present in eval_pool")
            if key in seen:
                problems.append(f"{path.name}: example reused across pools")
            seen.add(key)

    annotators = [row["annotator"] for row in read_jsonl(path)]
    return problems, {"n_pools": len(pools), "k": sorted(sizes),
                      "annotators": annotators}


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=EXPORT_ROOT)
    args = parser.parse_args()

    from open_cascade.data import read_jsonl, sample_key

    export_dir = Path(args.dir)
    if not export_dir.exists():
        print(f"No export at {export_dir}. Run `python -m prepare_data.run` first.")
        return 1

    all_problems: List[str] = []
    eval_keys: set = set()

    for name in ("eval_pool", "dev", "ties", "calibration", "test"):
        path = export_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        if name == "ties":
            # Ties carry no forced-choice label, so the label checks do not apply.
            rows = read_jsonl(path)
            print(f"{name:<14} {len(rows)} rows (no label checks; ties are unlabelled)")
            continue
        problems, summary = check_split(path, name)
        all_problems += problems
        print(f"{name:<14} {summary}")
        if name == "eval_pool":
            eval_keys = {sample_key(row) for row in read_jsonl(path)}

    for path in sorted(export_dir.glob("fewshot*.jsonl")):
        problems, summary = check_fewshot(path, eval_keys)
        all_problems += problems
        print(f"{path.stem:<14} {summary}")

    print()
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1

    print("Export looks consumable by run_open_vlm_judge.py / run_open_cascade.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
