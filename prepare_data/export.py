"""Phase 5: canonical records -> the jsonl format the judge scripts already read.

The repo's instance format is

    {"image_path", "instruction", "outputs": [a, b], "preferences": {...}, "source": {...}}

where a preference value of 1 means `outputs[0]` is better and 2 means
`outputs[1]` is better (see `cascaded_evaluation.util.prepare_data`). Real
annotator ids are used as the keys where a source has them, so the majority
vote that `util.merge_data` takes reproduces the label computed here.

What gets written:

    fewshot.<pool>.jsonl   one line per simulated annotator
    dev.jsonl              hyperparameter selection only (K, N, tau)
    eval_pool.jsonl        everything the 1000 random cal/test draws come from
    ties.jsonl             tied instances, for the confidence -> 0.5 test

Deliberately absent: a single calibration.jsonl / test.jsonl pair. One split
cannot show whether the (1 - delta) guarantee holds, so the splits are drawn at
evaluation time from eval_pool. `--legacy-split` writes one such pair anyway
for a quick smoke run against the existing scripts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from prepare_data.config import EXPORT_ROOT, OUT_ROOT, REPO_ROOT
from prepare_data.schema import Record

LABEL_TO_INDEX = {"A": 1, "B": 2}


def image_path_for(record: Record) -> str:
    """Path the judge will open, relative to the repo root (its working dir)."""
    absolute = (OUT_ROOT / record.media[0]).resolve()
    try:
        return absolute.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def to_instance(record: Record) -> Dict:
    """One canonical record as a repo-format evaluation instance.

    Tie records are exported with an empty `preferences` map. They carry no
    forced choice by construction - a VisionArena "tie (bothbad)" vote is not a
    hidden A or B - and inventing one would corrupt the very set that exists to
    test whether confidence falls to 0.5 under genuine indifference. Consumers
    read `source.is_tie` and score these for confidence only, never agreement.
    """
    preferences: Dict[str, int] = {}
    for annotation in record.annotations:
        if annotation["label"] in LABEL_TO_INDEX:
            preferences[annotation["annotator"]] = LABEL_TO_INDEX[annotation["label"]]

    if not preferences and not record.is_tie:
        # Sources without per-annotator labels (RLHF-V corrections, MM-RLHF
        # ranking-derived pairs) still carry a single aggregate preference.
        if record.label not in LABEL_TO_INDEX:
            raise ValueError(
                f"{record.uid}: non-tie record has no usable label ({record.label!r}); "
                "the adapter should have routed it to the tie set.")
        preferences = {"human": LABEL_TO_INDEX[record.label]}

    return {
        "image_path": image_path_for(record),
        "instruction": record.instruction,
        "outputs": [record.response_a, record.response_b],
        "preferences": preferences,
        "source": {
            "uid": record.uid,
            "dataset": record.source,
            "group_id": record.group_id,
            "media_hash": record.media_hash,
            "modality": record.modality,
            "label": record.label,
            "label_origin": record.label_origin,
            "swapped": record.swapped,
            "n_ann": record.n_ann,
            "agreement": record.agreement,
            "is_tie": record.is_tie,
            "model_a": record.model_a,
            "model_b": record.model_b,
            "meta": record.meta,
        },
    }


def write_jsonl(rows: Iterable[Dict], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit newline so the file is byte-identical on Windows and Linux,
    # matching open_cascade.data.write_jsonl.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_split(records: Sequence[Record], path: Path) -> int:
    return write_jsonl((to_instance(r) for r in records), path)


def export_fewshot(pools: Sequence[Dict], path: Path) -> int:
    """One line per simulated annotator: {"annotator", "evaluated_samples"}."""
    rows = [
        {
            "annotator": pool["annotator"],
            "evaluated_samples": [to_instance(r) for r in pool["records"]],
        }
        for pool in pools
    ]
    return write_jsonl(rows, path)


def export_all(splits: Dict[str, List[Record]], ties: Sequence[Record],
               fewshot_pools: Dict[str, Sequence[Dict]],
               out_dir: Path = EXPORT_ROOT,
               legacy_split: Optional[int] = None,
               seed: int = 42) -> Dict[str, int]:
    """Write every split and pool; return {filename: row count}."""
    out_dir = Path(out_dir)
    written: Dict[str, int] = {}

    for name in ("dev", "eval_pool"):
        if name in splits:
            written[f"{name}.jsonl"] = export_split(splits[name], out_dir / f"{name}.jsonl")

    written["ties.jsonl"] = export_split(list(ties), out_dir / "ties.jsonl")

    for pool_name, pools in fewshot_pools.items():
        filename = f"fewshot.{pool_name}.jsonl"
        written[filename] = export_fewshot(pools, out_dir / filename)

    # The judge scripts default to ./data/vlm/split/fewshot.jsonl; give them a
    # conventional name too so the common case needs no extra flag.
    if "maj" in fewshot_pools:
        written["fewshot.jsonl"] = export_fewshot(fewshot_pools["maj"],
                                                  out_dir / "fewshot.jsonl")

    if legacy_split:
        pool = list(splits.get("eval_pool", []))
        random.Random(f"{seed}-legacy").shuffle(pool)
        size = min(legacy_split, max(0, len(pool) - 1))
        written["calibration.jsonl"] = export_split(pool[:size], out_dir / "calibration.jsonl")
        written["test.jsonl"] = export_split(pool[size:], out_dir / "test.jsonl")

    return written
