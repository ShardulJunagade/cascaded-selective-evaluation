"""Phase 1: raw inspection, before any adapter touches the data.

This exists so the bias and bottleneck answers can be stated about the sources
*as published*, separately from what our own filtering did to them. It writes
`reports/raw_inspection.md`.

    python -m prepare_data.inspect_raw --source all
"""

from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from prepare_data.config import (ALL_SOURCES, REPORT_ROOT, SOURCES, raw_dir,
                                 use_utf8_stdout)
from prepare_data.download import human, read_log

MAX_ROWS_SCANNED = 4000
SEED = 42


def scan_indices(n_rows: int, limit: int = MAX_ROWS_SCANNED,
                 salt: str = "") -> List[int]:
    """Row indices to scan: a seeded random sample, never a prefix.

    Every one of these files is stored in a correlated order - MM-RLHF groups
    by subset, RLHF-V by origin dataset - so scanning the first N rows reports
    the statistics of whichever block happens to come first. Taking the prefix
    of MM-RLHF, for instance, reports zero video rows when the release has
    2,855 of them.
    """
    if n_rows <= limit:
        return list(range(n_rows))
    rng = random.Random(f"{SEED}-{salt}")
    return sorted(rng.sample(range(n_rows), limit))


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_No data._\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _quantiles(values: List[float]) -> Dict[str, float]:
    """Five-number summary, ignoring missing values.

    NaN must be filtered before sorting, not after: every comparison against
    NaN is False, so `sorted()` returns a silently non-monotonic list and the
    quantiles read out of it are nonsense rather than merely approximate.
    These CSVs do carry a handful of empty response cells, so this is a real
    case, not a defensive one.
    """
    finite = [float(v) for v in values if v is not None and v == v]
    if not finite:
        return {}

    ordered = sorted(finite)
    def at(q: float) -> float:
        return round(ordered[min(int(q * len(ordered)), len(ordered) - 1)], 1)

    summary = {"min": round(ordered[0], 1), "p25": at(0.25), "median": at(0.5),
               "p75": at(0.75), "max": round(ordered[-1], 1)}
    if len(finite) != len(values):
        summary["missing"] = len(values) - len(finite)
    return summary


# --------------------------------------------------------------------------
# per-source inspection
# --------------------------------------------------------------------------

def inspect_rlhf_v() -> Dict:
    from datasets import load_dataset

    files = sorted(str(p) for p in (raw_dir("rlhf_v") / "hf").rglob("*.parquet")
                   if ".cache" not in p.parts)
    dataset = load_dataset("parquet", data_files={"train": files}, split="train")

    lengths_chosen, lengths_rejected = [], []
    origins = Counter()
    longer_is_chosen = 0
    comparable = 0

    indices = scan_indices(len(dataset), salt="rlhf_v")
    for i in indices:
        row = dataset[i]
        origins[row.get("origin_dataset")] += 1
        try:
            text = json.loads(row["text"]) if isinstance(row["text"], str) else row["text"]
        except (json.JSONDecodeError, TypeError):
            continue
        chosen, rejected = text.get("chosen", ""), text.get("rejected", "")
        lengths_chosen.append(len(chosen))
        lengths_rejected.append(len(rejected))
        if len(chosen) != len(rejected):
            comparable += 1
            longer_is_chosen += len(chosen) > len(rejected)

    return {
        "rows": len(dataset),
        "fields": list(dataset.features),
        "label_values": {"always chosen (correction)": len(dataset)},
        "winner_position_balance": "n/a - chosen side is not positioned in the raw data",
        "annotator_ids": False,
        "annotations_per_item": 1,
        "length_chosen": _quantiles(lengths_chosen),
        "length_rejected": _quantiles(lengths_rejected),
        "p_longer_is_preferred": (round(longer_is_chosen / comparable, 4)
                                  if comparable else None),
        "origin_datasets": origins.most_common(10),
        "scanned": len(indices),
        "sampling": "seeded random sample" if len(indices) < len(dataset) else "census",
    }


def inspect_visit_bench() -> Dict:
    import pandas as pd

    path = raw_dir("visit_bench") / SOURCES["visit_bench"]["filename"]
    frame = pd.read_csv(path).rename(columns={"model_selection.A": "sel_a",
                                              "model_selection.B": "sel_b"})

    grouped = frame.groupby(["image_url", "instruction", "A", "B", "A_model", "B_model"],
                            dropna=False)
    per_tuple = grouped.size()

    lengths_a = frame["A"].astype(str).str.len().tolist()
    lengths_b = frame["B"].astype(str).str.len().tolist()

    # A handful of response cells are empty, and NaN compares unequal to
    # everything, so pairs involving one would otherwise be counted as
    # comparable and always score against the longer response.
    pairs = [(a, b, sel) for a, b, sel
             in zip(lengths_a, lengths_b, frame["sel_a"])
             if a == a and b == b and a != b]
    comparable = len(pairs)
    longer_wins = sum(1 for a, b, sel in pairs if (a > b) == bool(sel))

    return {
        "rows": len(frame),
        "fields": list(frame.columns),
        "unique_tuples": int(per_tuple.shape[0]),
        "judgments_per_tuple": {str(k): int(v) for k, v
                                in per_tuple.value_counts().sort_index().items()},
        "unique_images": int(frame["image_url"].nunique()),
        "unique_instructions": int(frame["instruction"].nunique()),
        "label_values": {"A selected": int(frame["sel_a"].sum()),
                         "B selected": int(frame["sel_b"].sum())},
        "winner_position_balance": round(float(frame["sel_a"].mean()), 4),
        "annotator_ids": False,
        "models": frame["A_model"].value_counts().to_dict(),
        "length_a": _quantiles(lengths_a),
        "length_b": _quantiles(lengths_b),
        "p_longer_is_preferred": (round(longer_wins / comparable, 4)
                                  if comparable else None),
        "images_are_urls": True,
    }


def inspect_mm_rlhf() -> Dict:
    from datasets import load_dataset

    files = sorted(str(p) for p in (raw_dir("mm_rlhf") / "hf").rglob("*.parquet")
                   if ".cache" not in p.parts)
    if not files:
        return {"error": "no parquet shards found"}
    dataset = load_dataset("parquet", data_files={"train": files}, split="train")

    indices = scan_indices(len(dataset), salt="mm_rlhf")
    n_image, n_video, n_outputs = 0, 0, Counter()
    subsets = Counter()
    for i in indices:
        row = dataset[i]
        image = row.get("image") or ""
        n_image += bool(image)
        n_video += bool(row.get("video"))
        n_outputs[len(row.get("models_output") or [])] += 1
        subsets[image.split("/")[0] if image else "<no image>"] += 1

    archives = [(p.name, p.stat().st_size)
                for p in (raw_dir("mm_rlhf") / "hf").rglob("*.zip")
                if ".cache" not in p.parts]

    return {
        "rows": len(dataset),
        "fields": list(dataset.features),
        "scanned": len(indices),
        "sampling": "seeded random sample" if len(indices) < len(dataset) else "census",
        "with_image": n_image,
        "with_video": n_video,
        "subset_distribution": subsets.most_common(),
        "outputs_per_prompt": dict(sorted(n_outputs.items())),
        "label_values": {"ranking, no pairwise label": len(dataset)},
        "annotator_ids": False,
        "image_archives": [(name, human(size)) for name, size in archives],
    }


def inspect_judge_anything() -> Dict:
    from prepare_data.adapters.judge_anything import JudgeAnythingAdapter
    from prepare_data.funnel import Funnel
    from prepare_data.media import MediaStore

    adapter = JudgeAnythingAdapter(MediaStore(), Funnel())
    rows = adapter._read_rows()          # noqa: SLF001 - inspection is the point
    if not rows:
        return {"error": f"no readable rows under {adapter.hf_dir()}"}

    fields = Counter()
    for row in rows[:MAX_ROWS_SCANNED]:
        fields.update(row.keys())

    return {
        "rows": len(rows),
        "fields": sorted(fields),
        "field_coverage": {k: v for k, v in fields.most_common(30)},
        "example": {k: str(v)[:160] for k, v in list(rows[0].items())[:15]},
    }


def inspect_visionarena() -> Dict:
    from datasets import load_dataset

    from prepare_data.adapters.visionarena import active_categories

    files = sorted(str(p) for p in (raw_dir("visionarena") / "hf").rglob("*.parquet")
                   if ".cache" not in p.parts)
    if not files:
        return {"error": "not downloaded (gated)"}

    # Read only the metadata columns. Touching a row through the full schema
    # decodes its image, which turns a few seconds of counting into many
    # minutes and tells us nothing about the labels.
    columns = ["judge", "winner", "num_turns", "language", "categories"]
    dataset = load_dataset("parquet", data_files={"train": files}, split="train",
                           columns=columns)

    winners = Counter(dataset["winner"])
    languages = Counter(dataset["language"])
    turns = Counter(dataset["num_turns"])
    voters = Counter(dataset["judge"])
    categories = Counter(active_categories(c) for c in dataset["categories"])

    # Every row is the whole corpus here, so this is a census, not a sample.
    votes_per_voter = Counter(voters.values())
    decided = winners.get("model_a", 0) + winners.get("model_b", 0)

    return {
        "rows": len(dataset),
        "fields": columns + ["images", "question_id", "model_a", "model_b",
                             "conversation_a", "conversation_b", "tstamp",
                             "conv_metadata"],
        "scanned": len(dataset),
        "label_values": dict(winners),
        "winner_position_balance": round(winners.get("model_a", 0) / max(1, decided), 4),
        "tie_fraction": round(
            sum(v for k, v in winners.items() if k and k.startswith("tie")) / len(dataset), 4),
        "annotator_ids": True,
        "unique_voters": len(voters),
        "mean_votes_per_voter": round(len(dataset) / max(1, len(voters)), 3),
        "votes_per_voter": dict(sorted(votes_per_voter.items())[:10]),
        # The Ind. pools need annotators with at least K+1 labels; this is the
        # supply, and the reason VisionArena is the only source that can feed them.
        "voters_with_at_least_k_plus_1": {
            k: sum(1 for n in voters.values() if n >= k + 1) for k in (1, 2, 3, 5)},
        "turns_per_conversation": dict(sorted(turns.items())[:8]),
        "single_turn": turns.get(1, 0),
        "languages": languages.most_common(12),
        "categories": categories.most_common(12),
    }


INSPECTORS = {
    "rlhf_v": inspect_rlhf_v,
    "visit_bench": inspect_visit_bench,
    "mm_rlhf": inspect_mm_rlhf,
    "judge_anything": inspect_judge_anything,
    "visionarena": inspect_visionarena,
}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render(results: Dict[str, Dict]) -> str:
    parts = ["# Phase 1: raw source inspection\n",
             "Statistics of each source **as published**, before any filtering, "
             "pairing or randomization by this pipeline. Produced by "
             "`python -m prepare_data.inspect_raw`.\n"]

    rows = []
    for source, result in results.items():
        log = read_log(source) or {}
        rows.append([
            source,
            result.get("rows", "-"),
            log.get("status", "-"),
            human(log.get("bytes", 0)),
            f"{log.get('seconds', 0):.0f}s",
            "yes" if result.get("annotator_ids") else "no",
        ])
    parts.append("\n## Overview\n")
    parts.append(_table(["Source", "Raw rows", "Download", "Size", "Time",
                         "Annotator ids"], rows))

    for source, result in results.items():
        parts.append(f"\n## {source}\n")
        spec = SOURCES.get(source, {})
        parts.append(f"`{spec.get('repo') or spec.get('url', '')}`\n\n")
        if "error" in result:
            parts.append(f"**Not inspected**: {result['error']}\n")
            continue
        parts.append("```json\n" + json.dumps(result, indent=2, default=str) + "\n```\n")

    return "".join(parts)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=["all"])
    parser.add_argument("--out", type=Path, default=REPORT_ROOT / "raw_inspection.md")
    return parser.parse_args()


def main(sources: Optional[Sequence[str]] = None, out: Optional[Path] = None) -> Dict:
    use_utf8_stdout()
    args = parse_args() if sources is None else None
    sources = list(sources or args.source)
    out = out or (args.out if args else REPORT_ROOT / "raw_inspection.md")
    if "all" in sources:
        sources = ALL_SOURCES

    results: Dict[str, Dict] = {}
    for source in sources:
        print(f"[{source}] inspecting ...")
        try:
            results[source] = INSPECTORS[source]()
        except Exception as exc:                   # noqa: BLE001 - recorded in the report
            results[source] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  ! {type(exc).__name__}: {exc}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {out}")
    return results


if __name__ == "__main__":
    main()
