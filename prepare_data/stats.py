"""Phase 4: the manifest and the bias tables.

Every number Assignment 2 asks for about bias and bottlenecks is produced here
from the records themselves, not from memory of what the pipeline did.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from prepare_data.config import K_VALUES, REPORT_ROOT, SOURCES
from prepare_data.funnel import Funnel
from prepare_data.schema import Record
from prepare_data.stages import annotator_capacity


def _pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}%" if denominator else "-"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_No data._\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# individual tables
# --------------------------------------------------------------------------

def label_balance(records: Sequence[Record]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0})
    for record in records:
        if record.label in ("A", "B"):
            out[record.source][record.label] += 1
    return {source: dict(counts) for source, counts in out.items()}


def length_bias(records: Sequence[Record]) -> Dict[str, Dict[str, object]]:
    """P(the longer response is the preferred one), per source.

    A judge can score well on a set where this is far from 0.5 simply by
    preferring length, so this is reported before any agreement number is.
    """
    out: Dict[str, Dict[str, object]] = {}
    by_source: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    for source, rows in by_source.items():
        longer_wins = 0
        comparable = 0
        for record in rows:
            len_a, len_b = len(record.response_a), len(record.response_b)
            if len_a == len_b:
                continue
            comparable += 1
            winner_is_a = record.label == "A"
            if (len_a > len_b) == winner_is_a:
                longer_wins += 1
        out[source] = {
            "n": len(rows),
            "comparable": comparable,
            "p_longer_wins": round(longer_wins / comparable, 4) if comparable else None,
        }
    return out


def agreement_histogram(records: Sequence[Record], bins: int = 5) -> Dict[str, int]:
    histogram: Dict[str, int] = Counter()
    for record in records:
        if record.agreement is None:
            histogram["n/a (single annotation)"] += 1
            continue
        index = min(int(record.agreement * bins), bins - 1)
        low, high = index / bins, (index + 1) / bins
        histogram[f"{low:.1f}-{high:.1f}"] += 1
    return dict(sorted(histogram.items()))


def meta_distribution(records: Sequence[Record], key: str,
                      top: int = 12) -> List[tuple]:
    counter = Counter(str(r.meta.get(key)) for r in records if r.meta.get(key) is not None)
    return counter.most_common(top)


def resolution_summary(records: Sequence[Record]) -> Dict[str, object]:
    pixels = []
    for record in records:
        res = record.meta.get("img_res")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            pixels.append(int(res[0]) * int(res[1]))
    if not pixels:
        return {}
    pixels.sort()
    return {
        "n": len(pixels),
        "min_megapixels": round(pixels[0] / 1e6, 3),
        "median_megapixels": round(pixels[len(pixels) // 2] / 1e6, 3),
        "max_megapixels": round(pixels[-1] / 1e6, 3),
    }


def swap_rate(records: Sequence[Record]) -> Dict[str, object]:
    by_source: Dict[str, List[bool]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record.swapped)
    return {source: {"n": len(flags), "swapped": sum(flags),
                     "rate": round(sum(flags) / len(flags), 4) if flags else None}
            for source, flags in by_source.items()}


# --------------------------------------------------------------------------
# manifest + report
# --------------------------------------------------------------------------

def build_manifest(funnel: Funnel,
                   pre_randomization: Sequence[Record],
                   splits: Dict[str, List[Record]],
                   ties: Sequence[Record],
                   fewshot_pools: Dict[str, Sequence[Dict]],
                   media_summary: Dict,
                   download_logs: Dict[str, Dict],
                   tau_sweep: Optional[List[Dict]] = None,
                   config: Optional[Dict] = None) -> Dict:
    main = [r for split in splits.values() for r in split]

    return {
        "config": config or {},
        "downloads": download_logs,
        "media": media_summary,
        "funnel": funnel.to_dict(),
        "counts": {
            "main_total": len(main),
            "ties": len(ties),
            **{name: len(rows) for name, rows in splits.items()},
        },
        "label_balance_before_randomization": label_balance(pre_randomization),
        "label_balance_after_randomization": label_balance(main),
        "position_swap_rate": swap_rate(main),
        "length_bias": length_bias(main),
        "agreement_histogram": agreement_histogram(main),
        "tie_types": dict(Counter(
            f"{r.source}: {r.meta.get('tie_type') or 'unspecified'}" for r in ties)),
        "annotator_capacity": {
            "all": annotator_capacity(main),
            "fewshot_split": annotator_capacity(splits.get("fewshot", [])),
        },
        "fewshot_pools": {
            name: [{"annotator": p["annotator"], "k": len(p["records"])} for p in pools]
            for name, pools in fewshot_pools.items()
        },
        "language_distribution": meta_distribution(main, "lang"),
        "category_distribution": meta_distribution(main, "category"),
        "resolution": resolution_summary(main),
        "over_context_budget": sum(1 for r in main if r.meta.get("over_context_budget")),
        "mm_rlhf_tau_sweep": tau_sweep or [],
        "has_reference_slice": sum(1 for r in main if r.meta.get("has_reference")),
    }


def render_bias_tables(manifest: Dict, funnel: Funnel) -> str:
    sources = funnel.sources()
    counts = manifest["counts"]
    parts: List[str] = []

    parts.append("# Dataset bias and funnel tables\n")
    parts.append("Generated by `python -m prepare_data.run`. Every number below comes "
                 "from the records the pipeline actually wrote.\n")

    # -- provenance -------------------------------------------------------
    parts.append("\n## 0. Sources\n")
    rows = []
    for source, log in manifest["downloads"].items():
        spec = SOURCES.get(source, {})
        size_mb = round(log.get("bytes", 0) / 1e6, 1)
        rows.append([source, spec.get("repo") or spec.get("url", "")[:48],
                     log.get("status", "?"), f"{size_mb} MB",
                     f"{log.get('seconds', 0):.0f}s"])
    parts.append(_table(["Source", "Location", "Status", "Downloaded", "Time"], rows))

    # -- funnel -----------------------------------------------------------
    parts.append("\n## 1. Funnel: rows surviving each stage\n")
    parts.append(funnel.markdown_table(sources))
    parts.append("\n### Rows dropped, by reason\n")
    parts.append(funnel.markdown_drops(sources))

    # -- label balance ----------------------------------------------------
    parts.append("\n## 2. Label balance, before and after position randomization\n")
    parts.append(
        "The old `prepare_vlm_dataset.py` wrote the preferred response into slot A for "
        "every row, so the label was the constant `1` and a judge answering \"A\" every "
        "time would have scored 100%. The 'before' column below shows that skew as the "
        "adapters inherit it; the 'after' column shows it removed.\n")
    before = manifest["label_balance_before_randomization"]
    after = manifest["label_balance_after_randomization"]
    rows = []
    for source in sources:
        b = before.get(source, {})
        a = after.get(source, {})
        b_total, a_total = sum(b.values()), sum(a.values())
        rows.append([
            source,
            f"{b.get('A', 0)} / {b.get('B', 0)}",
            _pct(b.get("A", 0), b_total),
            f"{a.get('A', 0)} / {a.get('B', 0)}",
            _pct(a.get("A", 0), a_total),
            manifest["position_swap_rate"].get(source, {}).get("rate", "-"),
        ])
    parts.append(_table(
        ["Source", "Before A/B", "Before %A", "After A/B", "After %A", "Swap rate"], rows))

    # -- length bias ------------------------------------------------------
    parts.append("\n## 3. Length bias: P(longer response is preferred)\n")
    parts.append("0.5 means length carries no signal. Values far from 0.5 mean a judge "
                 "can score well by preferring length alone.\n")
    rows = [[source, d["n"], d["comparable"], d["p_longer_wins"]]
            for source, d in manifest["length_bias"].items()]
    parts.append(_table(["Source", "Records", "Comparable", "P(longer wins)"], rows))

    # -- annotators -------------------------------------------------------
    parts.append("\n## 4. Annotators usable for Simulated Annotators (Ind.)\n")
    parts.append("An annotator needs at least K+1 labels: K go into the prompt and at "
                 "least one must be held out.\n")
    capacity = manifest["annotator_capacity"]
    rows = [[f"K={k}", capacity["all"].get(k, 0), capacity["fewshot_split"].get(k, 0)]
            for k in sorted(set(K_VALUES) | {3, 5})]
    parts.append(_table(["Requirement", "Annotators (all splits)",
                         "Annotators (fewshot split)"], rows))

    # -- agreement and ties -----------------------------------------------
    parts.append("\n## 5. Inter-annotator agreement and ties\n")
    rows = [[bucket, n] for bucket, n in manifest["agreement_histogram"].items()]
    parts.append(_table(["Agreement with majority", "Records"], rows))
    parts.append(f"\nTied instances routed out of the main protocol: "
                 f"**{counts['ties']}**.\n")
    if manifest["tie_types"]:
        parts.append(_table(["Tie type", "Count"],
                            [[k, v] for k, v in manifest["tie_types"].items()]))

    # -- splits -----------------------------------------------------------
    parts.append("\n## 6. Splits\n")
    parts.append("Splits are drawn over connected components of (group id, image hash), "
                 "so no image and no prompt group crosses a boundary. No single "
                 "calibration/test pair is written: the evaluation draws many random "
                 "splits from `eval_pool`.\n")
    rows = [[name, counts.get(name, 0)] for name in ("fewshot", "dev", "eval_pool")]
    rows.append(["ties (separate)", counts["ties"]])
    parts.append(_table(["Split", "Records"], rows))

    # -- multimodal metadata ----------------------------------------------
    parts.append("\n## 7. Multimodal metadata for the bias ablations\n")
    if manifest["resolution"]:
        res = manifest["resolution"]
        parts.append(f"Image resolution over {res['n']} records: min "
                     f"{res['min_megapixels']} MP, median {res['median_megapixels']} MP, "
                     f"max {res['max_megapixels']} MP.\n")
    if manifest["language_distribution"]:
        parts.append("\n**Languages**\n")
        parts.append(_table(["Language", "Records"],
                            manifest["language_distribution"]))
    if manifest["category_distribution"]:
        parts.append("\n**Categories**\n")
        parts.append(_table(["Category", "Records"],
                            manifest["category_distribution"]))
    parts.append(f"\nPairs involving the VisIT-Bench human-verified reference: "
                 f"**{manifest['has_reference_slice']}** (reportable as its own slice, "
                 "since the strongest model beats the reference only ~27% of the time).\n")
    parts.append(f"\nPairs flagged as over the judge context budget at K={max(K_VALUES)}: "
                 f"**{manifest['over_context_budget']}** (flagged, not dropped).\n")

    # -- tau --------------------------------------------------------------
    if manifest["mm_rlhf_tau_sweep"]:
        parts.append("\n## 8. MM-RLHF difficulty vs the score-gap threshold tau\n")
        parts.append("tau decides how easy the constructed pairs are, so it is a "
                     "selection-bias knob and is reported rather than tuned silently.\n")
        rows = [[d["tau"], d["prompts_examined"], d["prompts_with_pair"],
                 d["fraction"], d["mean_min_gap"]]
                for d in manifest["mm_rlhf_tau_sweep"]]
        parts.append(_table(["tau", "Prompts examined", "With a qualifying pair",
                             "Fraction", "Mean minimum gap"], rows))

    # -- bottlenecks ------------------------------------------------------
    parts.append("\n## 9. Bottlenecks observed\n")
    notes = funnel.to_dict().get("notes", [])
    blocked = [s for s, log in manifest["downloads"].items() if log.get("status") != "ok"]
    if blocked:
        for source in blocked:
            log = manifest["downloads"][source]
            parts.append(f"- **{source}**: not downloaded ({log.get('reason')}). "
                         f"{log.get('detail', '')}\n")
    media = manifest.get("media", {})
    if media.get("failed"):
        parts.append(f"- **Media fetch failures**: {media['failed']} images could not be "
                     "retrieved or decoded (see `media_failures.json`).\n")
    for note in notes:
        parts.append(f"- {note}\n")

    return "".join(parts)


def write_reports(manifest: Dict, funnel: Funnel,
                  out_dir: Path = REPORT_ROOT,
                  manifest_path: Optional[Path] = None) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = out_dir / "bias_tables.md"
    tables.write_text(render_bias_tables(manifest, funnel), encoding="utf-8")

    manifest_path = manifest_path or (out_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {"bias_tables": tables, "manifest": manifest_path}
