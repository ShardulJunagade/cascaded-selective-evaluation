#!/usr/bin/env python3
"""Write every finished evaluation in outputs/ as Markdown tables.

    python scripts/api_round/tables_md.py                 # -> outputs/api_round_tables.md
    python scripts/api_round/tables_md.py --suffix _quick # the --quick runs instead
    python scripts/api_round/tables_md.py --out results.md

Reads outputs/<run>/baseline_comparison.json and global_router_comparison.json for every
run folder that has them (this round's runs first, then any others). Safe to run while
experiments are still going: unfinished runs are simply missing.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KNOWN_RUNS = [
    ("vlm_v2_fast_4judge", "E0: 4 open judges"),
    ("fast_api_kimi", "E1: Kimi K2.5 alone"),
    ("fast_api_gemini", "E2: Gemini Flash alone"),
    ("fast_api_kimi_gemini", "E3: Kimi → Gemini"),
    ("fast_api_gemini_kimi", "E3b: Gemini → Kimi"),
    ("fast_mixed_6judge", "E4: 4 open + Kimi + Gemini"),
    ("fast_mixed_5judge_kimi", "E5: 4 open + Kimi"),
]
FILES = ("baseline_comparison.json", "global_router_comparison.json")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def mean(summary: dict, key: str):
    value = summary.get(key)
    return value.get("mean") if isinstance(value, dict) else value


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def kept(summary: dict):
    """Guarantee held: answered and accurate enough, or abstained (nothing answered)."""
    success, nonempty = summary.get("guarantee_success_rate"), summary.get("nonempty_split_rate")
    return None if success is None or nonempty is None else success + (1 - nonempty)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def baseline_rows(policies: dict) -> list[list[str]]:
    rows = []
    for name, s in policies.items():
        label = f"**{name}**" if name == "cascaded_selective" else f"`{name}`"
        rows.append([label, pct(mean(s, "agreement")), pct(mean(s, "macro_source_agreement")),
                     pct(mean(s, "coverage")), pct(s.get("nonempty_split_rate")),
                     pct(s.get("guarantee_success_rate")), pct(kept(s))])
    return rows


def router_rows(methods: dict) -> list[list[str]]:
    rows = []
    for name, s in methods.items():
        calls = mean(s, "mean_judges_called")
        rows.append([f"**{name}**", pct(mean(s, "agreement")),
                     pct(mean(s, "macro_source_agreement")), pct(mean(s, "coverage")),
                     "n/a" if calls is None else f"{calls:.2f}",
                     pct(s.get("nonempty_split_rate")), pct(s.get("certified_split_rate")),
                     pct(s.get("guarantee_success_rate")), pct(kept(s))])
    return rows


def find_runs(suffix: str) -> list[tuple[str, str]]:
    outputs = ROOT / "outputs"
    runs = [(name + suffix, label) for name, label in KNOWN_RUNS
            if any((outputs / f"{name}{suffix}" / f).is_file() for f in FILES)]
    known = {name for name, _ in runs}
    if outputs.is_dir():
        for folder in sorted(p for p in outputs.iterdir() if p.is_dir()):
            if folder.name not in known and folder.name.endswith(suffix) \
                    and any((folder / f).is_file() for f in FILES):
                runs.append((folder.name, folder.name))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suffix", default="", help="run-name suffix, e.g. _quick")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "api_round_tables.md")
    args = parser.parse_args()

    runs = find_runs(args.suffix)
    if not runs:
        raise SystemExit("No finished evaluations found in outputs/.")
    data = {name: [load(ROOT / "outputs" / name / f) for f in FILES] for name, _ in runs}
    first = next(d for pair in data.values() for d in pair if d)
    alpha, delta = first.get("alpha", 0.15), first.get("delta", 0.1)

    lines = [
        "# Results tables",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `outputs/`. "
        f"Target: at least {100 * (1 - alpha):.0f}% agreement with the human label on the "
        f"pairs a method answers, with probability at least {100 * (1 - delta):.0f}% "
        f"(α = {alpha}, δ = {delta}). Every number is a percentage averaged over "
        f"{first.get('n_splits', 'the')} random calibration/test splits.",
        "",
        "- **Accuracy**: agreement with the human label on the answered pairs, averaged "
        "over splits that answered anything. **Macro** averages the five data sources "
        "equally.",
        "- **Coverage**: share of pairs answered. Splits that answered nothing count as 0.",
        "- **Answered**: share of splits that answered at least one pair.",
        "- **Success**: share of splits that answered something *and* met the accuracy "
        "target. Splits that abstained on everything count as failures here.",
        "- **Kept**: share of splits where the guarantee held = Success + (100 − "
        f"Answered). Abstaining is safe, so this is the number to compare with "
        f"{100 * (1 - delta):.0f}%.",
        "- **Calls**: mean number of judges queried per pair. **Certified**: share of splits "
        "where a routing threshold was certified.",
        "",
        "## Overview",
        "",
    ]

    overview = []
    for name, label in runs:
        base, router = data[name]
        policies = (base or {}).get("policies", {})
        methods = (router or {}).get("methods", {})
        cells = [label]
        for s in (policies.get("cascaded_selective"), methods.get("global_raw_confidence"),
                  methods.get("global_learned_router")):
            cells += ([pct(mean(s, "coverage")), pct(mean(s, "agreement")), pct(kept(s))]
                      if s else ["–", "–", "–"])
        overview.append(cells)
    lines += table(["Run",
                    "Trust or Escalate: cover", "acc", "kept",
                    "Router (raw): cover", "acc", "kept",
                    "Router (learned): cover", "acc", "kept"], overview)
    lines += ["", "Trust or Escalate is `cascaded_selective` (one threshold per judge, δ split "
              "across judges). The routers are the end-to-end method (one threshold for the "
              "whole cascade, certified at the full δ).", ""]

    for name, label in runs:
        base, router = data[name]
        lines += [f"## {label}", "", f"`outputs/{name}/`", ""]
        if base:
            lines += ["**Trust or Escalate and baselines** (`baseline_comparison.json`)", ""]
            lines += table(["Policy", "Accuracy", "Macro", "Coverage", "Answered", "Success",
                            "Kept"], baseline_rows(base.get("policies", {})))
            lines.append("")
        if router:
            lines += ["**End-to-end router** (`global_router_comparison.json`)", ""]
            lines += table(["Method", "Accuracy", "Macro", "Coverage", "Calls", "Answered",
                            "Certified", "Success", "Kept"], router_rows(router.get("methods", {})))
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(runs)} runs: {', '.join(name for name, _ in runs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
