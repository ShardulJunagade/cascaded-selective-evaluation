#!/usr/bin/env python3
"""Fetch the source datasets and build the multimodal preference corpus.

No data is version controlled - the raw sources are ~43 GB - so this script is
how a fresh clone obtains it. It runs the three pipeline phases in order and
stops at the first failure:

    1. download  raw sources from HuggingFace / GitHub
    2. build     adapters, hygiene, dedup, tie routing, position
                 randomization, group-aware split, export
    3. verify    read the written splits back through the repo's own loaders

    python get_data.py                      # everything, with the defaults
    python get_data.py --limit 500          # a small slice, for a smoke test
    python get_data.py --skip-download      # rebuild from already-downloaded data
    python get_data.py --inspect            # also write reports/raw_inspection.md
    python get_data.py --tau 0.5 --skip-download   # more MM-RLHF pairs, easier ones

Before you start
----------------
* `pip install -r requirements-open.txt`
* VisionArena-Battle is gated. Accept the terms at
  https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle while signed
  in, then run `hf auth login`. Browser device login is enough; no read token
  is needed. Without it that source is skipped and the Simulated Annotators
  (Ind.) pools cannot be built - it is the only source with annotator ids.
* On a cluster, point the HF cache at scratch first: `export HF_HOME=/scratch/...`

Expect roughly 43 GB and several hours on a home connection for the full set,
dominated by MM-RLHF (30 GB) and VisionArena (15 GB). Interrupted transfers
resume: just run this again. Do not pass `--force` to resume - see
`prepare_data/download.py`.

See DATASET_HANDOFF.md for what the corpus contains and what the numbers mean.
"""

from __future__ import annotations

import subprocess
import sys
import time
from argparse import ArgumentParser, REMAINDER

SOURCES = ["rlhf_v", "visit_bench", "visionarena", "mm_rlhf", "judge_anything"]


def run(step: str, argv: list) -> None:
    print(f"\n{'=' * 70}\n{step}\n{'=' * 70}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, "-u", *argv])
    if result.returncode != 0:
        raise SystemExit(
            f"\n{step} failed (exit {result.returncode}). Nothing further was run.")
    print(f"\n{step} finished in {time.time() - started:.0f}s", flush=True)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=SOURCES,
                        help=f"sources to fetch and build (default: {' '.join(SOURCES)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap records per source; a seeded random subset, not a prefix")
    parser.add_argument("--k", type=int, default=2, help="few-shot examples per annotator")
    parser.add_argument("--n", type=int, default=5, help="number of simulated annotators")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="MM-RLHF score-gap threshold for building pairs. Lower "
                             "keeps more pairs but makes them easier: tau=0.5 yields "
                             "~11,062 pairs (mean min gap 0.95), tau=1.0 ~8,785 (1.44), "
                             "tau=1.5 ~4,083 (2.12). Reported in reports/bias_tables.md "
                             "section 8, because it is a difficulty-selection knob.")
    parser.add_argument("--skip-download", action="store_true",
                        help="rebuild from data already in data/raw/")
    parser.add_argument("--inspect", action="store_true",
                        help="also write reports/raw_inspection.md (Phase 1 statistics)")
    parser.add_argument("--legacy-split", type=int, default=500,
                        help="also write ONE fixed calibration.jsonl/test.jsonl pair of "
                             "this calibration size for an older single-split smoke test. "
                             "It is not the real "
                             "protocol: a single split cannot show whether the (1-delta) "
                             "guarantee holds, so the evaluation must draw many random "
                             "splits from eval_pool.jsonl. Pass 0 to skip it.")
    parser.add_argument("build_args", nargs=REMAINDER,
                        help="extra arguments passed through to prepare_data.run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = list(args.source)

    if not args.skip_download:
        run("1/3  Downloading raw sources",
            ["-m", "prepare_data.download", "--source", *sources])

    if args.inspect:
        run("Raw inspection (Phase 1 statistics)",
            ["-m", "prepare_data.inspect_raw", "--source", *sources])

    build = ["-m", "prepare_data.run", "--source", *sources,
             "--k", str(args.k), "--n", str(args.n), "--tau", str(args.tau)]
    if args.legacy_split:
        build += ["--legacy-split", str(args.legacy_split)]
    if args.limit is not None:
        build += ["--limit", str(args.limit)]
    build += [a for a in args.build_args if a != "--"]
    run("2/3  Building the corpus", build)

    run("3/3  Verifying the export", ["-m", "prepare_data.verify_export"])

    print("\n" + "=" * 70)
    print("Done. The corpus is in data/vlm_v2_export/:")
    print("  eval_pool.jsonl      draw calibration/test splits from this")
    print("  dev.jsonl            hyperparameter selection only (K, N, tau)")
    print("  ties.jsonl           unlabelled; for the confidence -> 0.5 test")
    print("  fewshot.maj.jsonl    Simulated Annotators (Maj.)")
    print("  fewshot.ind.jsonl    Simulated Annotators (Ind.)")
    print("\nReports: reports/bias_tables.md, data/vlm_v2/manifest.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
