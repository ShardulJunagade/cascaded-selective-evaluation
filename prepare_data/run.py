"""Phase 2-5 driver: raw sources -> canonical records -> repo-format splits.

    python -m prepare_data.run --source all
    python -m prepare_data.run --source rlhf_v visit_bench --limit 1500
    python -m prepare_data.run --source all --k 2 --n 5 --legacy-split 500

Sources whose raw download is missing are skipped with a message rather than
failing the run, so a gated source does not block the other four.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from prepare_data import stages, stats
from prepare_data.adapters import ADAPTERS
from prepare_data.config import (ALL_SOURCES, EXPORT_ROOT, K_MAX, MM_RLHF_TAU,
                                 MM_RLHF_TAU_SWEEP, N_MAX, OUT_ROOT, REPORT_ROOT,
                                 SEED, ensure_dirs, use_utf8_stdout)
from prepare_data.download import read_log
from prepare_data.export import export_all
from prepare_data.funnel import Funnel
from prepare_data.media import MediaStore
from prepare_data.schema import Record, ValidationError, validate


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=["all"],
                        help=f"one or more of {ALL_SOURCES}, or 'all'")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows per source (random subset, not a prefix)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--k", type=int, default=K_MAX,
                        help="few-shot examples per simulated annotator")
    parser.add_argument("--n", type=int, default=N_MAX,
                        help="number of simulated annotators")
    parser.add_argument("--tau", type=float, default=MM_RLHF_TAU,
                        help="MM-RLHF score-gap threshold for pair construction")
    parser.add_argument("--modalities", nargs="+", default=["image"],
                        help="modalities to keep from JudgeAnything (image/video/audio)")
    parser.add_argument("--media-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--exact-tokens", action="store_true",
                        help="count the context budget with the judge's own tokenizer "
                             "instead of ~4 chars/token (downloads the processor)")
    parser.add_argument("--legacy-split", type=int, default=None,
                        help="also write one calibration/test pair of this calibration "
                             "size, for a smoke run against the existing scripts")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_ROOT)
    return parser.parse_args()


def write_canonical(records: Sequence[Record], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return len(records)


def adapt_sources(sources: Sequence[str], store: MediaStore, funnel: Funnel,
                  args) -> List[Record]:
    """Phase 2: run each adapter, validating every record it emits."""
    records: List[Record] = []

    for name in sources:
        adapter_cls = ADAPTERS[name]
        kwargs = {}
        if name == "mm_rlhf":
            kwargs["tau"] = args.tau
        if name == "judge_anything":
            kwargs["modalities"] = tuple(args.modalities)

        adapter = adapter_cls(store, funnel, limit=args.limit, seed=args.seed, **kwargs)
        if not adapter.available():
            print(f"[{name}] skipped: {adapter.skip_reason()}")
            funnel.note(f"{name}: skipped, raw data absent.")
            continue

        print(f"[{name}] adapting ...")
        n_before = len(records)
        for record in adapter.iter_records():
            try:
                validate(record)
            except ValidationError as exc:
                funnel.drop("adapted", name, "schema validation failed")
                print(f"  ! {exc}")
                continue
            records.append(record)
        print(f"[{name}] -> {len(records) - n_before} records")

    return records


def build_fewshot_pools(fewshot_records: Sequence[Record], k: int, n: int,
                        seed: int, funnel: Funnel) -> Dict[str, List[Dict]]:
    pools: Dict[str, List[Dict]] = {}

    ind = stages.individual_pools(fewshot_records, k=k, n=n, seed=seed)
    if len(ind) < n:
        funnel.note(
            f"fewshot: only {len(ind)} annotators have >= K+1={k + 1} labels in the "
            f"fewshot split, so Simulated Annotators (Ind.) runs with N={len(ind)} "
            f"rather than the requested N={n}. This is the annotator-supply bottleneck."
        )
    if ind:
        pools["ind"] = ind

    try:
        pools["maj"] = stages.majority_pools(fewshot_records, k=k, n=n, seed=seed)
    except ValueError as exc:
        funnel.note(f"fewshot: {exc}")

    return pools


def main() -> Dict:
    use_utf8_stdout()
    args = parse_args()
    ensure_dirs()

    sources = ALL_SOURCES if "all" in args.source else list(args.source)
    unknown = [s for s in sources if s not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown source(s) {unknown}. Known: {ALL_SOURCES}")

    funnel = Funnel()
    store = MediaStore(image_format=args.media_format)

    # -- Phase 2: adapters ------------------------------------------------
    records = adapt_sources(sources, store, funnel, args)
    # Fetching is by far the most expensive step, so bank the index now rather
    # than only at the end; a later crash should not cost the downloads.
    store.save_index()
    if not records:
        raise SystemExit(
            "No records produced. Run `python -m prepare_data.download --source all` "
            "first, and check data/raw/*/download_log.json for what was blocked.")

    # -- Phase 3: shared stages -------------------------------------------
    tokenizer = None
    if args.exact_tokens:
        tokenizer = stages.load_judge_tokenizer()
        if tokenizer is None:
            print("  ! judge tokenizer unavailable; falling back to the character estimate")

    print(f"\nhygiene      {len(records)} in ...")
    records = stages.hygiene(records, funnel, tokenizer=tokenizer)
    print(f"dedup        {len(records)} in ...")
    records = stages.dedup(records, funnel)
    print(f"tie routing  {len(records)} in ...")
    records, ties = stages.route_ties(records, funnel)

    # Snapshot the label balance before it is broken, for the bias table.
    pre_randomization = [Record.from_dict(r.to_dict()) for r in records]

    print(f"randomizing  {len(records)} positions ...")
    records = stages.randomize_positions(records, seed=args.seed, funnel=funnel)

    print(f"splitting    {len(records)} records over leakage components ...")
    splits = stages.split_groups(records, seed=args.seed, funnel=funnel, k=args.k)
    for name, rows in splits.items():
        print(f"  {name:<10} {len(rows)}")

    fewshot_pools = build_fewshot_pools(splits.get("fewshot", []), k=args.k, n=args.n,
                                        seed=args.seed, funnel=funnel)

    # -- write canonical records ------------------------------------------
    out_dir = Path(args.out_dir)
    by_source: Dict[str, Dict[str, List[Record]]] = {}
    for split_name, rows in list(splits.items()) + [("ties", ties)]:
        for record in rows:
            by_source.setdefault(record.source, {}).setdefault(split_name, []).append(record)
    for source, split_map in by_source.items():
        for split_name, rows in split_map.items():
            write_canonical(rows, out_dir / source / f"{split_name}.jsonl")

    # -- Phase 5: export --------------------------------------------------
    written = export_all(splits, ties, fewshot_pools, out_dir=Path(args.export_dir),
                         legacy_split=args.legacy_split, seed=args.seed)

    # -- Phase 4: stats ---------------------------------------------------
    tau_sweep: List[Dict] = []
    if "mm_rlhf" in sources:
        adapter = ADAPTERS["mm_rlhf"](store, Funnel(), limit=args.limit, seed=args.seed,
                                      tau=args.tau)
        if adapter.available():
            tau_sweep = adapter.tau_sweep(MM_RLHF_TAU_SWEEP)

    store.save_index()
    store.write_failure_log(REPORT_ROOT / "media_failures.json")
    download_logs = {s: (read_log(s) or {"status": "not attempted", "bytes": 0})
                     for s in sources}

    manifest = stats.build_manifest(
        funnel=funnel,
        pre_randomization=pre_randomization,
        splits=splits,
        ties=ties,
        fewshot_pools=fewshot_pools,
        media_summary=store.summary(),
        download_logs=download_logs,
        tau_sweep=tau_sweep,
        config={
            "sources": sources, "limit": args.limit, "seed": args.seed,
            "k": args.k, "n": args.n, "tau": args.tau,
            "modalities": args.modalities, "media_format": args.media_format,
        },
    )
    paths = stats.write_reports(manifest, funnel, out_dir=REPORT_ROOT,
                                manifest_path=out_dir / "manifest.json")
    funnel.save(out_dir / "funnel.json")

    # -- summary ----------------------------------------------------------
    print("\nExported:")
    for filename, count in sorted(written.items()):
        print(f"  {filename:<28} {count}")
    print(f"\nManifest:    {paths['manifest']}")
    print(f"Bias tables: {paths['bias_tables']}")
    print(f"Media:       {json.dumps(store.summary())}")
    return manifest


if __name__ == "__main__":
    main()
