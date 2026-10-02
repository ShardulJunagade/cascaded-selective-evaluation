# `prepare_data` — pipeline internals

How the corpus is built. For the corpus itself and how to use it see
[DATASET_HANDOFF.md](../DATASET_HANDOFF.md); for the analysis behind the design
choices see [DATASET_FINDINGS.md](../DATASET_FINDINGS.md).

Normal use is `python get_data.py` from the repo root. The modules below are
the pieces it drives.

## Modules

| Module | Role |
|---|---|
| `config.py` | paths, source registry, constants |
| `download.py` | fetch raw sources; log bytes, time and failures |
| `inspect_raw.py` | statistics of each source *as published* |
| `schema.py` | the canonical `Record` and its validator |
| `media.py` | content-addressed image store + persisted index |
| `adapters/` | one per source: raw rows → canonical records |
| `stages.py` | hygiene, dedup, tie routing, randomization, split |
| `stats.py` | `manifest.json` and `reports/bias_tables.md` |
| `export.py` | canonical records → the format the judge scripts read |
| `verify_export.py` | pre-flight checks on the written splits |
| `run.py` | CLI driving adapt → stages → export → stats |

Run individually if you need to:

```bash
python -m prepare_data.download --source all
python -m prepare_data.inspect_raw --source all
python -m prepare_data.run --source all --k 2 --n 5
python -m prepare_data.verify_export
```

## Canonical record

```python
{
  "uid", "source", "group_id",            # group_id = leakage unit for splits
  "modality": "image" | "video" | "audio",
  "media": [path], "media_hash": [phash],
  "instruction", "context",
  "response_a", "response_b", "model_a", "model_b",
  "label": "A" | "B" | None,              # names the CURRENT slot contents
  "swapped": bool,
  "annotations": [{"annotator", "label"}],
  "n_ann", "agreement", "is_tie",
  "label_origin": "vote" | "majority" | "ranking_derived" | "correction",
  "meta": {"category", "lang", "score_gap", "len_a", "len_b", "img_res", ...}
}
```

`label` always names the current contents of `response_a` / `response_b`. The
randomization stage swaps the responses and flips the label together, so
downstream code never consults `swapped` to read the label correctly.

## Stages, in order

Adapters convert a source and nothing else — no randomizing, dedup or
splitting — so every source gets identical treatment and the funnel compares
like with like.

1. **Hygiene** — drop identical, empty and double-refusal pairs. Flag, never
   drop, pairs over the judge context budget.
2. **Cross-source dedup** — two passes: exact instance duplicates anywhere,
   then prompt-level overlap *only across sources*.
3. **Tie routing** — ties leave the main protocol and become the set for the
   confidence→0.5 test.
4. **Position randomization** — seeded by `uid`, so a record's assignment does
   not depend on which sources ran or in what order.
5. **Group-aware split** — over connected components of (group id, image
   hash); `assert_no_leakage` enforces it.
6. **Few-shot pools** — Ind. (one per real annotator with ≥ K+1 labels) and
   Maj. (N disjoint K-sets, stratified across sources).

Steps 2 and 5 and the fewshot construction all encode decisions that are not
obvious from the code alone; DATASET_FINDINGS.md §5 explains why each is
shaped the way it is.

## Media store

Images land in `data/vlm_v2/media/<ab>/<sha256>.jpg`, keyed on a hash of the
**decoded RGB pixels**, so the same picture delivered as PNG by one source and
JPEG by another is stored once — which is what makes cross-source dedup
possible. Nothing is resized on disk; the judge applies `max_pixels` at prompt
time and the original resolution is kept in `meta.img_res`.

`media_index.json` maps source identity (URL, archive path) to the stored
reference, so re-runs skip refetching. The content address is only knowable
*after* fetching, so without it every re-run would re-download every URL.

## Adding a source

1. Add an entry to `SOURCES` in `config.py` (include `allow_patterns` /
   `ignore_patterns` if the repo ships media you will not use).
2. Write `adapters/<name>.py` subclassing `Adapter`, implementing
   `iter_records()`; register it in `adapters/__init__.py`.
3. Set `meta["real_annotator_ids"]` truthfully — it gates the Ind. pools.
4. Add the source to `SOURCE_PRIORITY` for dedup precedence.
5. Add an inspector to `inspect_raw.py` if you want Phase 1 statistics.
