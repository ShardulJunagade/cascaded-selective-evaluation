# Dataset handoff

What the corpus is, where it lives, and how to use it.

For *why* it looks this way — bias analysis, filtering rationale, bottlenecks,
numbers for the report — see **[DATASET_FINDINGS.md](DATASET_FINDINGS.md)**.

**Status:** complete and validated. No judge has been run on it yet.

---

## 1. What you have

**31,635 labelled instances + 9,696 ties**, image-grounded, from five sources.

```
data/vlm_v2_export/
  eval_pool.jsonl        25,308   draw calibration/test splits from THIS
  dev.jsonl               3,163   hyperparameter selection only (K, N, tau)
  ties.jsonl              9,696   unlabelled; for the confidence -> 0.5 test
  fewshot.maj.jsonl           5   Simulated Annotators (Maj.), K=2
  fewshot.ind.jsonl           5   Simulated Annotators (Ind.), K=2
  calibration.jsonl         500 } smoke-test pair only, see section 4
  test.jsonl             24,808 }
```

Per source, in the main protocol:

| Source | Instances | Annotator ids | Notes |
|---|---|---|---|
| VisionArena-Battle | 15,964 | **yes** | only source that can feed Ind. pools |
| MM-RLHF | 8,650 | no | pairs built from rankings at τ=1.0 |
| RLHF-V | 5,716 | no | **pilot only** — human-edit vs model output |
| VisIT-Bench | 999 | no | includes a human-reference slice |
| JudgeAnything | 306 | no | Image2Text only; any-to-any models, weaker outputs |

Supporting files: `data/vlm_v2/manifest.json` (every number),
`reports/bias_tables.md`, `reports/raw_inspection.md`.

Images live in `data/vlm_v2/media/` (32,142 files, content-addressed). They are
**not** in git — `image_path` in each record is repo-relative, so run the judge
from the repo root.

---

## 2. Run a judge on it

```bash
pip install -r requirements-open.txt

python run_open_vlm_judge.py --model_name=qwen2.5-vl-3b-instruct \
  --in_filename=./data/vlm_v2_export/eval_pool.jsonl \
  --fewshot_in_filename=./data/vlm_v2_export/fewshot.maj.jsonl --resume

python run_open_vlm_judge.py --model_name=qwen2.5-vl-7b-instruct \
  --in_filename=./data/vlm_v2_export/eval_pool.jsonl \
  --fewshot_in_filename=./data/vlm_v2_export/fewshot.maj.jsonl --resume
```

Swap in `fewshot.ind.jsonl` for the Ind. variant of Simulated Annotators.

---

## 3. Record format

Unchanged from what the judge scripts already read:

```json
{"image_path": "data/vlm_v2/media/ab/<sha256>.jpg",
 "instruction": "...",
 "outputs": ["response A", "response B"],
 "preferences": {"arena_user_xxx": 1},
 "source": {"uid", "dataset", "group_id", "label", "label_origin",
            "swapped", "n_ann", "agreement", "is_tie",
            "meta": {"len_a", "len_b", "img_res", "lang", "category",
                     "score_gap", "has_reference", ...}}}
```

- `1` = `outputs[0]` preferred, `2` = `outputs[1]`.
- Keys are **real annotator ids** where a source has them. Use
  `open_cascade.data.preferred_index(sample)` to read the label, not
  `preferences["human"]` — that key only exists on sources without ids.
- `source.meta` carries everything needed to slice results (length, resolution,
  language, category, score gap, reference flag).
- **Ties have empty `preferences`** and `source.is_tie = true`. Score them for
  confidence only; they carry no forced choice.

---

## 4. Splits — read before writing the evaluation

**There is deliberately no canonical calibration/test split.** A single split
cannot show whether the (1−δ) guarantee holds, and that is exactly what the
Guarantee Success Rate measures. The evaluation must draw **many random
calibration/test splits from `eval_pool.jsonl`**.

`calibration.jsonl` / `test.jsonl` exist only so the current
`run_open_cascade.py` runs unchanged for a smoke test. Rebuild with
`--legacy-split 0` to omit them.

> **This is the main remaining code task:** `run_open_cascade.py` reads that
> one fixed pair and needs to resample instead.

Other rules:

- `dev.jsonl` selects K, N and τ. Never calibrate or test on it.
- Splits are drawn over connected components of (group id, perceptual hash), so
  no image or prompt group crosses a boundary. `assert_no_leakage` enforces it.
- `fewshot` is prompt material only and is never evaluated on.

---

## 5. Rebuild or re-tune

```bash
python get_data.py                             # download + build + verify
python get_data.py --skip-download             # rebuild from data/raw/ (minutes)
python get_data.py --tau 0.5 --skip-download   # more MM-RLHF pairs, easier ones
python get_data.py --limit 500                 # small slice for a smoke test
python get_data.py --legacy-split 0            # no fixed cal/test pair
```

Knobs: `--k` (few-shot examples, 1–2), `--n` (simulated annotators), `--tau`
(MM-RLHF difficulty: 0.5 → ~11,062 pairs, 1.0 → ~8,785, 1.5 → ~4,083),
`--source`, `--limit`.

Checks:

```bash
python -m prepare_data.verify_export   # run before spending GPU time
pytest tests/test_prepare_data.py -q   # 51 tests, no network or GPU
```

`verify_export` catches unopenable image paths, labels disagreeing with
`preferences`, near-constant labels, few-shot leaking into `eval_pool`, and
`sample_key` collisions that would make `merge_data` silently drop rows.

**Downloading:** ~43 GB. VisionArena is gated — accept the terms on its
HuggingFace page, then `hf auth login` (browser device login is enough).
To resume an interrupted download just re-run; **do not pass `--force`**, which
starts a new partial and abandons the one on disk. On a cluster set `HF_HOME`
to scratch first.

---

## 6. Next tasks

1. **Make `run_open_cascade.py` resample** calibration/test from `eval_pool`.
   Highest priority; the Guarantee Success Rate depends on it.
2. **Re-run the 3B→7B cascade** and report **per-source** agreement, not a
   pooled number. The old 85.95% / 29.51% should not be carried forward
   (DATASET_FINDINGS.md §1).
3. **Length-matched ablation.** P(longer preferred) is 0.63–0.72 on four of
   five sources, so a length heuristic alone scores in that range
   (DATASET_FINDINGS.md §3.2). Slice on `meta.len_a` / `meta.len_b`.
4. Optional: speed up re-runs by casting VisionArena's `images` column to
   `decode=False`, so cached keys short-circuit before the JPEG is decoded.
5. Optional: filter the one near-1×1 image (`min 0.0 MP` in the resolution
   summary).

---

## 7. Layout and cost

```
data/raw/<source>/          raw downloads + download_log.json   (43 GB, gitignored)
data/vlm_v2/
  media/<ab>/<sha256>.jpg   content-addressed store              (~4 GB, gitignored)
  media_index.json          lets re-runs skip refetching
  <source>/*.jsonl          canonical records per source
  manifest.json             every number in one place
data/vlm_v2_export/         the files you actually run on
reports/                    bias_tables.md, raw_inspection.md
prepare_data/               the pipeline (see prepare_data/README.md)
```

| | |
|---|---|
| Raw downloads | 43 GB |
| Media store | ~4 GB, 32,086 images |
| Full rebuild | ~15 min (VisionArena image decode dominates) |
| Download time | VisionArena ~3 h, MM-RLHF ~3 h |

All CPU-only.
