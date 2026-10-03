# Balanced small benchmark

`run_small_benchmark.py` is the practical cluster profile. It draws a deterministic,
seeded streaming sample of at most 1,000 raw rows from each source before adaptation.
JudgeAnything has fewer applicable image prompts and is retained in full. Streaming stops
at the cap and does not materialize every Hugging Face parquet shard into Arrow training
batches.

The design retains all five data sources, group-safe splitting, position randomization,
five simulated annotators with two examples each, a 500-item calibration set, 1,000
repeated calibration/test draws, all individual-model and judge-setup baselines, and both
the three-family and four-family cascades. Macro source metrics prevent larger sources from
dominating the comparison.

The resulting evaluation pool is expected to contain roughly 2,500--3,000 pairs after
source-specific filtering and the 80/10/10 split. For an evaluation set of 2,000 independent
pairs, the worst-case 95% binomial margin of error for overall accuracy is about 2.2
percentage points. Per-source estimates are less precise and must be reported with their
sample counts. Repeated splits measure calibration-procedure stability; they do not create
additional independent examples.

Run inside an existing GPU allocation:

```bash
python -u run_small_benchmark.py
```

If an older unpinned install reports a missing `PixtralRotaryEmbedding`, repair the
judge runtime once and rerun the same command. The completed corpus is reused:

```bash
python -m pip install --no-cache-dir --upgrade "transformers==5.5.3"
python -m pip check
python -u run_small_benchmark.py
```

Judge caches are written to `result/vlm_v2_balanced_1k`, separate from the full-corpus
caches. Results are written under `outputs/vlm_v2_small_{3judge,4judge}`.
