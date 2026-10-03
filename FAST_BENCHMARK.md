# Fast VLM benchmark

`run_fast_benchmark.py` is the time-focused profile for one GPU. It derives a
deterministic sample of at most 1,000 rows from the balanced-small export, selecting
complete prompt/image leakage components and approximately preserving source shares.

The scoring protocol uses three simulated annotators, one demonstration per annotator,
and both response orderings. This is 6,000 judge requests rather than 27,740 per judge
in the balanced-small N=5, K=2 profile. Keeping both orderings retains the position-bias
control used by the main method.

Run inside an existing GPU allocation:

```bash
python -u run_fast_benchmark.py
```

The runner reuses `data/vlm_v2_small_export`, model downloads, and compilation caches.
It writes separate judge caches to `result/vlm_v2_fast_1k_n3_k1` and separate results
under `outputs/vlm_v2_fast_{3judge,4judge}`. Interrupted scoring resumes by sample ID.
