# API judges (Kimi K2.5, Gemini)

How to run hosted models as judges in the cascade, with no GPU needed. They use the same
`score` → `evaluate` pipeline as the local Qwen judges (see [PIPELINE.md](PIPELINE.md))
and write judgements in the same format, so the two kinds can be mixed in one cascade.

---

## 1. Setup

```bash
pip install -r requirements-open.txt   # needs httpx and pillow; vLLM is not required
```

Set API keys as environment variables. They are never read from configs, so keep them
out of git.

| Provider | Variable | Used by |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `kimi-k2.5`, `kimi-k2.5-votes`, `gemini-flash-openrouter` |
| Google AI Studio | `GEMINI_API_KEY` | `gemini-flash` |

```powershell
# PowerShell, current window only
$env:OPENROUTER_API_KEY = "sk-or-..."
echo $env:OPENROUTER_API_KEY         # check it (note the env: prefix)

# persist for new windows (then open a new terminal)
setx OPENROUTER_API_KEY "sk-or-..."
```
```bash
export OPENROUTER_API_KEY=sk-or-...  # bash
```

Run everything from the repo root: `image_path` values in the corpus are repo-relative.

## 2. Registered judges

| Judge | Model | Confidence | Status |
|---|---|---|---|
| `kimi-k2.5` | `moonshotai/kimi-k2.5` via OpenRouter, pinned to Novita | logprobs | **verified** 2026-11-07 |
| `kimi-k2.5-votes` | same, any provider | votes | fallback, untested |
| `gemini-flash` | `gemini-flash-latest`, Google direct | votes | awaiting key |
| `gemini-flash-openrouter` | same model via OpenRouter | votes | awaiting key |

`python -m open_cascade list` shows all judges.

**Confidence modes.** Each sample is judged 2N times (N simulated annotators × both
response orders), then the results are averaged into `probs`.

- **logprobs**: P(A) and P(B) are read from the token probabilities at the verdict.
  This gives graded confidence per call, closest to the local judges. Use it whenever the
  provider returns logprobs.
- **votes**: each call gives a hard A or B, and confidence is the vote share across the
  2N calls (with N=5: 0.5, 0.6, … 1.0). That is coarser, so thresholds are less fine-grained.
  Gemini only supports this mode: it returns no logprobs, either directly or through OpenRouter.

**Why Kimi is pinned to Novita.** Only Novita and Venice return logprobs for Kimi K2.5,
and Venice ignores every way of switching off thinking. It reasons in the answer text and
runs out of tokens before writing the verdict. Novita answers `[[A]]` in 4 tokens. Fallback
is disabled, so a Novita outage stops the run with an error rather than quietly routing to
Venice. If that happens, see §6.

## 3. Smoke test (always do this first)

```bash
python test_api_judge.py --judge kimi-k2.5 --n 3
```

1. **ping**: one tiny text request. Checks the key, the model id, and that the
   answer is a clean `[[A]]` (with logprobs, in logprobs mode).
2. **score**: the first `--n` samples of the calibration split go through the real
   scoring path, printed next to the human label.
3. **report**: request count and failures, tokens, $ cost, and the projected cost per
   1,000 samples and for the full pool.

Reference run (Kimi, N=5, K=1): 30/30 requests ok, about 7 s/sample,
**~$0.008/sample → ~$8 per 1,000 samples, ~$205 for the full 25,308-sample pool.**

Other options:

| Flag | Purpose |
|---|---|
| `--dry-run` | print one full request and exit; no network, no key |
| `--probe` | try several provider / thinking-off settings and report which work (§6) |
| `--k 2` | few-shot examples per annotator (each adds an image to every request) |
| `--annotators 3` | use only the first N annotators (fewer requests per sample) |
| `--fewshot data/vlm_v2_export/fewshot.ind.jsonl` | Simulated Annotators (Ind.) instead of (Maj.) |
| `--split data/vlm_v2_export/dev.jsonl` | draw the samples from a different file |
| `--out sample.jsonl` | save the judgements |

## 4. Full run

```bash
python -m open_cascade score    configs/vlm_api_kimi.yaml
python -m open_cascade evaluate configs/vlm_api_kimi.yaml
```

- `score` is the only step that costs money. It writes
  `result/vlm-api/kimi-k2.5.{calibration,test}.jsonl`, appending after every
  `chunk_size` samples. **If it is interrupted, re-run the same command**: already-scored
  samples are skipped. It prints the total requests, failures, tokens and cost at the end.
- `evaluate` is free and instant. It picks thresholds on the calibration split and reports
  agreement and coverage on test. Re-run it as often as you like, e.g.
  `--set cascade.alpha=0.10`.

When Gemini is available, run the two-judge cascade. Kimi's judgements are reused, so only
Gemini is scored:

```bash
python test_api_judge.py --judge gemini-flash --n 3
python -m open_cascade score    configs/vlm_api_kimi_gemini.yaml
python -m open_cascade evaluate configs/vlm_api_kimi_gemini.yaml
```

With no Google key, change the second judge in that config to `gemini-flash-openrouter`.

### Settings that control cost

In `configs/vlm_api_kimi.yaml`; override any of them with `--set key=value`.

| Setting | Default | Effect |
|---|---|---|
| `scoring.max_samples` | 500 | score only the first n of each split. Cost scales linearly |
| `scoring.max_fewshot_examples` | 1 | K. Each example adds an image and a pair of responses to **every** request |
| `data.fewshot_name` | `fewshot.maj.jsonl` | the few-shot file has 5 annotators → 10 requests per sample |
| `scoring.chunk_size` | 8 | samples per batch; also how often progress is saved |

Example: `--set scoring.max_samples=100` for a cheap first pass.

## 5. Calibration vs test, briefly

The cascade only trusts a judge when its confidence is at least a threshold λ. Otherwise the
sample escalates to the next judge, or the cascade abstains after the last one.

- **calibration** split: λ is chosen per judge so that, on samples above λ, agreement with
  humans is at least 1−α (config: α=0.15 → 85%), with probability at least 1−δ (δ=0.1).
  At least 30 samples above a threshold are needed to accept it.
- **test** split: λ is frozen, and **agreement** (should be ≥ 1−α) and **coverage**
  (fraction not abstained) are measured on held-out data.

Note: `calibration.jsonl` / `test.jsonl` in `data/vlm_v2_export/` are a single fixed split,
the "smoke-test pair" in [DATASET_HANDOFF.md](DATASET_HANDOFF.md). The (1−δ) guarantee is
about repeated random splits, so the final experiment should score `eval_pool.jsonl` and
re-split it many times.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Set OPENROUTER_API_KEY ...` | key not in this terminal | §1. In PowerShell check with `$env:`, not `$OPENROUTER_API_KEY` |
| `rejected the API key (401)` | wrong or revoked key | regenerate it at openrouter.ai/keys |
| ping: `cut off before the verdict: the model is thinking first` | the provider ignores "reasoning off" | `python test_api_judge.py --probe`, then copy a working setting into the judge's `extra_body` in `open_cascade/registry.py` |
| `no logprobs in response` / fails after 20 requests | no logprob-capable provider is serving the model | `--probe`; otherwise switch the config to `kimi-k2.5-votes` |
| many `HTTP 404/503`, run stops | the pinned provider (Novita) is down | wait, or `--probe` to find another provider |
| frequent `retries` | rate limited (429) | lower `concurrency` in the judge's `api` options |
| `Missing image data/vlm_v2/media/...` | media not built, or not run from repo root | `python get_data.py`; `cd` to the repo root |

Failures are counted and the first few are printed in the usage report at the end of
`score` and of the smoke test. A sample is dropped only if all 2N of its calls fail.

## 7. Adding another API model

Any OpenAI-compatible chat endpoint works. Add an entry to `JUDGE_REGISTRY` in
[open_cascade/registry.py](open_cascade/registry.py):

```python
"my-judge": JudgeConfig(
    hf_name="vendor/model-id",      # the provider's model id
    modality="vlm",                 # or "text"
    provider="openrouter",          # key in judges/api.py PROVIDERS
    api={
        "confidence": "logprobs",   # or "votes"
        "max_tokens": 16,
        "extra_body": {...},        # merged into every request: routing, reasoning, ...
    },
),
```

Every `api` option, with its default, is in `DEFAULT_API_OPTIONS` in
[open_cascade/judges/api.py](open_cascade/judges/api.py): `confidence`, `max_tokens`,
`temperature`, `n_votes`, `top_logprobs`, `concurrency`, `max_image_side`, `timeout`,
`max_retries`, `extra_body`. To support a new provider (base URL and key variable), add
it to `PROVIDERS` in the same file. Then smoke-test it with `test_api_judge.py --judge my-judge`.

Tests: `python -m pytest tests/test_api_judge.py`. All HTTP is mocked, so no key is needed.
