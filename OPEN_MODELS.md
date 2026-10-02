# Open-model replication

Reproducing *Trust or Escalate* without API access, by replacing the two OpenAI judges with
open-weight models.

| Tier | Paper | Open replacement |
|---|---|---|
| weak | `mistral-7b-instruct` | unchanged — already open |
| mid | `gpt-3.5-turbo` | `qwen2.5-7b-instruct` |
| strong | `gpt-4-turbo` | `qwen2.5-72b-instruct` |

The cascade only works if each judge is genuinely stronger than the last. On the released
judgements, full-coverage human agreement is Mistral-7B **73.3%** → GPT-3.5 **76.4%** →
GPT-4 **78.6%**, so the replacements must preserve that ordering.

## Design

The authors' statistical code is left behaviour-compatible so the original text results stay
reproducible. The only shared utility extension is that sample identity now includes
`image_path` when present, which prevents VLM samples with identical text but different
images from colliding.

The statistical core — `SelectiveClassificationUtil` (fixed-sequence testing over the exact
binomial upper confidence bound), `merge_data`, `prepare_data` — is *imported* from
`cascaded_evaluation.util` and reused for both text and VLM cascades. That module depends only on `torch` and
`scipy`, so it imports cleanly without vLLM or an API key. Only judge inference and the
cascade loop are reimplemented.

Everything runs through one CLI, `python -m open_cascade <command> <config.yaml>`, with one
YAML file per experiment in `configs/`. [PIPELINE.md](PIPELINE.md) describes the commands,
the config format, the code layout and how to add judges, datasets and experiments.

Judge output is written in the authors' `./result/{model}.{split}.jsonl` format, so it also
works with their `example_apply_decision_rule.py` unchanged.

## Requirements

`pip install -r requirements-open.txt`

Scoring needs vLLM, hence **Linux** (Colab / Kaggle / a cluster) — vLLM has no Windows
build. Calibration and analysis need only numpy/scipy/torch/pyyaml and run anywhere.

## Step 0 — rebuild the dataset

```shell
python -m open_cascade prepare-data configs/text_open_cascade.yaml
```

The repo ships judgements but not the data. Every released file carries the full instance
next to its `probs`, and all three released judges cover exactly the same 5218 instances, so
stripping `probs` recovers it (`open_cascade/builders/released_text.py`). Writes
`data/split/{calibration,test}.jsonl` (500 / 4718) and `data/preprocessed/data.jsonl`,
**preserving the original split** — which is what makes a newly scored judge directly
comparable against the released GPT-4 / GPT-3.5 numbers. The builder asserts the splits are
disjoint and that all released judges cover identical instances. All text configs share
this data, so this runs once.

## Step 1 — validate before spending GPU time

Re-score the one judge whose released results already exist, into `./result/_repro/`:

```shell
python -m open_cascade score    configs/validate_mistral.yaml
python -m open_cascade validate configs/validate_mistral.yaml
```

500 samples, a few minutes. Exits non-zero if label agreement or human agreement drift past
the thresholds in the config (`validation:` section; `strict_confidence: true` also requires
confidence correlation ≥ 0.90). **Do not score the larger judges until this passes** — a
wrong label token id or a broken chat template produces plausible-looking output, not an
error.

## Step 2 — score the judges

```shell
python -m open_cascade score configs/text_open_cascade.yaml
# or one judge per job:
python -m open_cascade score configs/text_open_cascade.yaml --judge qwen2.5-72b-instruct
```

Each sample costs `2N` forward passes — N simulated annotators × the A/B-swap position-bias
control. With the shipped `fewshot.jsonl` (N=K=3) that is 6 per sample, so 31,308 per judge.

Scoring skips already-scored instances (`scoring.resume: true`), so a preempted run
continues where it stopped. Set `tensor_parallel_size` per judge in
`open_cascade/registry.py` to match your GPUs.

Two things matter for throughput, both on by default:

* **Prefix caching.** Only N distinct few-shot prefixes exist across an entire run. Measured
  on this dataset the prefixes are 1382 / 1027 / 966 tokens against a median total prompt of
  1752, so caching removes ~64% of prefill — roughly a 2.5-3× saving. Prompts are emitted
  grouped by annotator to keep the cache hot. `--set scoring.prefix_caching=false`
  disables it.
* **One forward pass per simulation, not a generation.** The assistant turn is opened with
  `[[` and the next-token distribution is read directly.

## Step 3 — calibrate and evaluate

```shell
python -m open_cascade evaluate configs/text_open_cascade.yaml
python -m open_cascade evaluate configs/text_open_cascade.yaml --set cascade.alpha=0.1
python -m open_cascade evaluate configs/text_mistral_qwen7b.yaml
```

Because scoring is cached per judge, **any cascade subset or ordering can be evaluated
post-hoc for free** — that is the entire Table 6 ablation. Score a judge once, reuse it
everywhere: a new cascade is just a new config (or `--set "judges=[...]"`) pointing at the
same `result_dir`.

## Vision-language cascade

The VLM path mirrors the open text-judge path, but each instance also carries an
`image_path`. The default dataset is `openbmb/RLHF-V-Dataset`, which has the right shape for
this project: one image, one question/instruction, and a human-preferred `chosen` response
paired with a `rejected` response.

Prepare the VLM data and local image files, score, and evaluate:

```shell
python -m open_cascade prepare-data configs/vlm_qwen_3b_7b.yaml
python -m open_cascade score        configs/vlm_qwen_3b_7b.yaml
python -m open_cascade evaluate     configs/vlm_qwen_3b_7b.yaml
```

For a quick smoke test before spending GPU time, run the same three commands with
`configs/vlm_smoke.yaml` (64 samples, 16 for calibration).

Every few-shot example adds an image to the prompt, so VLM configs set
`scoring.max_fewshot_examples: 1`. vLLM's per-prompt image limit follows from it
(`max_fewshot_examples + 1`), so raising it needs no other change.

`qwen2.5-vl-72b-instruct` is configured for 4-way tensor parallelism in
`open_cascade/registry.py`; request four GPUs before scoring it. The three-judge cascade
reuses the 3B/7B judgements:

```shell
python -m open_cascade score    configs/vlm_qwen_3b_7b_72b.yaml --judge qwen2.5-vl-72b-instruct
python -m open_cascade evaluate configs/vlm_qwen_3b_7b_72b.yaml
```

## Optional — raise N and K to 5

The paper used N=K=5 on ChatArena; the shipped `fewshot.jsonl` is N=K=3. Table 5 shows
coverage rises monotonically with N while the guarantee holds throughout. The paper capped
at 5 because each simulation was a GPT-4 call — that constraint is gone with local models.

Copy a text config, then change its `data` section to draw a fresh split, and give it its
own `data.out_dir` and `result_dir`:

```yaml
data:
  builder: released_text
  out_dir: ./data/n5k5
  n_annotators: 5
  k_shot: 5
  calibration_size: 500
  options:
    keep_original_split: false
result_dir: ./result/n5k5
```

Costs 1.67× compute. Note this *reshuffles* the split, so results are no longer directly
comparable against the released judgements — keep the Step 0 split for that comparison.

(The authors' `prepare_data_splits.py` does the same split but declares `--N` / `--K`
without `type=int`, so `args.N * args.K` performs string repetition; the builder avoids it.)

---

## Deviations from the authors' implementation

These mostly live in `open_cascade/` and are deliberate. The shared utility change in
`cascaded_evaluation/util.py` only broadens sample identity to include `image_path` when a
VLM dataset provides one.

### Judge inference (`open_cascade/judges/` vs `model/vllm_model.py`)

**Label token ids are derived from the tokenizer**, not hardcoded to `28741` / `28760`
(Mistral's `A`/`B`). This is the change that makes other model families work at all, and the
obvious implementation of it is wrong: Mistral encodes a bare `"A"` as `330` — the
space-prefixed variant — while the token following `[[` is `28741`. Reading logprobs for
`330` returns `-inf` on every sample and produces an empty output file with no error raised.
The ids are derived by diffing against the real prompt prefix, with the token boundary
asserted so an unexpected tokenizer fails loudly at construction.

Verified for both families:

| | id after `[[` | naive `encode("A")` |
|---|---|---|
| Qwen2.5 | `A:32, B:33` | 32 — coincides |
| Mistral | `A:28741, B:28760` | **330 — wrong** |

**`add_generation_prompt=True`** on the chat template. Omitting it is harmless for Mistral,
whose template appends `[/INST]` regardless, but fatal for Qwen: the ChatML prompt would end
at `<|im_end|>` with no `<|im_start|>assistant`, leaving the model outside answering
position entirely.

**The verdict is teacher-forced.** The prompt ends at `[[` and one decode step is read. The
original generates 5 tokens, requires the text to equal `"[[A]]"` exactly, then searches for
the position after `[[`, dropping the sample on any deviation — silently, at
`evaluate_calibration_vllm.py:64`. Weaker judges deviate often, so a sample can quietly end
up averaged over fewer than N annotators. Teacher forcing is also ~5× cheaper and fully
deterministic.

**One BOS.** Prompts are built once as token ids. Re-tokenising an already-templated string,
as the original does, emits a second BOS for Mistral.

**Probabilities renormalised over `{A, B}`.** The released values are raw `exp(logprob)`.
In practice this is nearly identical — they sum to 1.000 at the median and 0.964 at worst,
since A/B absorb essentially all the mass in this position — but renormalising makes the
confidence exactly a two-way probability.

### Cascade (`open_cascade/cascade.py` vs `cascaded_evaluation/cascaded_classifier.py`)

**δ is split across the cascade.** Algorithm 2 (§A.2) calibrates each judge at `δ/|M|`, so
the union bound over per-judge risks gives `P(R_cascades > α) ≤ δ`. The released code passes
the full `δ` to every judge, which inflates the true error level to `|M|·δ` — with three
judges and `δ=0.1`, the guarantee holds at 1−0.3 rather than the requested 1−0.1.
`--set cascade.split_delta=false` reproduces the original behaviour for comparison.

**Imports are lazy.** `cascaded_evaluation.cascaded_classifier` imports `openai` and `vllm`
at module level, so it cannot be imported for calibration-only work without both installed
— even though calibration touches neither.

### Environment

`scipy` is imported by `cascaded_evaluation/util.py` but missing from the authors'
`requirements.txt`; a clean install per that file fails. It is listed in
`requirements-open.txt`.

## Repeated evaluation and baselines

The `baseline_comparison` experiment now draws leakage-safe random splits from cached
`eval_pool` judgements and reports the paper's direct, heuristic, cascaded heuristic,
point-estimate and fixed-sequence policies. See `BASELINES.md` for the multimodal model
baselines and common metrics.
