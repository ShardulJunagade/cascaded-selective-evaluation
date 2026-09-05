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

```
open_cascade/
    registry.py   judge configs (HF name, dtype, tensor parallelism, quantisation)
    judge.py      teacher-forced vLLM scoring for Simulated Annotators
    vlm_registry.py   VLM judge configs
    vlm_judge.py      teacher-forced vLLM scoring with images
    cascade.py    cascade calibration + decision rule
    data.py       jsonl / split helpers

reconstruct_dataset.py        rebuild data + splits from ./result/
prepare_vlm_dataset.py        build VLM data + splits from openbmb/RLHF-V-Dataset
run_open_judge.py             score one judge over a split
run_open_vlm_judge.py         score one VLM judge over a split
validate_reimplementation.py  check a re-scored judge against ./result/
run_open_cascade.py           calibrate + evaluate a cascade
requirements-open.txt
```

Judge output is written in the authors' `./result/{model}.{split}.jsonl` format, so it also
works with their `example_apply_decision_rule.py` unchanged.

## Requirements

`pip install -r requirements-open.txt`

Scoring needs vLLM, hence **Linux** (Colab / Kaggle / a cluster) — vLLM has no Windows
build. Calibration and analysis need only numpy/scipy/torch/jsonlines and run anywhere.

## Step 0 — rebuild the dataset

```shell
python reconstruct_dataset.py
```

The repo ships judgements but not the data. Every released file carries the full instance
next to its `probs`, and all three released judges cover exactly the same 5218 instances, so
stripping `probs` recovers it. Writes `data/split/{calibration,test}.jsonl` (500 / 4718) and
`data/preprocessed/data.jsonl`, **preserving the original split** — which is what makes a
newly scored judge directly comparable against the released GPT-4 / GPT-3.5 numbers. The
script asserts the splits are disjoint and that all released judges cover identical
instances.

## Step 1 — validate before spending GPU time

Re-score the one judge whose released results already exist, to a separate file:

```shell
python run_open_judge.py \
  --model_name=mistral-7b-instruct \
  --in_filename=./data/split/calibration.jsonl \
  --out_filename=./result/_repro.mistral-7b-instruct.calibration.jsonl

python validate_reimplementation.py \
  --reproduced=./result/_repro.mistral-7b-instruct.calibration.jsonl
```

500 samples, a few minutes. Exits non-zero if label agreement < 0.95 or confidence
correlation < 0.90. **Do not score the larger judges until this passes** — a wrong label
token id or a broken chat template produces plausible-looking output, not an error.

## Step 2 — score the judges

```shell
for split in calibration test; do
  for judge in mistral-7b-instruct qwen2.5-7b-instruct qwen2.5-72b-instruct; do
    python run_open_judge.py --model_name=$judge \
      --in_filename=./data/split/$split.jsonl --resume
  done
done
```

Each sample costs `2N` forward passes — N simulated annotators × the A/B-swap position-bias
control. With the shipped `fewshot.jsonl` (N=K=3) that is 6 per sample, so 31,308 per judge.

`--resume` skips already-scored instances, so a preempted run continues where it stopped.
Set `tensor_parallel_size` per judge in `open_cascade/registry.py` to match your GPUs.

Two things matter for throughput, both on by default:

* **Prefix caching.** Only N distinct few-shot prefixes exist across an entire run. Measured
  on this dataset the prefixes are 1382 / 1027 / 966 tokens against a median total prompt of
  1752, so caching removes ~64% of prefill — roughly a 2.5-3× saving. Prompts are emitted
  grouped by annotator to keep the cache hot. `--no_prefix_caching` disables it.
* **One forward pass per simulation, not a generation.** The assistant turn is opened with
  `[[` and the next-token distribution is read directly.

## Step 3 — calibrate and evaluate

```shell
python run_open_cascade.py --alpha=0.15
python run_open_cascade.py --alpha_sweep
python run_open_cascade.py --model_names mistral-7b-instruct qwen2.5-7b-instruct
```

Because scoring is cached per judge, **any cascade subset or ordering can be evaluated
post-hoc for free** — that is the entire Table 6 ablation. Score a judge once, reuse it
everywhere.

## Vision-language cascade

The VLM path mirrors the open text-judge path, but each instance also carries an
`image_path`. The default dataset is `openbmb/RLHF-V-Dataset`, which has the right shape for
this project: one image, one question/instruction, and a human-preferred `chosen` response
paired with a `rejected` response.

Prepare the VLM data and local image files:

```shell
python prepare_vlm_dataset.py
```

For a quick smoke test before spending GPU time, cap the data:

```shell
python prepare_vlm_dataset.py --out_dir ./data/vlm-smoke --max_samples 64 --calibration_set_size 16
```

Score VLM judges with vLLM:

```shell
for split in calibration test; do
  for judge in qwen2.5-vl-3b-instruct qwen2.5-vl-7b-instruct; do
    python run_open_vlm_judge.py --model_name=$judge \
      --in_filename=./data/vlm/split/$split.jsonl --resume
  done
done
```

`qwen2.5-vl-72b-instruct` is configured for 4-way tensor parallelism in
`open_cascade/vlm_registry.py`; request four GPUs before scoring it.

Once every VLM judge has calibration and test files under `./result/vlm`, reuse the same
cascade evaluator:

```shell
python run_open_cascade.py \
  --result_dir ./result/vlm \
  --model_names qwen2.5-vl-3b-instruct qwen2.5-vl-7b-instruct qwen2.5-vl-72b-instruct \
  --alpha=0.15
```

## Optional — raise N and K to 5

The paper used N=K=5 on ChatArena; the shipped `fewshot.jsonl` is N=K=3. Table 5 shows
coverage rises monotonically with N while the guarantee holds throughout. The paper capped
at 5 because each simulation was a GPT-4 call — that constraint is gone with local models.

```shell
python prepare_data_splits.py \
  --in_filename=./data/preprocessed/data.jsonl \
  --fewshot_out_filename=./data/split/fewshot_n5k5.jsonl \
  --calibration_out_filename=./data/split/calibration_n5k5.jsonl \
  --test_out_filename=./data/split/test_n5k5.jsonl \
  --N=5 --K=5 --calibration_set_size=500
```

Costs 1.67× compute. Note this *reshuffles* the split, so results are no longer directly
comparable against the released judgements — keep the Step 0 split for that comparison.

Note also that `prepare_data_splits.py` declares `--N` / `--K` without `type=int`, so
`args.N * args.K` performs string repetition rather than multiplication. Pass the values as
Python ints if calling it programmatically, or fix it locally.

---

## Deviations from the authors' implementation

These mostly live in `open_cascade/` and are deliberate. The shared utility change in
`cascaded_evaluation/util.py` only broadens sample identity to include `image_path` when a
VLM dataset provides one.

### Judge inference (`open_cascade/judge.py` vs `model/vllm_model.py`)

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
`--no_split_delta` reproduces the original behaviour for comparison.

**Imports are lazy.** `cascaded_evaluation.cascaded_classifier` imports `openai` and `vllm`
at module level, so it cannot be imported for calibration-only work without both installed
— even though calibration touches neither.

### Environment

`scipy` is imported by `cascaded_evaluation/util.py` but missing from the authors'
`requirements.txt`; a clean install per that file fails. It is listed in
`requirements-open.txt`.

## Not done yet

* **Multi-split harness.** The paper's headline metric is Guarantee Success Rate over 1000
  random calibration/test splits (Table 3, Figure 4); the current scripts evaluate a single
  split. All inference is cached, so this is pure numpy and cheap.
* **Selective baselines** — No Selection, Heuristic Selection, Cascaded Heuristic, and
  Point-Estimate Calibration, needed for the Table 3 comparison.
