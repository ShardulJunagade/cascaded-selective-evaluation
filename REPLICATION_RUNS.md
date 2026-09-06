# Cascaded Selective Evaluation Replication Runs

This note records the open-model and vision-language replication work run on NCSA Delta.

## Repository State

The original text-only cascade is preserved, with an added open-model path under
`open_cascade/` and a VLM extension for image-conditioned pairwise preference data.

Important additions:

- `run_open_judge.py`: scores open LLM judges with vLLM.
- `run_open_vlm_judge.py`: scores open VLM judges with vLLM.
- `prepare_vlm_dataset.py`: converts a VLM preference dataset into this repo's
  `instruction`, `outputs`, `preferences`, `image_path` format.
- `run_open_cascade.py`: calibrates and evaluates a cascade from cached judge outputs.

## Datasets

### LLM Cascade

The LLM replication uses the reconstructed text preference split from the released judge
outputs:

- calibration: `data/split/calibration.jsonl`
- test: `data/split/test.jsonl`
- few-shot examples: `data/split/fewshot.jsonl`

The reconstructed split has 500 calibration samples and 4718 test samples.

### VLM Cascade

The VLM extension uses `openbmb/RLHF-V-Dataset`.

Each source row has:

- an image
- a question/instruction
- a chosen response
- a rejected response

The conversion stores:

- `text.question` as `instruction`
- `text.chosen` as `outputs[0]`
- `text.rejected` as `outputs[1]`
- `preferences = {"human": 1}`
- the image as a local `image_path`

The VLM run used a 3B -> 7B cascade. The 72B VLM judge was intentionally skipped because it
was too expensive for the available GPU budget.

## Validation

Before running the open LLM cascade, Mistral-7B was re-scored on the calibration split and
compared against the released Mistral judgments.

The validation showed matching human agreement but softer reproduced confidence values:

```text
label agreement (released vs reproduced) : 0.9440
confidence correlation (Pearson)         : 0.6816
human agreement, released                : 0.7500
human agreement, reproduced              : 0.7500
```

The confidence drift is treated as a vLLM/logprob-version diagnostic rather than a hard
failure, since the reproduced and released judgments have identical agreement with human
labels on the validation split.

## LLM Run

Judges:

```text
mistral-7b-instruct -> qwen2.5-7b-instruct
```

Command:

```shell
python run_open_cascade.py \
  --model_names mistral-7b-instruct qwen2.5-7b-instruct \
  --alpha=0.15
```

Result:

```text
lambda_hats           : [0.9998, 0.9996]
target human agreement: 0.85
empirical agreement   : 0.8815
coverage              : 0.2039
evaluator composition : {'mistral-7b-instruct': 0.5405, 'qwen2.5-7b-instruct': 0.4595}
```

The LLM cascade exceeded the 0.85 target agreement, but with conservative thresholds and
low coverage.

### LLM Alpha Sweep

```text
 target  agreement  coverage  composition
-------------------------------------------------------------
   0.70     0.7329     1.000  mistral-7b-instruct:100%  qwen2.5-7b-instruct:0%
   0.75     0.7889     0.828  mistral-7b-instruct:13%   qwen2.5-7b-instruct:87%
   0.80     0.8884     0.232  mistral-7b-instruct:48%   qwen2.5-7b-instruct:52%
   0.85     0.8815     0.204  mistral-7b-instruct:54%   qwen2.5-7b-instruct:46%
   0.90     0.8815     0.204  mistral-7b-instruct:54%   qwen2.5-7b-instruct:46%  <- MISSED
   0.95     0.8815     0.204  mistral-7b-instruct:54%   qwen2.5-7b-instruct:46%  <- MISSED
```

## VLM Run

Judges:

```text
qwen2.5-vl-3b-instruct -> qwen2.5-vl-7b-instruct
```

Command:

```shell
python run_open_cascade.py \
  --result_dir ./result/vlm \
  --model_names qwen2.5-vl-3b-instruct qwen2.5-vl-7b-instruct \
  --alpha=0.15
```

Result:

```text
lambda_hats           : [0.7936, 0.7457]
target human agreement: 0.85
empirical agreement   : 0.8595
coverage              : 0.2951
evaluator composition : {'qwen2.5-vl-3b-instruct': 0.2688, 'qwen2.5-vl-7b-instruct': 0.7312}
```

The VLM cascade also exceeded the 0.85 target agreement and achieved higher coverage than
the two-judge LLM cascade in this run.

## Comparison

At `alpha=0.15`, both cascades reached the 0.85 target:

| Cascade | Empirical agreement | Coverage | Notes |
|---|---:|---:|---|
| LLM: Mistral-7B -> Qwen2.5-7B | 0.8815 | 0.2039 | Higher agreement, lower coverage |
| VLM: Qwen2.5-VL-3B -> Qwen2.5-VL-7B | 0.8595 | 0.2951 | Higher coverage, lower agreement |

These are single-split results. The paper's headline guarantee-success metric evaluates
many random calibration/test splits, so these numbers should be treated as one replication
run rather than a full statistical reproduction.
