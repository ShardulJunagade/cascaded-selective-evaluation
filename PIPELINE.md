# Experiment pipeline

How to run experiments with the `open_cascade` package, and where to plug in new ones.
For *why* the open-model judges work the way they do, see [OPEN_MODELS.md](OPEN_MODELS.md).
For results already obtained, see [REPLICATION_RUNS.md](REPLICATION_RUNS.md).

## The idea in one paragraph

An **experiment is one YAML file** in `configs/`. It names the dataset, the judges (weakest
first), and the cascade settings. Every stage is one command that reads that file:

```
prepare-data  ->  score  ->  evaluate
```

`score` is the only expensive step, and its output is cached per judge in
`{result_dir}/{judge}.{split}.jsonl`. After that, any number of cascade experiments
(different alpha, judge subsets, calibration methods, metrics) run on the cached judgements
without touching a GPU.

## Running

```shell
pip install -r requirements-open.txt        # vLLM part is Linux-only

python -m open_cascade list                 # what judges / datasets / experiments exist

python -m open_cascade prepare-data configs/vlm_qwen_3b_7b.yaml
python -m open_cascade score        configs/vlm_qwen_3b_7b.yaml
python -m open_cascade evaluate     configs/vlm_qwen_3b_7b.yaml
```

* `score` scores every judge in the config, one process per judge (vLLM does not reliably free
  GPU memory in-process). To score just one, e.g. in its own SLURM job:
  `--judge qwen2.5-vl-7b-instruct`. Scoring resumes where it stopped if a job is preempted.
* `evaluate` prints its tables and saves them to `outputs/{name}/{experiment}.json`, together
  with the exact config used (`outputs/{name}/config.yaml`).
* Any config value can be overridden without editing the file (values are parsed as YAML):

  ```shell
  python -m open_cascade evaluate configs/text_mistral_qwen7b.yaml --set cascade.alpha=0.1
  python -m open_cascade evaluate configs/text_paper_released.yaml \
      --set cascade.split_delta=false --set "evaluate.experiments=[cascade]"
  ```

  Overridden runs save to the same `outputs/{name}/` folder; add `--set name=my_variant` to
  keep them apart.

### Shipped configs (everything run for Assignment 1)

| Config | What it is |
|---|---|
| `text_paper_released.yaml` | Paper cascade (Mistral → GPT-3.5 → GPT-4) on the released judgements. Evaluate only. |
| `text_mistral_qwen7b.yaml` | Open text cascade Mistral-7B → Qwen2.5-7B (REPLICATION_RUNS "LLM Run"). |
| `text_open_cascade.yaml` | Full open replacement Mistral-7B → Qwen2.5-7B → Qwen2.5-72B. |
| `validate_mistral.yaml` | Re-score Mistral-7B and compare with the released file (`validate`). |
| `vlm_qwen_3b_7b.yaml` | VLM cascade Qwen2.5-VL-3B → 7B on RLHF-V (REPLICATION_RUNS "VLM Run"). |
| `vlm_qwen_3b_7b_72b.yaml` | Same plus Qwen2.5-VL-72B (4 GPUs). |
| `vlm_smoke.yaml` | 64-sample end-to-end check of the VLM pipeline. |

### What the config sections mean

```yaml
name: my_experiment            # output folder name; keep equal to the file name
judges: [small, medium, big]   # weakest (cheapest) first
result_dir: ./result/vlm       # where cached judgements are read/written

data:                          # prepare-data
  builder: rlhf_v              # which dataset builder (python -m open_cascade list)
  out_dir: ./data/vlm          # splits go to {out_dir}/split/{fewshot,calibration,test}.jsonl
  n_annotators: 3              # N simulated annotators
  k_shot: 3                    # K few-shot examples per annotator
  calibration_size: 500
  seed: 42
  options: {...}               # builder-specific, see the builder's docstring

scoring:                       # score
  splits: [calibration, test]
  chunk_size: 16
  max_fewshot_examples: 1      # use at most this many of the K examples (VLM: 1 image each)
  prefix_caching: true
  resume: true
  released_mistral_compat: false

cascade:                       # evaluate
  alpha: 0.15                  # target human agreement = 1 - alpha
  delta: 0.1                   # guarantee holds with probability 1 - delta
  split_delta: true            # delta/|judges| per judge (Algorithm 2); false = released code
  threshold_method: fixed_sequence_testing

evaluate:
  experiments: [cascade, alpha_sweep]
  alphas: [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
  output_dir: ./outputs
```

Every key and its default is defined in [open_cascade/config.py](open_cascade/config.py).
A misspelt key is an error, not silently ignored.

## Code layout

```
open_cascade/
    cli.py            the commands above
    config.py         YAML -> dataclasses (all settings + defaults live here)
    data.py           jsonl, few-shot and split helpers

    builders/         prepare-data: raw dataset -> common format -> splits
    registry.py       judge models (HF name, dtype, number of GPUs, ...)
    judges/           vLLM judges: base.py (shared), text.py, vlm.py
    scoring.py        score: run a judge over the splits, cache results

    thresholds.py     how each judge's lambda is chosen
    cascade.py        calibrate lambdas down the cascade, apply the decision rule
    metrics.py        human agreement, coverage, composition
    experiments/      what evaluate runs (single_split.py: cascade, alpha_sweep)
    validation.py     validate: re-scored judge vs released judgements

configs/              one YAML per experiment
tests/                python -m pytest   (CPU only, ~30 s)
```

The authors' original code (`cascaded_evaluation/`, `model/`, `evaluate_calibration_*.py`,
`example_*.py`, `prepare_data_splits.py`) is left untouched. We import two things from it:
the statistics in `cascaded_evaluation/util.py` and the prompt templates in `model/`.

## Adding things

Every plug-in point works the same way: **write a function or class, add it to the
dictionary in that module, refer to it by name in a config.** `python -m open_cascade list`
shows what is registered.

| To add... | Write | Register in | Select with |
|---|---|---|---|
| a judge model | a `JudgeConfig` entry | `registry.py` → `JUDGE_REGISTRY` | `judges:` |
| a dataset | a `DatasetBuilder` subclass (new file in `builders/`) | `builders/__init__.py` → `BUILDERS` | `data.builder` |
| an experiment | a function `(config, calibration, test) -> dict` (new file in `experiments/`) | `experiments/__init__.py` → `EXPERIMENTS` | `evaluate.experiments` |
| a calibration method / baseline | a function `(phats, yhats, labels, alpha, delta) -> lambda` | `thresholds.py` → `THRESHOLD_METHODS` | `cascade.threshold_method` |
| a metric | a function over `CascadeResult` tensors | `metrics.py` | call it from an experiment |
| a new kind of judge (new prompt format / modality) | a `BaseJudge` subclass with 3 hooks | `judges/__init__.py` → `JUDGE_CLASSES` | `modality=` in `JudgeConfig` |

If an experiment needs a new setting (say `n_splits`), add a field with a default to the
matching dataclass in `config.py`; existing configs keep working.

### Example: a new dataset

```python
# open_cascade/builders/my_dataset.py
from open_cascade.builders.base import DatasetBuilder

class MyDatasetBuilder(DatasetBuilder):
    def __init__(self, config, hf_name="org/my-dataset"):   # options from data.options
        super().__init__(config)
        self.hf_name = hf_name

    def load_instances(self):
        # return a list of {"instruction", "outputs": [a, b], "preferences": {"human": 1|2},
        #                   "image_path" (VLM only), "source" (optional)}
        ...
```

Add `"my_dataset": MyDatasetBuilder` to `BUILDERS`, copy a config, set
`data.builder: my_dataset` and a new `data.out_dir`. `build()` in the base class does the
N×K few-shot sampling and the calibration/test split for you.

### Example: a new experiment

```python
# open_cascade/experiments/my_experiment.py
from open_cascade.experiments.single_split import calibrate_and_evaluate

def run_my_experiment(config, calibration, test):
    """One-line description shown by `python -m open_cascade list`."""
    result = calibrate_and_evaluate(config, calibration, test, alpha=config.cascade.alpha)
    print(f"coverage: {result.coverage:.3f}")
    return {"coverage": result.coverage}
```

Add `"my_experiment": run_my_experiment` to `EXPERIMENTS` and list it under
`evaluate.experiments`. `result` is a `CascadeResult` (see `cascade.py`): it carries every
judge's `phats`/`yhats` on the test split, the human `labels`, and which judge answered each
instance, so most analyses need nothing else. For experiments that re-split the data
(e.g. many random calibration/test splits), pool `calibration` and `test` — they are plain
`{judge: [judged samples]}` dicts.

## Tests

```shell
python -m pytest
```
