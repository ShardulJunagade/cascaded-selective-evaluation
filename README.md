# Cascaded Selective Evaluation

This is the supplementary code for the paper **Trust or Escalate: LLM Judges with Provable Guarantees for Human Judgement**.

---

## Multimodal extension (CS 613 project)

Extends the method from text-only LLM judges to vision-language judges. The
dataset pipeline lives in [`prepare_data/`](prepare_data/) and builds a
31,635-instance image-grounded preference corpus from five public sources.

**No data is in this repository** — the raw sources are ~43 GB. To obtain it:

```shell
pip install -r requirements-open.txt
python get_data.py
```

That downloads the sources, builds the corpus into `data/vlm_v2_export/`, and
verifies it. One source (VisionArena-Battle) is gated and needs
`hf auth login`; see [`get_data.py`](get_data.py) for the details.

Start here before running experiments:

| Document | What it covers |
|---|---|
| [DATASET_HANDOFF.md](DATASET_HANDOFF.md) | what the corpus is and how to run on it |
| [DATASET_FINDINGS.md](DATASET_FINDINGS.md) | bias analysis, filtering rationale, bottlenecks — report material |
| [prepare_data/README.md](prepare_data/README.md) | pipeline internals |

## Current VLM benchmark

The reported fast benchmark uses a deterministic, approximately source-stratified sample
of 1,000 image-grounded preference pairs. Each judge sees an image, an instruction, and
two candidate responses and predicts the response preferred by human annotators.

The four-model cascade contains models from different families, ordered from smaller to
larger judges:

1. Granite Vision 3.3 2B
2. Qwen2.5-VL 3B Instruct
3. Phi-3.5 Vision Instruct
4. InternVL3 8B

Scoring uses three simulated annotator prompts, one few-shot example per prompt, and both
response orderings. A completed model judgment is cached in
`result/vlm_v2_fast_1k_n3_k1`. All policy comparisons below operate on those cached
probabilities; evaluation does not load a VLM or repeat inference.

The target selective accuracy is 85%, corresponding to `alpha=0.15`. Statistical
certificates use `delta=0.1`, giving a target confidence level of 90%. Every repeated split
keeps connected prompt and perceptually duplicate image groups together to prevent leakage.

### Original methods and baselines

`baseline_comparison` evaluates the following policies over 1,000 random, group-safe
calibration/test splits:

| Method | Decision rule |
|---|---|
| `direct:<judge>` | Always use one judge; this is the standard full-coverage VLM-as-a-judge baseline. |
| `heuristic:<judge>` | Answer when model confidence is at least `1 - alpha`. |
| `cascaded_heuristic` | Apply the same confidence heuristic sequentially to all judges. |
| `point_estimate:<judge>` | Select the lowest threshold whose calibration error is at most `alpha`, without a confidence bound. |
| `selective:<judge>` | Certify a single judge using a one-sided exact binomial risk bound. |
| `cascaded_selective` | Certify each cascade stage separately and divide `delta` across the judges. |
| Judge setup baselines | Test a single prompt, response-order averaging, prompt/order voting, swap consistency, and consistency-based escalation. |

The key four-judge results from the original comparison were:

| Policy | Accuracy | Macro accuracy | Coverage | Nonempty splits | Target success |
|---|---:|---:|---:|---:|---:|
| `direct:internvl3-8b` | 70.51% | 73.25% | 100.00% | 100.00% | 0.00% |
| `heuristic:internvl3-8b` | 88.72% | 85.73% | 33.80% | 100.00% | 99.90% |
| `cascaded_heuristic` | 86.95% | 84.77% | 36.11% | 100.00% | 96.60% |
| `point_estimate:internvl3-8b` | 84.77% | 83.68% | 42.74% | 100.00% | 45.70% |
| `cascaded_selective` | 93.62% | 87.93% | 13.06% | 66.80% | 63.70% |

The heuristic policies provide useful coverage but do not carry a finite-sample risk
certificate. The original certified cascade exceeds the accuracy target when it answers,
but its per-stage allocation of `delta/4` makes the thresholds conservative and leaves
many splits with no selected test examples.

The three-judge chain showed this effect more strongly. `cascaded_selective` reached
99.50% accuracy but only 0.36% mean coverage and answered in 3.7% of splits. Adding
InternVL3 raised certified coverage to 13.06%, showing that judge strength and confidence
quality materially affect selective coverage.

### End-to-end certified routing

#### Why change the certificate?

The previous certificate is statistically valid, but it controls a stronger collection of
events than the deployed system actually needs. For judge $j$, it separately requires
the conditional selective risk

$$
R_j = \Pr(\widehat{Y}_j \ne Y \mid \text{the cascade reaches and accepts at stage }j)
$$

to be at most `alpha`. With $M$ judges, all $M$ statements must hold simultaneously.
The method therefore assigns `delta/M` to each stage and uses the union bound:

$$
\Pr(\text{any stage certificate fails})
\leq \sum_{j=1}^{M}\delta_j = \delta.
$$

This construction creates three sources of conservatism:

1. With four judges and `delta=0.1`, each stage must certify at `delta/4=0.025`, producing
   wider upper risk bounds than a test at `delta=0.1`.
2. Later judges are calibrated only on examples rejected by earlier judges. Their residual
   calibration samples can be much smaller, which further widens the binomial bounds.
3. Requiring every stage to have risk at most `alpha` is sufficient but not necessary for
   the final cascade to meet the target. The deployed system's risk is a coverage-weighted
   mixture of stage errors, so a low-coverage stage can exceed `alpha` while the complete
   policy still remains below `alpha`.

The theoretical limitation is therefore loss of power and coverage rather than an invalid
guarantee. The four-judge result illustrates it: the stage-wise method has 93.62% accuracy
when it answers, considerably above the required 85%, but answers only 13.06% of examples.
The unused accuracy margin indicates that the certificate is more conservative than the
application requires.

#### What the global certificate changes

Let $\pi_\tau$ denote the complete frozen policy at threshold $\tau$, including all
early accept, escalation, and final abstention decisions. The global method directly
controls the quantity used at deployment:

$$
R(\pi_\tau) =
\Pr(\widehat{Y}_{\pi_\tau} \ne Y \mid \pi_\tau\text{ answers})
\leq \alpha.
$$

This is one end-to-end claim, so its one-sided binomial test can use the full `delta=0.1`.
Errors and coverage from every stage enter the same final-policy risk calculation, avoiding
the requirement that each stage independently satisfy the target. It also avoids computing
a separate certificate from the progressively smaller residual set at every stage.

Using the full `delta` is justified only after the policy has been frozen independently of
the certification labels. For that reason, each repetition uses a three-way split rather
than learning and certifying on the same examples:

`global_router_comparison` tests a modified certification strategy designed to improve
coverage. Each repetition uses three disjoint, group-safe partitions:

1. **Router training:** fit or define the routing score and freeze the candidate threshold
   grid.
2. **Certification:** test the risk of the complete routing policy with fixed-sequence
   testing and the full `delta=0.1`.
3. **Testing:** measure accuracy, macro accuracy, coverage, target success, judge calls,
   early exits, and model composition on unseen examples.

Certifying the complete frozen routing policy produces one statistical claim, so it does
not require a union-bound allocation across four stage-specific claims. Threshold
candidates are quantiles fixed using only the router-training partition; certification
labels are not used to construct the score or threshold grid. Fixed-sequence testing then
examines the frozen candidates from strict to permissive and stops at the first candidate
that cannot be certified. The untouched test partition is used only to measure whether the
claimed risk and coverage generalize.

Two end-to-end policies are implemented:

| Method | Routing score |
|---|---|
| `global_raw_confidence` | The cached confidence of the current judge. This isolates the effect of end-to-end certification. |
| `global_learned_router` | A stage-specific logistic correctness score trained from cached confidence, margin, entropy, simulated-annotator vote support, probability dispersion, response-order consistency, response-order probability shift, agreement among visited model families, and earlier confidence summaries. |

No new examples or VLM calls are needed. The learned router and both certificates run on
CPU from the previously saved judgment files.

### End-to-end results

The four-judge end-to-end experiment used 1,000 repeated splits, with approximately 200
router-training examples, 300 certification examples, and 500 test examples per split.

| Method | Accuracy | Macro accuracy | Coverage | Mean judges called | Nonempty splits | Certified splits | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global_raw_confidence` | **91.08%** | **86.58%** | **25.20%** | 3.891 | **96.00%** | **96.00%** | **86.50%** |
| `global_learned_router` | 90.42% | 86.04% | 15.43% | 3.909 | 61.80% | 61.80% | 53.70% |

Both experiments use the same cached 1,000-example pool. The original comparison assigns
about 300 examples to calibration and 700 to testing; the global-router comparison also
reserves about 200 examples for policy learning, leaving about 500 for testing. The two
global methods use identical partitions within every repetition, while comparisons against
the original table are repeated-split aggregate comparisons rather than paired test-set
differences.

Relative to `cascaded_selective`, raw-confidence end-to-end certification increased mean
coverage from 13.06% to 25.20%, a 12.14 percentage-point or 1.93x increase. Selective
accuracy remained above the 85% target. Among splits where the raw policy found a
certificate, it met the target in `86.5 / 96.0 = 90.1%` of splits, consistent with the
intended 90% confidence level in this experiment.

The 11-feature learned router is a negative ablation at this sample size: it certifies
fewer splits and provides less coverage than raw confidence. The likely issue is variance
from fitting a relatively rich correctness model on roughly 200 training examples. The
mean of 3.891 judge calls also shows that the successful raw policy usually reaches the
final judge. The current gain is therefore a coverage improvement from end-to-end
certification, with little reduction in judge computation.

Accuracy in these tables is conditional on answering and excludes empty selections.
Coverage is the fraction of test examples answered. `Nonempty splits` is the fraction of
repetitions answering at least one example. `Certified splits` is the fraction where the
calibration procedure found a valid threshold. `Target success` is the fraction of all
repetitions that both answered and achieved at least 85% test accuracy. For heuristic and
point-estimate methods, target success is an empirical metric and not a certificate.

### Reproduce from cached judgments

Run both the original comparison and the new end-to-end experiment:

```shell
python -u -m open_cascade evaluate configs/vlm_v2_fast_4judge.yaml
```

Run only the new CPU experiment without rescoring any model:

```shell
python -u -m open_cascade evaluate configs/vlm_v2_fast_4judge.yaml \
  --set 'evaluate.experiments=[global_router_comparison]'
```

Results are saved to:

- `outputs/vlm_v2_fast_4judge/baseline_comparison.json`
- `outputs/vlm_v2_fast_4judge/global_router_comparison.json`

See [BASELINES.md](BASELINES.md) for the full baseline definitions and certificate details,
and [FAST_BENCHMARK.md](FAST_BENCHMARK.md) for the scoring profile and cluster workflow.

> **Note on the Assignment 1 VLM result.** The 85.95% / 29.51% figures came
> from a dataset in which the preferred response was always in slot A, so
> every few-shot demonstration showed `[[A]]` as the answer. The preferences
> themselves were genuine and the judge averages both orderings, so the number
> is not an artefact of position — but the prompts were degenerate and the
> result should be re-run on this corpus before being reported.
> DATASET_FINDINGS.md §1 has the detail.

---

## ***Installation***
```shell
pip install -r requirements.txt
```

## ***Step 1. Preparing LLM Judgements***
**NOTE: We share the pre-generated LLM judgements necessary for running Cascaded Selective Eval in `./result/`. To learn to run the main code, ignore this section and go to Step 2.**
\
\
Prepare the few-shot examples (for simulated annotators), calibration split and test split by running `prepare_data_utils.py`. An example usage is as following:
```shell
python prepare_data_splits.py \
  --input_filename={DATA FILE directory} \
  --fewshot_out_filename={FEWSHOT FILE save directory} \
  --calibration_out_filename={CALIBRATION FILE save directory} \
  --test_out_filename={TEST FILE save directory} \
  --N=3 --K=3 --calibration_set_size=500
```
Next, prompt LLM-as-a-judge on the calibration set by running `evaluate_calibration_*.py`. We provide two versions `evaluate_calibration_openai.py` and `evaluate_calibration_vllm.py`.
An example usage is as following:
```shell
python evaluate_calibration_openai.py \
  --model_name=gpt-4-turbo \
  --in_filename={CALIBRATION FILE directory} \
  --fewshot_in_filename={FEWSHOT FILE directory}
```
Running the script will store generations in `./result/{MODEL NAME}.{CALIBRATION FILENAME}.jsonl`.

For those without API or GPU access, we provide LLM judgements with `gpt-4-turbo`, `gpt-3.5-turbo`, `Mistral-7b-instruct` as judge model in `./result/*.jsonl` files.

## ***Step 2. Cascaded Selective Evaluation***
Cascaded Selective Evaluation is implemented as `CascadedClassifier` class in `cascaded_evaluation` folder.

### Basic Usage
An example script to load model judgements on calibration file, calibrate the abstention policy, and evaluate an unseen sample is in `example_run_inference.py`.

First, load the generations from calibrated samples as follows:
```python
import jsonlines

model_names = ["mistral-7b-instruct", "gpt-3.5-turbo", "gpt-4-turbo"]
calibration_filenames = [
    "./result/mistral-7b-instruct.calibration.jsonl",
    "./result/gpt-3.5-turbo.calibration.jsonl",
    "./result/gpt-4-turbo.calibration.jsonl",
]

calibration_samples = {}
for model_name, input_filename in zip(model_names, calibration_filenames):
    with jsonlines.open(input_filename) as f:
        calibration_samples[model_name] = list(f)
```

Load the cascaded classifier - the classifier will automatically calibrate decision rule upon initialization. In addition, load the provided few-shot examples:
```python
from cascaded_evaluation.cascaded_classifier import CascadedClassifier
cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)

# in this example, use the fixed provided fewshot examples.
# alternatively, one may define a pool of held-out samples to dynamically sample few-shot demonstrations.
with jsonlines.open("./data/split/fewshot.jsonl") as f:
    fewshot_examples_list = [sample["evaluated_samples"] for sample in list(f)]
```

Now the classifier can perform cascaded inference given any unseen example.
```python
result = cascaded_classifier.evaluate_sample(
    sample={
        "instruction": "How many rs are in the word strawberry?",
        "outputs": [
            "There are 3 rs in the word strawberry.",
            "There is no r in the word strawberry.",
        ],
    },
    fewshot_examples_list=fewshot_examples_list,
)  # evaluation done by `gpt-3.5-turbo` in this example
```

### Applying decision rule 
Alternatively, in case one already has full evaluation results across all judge models, `CascadedClassifier` can apply calibrated abstention rules on the model predictions. An example script is provided in `example_apply_decision_rule.py`.

Load predictions for both calibration and test set as follows:
```python
import jsonlines

model_names = ["mistral-7b-instruct", "gpt-3.5-turbo", "gpt-4-turbo"]
calibration_filenames = [
    "./result/mistral-7b-instruct.calibration.jsonl",
    "./result/gpt-3.5-turbo.calibration.jsonl",
    "./result/gpt-4-turbo.calibration.jsonl",
]
test_filenames = [
    "./result/mistral-7b-instruct.test.jsonl",
    "./result/gpt-3.5-turbo.test.jsonl",
    "./result/gpt-4-turbo.test.jsonl",
]

calibration_samples = {}
for model_name, input_filename in zip(model_names, calibration_filenames):
    with jsonlines.open(input_filename) as f:
        calibration_samples[model_name] = list(f)

test_samples = {}
for model_name, input_filename in zip(model_names, test_filenames):
    with jsonlines.open(input_filename) as f:
        test_samples[model_name] = list(f)
```

Load the cascaded classifier:
```python
from cascaded_evaluation.cascaded_classifier import CascadedClassifier

cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)
```

Apply abstention rule to the test predictions and report the metrics.
```python
evaluator_composition, selective_acc, coverage, evaluators = cascaded_classifier.apply_decision_rule(test_samples, model_names)
print(f"Evaluator Composition: {evaluator_composition}")
print(f"Acc: {selective_acc}")
print(f"Coverage: {coverage}")
```

The selective accuracy is around 0.87 in this example, which satisfies the user-prescribed risk of `alpha=0.15`.
