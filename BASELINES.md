# Baseline plan for the multimodal judge experiment

## Common comparison task and metrics

Every judge sees the image, instruction, and two candidate responses, then predicts the
human-preferred response. This matches the pairwise setup used by
[Multimodal RewardBench](https://arxiv.org/abs/2502.14191) and
[VL-RewardBench](https://arxiv.org/abs/2411.17451).

The primary model metric is **pairwise preference accuracy**, called human agreement in
this repository. Because the five sources have very different sizes, also report
**macro source accuracy**: compute accuracy independently for each source, then average
the five values equally. This follows VL-RewardBench's use of macro accuracy for an
imbalanced benchmark. Selective policies additionally report **coverage** and
**guarantee success rate** over 1,000 calibration/test draws.

## Judge models to call out

1. **General-purpose VLM judge: Qwen2.5-VL-7B-Instruct.** This is already scored by the
   harness. It represents the common approach of prompting an instruction-tuned VLM to
   choose between two responses without judge-specific training. The direct baseline
   always uses its prediction and therefore has 100% coverage.
2. **Critic-trained VLM judge: LLaVA-Critic-7B.** LLaVA-Critic is explicitly trained for
   pointwise scoring and pairwise ranking. It is the clearest open-weight specialist
   comparison at roughly the same language-model scale. Its official implementation uses
   LLaVA-NeXT generation and a free-form verdict, so it should be scored with its official
   prompt/parser and then exported to this repository's JSONL judgement format. It should
   not be passed through the Qwen teacher-forced `[[A]]` scorer without validating that
   altered inference protocol.

[Prometheus-Vision](https://arxiv.org/abs/2401.06591) is relevant prior work, but its
headline evaluation is fine-grained pointwise scoring measured by correlation. Converting
those scores into pairwise preferences would change the protocol, making it a less direct
baseline for the present A/B preference dataset.

## Reliability setups from prior work

Model choice and judge setup are separate comparison axes. The following papers change the
prompt, repeat the judgment, or train the model specifically for evaluation. They are the
closest literature comparisons to this repository's simulated-annotator and selective-cascade
setup.

| Paper / setup | Reliability mechanism | Common metric here | Repository status |
|---|---|---|---|
| [MLLM-as-a-Judge](https://arxiv.org/abs/2402.04788), [Multimodal RewardBench](https://arxiv.org/abs/2502.14191) | One fixed pairwise comparison prompt | pairwise accuracy | `setup:vanilla_single:<judge>` |
| [MMRB2](https://github.com/facebookresearch/MMRB2) | Require a position-consistent result across both response orders | accuracy; consistency filtering also has coverage | `setup:position_swap:<judge>` and `setup:swap_consistency:<judge>` |
| [VL-RewardBench](https://arxiv.org/abs/2411.17451) | Repeat evaluation with randomized response order and use majority vote (five calls in its main protocol) | overall and macro accuracy | `setup:prompt_order_vote:<judge>` is a deterministic analogue over this harness's prompt/order runs |
| [Prometheus-Vision](https://arxiv.org/abs/2401.06591) | Train an evaluator to follow user-defined, fine-grained score rubrics | originally correlation; pairwise accuracy after protocol adaptation | requires a rubric-aware scorer and separately cached judgments |
| [LLaVA-Critic](https://arxiv.org/abs/2410.02712) | Critic instruction tuning across evaluation criteria and scenarios | pairwise accuracy | requires official model inference and separately cached judgments |
| [Learning While Evaluating](https://aclanthology.org/2026.eacl-short.50/) | Generate sample-specific criteria, evolve a meta-prompt, and spend extra computation on self-inconsistent cases | pairwise accuracy and consistency | `setup:consistency_escalation` tests its routing heuristic; full LWE needs the paper's sequential meta-prompt loop |

The labels above deliberately distinguish faithful implementations from operational
analogues. `prompt_order_vote` is not VL-RewardBench's stochastic five-call protocol: the
current scorer is deterministic and votes over few-shot prompt variants and both response
orders. `consistency_escalation` implements the reusable routing rule from Selective LWE,
but does not claim to implement its evolving meta-prompt or generated per-sample rubric.

New scoring runs retain every prompt/order probability under `judge_details.simulations`.
This makes the setup comparisons post-hoc ablations: all variants use the same judge forward
passes, dataset split, and human labels. Existing JSONL caches remain valid for the original
aggregate baselines. Their result records `judge_setup_baselines.status = unavailable` and
lists the judges that must be rescored before setup ablations can be reported. Because resume
mode skips completed rows, rescore into a new `result_dir` or remove the affected judge's
`eval_pool` cache first.

## Policy baselines implemented here

The `baseline_comparison` experiment reuses the same cached judge outputs and the same
leakage-safe splits for every policy:

| Policy | Judges | Selection rule |
|---|---|---|
| `direct:<judge>` | each judge separately | no abstention; standard VLM-as-a-judge accuracy |
| `heuristic:<strongest>` | strongest judge | accept when confidence is at least `1 - alpha` |
| `cascaded_heuristic` | full cascade | apply the same heuristic at every stage |
| `point_estimate:<judge>` | each judge separately | calibrate on empirical selective risk, without a confidence bound |
| `cascaded_selective` | full cascade | fixed-sequence testing with the binomial upper bound |

When per-run details are available, the same result also contains:

| Policy | Decision rule |
|---|---|
| `setup:vanilla_single:<judge>` | first prompt, original response order |
| `setup:position_swap:<judge>` | average the first prompt's probabilities across both response orders |
| `setup:prompt_order_vote:<judge>` | majority vote across every prompt/order run |
| `setup:swap_consistency:<judge>` | answer only when the first prompt chooses the same original response in both orders |
| `setup:consistency_escalation` | use each judge only when swap-consistent; route disagreement onward, with the final judge resolving all remaining pairs |

For every policy, the saved result contains mean and standard deviation of overall
accuracy, macro source accuracy, coverage, macro source coverage, per-source results, and
guarantee success rate. The complete per-split records are retained for confidence
intervals and plots in the report.

## Run

```shell
python -m open_cascade score configs/vlm_v2_guarantee.yaml
python -m open_cascade evaluate configs/vlm_v2_guarantee.yaml
```

The comparison is saved to
`outputs/vlm_v2_guarantee/baseline_comparison.json`. A quick CPU evaluation after judge
scoring can use `--set evaluate.n_splits=3`; the report run should retain 1,000 splits.
