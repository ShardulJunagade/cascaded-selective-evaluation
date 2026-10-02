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

## Two model baselines to call out

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
