"""Cascaded Selective Evaluation with open-weight text and vision-language judges.

The paper's cascade is Mistral-7B-Instruct-v0.2 -> gpt-3.5-turbo -> gpt-4-turbo. This
package replaces the OpenAI judges with open-weight models so experiments run without API
access, and extends the method to vision-language judges.

The statistical core -- `SelectiveClassificationUtil`, `merge_data`, `prepare_data` -- is
imported from the authors' `cascaded_evaluation.util` (whose only change is that sample
identity includes `image_path`). Judge inference and the cascade loop are reimplemented here.

Run it with `python -m open_cascade <command> <config.yaml>`; see PIPELINE.md.

Layout:
    cli.py          commands: prepare-data, score, evaluate, validate, list
    config.py       experiment config (YAML -> dataclasses)
    data.py         jsonl / few-shot / split helpers

    builders/       dataset builders            (plug-in point: new datasets)
    registry.py     judge model configurations  (plug-in point: new judge models)
    judges/         vLLM judges, text and VLM   (plug-in point: new judge types)
    scoring.py      the `score` stage

    thresholds.py   per-judge lambda selection  (plug-in point: new calibration methods)
    cascade.py      cascade calibration + decision rule
    metrics.py      agreement, coverage, ...    (plug-in point: new metrics)
    experiments/    what `evaluate` runs        (plug-in point: new experiments)
    validation.py   the `validate` stage
"""
