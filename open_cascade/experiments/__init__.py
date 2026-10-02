"""Experiments run by `python -m open_cascade evaluate`.

An experiment is a plain function

    experiment(config, calibration, test) -> dict

  config       the ExperimentConfig (judges, cascade settings, evaluate settings, ...)
  calibration  {judge_name: judged samples} for the calibration split
  test         {judge_name: judged samples} for the test split

It may print a human-readable table, and returns a JSON-serialisable dict which is saved to
{evaluate.output_dir}/{config name}/{experiment}.json.

To add one: write the function in a new file in this folder, add it to EXPERIMENTS below,
and list its key under `evaluate.experiments` in a config. If it needs new settings, add
fields to EvaluateConfig in open_cascade/config.py.
"""
from open_cascade.experiments.single_split import run_alpha_sweep, run_cascade

EXPERIMENTS = {
    "cascade": run_cascade,
    "alpha_sweep": run_alpha_sweep,
}
