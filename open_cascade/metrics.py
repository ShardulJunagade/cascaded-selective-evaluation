"""Metrics computed from a cascade's decisions on the test split.

Every function takes 1-D tensors, one entry per test instance, as stored in CascadeResult:

  labels       human label (0 = outputs[0] preferred, 1 = outputs[1] preferred)
  predictions  label from the judge that answered, -1 where the whole cascade abstained
  evaluators   index of the judge that answered,    -1 where the whole cascade abstained

Add new metrics here as plain functions (e.g. ECE / AUROC on `CascadeResult.phats`).
"""
from typing import Dict, Sequence

import torch


def coverage(evaluators: torch.Tensor) -> float:
    """Fraction of instances the cascade did not abstain on."""
    if evaluators.numel() == 0:
        return 0.0
    return int(torch.sum(evaluators >= 0)) / evaluators.size(-1)


def human_agreement(predictions: torch.Tensor, labels: torch.Tensor,
                    evaluators: torch.Tensor) -> float:
    """Agreement with humans on the non-abstained instances (a.k.a. selective accuracy).

    1.0 when nothing was evaluated, following the released code: an empty selection
    has no errors.
    """
    n_evaluated = int(torch.sum(evaluators >= 0))
    if n_evaluated == 0:
        return 1.0
    return float(torch.sum(predictions == labels)) / n_evaluated


def evaluator_composition(evaluators: torch.Tensor, judge_names: Sequence[str]) -> Dict[str, float]:
    """Share of the *evaluated* instances each judge answered."""
    n_evaluated = int(torch.sum(evaluators >= 0))
    if n_evaluated == 0:
        return {name: 0.0 for name in judge_names}
    return {
        name: float(torch.sum(evaluators == idx)) / n_evaluated
        for idx, name in enumerate(judge_names)
    }
