"""Metrics computed from a cascade's decisions on the test split.

Every function takes 1-D tensors, one entry per test instance, as stored in CascadeResult:

  labels       human label (0 = outputs[0] preferred, 1 = outputs[1] preferred)
  predictions  label from the judge that answered, -1 where the whole cascade abstained
  evaluators   index of the judge that answered,    -1 where the whole cascade abstained

Add new metrics here as plain functions (e.g. ECE / AUROC on `CascadeResult.phats`).
"""
from collections import defaultdict
from typing import Dict, Optional, Sequence

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


def grouped_agreement(predictions: torch.Tensor, labels: torch.Tensor,
                      evaluators: torch.Tensor, groups: Sequence[str]) -> Dict:
    """Agreement and coverage per source, plus equally weighted macro averages."""
    if len(groups) != labels.numel():
        raise ValueError("groups must have one entry per prediction")
    indices = defaultdict(list)
    for index, group in enumerate(groups):
        indices[str(group)].append(index)

    per_group = {}
    agreements = []
    coverages = []
    for group in sorted(indices):
        idx = torch.tensor(indices[group], dtype=torch.long, device=labels.device)
        selected = evaluators[idx] >= 0
        n_total = len(indices[group])
        n_selected = int(selected.sum())
        agreement: Optional[float] = None
        if n_selected:
            agreement = float((predictions[idx][selected] == labels[idx][selected]).float().mean())
            agreements.append(agreement)
        group_coverage = n_selected / n_total
        coverages.append(group_coverage)
        per_group[group] = {
            "agreement": agreement,
            "coverage": group_coverage,
            "n": n_total,
            "n_evaluated": n_selected,
        }

    return {
        "macro_agreement": sum(agreements) / len(agreements) if agreements else None,
        "macro_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "per_group": per_group,
    }

import numpy as np
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

def auroc(phats: torch.Tensor, correct: torch.Tensor) -> float:
    """Area Under the Receiver Operating Characteristic curve for the confidence scores."""
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for AUROC")
    
    y_true = correct.cpu().numpy().astype(int)
    y_score = phats.cpu().numpy()
    
    if len(np.unique(y_true)) < 2:
        return float('nan')
        
    return roc_auc_score(y_true, y_score)


def ece(phats: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    phats_np = phats.cpu().numpy()
    correct_np = correct.cpu().numpy().astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece_val = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (phats_np > bin_lower) & (phats_np <= bin_upper)
        if bin_lower == 0.0:
            in_bin = (phats_np >= bin_lower) & (phats_np <= bin_upper)
            
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = correct_np[in_bin].mean()
            avg_confidence_in_bin = phats_np[in_bin].mean()
            ece_val += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece_val)
