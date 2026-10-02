"""How each judge's abstention threshold (lambda) is chosen.

A threshold method is a plain function

    method(phats, yhats, labels, alpha, delta) -> Optional[float]

  phats   confidence of this judge on each calibration instance it sees
  yhats   its predicted label (0 / 1) on those instances
  labels  the human label (0 / 1)
  alpha   risk tolerance (target human agreement is 1 - alpha)
  delta   error level available to *this* judge (already split across the cascade)

The cascade only passes a judge the calibration instances every earlier judge abstained on.
Return None when there is too little data to pick a threshold; the cascade then reuses the
previous judge's threshold.

To add a method (e.g. a baseline such as heuristic lambda = 1 - alpha, or point-estimate
calibration), write the function here and add it to THRESHOLD_METHODS. Select it with
`cascade.threshold_method` in the experiment config.
"""
from typing import Optional

import torch

from cascaded_evaluation.util import SelectiveClassificationUtil


def no_selection(phats: torch.Tensor, yhats: torch.Tensor, labels: torch.Tensor,
                 alpha: float, delta: float) -> float:
    """Always answer. This is the standard full-coverage VLM-as-a-judge baseline."""
    return 0.0


def heuristic_selection(phats: torch.Tensor, yhats: torch.Tensor, labels: torch.Tensor,
                        alpha: float, delta: float) -> float:
    """Assume confidence is calibrated and accept when confidence >= 1 - alpha."""
    return 1.0 - alpha


def point_estimate_calibration(phats: torch.Tensor, yhats: torch.Tensor,
                               labels: torch.Tensor, alpha: float,
                               delta: float) -> Optional[float]:
    """Smallest threshold whose empirical selective risk is at most alpha.

    This follows the point-estimate baseline from Trust or Escalate: it uses
    calibration error directly, without an upper confidence bound or hypothesis
    testing. As in the proposed method, a threshold must cover at least 30
    calibration examples so a tiny selected set cannot look perfect by chance.
    """
    if phats.numel() == 0:
        return None
    candidates = torch.unique(phats).sort().values
    valid = []
    for threshold in candidates:
        selected = phats >= threshold
        if int(selected.sum()) < 30:
            continue
        risk = float((yhats[selected] != labels[selected]).float().mean())
        if risk <= alpha:
            valid.append(float(threshold))
    return min(valid) if valid else None


def fixed_sequence_testing(phats: torch.Tensor, yhats: torch.Tensor, labels: torch.Tensor,
                           alpha: float, delta: float) -> Optional[float]:
    """The paper's method (S2.3): the smallest lambda whose exact binomial upper confidence
    bound on the selective risk stays <= alpha, testing from the highest lambda down."""
    selective_util = SelectiveClassificationUtil(
        cal_phats=phats, cal_yhats=yhats, cal_labels=labels, delta=delta)

    if selective_util.lambdas.size(-1) == 0:
        return None  # too few instances to test any threshold
    return float(selective_util.lambda_hat(alpha=alpha))


THRESHOLD_METHODS = {
    "fixed_sequence_testing": fixed_sequence_testing,
    "no_selection": no_selection,
    "heuristic_selection": heuristic_selection,
    "point_estimate_calibration": point_estimate_calibration,
}
