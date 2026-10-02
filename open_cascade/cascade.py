"""Cascaded Selective Evaluation over cached judgements (paper S2.3, Algorithms 1 and 2).

The statistical core is the authors' -- `SelectiveClassificationUtil` (fixed-sequence testing
over the exact binomial upper confidence bound), `merge_data` and `prepare_data` are imported
from `cascaded_evaluation.util`. This module only re-implements the cascade loop, for two
reasons:

  1. `cascaded_evaluation.cascaded_classifier` imports `openai` and `vllm` at module level,
     so it cannot be imported for calibration-only work without both installed.

  2. It calibrates every judge at the full `delta`. Algorithm 2 (S A.2) calibrates each judge
     at `delta / |M|`, so that the union bound over per-judge risks gives

         P(R_cascades > alpha) <= sum_i P(R_i > alpha) <= (delta / |M|) * |M| = delta.

     Passing the full `delta` inflates the true error level to `|M| * delta` -- with three
     judges and delta=0.1 the guarantee holds at 1-0.3, not the requested 1-0.1.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from cascaded_evaluation.util import merge_data, prepare_data
from open_cascade import metrics
from open_cascade.thresholds import THRESHOLD_METHODS


def merge_judgement_rows(samples: Dict[str, List[Dict]], model_names: Sequence[str]) -> List[Dict]:
    """Line up rows scored by every requested judge, preserving their metadata."""
    missing = [name for name in model_names if name not in samples]
    if missing:
        raise KeyError(f"Judgements missing for: {missing}")
    merged_samples = merge_data(samples, list(model_names))
    if not merged_samples:
        raise ValueError(
            "No instances are covered by every judge. Judges must be scored on the same split."
        )
    return merged_samples


def merge_judgements(samples: Dict[str, List[Dict]], model_names: Sequence[str]
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """Line up judgements from every judge and turn them into tensors.

    Returns `(phats, yhats, labels)`: per judge, confidence and predicted label on each
    instance scored by *all* judges, plus the human label (0 / 1).
    """
    return prepare_data(merge_judgement_rows(samples, model_names), list(model_names))


@dataclass
class CascadeResult:
    """What the cascade decided on each test instance. Metrics are computed from this."""

    judge_names: List[str]
    lambda_hats: List[float]
    phats: Dict[str, torch.Tensor]   # each judge's confidence on every test instance
    yhats: Dict[str, torch.Tensor]   # each judge's predicted label on every test instance
    labels: torch.Tensor             # human label
    predictions: torch.Tensor        # label of the judge that answered, -1 = abstained
    evaluators: torch.Tensor         # index of the judge that answered, -1 = abstained
    samples: List[Dict]              # merged rows, in tensor order, with source metadata

    @property
    def coverage(self) -> float:
        return metrics.coverage(self.evaluators)

    @property
    def human_agreement(self) -> float:
        return metrics.human_agreement(self.predictions, self.labels, self.evaluators)

    @property
    def composition(self) -> Dict[str, float]:
        return metrics.evaluator_composition(self.evaluators, self.judge_names)

    def summary(self) -> Dict:
        """JSON-serialisable headline numbers."""
        return {
            "lambda_hats": [round(float(lam), 4) for lam in self.lambda_hats],
            "human_agreement": self.human_agreement,
            "coverage": self.coverage,
            "composition": self.composition,
            "n_test": int(self.labels.numel()),
        }


class OpenCascadedClassifier:
    """Calibrate per-judge abstention thresholds, then evaluate with escalation.

    Args:
        model_names: judges ordered weakest (cheapest) first.
        calibration_samples: `{model_name: judged samples}` for the calibration split.
        alpha: risk tolerance; the target human agreement level is `1 - alpha`.
        delta: error level for the guarantee, split across judges per Algorithm 2.
        split_delta: set False to reproduce the released code's behaviour instead.
        threshold_method: key in `thresholds.THRESHOLD_METHODS`.
    """

    def __init__(self, model_names: Sequence[str], calibration_samples: Dict[str, List[Dict]],
                 alpha: float, delta: float, split_delta: bool = True,
                 threshold_method: str = "fixed_sequence_testing"):
        if threshold_method not in THRESHOLD_METHODS:
            raise KeyError(f"Unknown threshold method '{threshold_method}'. "
                           f"Known: {sorted(THRESHOLD_METHODS)}")

        self.model_names = list(model_names)
        self.alpha = alpha
        self.delta = delta
        self.split_delta = split_delta
        self.threshold_method = threshold_method

        self.lambda_hats = self.calibrate(calibration_samples, alpha, delta)

    @property
    def per_model_delta(self) -> float:
        return self.delta / len(self.model_names) if self.split_delta else self.delta

    def calibrate(self, calibration_samples: Dict[str, List[Dict]],
                  alpha: float, delta: float) -> List[float]:
        """Set and return `self.lambda_hats`, one threshold per judge."""
        self.alpha, self.delta = alpha, delta
        choose_threshold = THRESHOLD_METHODS[self.threshold_method]

        phats, yhats, labels = merge_judgements(calibration_samples, self.model_names)

        self.lambda_hats = []

        # Each judge is calibrated only on the instances the previous judges abstained on.
        predicted = torch.zeros_like(labels)
        for name in self.model_names:
            abstained = ~predicted.bool()
            lam_hat = choose_threshold(
                phats[name][abstained], yhats[name][abstained], labels[abstained],
                alpha, self.per_model_delta)

            if lam_hat is None:
                # too few remaining instances to test a threshold; inherit the previous one
                lam_hat = self.lambda_hats[-1] if self.lambda_hats else 1.0

            self.lambda_hats.append(lam_hat)
            predicted[phats[name] >= lam_hat] = 1

        return self.lambda_hats

    def apply_decision_rule(self, test_samples: Dict[str, List[Dict]]) -> CascadeResult:
        """Escalate every test instance through the cascade until a judge is confident."""
        samples = merge_judgement_rows(test_samples, self.model_names)
        phats, yhats, labels = prepare_data(samples, self.model_names)

        predictions = torch.ones_like(labels) * -1
        evaluators = torch.ones_like(labels) * -1  # index of the judge that evaluated each

        for idx, name in enumerate(self.model_names):
            # evaluate where this judge is confident and no earlier judge already answered
            evaluated = torch.logical_and(phats[name] >= self.lambda_hats[idx], evaluators < 0)
            evaluators[evaluated] = idx
            predictions[evaluated] = yhats[name][evaluated].float()

        return CascadeResult(
            judge_names=self.model_names,
            lambda_hats=list(self.lambda_hats),
            phats=phats,
            yhats=yhats,
            labels=labels,
            predictions=predictions,
            evaluators=evaluators,
            samples=samples,
        )
