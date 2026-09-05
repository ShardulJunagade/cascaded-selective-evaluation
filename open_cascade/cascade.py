"""Cascaded Selective Evaluation over open judges (paper S2.3, Algorithms 1 and 2).

The statistical core is the authors' -- `SelectiveClassificationUtil` (fixed-sequence testing
over the exact binomial upper confidence bound), `merge_data` and `prepare_data` are imported
from `cascaded_evaluation.util` and used unchanged. This module only re-implements the
cascade loop, for two reasons:

  1. `cascaded_evaluation.cascaded_classifier` imports `openai` and `vllm` at module level,
     so it cannot be imported for calibration-only work without both installed.

  2. It calibrates every judge at the full `delta`. Algorithm 2 (S A.2) calibrates each judge
     at `delta / |M|`, so that the union bound over per-judge risks gives

         P(R_cascades > alpha) <= sum_i P(R_i > alpha) <= (delta / |M|) * |M| = delta.

     Passing the full `delta` inflates the true error level to `|M| * delta` -- with three
     judges and delta=0.1 the guarantee holds at 1-0.3, not the requested 1-0.1.
"""
from typing import Dict, List, Sequence, Tuple

import torch

from cascaded_evaluation.util import SelectiveClassificationUtil, merge_data, prepare_data


class OpenCascadedClassifier:
    """Calibrate per-judge abstention thresholds, then evaluate with escalation.

    Args:
        model_names: judges ordered weakest (cheapest) first.
        calibration_samples: `{model_name: judged samples}` for the calibration split.
        alpha: risk tolerance; the target human agreement level is `1 - alpha`.
        delta: error level for the guarantee, split across judges per Algorithm 2.
        split_delta: set False to reproduce the released code's behaviour instead.
    """

    def __init__(self, model_names: Sequence[str], calibration_samples: Dict[str, List[Dict]],
                 alpha: float, delta: float, split_delta: bool = True):
        self.model_names = list(model_names)
        self.alpha = alpha
        self.delta = delta
        self.split_delta = split_delta

        self.lambda_hats = self.calibrate(calibration_samples, alpha, delta)

    @property
    def per_model_delta(self) -> float:
        return self.delta / len(self.model_names) if self.split_delta else self.delta

    def calibrate(self, calibration_samples: Dict[str, List[Dict]],
                  alpha: float, delta: float) -> List[float]:
        missing = [name for name in self.model_names if name not in calibration_samples]
        if missing:
            raise KeyError(f"Calibration samples missing for: {missing}")

        self.alpha, self.delta = alpha, delta

        merged_samples = merge_data(calibration_samples, self.model_names)
        if not merged_samples:
            raise ValueError(
                "No calibration instances are covered by every judge. Judges must be scored "
                "on the same split."
            )
        phats, yhats, labels = prepare_data(merged_samples, self.model_names)

        self.lambda_hats = []

        # Each judge is calibrated only on the instances the previous judges abstained on.
        predicted = torch.zeros_like(labels)
        for name in self.model_names:
            abstained = ~predicted.bool()
            selective_util = SelectiveClassificationUtil(
                cal_phats=phats[name][abstained],
                cal_yhats=yhats[name][abstained],
                cal_labels=labels[abstained],
                delta=self.per_model_delta,
            )

            if selective_util.lambdas.size(-1) == 0:
                # too few remaining instances to test a threshold; inherit the previous one
                lam_hat = self.lambda_hats[-1] if self.lambda_hats else 1.0
            else:
                lam_hat = selective_util.lambda_hat(alpha=alpha)

            self.lambda_hats.append(lam_hat)
            predicted[phats[name] >= lam_hat] = 1

        return self.lambda_hats

    def apply_decision_rule(self, test_samples: Dict[str, List[Dict]]
                            ) -> Tuple[Dict[str, float], float, float, torch.Tensor]:
        """Escalate through the cascade and report composition, selective accuracy, coverage."""
        missing = [name for name in self.model_names if name not in test_samples]
        if missing:
            raise KeyError(f"Test samples missing for: {missing}")

        merged_samples = merge_data(test_samples, self.model_names)
        phats, yhats, labels = prepare_data(merged_samples, self.model_names)

        predictions = torch.ones_like(labels) * -1
        evaluators = torch.ones_like(labels) * -1  # index of the judge that evaluated each

        for idx, name in enumerate(self.model_names):
            # evaluate where this judge is confident and no earlier judge already answered
            evaluated = torch.logical_and(phats[name] >= self.lambda_hats[idx], evaluators < 0)
            evaluators[evaluated] = idx
            predictions[evaluated] = yhats[name][evaluated].float()

        n_evaluated = int(torch.sum(evaluators >= 0))
        if n_evaluated == 0:
            return {name: 0.0 for name in self.model_names}, 1.0, 0.0, evaluators

        evaluator_composition = {
            name: float(torch.sum(evaluators == idx)) / n_evaluated
            for idx, name in enumerate(self.model_names)
        }
        selective_acc = float(torch.sum(predictions == labels)) / n_evaluated
        coverage = n_evaluated / evaluators.size(-1)

        return evaluator_composition, selective_acc, coverage, evaluators
