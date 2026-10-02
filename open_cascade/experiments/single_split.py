"""Experiments on the fixed calibration/test split: one alpha, or a sweep over alpha."""
from typing import Dict, List

from open_cascade.cascade import CascadeResult, OpenCascadedClassifier
from open_cascade.config import ExperimentConfig

Judgements = Dict[str, List[Dict]]


def calibrate_and_evaluate(config: ExperimentConfig, calibration: Judgements, test: Judgements,
                           alpha: float) -> CascadeResult:
    """Calibrate the config's cascade at `alpha` and apply it to the test split."""
    classifier = OpenCascadedClassifier(
        config.judges,
        calibration_samples=calibration,
        alpha=alpha,
        delta=config.cascade.delta,
        split_delta=config.cascade.split_delta,
        threshold_method=config.cascade.threshold_method,
    )
    return classifier.apply_decision_rule(test)


def run_cascade(config: ExperimentConfig, calibration: Judgements, test: Judgements) -> Dict:
    """Calibrate at `cascade.alpha` and report agreement, coverage and composition."""
    alpha = config.cascade.alpha
    result = calibrate_and_evaluate(config, calibration, test, alpha)
    summary = {"alpha": alpha, "target_agreement": 1 - alpha, **result.summary()}

    print(f"lambda_hats           : {summary['lambda_hats']}")
    print(f"target human agreement: {1 - alpha:.2f}")
    print(f"empirical agreement   : {result.human_agreement:.4f}")
    print(f"coverage              : {result.coverage:.4f}")
    print(f"evaluator composition : "
          f"{ {name: round(share, 4) for name, share in result.composition.items()} }")
    return summary


def run_alpha_sweep(config: ExperimentConfig, calibration: Judgements, test: Judgements) -> Dict:
    """Repeat `cascade` for every alpha in `evaluate.alphas`."""
    header = f"{'target':>7} {'agreement':>10} {'coverage':>9}  composition"
    print(header)
    print("-" * (len(header) + 20))

    rows = []
    for alpha in config.evaluate.alphas:
        result = calibrate_and_evaluate(config, calibration, test, alpha)
        met = result.human_agreement >= 1 - alpha
        rows.append({"alpha": alpha, "target_agreement": 1 - alpha, "met_target": met,
                     **result.summary()})

        shares = "  ".join(f"{n}:{result.composition[n]:.0%}" for n in config.judges)
        flag = "" if met else "  <- MISSED"
        print(f"{1 - alpha:>7.2f} {result.human_agreement:>10.4f} "
              f"{result.coverage:>9.3f}  {shares}{flag}")

    print("\nNote: this is a single split. The paper's headline metric is the guarantee")
    print("success rate over 1000 random calibration/test splits.")
    return {"rows": rows}
