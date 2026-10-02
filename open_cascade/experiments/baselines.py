"""Fair policy baselines over the same cached VLM judgments and random splits."""
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

from open_cascade import metrics
from open_cascade.cascade import OpenCascadedClassifier
from open_cascade.experiments.multi_split import align_pool, draw_keys


Judgements = Dict[str, List[Dict]]
Policy = Tuple[str, Sequence[str], str]


def _policies(judges: Sequence[str]) -> List[Policy]:
    """Baselines from VLM judge benchmarks and Trust or Escalate."""
    policies: List[Policy] = []
    # Standard VLM-as-a-judge: every model decides every pair.
    policies.extend((f"direct:{judge}", [judge], "no_selection") for judge in judges)
    # Selective baselines from the original cascade paper.
    policies.append((f"heuristic:{judges[-1]}", [judges[-1]], "heuristic_selection"))
    policies.append(("cascaded_heuristic", list(judges), "heuristic_selection"))
    policies.extend(
        (f"point_estimate:{judge}", [judge], "point_estimate_calibration")
        for judge in judges
    )
    # Proposed method, included here so every number uses identical splits.
    policies.append(("cascaded_selective", list(judges), "fixed_sequence_testing"))
    return policies


def _run_policy(config, calibration: Judgements, test: Judgements,
                judges: Sequence[str], threshold_method: str):
    classifier = OpenCascadedClassifier(
        judges,
        calibration_samples={judge: calibration[judge] for judge in judges},
        alpha=config.cascade.alpha,
        delta=config.cascade.delta,
        split_delta=config.cascade.split_delta,
        threshold_method=threshold_method,
    )
    return classifier.apply_decision_rule({judge: test[judge] for judge in judges})


def _source_metrics(result) -> Dict:
    sources = [sample.get("source", {}).get("dataset", "unknown") for sample in result.samples]
    return metrics.grouped_agreement(
        result.predictions, result.labels, result.evaluators, sources)


def _aggregate_policy(rows: List[Dict]) -> Dict:
    per_source = defaultdict(lambda: {"agreement": [], "coverage": []})
    for row in rows:
        for source, values in row["per_source"].items():
            if values["agreement"] is not None:
                per_source[source]["agreement"].append(values["agreement"])
            per_source[source]["coverage"].append(values["coverage"])

    def stats(field: str) -> Dict[str, float]:
        values = [row[field] for row in rows if row[field] is not None]
        if not values:
            return {"mean": None, "std": None}
        return {"mean": mean(values), "std": pstdev(values)}

    return {
        "guarantee_success_rate": mean(row["met_target"] for row in rows),
        "agreement": stats("agreement"),
        "macro_source_agreement": stats("macro_source_agreement"),
        "coverage": stats("coverage"),
        "macro_source_coverage": stats("macro_source_coverage"),
        "per_source": {
            source: {
                "mean_agreement": (mean(values["agreement"])
                                   if values["agreement"] else None),
                "mean_coverage": mean(values["coverage"]),
            }
            for source, values in sorted(per_source.items())
        },
        "splits": rows,
    }


def run_baseline_comparison(config, pool: Judgements, raw_rows=None) -> Dict:
    """Compare direct and selective judges on accuracy, macro accuracy and coverage."""
    if config.evaluate.n_splits <= 0:
        raise ValueError("evaluate.n_splits must be positive")
    indexed, components, inventory = align_pool(pool, config.judges, raw_rows)
    if config.data.calibration_size >= inventory["n_common"]:
        raise ValueError("calibration_size must be smaller than the common scored pool")

    policy_rows = {name: [] for name, _, _ in _policies(config.judges)}
    for number in range(config.evaluate.n_splits):
        seed = config.evaluate.seed + number
        cal_keys, test_keys = draw_keys(components, config.data.calibration_size, seed)
        calibration = {
            judge: [indexed[judge][key] for key in cal_keys] for judge in config.judges
        }
        test = {judge: [indexed[judge][key] for key in test_keys] for judge in config.judges}

        for name, judges, method in _policies(config.judges):
            result = _run_policy(config, calibration, test, judges, method)
            grouped = _source_metrics(result)
            policy_rows[name].append({
                "seed": seed,
                "n_calibration": len(cal_keys),
                "n_test": len(test_keys),
                "agreement": result.human_agreement,
                "macro_source_agreement": grouped["macro_agreement"],
                "coverage": result.coverage,
                "macro_source_coverage": grouped["macro_coverage"],
                "met_target": (result.coverage > 0
                               and result.human_agreement >= 1 - config.cascade.alpha),
                "lambda_hats": result.summary()["lambda_hats"],
                "composition": result.composition,
                "per_source": grouped["per_group"],
            })

    policies = {name: _aggregate_policy(rows) for name, rows in policy_rows.items()}
    print(f"{'policy':<45} {'accuracy':>9} {'macro':>9} {'coverage':>9} {'success':>9}")
    print("-" * 85)
    for name, summary in policies.items():
        macro = summary["macro_source_agreement"]["mean"]
        macro_text = f"{macro:.4f}" if macro is not None else "n/a"
        print(f"{name:<45} {summary['agreement']['mean']:>9.4f} "
              f"{macro_text:>9} "
              f"{summary['coverage']['mean']:>9.4f} "
              f"{summary['guarantee_success_rate']:>9.4f}")

    return {
        "alpha": config.cascade.alpha,
        "delta": config.cascade.delta,
        "target_agreement": 1 - config.cascade.alpha,
        "n_splits": config.evaluate.n_splits,
        **inventory,
        "policies": policies,
    }
