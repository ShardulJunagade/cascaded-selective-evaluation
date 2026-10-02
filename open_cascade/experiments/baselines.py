"""Fair policy baselines over the same cached VLM judgments and random splits."""
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from open_cascade import metrics
from open_cascade.cascade import OpenCascadedClassifier
from open_cascade.data import preferred_index
from open_cascade.experiments.multi_split import align_pool, draw_keys


Judgements = Dict[str, List[Dict]]
Policy = Tuple[str, Sequence[str], str]
SETUP_METHODS = ("vanilla_single", "position_swap", "prompt_order_vote",
                 "swap_consistency")


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


def _simulation_index(row: Dict) -> Dict[Tuple[int, int], List[float]]:
    """Index persisted per-run probabilities by (prompt/annotator, response order)."""
    simulations = (row.get("judge_details") or {}).get("simulations")
    if not isinstance(simulations, list):
        return {}
    indexed = {}
    for run in simulations:
        probs = run.get("probs", []) if isinstance(run, dict) else []
        if (len(probs) != 2 or "annotator" not in run or "ordering" not in run):
            continue
        try:
            key = (int(run["annotator"]), int(run["ordering"]))
            indexed[key] = [float(probs[0]), float(probs[1])]
        except (TypeError, ValueError):
            continue
    return indexed


def _setup_probs(row: Dict, method: str) -> Optional[List[float]]:
    """Derive one published judge-setup ablation from cached per-run signals."""
    runs = _simulation_index(row)
    annotators = sorted({annotator for annotator, _ in runs})
    if not annotators:
        return None
    first = annotators[0]
    original = runs.get((first, 0))
    swapped = runs.get((first, 1))

    if method == "vanilla_single":
        return original
    if method in ("position_swap", "swap_consistency"):
        if original is None or swapped is None:
            return None
        if (method == "swap_consistency"
                and original.index(max(original)) != swapped.index(max(swapped))):
            return None
        return [(original[0] + swapped[0]) / 2, (original[1] + swapped[1]) / 2]
    if method == "prompt_order_vote":
        votes = [probs.index(max(probs)) for probs in runs.values()]
        return [votes.count(0) / len(votes), votes.count(1) / len(votes)]
    raise KeyError(f"Unknown judge setup method: {method}")


def _setup_available(rows: Sequence[Dict]) -> bool:
    """A complete setup record needs both orderings for at least one prompt."""
    for row in rows:
        runs = _simulation_index(row)
        annotators = {annotator for annotator, _ in runs}
        if not any((annotator, 0) in runs and (annotator, 1) in runs
                   for annotator in annotators):
            return False
    return True


def _setup_result(test: Judgements, judges: Sequence[str], method: str,
                  cascade: bool = False) -> Dict:
    """Evaluate a setup heuristic, optionally escalating inconsistent pairs."""
    rows = test[judges[0]]
    labels = torch.tensor([preferred_index(row) - 1 for row in rows], dtype=torch.long)
    predictions = torch.full_like(labels, -1)
    evaluators = torch.full_like(labels, -1)

    if cascade:
        for judge_index, judge in enumerate(judges):
            for row_index, row in enumerate(test[judge]):
                if evaluators[row_index] >= 0:
                    continue
                # The last judge resolves every pair using its full prompt ensemble.
                probs = (row["probs"] if judge_index == len(judges) - 1
                         else _setup_probs(row, "swap_consistency"))
                if probs is not None:
                    predictions[row_index] = probs.index(max(probs))
                    evaluators[row_index] = judge_index
    else:
        judge = judges[0]
        for row_index, row in enumerate(test[judge]):
            probs = _setup_probs(row, method)
            if probs is not None:
                predictions[row_index] = probs.index(max(probs))
                evaluators[row_index] = 0

    sources = [row.get("source", {}).get("dataset", "unknown") for row in rows]
    grouped = metrics.grouped_agreement(predictions, labels, evaluators, sources)
    return {
        "agreement": metrics.human_agreement(predictions, labels, evaluators),
        "macro_source_agreement": grouped["macro_agreement"],
        "coverage": metrics.coverage(evaluators),
        "macro_source_coverage": grouped["macro_coverage"],
        "composition": metrics.evaluator_composition(evaluators, judges),
        "per_source": grouped["per_group"],
    }


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

    common_keys = {key for component in components for key in component}
    setup_available = {
        judge: _setup_available([indexed[judge][key] for key in common_keys])
        for judge in config.judges
    }
    setup_policy_names = [
        f"setup:{method}:{judge}"
        for judge in config.judges if setup_available[judge]
        for method in SETUP_METHODS
    ]
    if len(config.judges) > 1 and all(setup_available[judge] for judge in config.judges[:-1]):
        setup_policy_names.append("setup:consistency_escalation")

    policy_rows = {name: [] for name, _, _ in _policies(config.judges)}
    policy_rows.update({name: [] for name in setup_policy_names})
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

        for judge in config.judges:
            if not setup_available[judge]:
                continue
            for method in SETUP_METHODS:
                name = f"setup:{method}:{judge}"
                values = _setup_result(test, [judge], method)
                policy_rows[name].append({
                    "seed": seed,
                    "n_calibration": len(cal_keys),
                    "n_test": len(test_keys),
                    **values,
                    "met_target": (values["coverage"] > 0 and values["agreement"]
                                   >= 1 - config.cascade.alpha),
                    "lambda_hats": [],
                })

        escalation_name = "setup:consistency_escalation"
        if escalation_name in policy_rows:
            values = _setup_result(test, config.judges, "swap_consistency", cascade=True)
            policy_rows[escalation_name].append({
                "seed": seed,
                "n_calibration": len(cal_keys),
                "n_test": len(test_keys),
                **values,
                "met_target": (values["coverage"] > 0 and values["agreement"]
                               >= 1 - config.cascade.alpha),
                "lambda_hats": [],
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
        "judge_setup_baselines": {
            "status": ("available" if all(setup_available.values()) else
                       "partial" if any(setup_available.values()) else "unavailable"),
            "available_by_judge": setup_available,
            "rescore_required": [judge for judge, available in setup_available.items()
                                 if not available],
            "note": ("Cached rows need judge_details.simulations. Re-run scoring for judges "
                     "listed in rescore_required; older aggregate-only caches remain valid for "
                     "the model and selective-policy baselines."),
        },
        "policies": policies,
    }
