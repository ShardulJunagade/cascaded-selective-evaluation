"""End-to-end certified routing over cached multimodal judge signals.

The original cascade certifies every stage separately and divides delta across
judges. This experiment learns a frozen sequential router on one group-safe
split, calibrates one global acceptance threshold on a second split using the
full delta, and evaluates on a third split. No judge is called by this module.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit
from scipy.stats import beta
import torch

from open_cascade import metrics
from open_cascade.data import preferred_index
from open_cascade.experiments.multi_split import align_pool


@dataclass
class _LogisticScore:
    location: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    constant: Optional[float] = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(features.shape[0], self.constant, dtype=float)
        standard = (features - self.location) / self.scale
        design = np.column_stack([np.ones(features.shape[0]), standard])
        return expit(design @ self.weights)


def _fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float) -> _LogisticScore:
    """Small deterministic correctness model; keeps the CPU experiment self-contained."""
    location = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    unique = np.unique(labels)
    if len(unique) == 1:
        return _LogisticScore(location, scale, np.zeros(features.shape[1] + 1),
                              constant=float(unique[0]))

    standard = (features - location) / scale
    design = np.column_stack([np.ones(features.shape[0]), standard])
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * l2
    penalty[0, 0] = 0.0
    n = max(1, len(labels))
    for _ in range(40):
        probabilities = expit(design @ weights)
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-5, None)
        gradient = design.T @ (probabilities - labels) / n + penalty @ weights
        hessian = design.T @ (variance[:, None] * design) / n + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return _LogisticScore(location, scale, weights)


def _binary_entropy(probability: float) -> float:
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(probability * math.log2(probability)
             + (1.0 - probability) * math.log2(1.0 - probability))


def _judge_signal(row: Dict) -> Tuple[int, float, List[float]]:
    """Prediction, confidence and reliability features from already cached runs."""
    probs = np.asarray(row["probs"], dtype=float)
    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])
    runs = (row.get("judge_details") or {}).get("simulations") or []
    valid = [run for run in runs
             if isinstance(run, dict) and len(run.get("probs", [])) == 2]

    if valid:
        run_probs = np.asarray([run["probs"] for run in valid], dtype=float)
        support = float(np.mean(np.argmax(run_probs, axis=1) == prediction))
        predicted_probs = run_probs[:, prediction]
        dispersion = float(np.std(predicted_probs))
        by_annotator = defaultdict(dict)
        for run in valid:
            if "annotator" in run and "ordering" in run:
                by_annotator[int(run["annotator"])][int(run["ordering"])] = run["probs"]
        consistent, gaps = [], []
        for orders in by_annotator.values():
            if 0 not in orders or 1 not in orders:
                continue
            first, second = np.asarray(orders[0]), np.asarray(orders[1])
            consistent.append(int(np.argmax(first) == np.argmax(second)))
            gaps.append(abs(float(first[prediction] - second[prediction])))
        swap_consistency = float(np.mean(consistent)) if consistent else 0.5
        order_gap = float(np.mean(gaps)) if gaps else 0.5
    else:
        support, dispersion, swap_consistency, order_gap = 0.5, 0.5, 0.5, 0.5

    features = [
        confidence,
        abs(float(probs[0] - probs[1])),
        _binary_entropy(float(probs[0])),
        support,
        dispersion,
        swap_consistency,
        order_gap,
    ]
    return prediction, confidence, features


def _cached_arrays(indexed: Dict[str, Dict[str, Dict]], judges: Sequence[str],
                   keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_judges = len(keys), len(judges)
    predictions = np.zeros((n_rows, n_judges), dtype=int)
    confidence = np.zeros((n_rows, n_judges), dtype=float)
    base_features = np.zeros((n_rows, n_judges, 7), dtype=float)
    labels = np.zeros(n_rows, dtype=int)
    for row_index, key in enumerate(keys):
        labels[row_index] = preferred_index(indexed[judges[0]][key]) - 1
        for judge_index, judge in enumerate(judges):
            prediction, conf, features = _judge_signal(indexed[judge][key])
            predictions[row_index, judge_index] = prediction
            confidence[row_index, judge_index] = conf
            base_features[row_index, judge_index] = features

    # Each stage sees its own stability plus agreement/confidence from judges
    # already evaluated. No future-model feature leaks into an earlier route.
    stage_features = np.zeros((n_rows, n_judges, 11), dtype=float)
    for stage in range(n_judges):
        stage_features[:, stage, :7] = base_features[:, stage]
        seen_predictions = predictions[:, :stage + 1]
        agreement = np.mean(seen_predictions == predictions[:, [stage]], axis=1)
        stage_features[:, stage, 7] = agreement
        stage_features[:, stage, 8] = np.all(
            seen_predictions == seen_predictions[:, [0]], axis=1).astype(float)
        if stage:
            stage_features[:, stage, 9] = np.max(confidence[:, :stage], axis=1)
            stage_features[:, stage, 10] = np.mean(confidence[:, :stage], axis=1)
    return predictions, confidence, stage_features, labels


def draw_three_way_keys(components: List[List[str]], train_size: int,
                        calibration_size: int, seed: int) -> Tuple[List[str], List[str], List[str]]:
    total = sum(map(len, components))
    if train_size <= 0 or calibration_size <= 0 or train_size + calibration_size >= total:
        raise ValueError("router train and calibration sizes must be positive and leave test rows")
    groups = list(components)
    random.Random(seed).shuffle(groups)
    train, calibration, test = [], [], []
    for index, group in enumerate(groups):
        remaining_groups = len(groups) - index - 1
        if len(train) < train_size and remaining_groups >= 2:
            train.extend(group)
        elif len(calibration) < calibration_size and remaining_groups >= 1:
            calibration.extend(group)
        else:
            test.extend(group)
    if not train or not calibration or not test:
        raise ValueError("Cannot construct nonempty group-safe router/train/test partitions")
    return train, calibration, test


def _route(scores: np.ndarray, predictions: np.ndarray,
           threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_judges = scores.shape
    chosen = np.full(n_rows, -1, dtype=int)
    output = np.full(n_rows, -1, dtype=int)
    for stage in range(n_judges):
        selected = (chosen < 0) & (scores[:, stage] >= threshold)
        chosen[selected] = stage
        output[selected] = predictions[selected, stage]
    return output, chosen


def _risk_upper(errors: int, selected: int, delta: float) -> float:
    if selected <= 0:
        return 1.0
    if errors >= selected:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, selected - errors))


def _threshold_grid(training_scores: np.ndarray, count: int) -> np.ndarray:
    """Freeze candidate thresholds without inspecting certification examples."""
    if count < 2:
        raise ValueError("evaluate.router_thresholds must be at least 2")
    quantiles = np.linspace(1.0, 0.0, count)
    return np.unique(np.quantile(training_scores.reshape(-1), quantiles))[::-1]


def _calibrate_global(scores: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
                      candidates: np.ndarray, alpha: float, delta: float, min_selected: int
                      ) -> Tuple[Optional[float], Dict]:
    last_passing = None
    diagnostic = {"selected": 0, "errors": 0, "risk_upper": 1.0}
    started = False
    for threshold in candidates:
        output, chosen = _route(scores, predictions, float(threshold))
        selected = chosen >= 0
        count = int(selected.sum())
        if count < min_selected:
            continue
        errors = int(np.sum(output[selected] != labels[selected]))
        upper = _risk_upper(errors, count, delta)
        if not started:
            started = True
            if upper > alpha:
                return None, {"selected": count, "errors": errors, "risk_upper": upper}
        if upper > alpha:
            break
        last_passing = float(threshold)
        diagnostic = {"selected": count, "errors": errors, "risk_upper": upper}
    return last_passing, diagnostic


def _evaluate(scores: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
              sources: Sequence[str], threshold: Optional[float],
              judges: Sequence[str]) -> Dict:
    if threshold is None:
        output = np.full(len(labels), -1, dtype=int)
        chosen = np.full(len(labels), -1, dtype=int)
    else:
        output, chosen = _route(scores, predictions, threshold)
    selected = chosen >= 0
    count = int(selected.sum())
    agreement = (float(np.mean(output[selected] == labels[selected])) if count else None)
    composition = {
        judge: (float(np.mean(chosen[selected] == index)) if count else 0.0)
        for index, judge in enumerate(judges)
    }
    grouped = metrics.grouped_agreement(
        torch.as_tensor(output), torch.as_tensor(labels), torch.as_tensor(chosen), sources)
    calls = np.where(selected, chosen + 1, len(judges))
    return {
        "agreement": agreement,
        "macro_source_agreement": grouped["macro_agreement"],
        "coverage": count / len(labels),
        "macro_source_coverage": grouped["macro_coverage"],
        "n_test": len(labels),
        "n_selected": count,
        "mean_judges_called": float(np.mean(calls)),
        "early_exit_rate": float(np.mean(selected & (chosen < len(judges) - 1))),
        "composition": composition,
        "per_source": grouped["per_group"],
    }


def _summary(rows: List[Dict]) -> Dict:
    def stats(field: str) -> Dict:
        values = [row[field] for row in rows if row[field] is not None]
        return ({"mean": mean(values), "std": pstdev(values)} if values
                else {"mean": None, "std": None})

    composition = {}
    per_source = defaultdict(lambda: {"agreement": [], "coverage": []})
    for row in rows:
        for source, values in row["per_source"].items():
            if values["agreement"] is not None:
                per_source[source]["agreement"].append(values["agreement"])
            per_source[source]["coverage"].append(values["coverage"])
    if rows:
        for judge in rows[0]["composition"]:
            composition[judge] = mean(row["composition"][judge] for row in rows)
    return {
        "agreement": stats("agreement"),
        "macro_source_agreement": stats("macro_source_agreement"),
        "coverage": stats("coverage"),
        "macro_source_coverage": stats("macro_source_coverage"),
        "mean_judges_called": stats("mean_judges_called"),
        "early_exit_rate": stats("early_exit_rate"),
        "nonempty_split_rate": mean(row["coverage"] > 0 for row in rows),
        "guarantee_success_rate": mean(row["met_target"] for row in rows),
        "certified_split_rate": mean(row["threshold"] is not None for row in rows),
        "mean_composition": composition,
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


def run_global_router_comparison(config, pool: Dict[str, List[Dict]], raw_rows=None) -> Dict:
    """Compare raw and learned end-to-end certified routers using cached judge runs."""
    if config.evaluate.n_splits <= 0:
        raise ValueError("evaluate.n_splits must be positive")
    if config.evaluate.router_l2 < 0:
        raise ValueError("evaluate.router_l2 must be nonnegative")
    if config.evaluate.router_min_selected <= 0:
        raise ValueError("evaluate.router_min_selected must be positive")
    indexed, components, inventory = align_pool(pool, config.judges, raw_rows)
    keys = [key for component in components for key in component]
    key_index = {key: index for index, key in enumerate(keys)}
    predictions, confidence, features, labels = _cached_arrays(indexed, config.judges, keys)
    sources = np.asarray([
        indexed[config.judges[0]][key].get("source", {}).get("dataset", "unknown")
        for key in keys
    ])
    train_size = config.evaluate.router_train_size
    calibration_size = config.data.calibration_size
    methods = {"global_raw_confidence": [], "global_learned_router": []}

    for number in range(config.evaluate.n_splits):
        seed = config.evaluate.seed + number
        train_keys, calibration_keys, test_keys = draw_three_way_keys(
            components, train_size, calibration_size, seed)
        train = np.asarray([key_index[key] for key in train_keys], dtype=int)
        calibration = np.asarray([key_index[key] for key in calibration_keys], dtype=int)
        test = np.asarray([key_index[key] for key in test_keys], dtype=int)

        learned_scores = np.zeros_like(confidence)
        for stage in range(len(config.judges)):
            correctness = (predictions[train, stage] == labels[train]).astype(float)
            model = _fit_logistic(features[train, stage], correctness,
                                  config.evaluate.router_l2)
            learned_scores[:, stage] = model.predict(features[:, stage])

        for method, all_scores in (
                ("global_raw_confidence", confidence),
                ("global_learned_router", learned_scores)):
            candidates = _threshold_grid(
                all_scores[train], config.evaluate.router_thresholds)
            threshold, certification = _calibrate_global(
                all_scores[calibration], predictions[calibration], labels[calibration],
                candidates,
                config.cascade.alpha, config.cascade.delta,
                config.evaluate.router_min_selected)
            values = _evaluate(
                all_scores[test], predictions[test], labels[test], sources[test],
                threshold, config.judges)
            methods[method].append({
                "seed": seed,
                "n_router_train": len(train),
                "n_calibration": len(calibration),
                "threshold": threshold,
                "certification": certification,
                **values,
                "met_target": (values["coverage"] > 0
                               and values["agreement"] >= 1 - config.cascade.alpha),
            })

    summaries = {method: _summary(rows) for method, rows in methods.items()}
    print(f"{'method':<30} {'accuracy':>9} {'macro':>9} {'coverage':>9} "
          f"{'calls':>7} {'nonempty':>9} {'certified':>9} {'success':>9}")
    print("-" * 97)
    for method, summary in summaries.items():
        accuracy = summary["agreement"]["mean"]
        accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "n/a"
        macro = summary["macro_source_agreement"]["mean"]
        macro_text = f"{macro:.4f}" if macro is not None else "n/a"
        print(f"{method:<30} {accuracy_text:>9} "
              f"{macro_text:>9} "
              f"{summary['coverage']['mean']:>9.4f} "
              f"{summary['mean_judges_called']['mean']:>7.3f} "
              f"{summary['nonempty_split_rate']:>9.4f} "
              f"{summary['certified_split_rate']:>9.4f} "
              f"{summary['guarantee_success_rate']:>9.4f}")

    return {
        "alpha": config.cascade.alpha,
        "delta": config.cascade.delta,
        "target_agreement": 1 - config.cascade.alpha,
        "n_splits": config.evaluate.n_splits,
        "router_train_size": train_size,
        "calibration_size": calibration_size,
        "router_thresholds": config.evaluate.router_thresholds,
        "certificate_scope": "end_to_end_routing_policy",
        "threshold_candidates_source": "router_training_split_quantiles",
        "uses_cached_judgements_only": True,
        "new_model_inference_required": False,
        "feature_names": [
            "confidence", "margin", "entropy", "simulation_vote_support",
            "simulation_probability_std", "swap_consistency", "swap_probability_gap",
            "seen_model_agreement", "seen_model_unanimity",
            "max_previous_confidence", "mean_previous_confidence",
        ],
        **inventory,
        "methods": summaries,
    }
