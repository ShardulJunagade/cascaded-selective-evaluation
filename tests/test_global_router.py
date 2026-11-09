"""End-to-end routing uses three leakage-safe splits and cached judge signals."""
import numpy as np
import pytest

from open_cascade.config import config_from_dict
from open_cascade.experiments.global_router import (
    _risk_upper,
    _threshold_grid,
    draw_three_way_keys,
    run_global_router_comparison,
)


def _row(index, correct=True, source="source_a"):
    label = index % 2
    prediction = label if correct else 1 - label
    probs = [0.1, 0.1]
    probs[prediction] = 0.9
    support = 0.9 if correct else 0.1
    run_probs = [1.0 - support, support] if prediction == 1 else [support, 1.0 - support]
    return {
        "instruction": f"question {index}",
        "outputs": [f"a{index}", f"b{index}"],
        "preferences": {"worker": label + 1},
        "image_path": f"images/{index}.jpg",
        "source": {"group_id": f"g{index}", "media_hash": [f"h{index}"],
                   "dataset": source},
        "probs": probs,
        "judge_details": {"simulations": [
            {"annotator": 0, "ordering": 0, "probs": run_probs},
            {"annotator": 0, "ordering": 1, "probs": run_probs},
        ]},
    }


def _config():
    return config_from_dict({
        "name": "router_test", "judges": ["small", "large"],
        "result_dir": "unused",
        "data": {"builder": "prepared_export", "out_dir": "unused",
                 "calibration_size": 30},
        "cascade": {"alpha": 0.15, "delta": 0.1},
        "evaluate": {"n_splits": 2, "seed": 7, "router_train_size": 30,
                     "router_min_selected": 20, "router_thresholds": 21},
    })


def test_three_way_draw_keeps_every_component_in_one_partition():
    components = [[f"k{number}.a", f"k{number}.b"] for number in range(20)]
    train, calibration, test = draw_three_way_keys(components, 10, 12, seed=4)
    partitions = [set(train), set(calibration), set(test)]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2]
                or partitions[1] & partitions[2])
    assert sum(map(len, partitions)) == 40
    for component in components:
        assert sum(set(component) <= partition for partition in partitions) == 1


def test_exact_upper_bound_and_policy_only_threshold_grid():
    assert _risk_upper(0, 30, 0.1) == pytest.approx(1 - 0.1 ** (1 / 30))
    training = np.asarray([[0.2, 0.8], [0.4, 0.6]])
    assert _threshold_grid(training, 3).tolist() == pytest.approx([0.8, 0.5, 0.2])


def test_learned_router_uses_cached_signals_and_can_improve_coverage():
    raw = [_row(i, source="source_a" if i % 2 else "source_b") for i in range(120)]
    # The first judge is confidently wrong on half the rows. The second judge is
    # always correct, so a correctness-aware sequential router can safely defer.
    pool = {
        "small": [_row(i, correct=i % 2 == 0,
                       source="source_a" if i % 2 else "source_b")
                  for i in range(120)],
        "large": [_row(i, correct=True,
                       source="source_a" if i % 2 else "source_b")
                  for i in range(120)],
    }
    result = run_global_router_comparison(_config(), pool, raw)

    assert result["uses_cached_judgements_only"] is True
    assert result["new_model_inference_required"] is False
    assert set(result["methods"]) == {"global_raw_confidence", "global_learned_router"}
    learned = result["methods"]["global_learned_router"]
    assert learned["coverage"]["mean"] == 1.0
    assert learned["agreement"]["mean"] == 1.0
    assert learned["certified_split_rate"] == 1.0
    assert learned["mean_judges_called"]["mean"] >= 1.0
    assert set(learned["per_source"]) == {"source_a", "source_b"}
    assert all(split["n_router_train"] >= 30 for split in learned["splits"])
    assert sum(len(rows) for rows in pool.values()) == 240  # inputs were not changed
