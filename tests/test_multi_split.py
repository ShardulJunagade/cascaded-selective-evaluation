"""Repeated evaluation must keep image/prompt components intact and align judges."""
from types import SimpleNamespace

import pytest

from open_cascade.config import config_from_dict
from open_cascade.data import sample_key
from open_cascade.experiments import multi_split


def row(i, group=None, image=None, phash=None):
    return {
        "instruction": f"question {i}",
        "outputs": [f"a{i}", f"b{i}"],
        "preferences": {"worker": 1 + i % 2},
        "image_path": image or f"images/{i}.jpg",
        "source": {"group_id": group or f"g{i}", "media_hash": [phash or f"h{i}"],
                   "dataset": "source_a" if i % 2 else "source_b"},
        "probs": [0.8, 0.2],
    }


def config(n_splits=3):
    return config_from_dict({
        "name": "test", "judges": ["small", "large"], "result_dir": "unused",
        "data": {"builder": "prepared_export", "out_dir": "unused", "calibration_size": 2},
        "evaluate": {"n_splits": n_splits, "seed": 17},
    })


def test_draws_keep_transitive_image_and_prompt_groups_together():
    rows = [row(0, group="shared"), row(1, group="shared", phash="other"),
            row(2, phash="other"), row(3), row(4), row(5)]
    groups = multi_split._components(rows)
    assert sorted(map(len, groups)) == [1, 1, 1, 3]
    for seed in range(20):
        cal, test = multi_split.draw_keys(groups, 2, seed)
        assert not set(cal) & set(test)
        assert len(cal) + len(test) == len(rows)
        related = {sample_key(r) for r in rows[:3]}
        assert related <= set(cal) or related <= set(test)
        assert multi_split.draw_keys(groups, 2, seed) == (cal, test)


def test_missing_judge_rows_are_excluded_and_mismatched_labels_fail():
    rows = [row(i) for i in range(5)]
    pool = {"small": rows, "large": rows[:-1]}
    _, _, inventory = multi_split.align_pool(pool, ["small", "large"])
    assert inventory["n_common"] == 4
    assert inventory["excluded_by_judge"] == {"small": 1, "large": 0}
    pool["large"] = [dict(r) for r in rows]
    pool["large"][0] = dict(rows[0], preferences={"worker": 2})
    with pytest.raises(ValueError, match="Human labels differ"):
        multi_split.align_pool(pool, ["small", "large"])


def test_unscored_bridge_still_keeps_scored_rows_together():
    raw = [row(0, group="left", phash="first"),
           row(1, group="left", phash="second"),
           row(2, group="right", phash="second"), row(3)]
    pool = {"small": raw, "large": [raw[0], raw[2], raw[3]]}
    _, groups, _ = multi_split.align_pool(pool, ["small", "large"], raw)
    assert any({sample_key(raw[0]), sample_key(raw[2])} <= set(group) for group in groups)


def test_guarantee_success_uses_same_split_for_both_judges(monkeypatch):
    rows = [row(i) for i in range(7)]
    pool = {"small": rows, "large": [dict(r) for r in reversed(rows)]}
    seen = []

    def evaluate(cfg, calibration, test, alpha):
        cal_keys = {sample_key(r) for r in calibration["small"]}
        test_keys = {sample_key(r) for r in test["small"]}
        assert cal_keys == {sample_key(r) for r in calibration["large"]}
        assert test_keys == {sample_key(r) for r in test["large"]}
        assert not cal_keys & test_keys
        seen.append(cal_keys)
        return SimpleNamespace(coverage=0.5, human_agreement=0.9,
                               summary=lambda: {"lambda_hats": [0.7, 0.8]})

    monkeypatch.setattr(multi_split, "calibrate_and_evaluate", evaluate)
    result = multi_split.run_guarantee_success(config(), pool)
    assert len(seen) == 3
    assert result["guarantee_success_rate"] == 1.0
    assert result["mean_coverage"] == 0.5
    assert [r["seed"] for r in result["splits"]] == [17, 18, 19]


def test_older_exports_recover_phash_from_image(monkeypatch):
    rows = [row(0), row(1)]
    for entry in rows:
        del entry["source"]["media_hash"]
    monkeypatch.setattr(multi_split, "_image_phash", lambda path: "same")
    assert len(multi_split._components(rows)) == 1


def test_stale_cache_rows_and_labels_are_rejected():
    raw = [row(0), row(1)]
    pool = {"small": [dict(r) for r in raw], "large": [dict(r) for r in raw]}
    multi_split.validate_scored_pool(raw, pool, ["small", "large"])
    pool["large"].append(row(2))
    with pytest.raises(ValueError, match="outside the current eval_pool"):
        multi_split.validate_scored_pool(raw, pool, ["small", "large"])
    pool["large"].pop()
    pool["large"][0] = dict(raw[0], preferences={"worker": 2})
    with pytest.raises(ValueError, match="disagrees"):
        multi_split.validate_scored_pool(raw, pool, ["small", "large"])


def test_baseline_comparison_reports_common_metrics():
    from open_cascade.experiments.baselines import run_baseline_comparison

    rows = [row(i) for i in range(42)]
    # Make the two judges disagree so their direct baselines are distinguishable.
    pool = {
        "small": [dict(r, probs=[0.8, 0.2]) for r in rows],
        "large": [dict(r, probs=[0.2, 0.8]) for r in rows],
    }
    cfg = config(n_splits=1)
    cfg.data.calibration_size = 31
    result = run_baseline_comparison(cfg, pool, rows)
    assert result["n_splits"] == 1
    assert set(result["policies"]) == {
        "direct:small", "direct:large", "heuristic:large", "cascaded_heuristic",
        "point_estimate:small", "point_estimate:large", "cascaded_selective",
    }
    direct = result["policies"]["direct:large"]
    assert direct["coverage"]["mean"] == 1.0
    assert direct["macro_source_agreement"]["mean"] is not None
    assert set(direct["per_source"]) == {"source_a", "source_b"}
