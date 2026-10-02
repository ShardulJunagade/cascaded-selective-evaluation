"""Every shipped config must load and only refer to things that exist."""
from pathlib import Path

import pytest

from open_cascade.builders import BUILDERS
from open_cascade.config import apply_override, config_from_dict, load_config
from open_cascade.experiments import EXPERIMENTS
from open_cascade.registry import JUDGE_REGISTRY
from open_cascade.thresholds import THRESHOLD_METHODS

CONFIGS = sorted(Path("configs").glob("*.yaml"))


def test_configs_exist():
    assert CONFIGS, "no configs found -- run pytest from the repo root"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_is_valid(path):
    config = load_config(path)

    assert config.name == path.stem, "keep `name` equal to the file name"
    assert config.data.builder in BUILDERS
    assert config.cascade.threshold_method in THRESHOLD_METHODS
    for experiment in config.evaluate.experiments:
        assert experiment in EXPERIMENTS

    # a judge must be scoreable, or (evaluation-only) already have released judgements
    for judge in config.judges:
        assert (judge in JUDGE_REGISTRY
                or config.result_file(judge, "calibration").exists()), judge


def test_override_parses_yaml_values():
    raw = {"cascade": {"alpha": 0.15}}
    apply_override(raw, "cascade.alpha=0.1")
    apply_override(raw, "cascade.split_delta=false")
    apply_override(raw, "judges=[a, b]")
    assert raw == {"cascade": {"alpha": 0.1, "split_delta": False}, "judges": ["a", "b"]}


def test_unknown_key_is_an_error():
    raw = {"name": "x", "judges": ["a"], "result_dir": "r",
           "data": {"builder": "rlhf_v", "out_dir": "d"},
           "cascade": {"alhpa": 0.1}}
    with pytest.raises(ValueError, match="alhpa"):
        config_from_dict(raw)


def test_missing_data_section_is_an_error():
    with pytest.raises(ValueError, match="data"):
        config_from_dict({"name": "x", "judges": ["a"], "result_dir": "r"})
