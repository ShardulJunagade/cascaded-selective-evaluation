"""Data helpers and dataset builders."""
from typing import Dict, List

import pytest

from cascaded_evaluation.util import merge_data
from open_cascade.builders.base import DatasetBuilder
from open_cascade.builders.released_text import ReleasedTextBuilder
from open_cascade.config import DataConfig
from open_cascade.data import load_fewshot, make_splits, read_jsonl, sample_key, write_jsonl


def fake_instances(n: int) -> List[Dict]:
    return [{"instruction": f"q{i}", "outputs": [f"a{i}", f"b{i}"],
             "preferences": {"human": 1 + i % 2}} for i in range(n)]


def test_jsonl_roundtrip(tmp_path):
    samples = fake_instances(3)
    path = tmp_path / "x.jsonl"
    write_jsonl(samples[:2], path)
    write_jsonl(samples[2:], path, mode="a")
    assert read_jsonl(path) == samples


def test_make_splits_is_deterministic_and_disjoint():
    samples = fake_instances(100)
    fewshot, calibration, test = make_splits(samples, n_annotators=3, k_shot=2,
                                             calibration_size=30, seed=7)
    assert [len(s) for s in fewshot] == [2, 2, 2]
    assert len(calibration) == 30 and len(test) == 70
    assert not {sample_key(s) for s in calibration} & {sample_key(s) for s in test}
    assert make_splits(samples, 3, 2, 30, seed=7) == (fewshot, calibration, test)
    assert samples == fake_instances(100), "input list must not be shuffled in place"


def test_make_splits_needs_enough_samples():
    with pytest.raises(ValueError, match="at least"):
        make_splits(fake_instances(10), n_annotators=3, k_shot=3, calibration_size=5, seed=0)


def test_sample_key_matches_merge_data():
    """Resume and validation use sample_key; the cascade uses merge_data. They must agree."""
    samples = fake_instances(5)
    samples[0]["image_path"] = "img/0.jpg"
    judged = [dict(s, probs=[0.6, 0.4]) for s in samples]
    merged = merge_data({"j1": judged, "j2": judged}, ["j1", "j2"])
    assert len(merged) == 5
    assert {sample_key(s) for s in merged} == {sample_key(s) for s in samples}


class InMemoryBuilder(DatasetBuilder):
    def load_instances(self):
        return fake_instances(40)


def test_builder_writes_all_files_and_refuses_to_overwrite(tmp_path):
    config = DataConfig(builder="memory", out_dir=str(tmp_path), n_annotators=2, k_shot=2,
                        calibration_size=10)
    InMemoryBuilder(config).build()

    assert len(load_fewshot(config.fewshot_file)) == 2
    assert len(read_jsonl(config.split_file("calibration"))) == 10
    assert len(read_jsonl(config.split_file("test"))) == 30
    assert len(read_jsonl(tmp_path / "preprocessed" / "data.jsonl")) == 40

    with pytest.raises(FileExistsError):
        InMemoryBuilder(config).build()


def test_released_text_builder_keeps_original_split(tmp_path):
    config = DataConfig(builder="released_text", out_dir=str(tmp_path))
    ReleasedTextBuilder(config).build()

    calibration = read_jsonl(config.split_file("calibration"))
    test = read_jsonl(config.split_file("test"))
    assert (len(calibration), len(test)) == (500, 4718)
    assert "probs" not in calibration[0]
    released = read_jsonl("./result/gpt-4-turbo.calibration.jsonl")
    assert [sample_key(s) for s in calibration] == [sample_key(s) for s in released]
