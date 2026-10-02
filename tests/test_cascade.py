"""Regression tests: the cascade on the released judgements in ./result/ must keep giving
the numbers the pipeline produced before the harness refactor. If one of these fails after a
change to cascade.py / thresholds.py / metrics.py, the change altered the results."""
import pytest

from open_cascade.cascade import OpenCascadedClassifier
from open_cascade.data import load_judgements

PAPER_JUDGES = ["mistral-7b-instruct", "gpt-3.5-turbo", "gpt-4-turbo"]


@pytest.fixture(scope="module")
def judgements():
    return (load_judgements(PAPER_JUDGES, "calibration", "./result"),
            load_judgements(PAPER_JUDGES, "test", "./result"))


def run(judgements, judges, alpha=0.15, split_delta=True):
    calibration, test = judgements
    classifier = OpenCascadedClassifier(judges, calibration, alpha=alpha, delta=0.1,
                                        split_delta=split_delta)
    return classifier.apply_decision_rule(test)


def test_paper_cascade_split_delta(judgements):
    result = run(judgements, PAPER_JUDGES)
    assert result.summary()["lambda_hats"] == [0.9998, 0.834, 0.9998]
    assert result.human_agreement == pytest.approx(0.8843, abs=5e-5)
    assert result.coverage == pytest.approx(0.5606, abs=5e-5)
    composition = result.composition
    assert composition["mistral-7b-instruct"] == pytest.approx(0.1966, abs=5e-5)
    assert composition["gpt-3.5-turbo"] == pytest.approx(0.7161, abs=5e-5)
    assert composition["gpt-4-turbo"] == pytest.approx(0.0873, abs=5e-5)


def test_paper_cascade_full_delta(judgements):
    result = run(judgements, PAPER_JUDGES, split_delta=False)
    assert result.summary()["lambda_hats"] == [0.9998, 0.7928, 0.9996]
    assert result.human_agreement == pytest.approx(0.8735, abs=5e-5)
    assert result.coverage == pytest.approx(0.6117, abs=5e-5)


def test_two_judge_subset(judgements):
    result = run(judgements, ["mistral-7b-instruct", "gpt-4-turbo"], alpha=0.1)
    assert result.summary()["lambda_hats"] == [0.9998, 0.9958]
    assert result.human_agreement == pytest.approx(0.9157, abs=5e-5)
    assert result.coverage == pytest.approx(0.5254, abs=5e-5)


def test_missing_judge_is_reported(judgements):
    calibration, _ = judgements
    with pytest.raises(KeyError, match="qwen"):
        OpenCascadedClassifier(["mistral-7b-instruct", "qwen2.5-7b-instruct"], calibration,
                               alpha=0.15, delta=0.1)


def test_unknown_threshold_method(judgements):
    calibration, _ = judgements
    with pytest.raises(KeyError, match="Unknown threshold method"):
        OpenCascadedClassifier(PAPER_JUDGES, calibration, alpha=0.15, delta=0.1,
                               threshold_method="does_not_exist")
