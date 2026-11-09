"""Regression tests for safe calibration on the released judgements in ./result/."""
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
    assert result.summary()["lambda_hats"] == [1.0001, 0.834, 1.0001]
    assert result.human_agreement == pytest.approx(0.88874, abs=5e-5)
    assert result.coverage == pytest.approx(0.48389, abs=5e-5)
    composition = result.composition
    assert composition["mistral-7b-instruct"] == 0.0
    assert composition["gpt-3.5-turbo"] == 1.0
    assert composition["gpt-4-turbo"] == 0.0


def test_paper_cascade_full_delta(judgements):
    result = run(judgements, PAPER_JUDGES, split_delta=False)
    assert result.summary()["lambda_hats"] == [1.0001, 0.7928, 1.0001]
    assert result.human_agreement == pytest.approx(0.87475, abs=5e-5)
    assert result.coverage == pytest.approx(0.52798, abs=5e-5)


def test_two_judge_subset(judgements):
    result = run(judgements, ["mistral-7b-instruct", "gpt-4-turbo"], alpha=0.1)
    assert result.summary()["lambda_hats"] == [1.0001, 1.0001]
    assert result.human_agreement == 1.0
    assert result.coverage == 0.0


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


def test_uncertifiable_stage_abstains_instead_of_reusing_threshold(judgements, monkeypatch):
    from open_cascade import cascade

    monkeypatch.setitem(cascade.THRESHOLD_METHODS, "never_certified",
                        lambda phats, yhats, labels, alpha, delta: None)
    calibration, test = judgements
    classifier = OpenCascadedClassifier(
        PAPER_JUDGES[:2], calibration, alpha=0.15, delta=0.1,
        threshold_method="never_certified")
    result = classifier.apply_decision_rule(test)

    assert classifier.lambda_hats == [1.0001, 1.0001]
    assert result.coverage == 0.0
