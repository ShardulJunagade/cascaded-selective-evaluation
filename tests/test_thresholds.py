"""Calibration baselines used in the repeated policy comparison."""
import pytest
import torch

from open_cascade.thresholds import (heuristic_selection, no_selection,
                                     point_estimate_calibration)


def test_fixed_baseline_thresholds():
    empty = torch.tensor([])
    assert no_selection(empty, empty, empty, alpha=0.15, delta=0.1) == 0.0
    assert heuristic_selection(empty, empty, empty, alpha=0.15, delta=0.1) == 0.85


def test_point_estimate_chooses_smallest_passing_threshold():
    phats = torch.tensor([0.6] * 10 + [0.8] * 10 + [0.9] * 30)
    labels = torch.zeros(50)
    yhats = torch.tensor([1] * 10 + [0] * 40)
    threshold = point_estimate_calibration(phats, yhats, labels, alpha=0.05, delta=0.1)
    assert threshold == pytest.approx(0.8)


def test_point_estimate_requires_thirty_selected_examples():
    phats = torch.tensor([0.9] * 29)
    labels = torch.zeros(29)
    yhats = torch.zeros(29)
    assert point_estimate_calibration(phats, yhats, labels, alpha=0.1, delta=0.1) is None
