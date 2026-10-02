"""The GPU-free parts of judge scoring: reading label probabilities and averaging them."""
from types import SimpleNamespace

import pytest

from open_cascade.judges.base import average_simulations, normalise_label_logprobs

LABEL_IDS = {"A": 10, "B": 11}


def step(**logprobs_by_id):
    """Fake vLLM step logprobs: {token_id: object with .logprob}."""
    return {int(k[1:]): SimpleNamespace(logprob=v) for k, v in logprobs_by_id.items()}


def test_normalise_renormalises_over_two_labels():
    probs = normalise_label_logprobs(step(t10=-0.1, t11=-2.0, t99=-3.0), LABEL_IDS)
    assert probs["A"] + probs["B"] == pytest.approx(1.0)
    assert probs["A"] > probs["B"]


def test_normalise_floors_a_missing_label():
    probs = normalise_label_logprobs(step(t10=-0.1, t99=-3.0), LABEL_IDS)
    assert probs["A"] > 0.9 and probs["B"] > 0.0


def test_normalise_returns_none_when_off_format():
    assert normalise_label_logprobs(step(t99=-0.1), LABEL_IDS) is None


def test_average_maps_swapped_ordering_back():
    # sample 0: ordering 0 says A (= outputs[0]) at 0.8; ordering 1 says B (= outputs[0]) at 0.6
    provenance = [(0, 0), (0, 1)]
    scored = [{"A": 0.8, "B": 0.2}, {"A": 0.4, "B": 0.6}]
    assert average_simulations(1, provenance, scored) == [pytest.approx([0.7, 0.3])]


def test_average_skips_failed_simulations():
    provenance = [(0, 0), (0, 1), (1, 0)]
    scored = [{"A": 0.9, "B": 0.1}, None, None]
    assert average_simulations(2, provenance, scored) == [pytest.approx([0.9, 0.1]), []]
