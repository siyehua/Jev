"""Unit tests: stable softmax, temperature, tie-breaking (SPEC.md §14)."""

from __future__ import annotations

import math

import pytest

from jev.scoring import argmax_first, softmax


def test_probabilities_sum_to_one():
    probs = softmax([2.0, 1.0, -0.5])
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-12)


def test_softmax_is_numerically_stable_for_large_logits():
    probs = softmax([10000.0, 9999.0, -10000.0])
    assert all(math.isfinite(p) for p in probs)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-12)
    assert probs[0] > probs[1] > probs[2]


def test_softmax_matches_manual_computation():
    logits = [1.0, 2.0, 3.0]
    expected = [math.exp(x) for x in logits]
    total = sum(expected)
    expected = [x / total for x in expected]
    assert softmax(logits) == pytest.approx(expected)


def test_temperature_flattens_distribution():
    sharp = softmax([1.0, 3.0], temperature=0.1)
    flat = softmax([1.0, 3.0], temperature=10.0)
    assert sharp[1] > flat[1]
    assert abs(flat[1] - 0.5) < abs(sharp[1] - 0.5)


def test_temperature_one_matches_plain_softmax():
    assert softmax([0.3, -2.0, 5.0], temperature=1.0) == pytest.approx(
        softmax([0.3, -2.0, 5.0])
    )


def test_non_positive_temperature_rejected():
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)
    with pytest.raises(ValueError):
        softmax([1.0], temperature=-1.0)


def test_argmax_first_wins_exact_ties():
    assert argmax_first([1.0, 3.0, 3.0, 2.0]) == 1
    assert argmax_first([5.0, 5.0]) == 0
    assert argmax_first([1.0, 2.0, 3.0]) == 2


def test_equal_logits_give_equal_probabilities():
    probs = softmax([7.0, 7.0, 7.0])
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    # First label wins the tie under input-order tie-breaking.
    assert argmax_first(probs) == 0
