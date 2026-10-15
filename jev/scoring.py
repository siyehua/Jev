"""Numerically stable label softmax with temperature (SPEC.md §7.5)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def argmax_first(values: Sequence[float]) -> int:
    """Index of the maximum; the first occurrence wins exact ties."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best
