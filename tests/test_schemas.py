"""Unit tests: request schema validation (SPEC.md §14)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev.schemas import DecisionRequest


def _base(**overrides):
    payload = {
        "model": "local-model",
        "input": "some context",
        "instructions": "pick a label",
        "criteria": {"a": "meaning a", "b": "meaning b"},
    }
    payload.update(overrides)
    return payload


def test_valid_request_parses():
    request = DecisionRequest(**_base())
    assert request.scoring.strategy == "label_logits"
    assert request.scoring.temperature == 1.0
    assert request.request_id is None


def test_defaults_applied():
    request = DecisionRequest(**_base())
    assert request.scoring.strategy == "label_logits"
    assert request.scoring.temperature == 1.0


@pytest.mark.parametrize("missing", ["model", "input", "instructions", "criteria"])
def test_missing_required_fields(missing):
    payload = _base()
    del payload[missing]
    with pytest.raises(ValidationError):
        DecisionRequest(**payload)


@pytest.mark.parametrize("blank", ["   ", "", "\n\t "])
def test_blank_input_or_instructions_rejected(blank):
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(input=blank))
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(instructions=blank))


def test_criteria_requires_two_entries():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(criteria={"a": "only one"}))


def test_criteria_rejects_too_many_entries():
    criteria = {chr(97 + i): f"meaning {i}" for i in range(27)}
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(criteria=criteria))


def test_criteria_allows_up_to_26_entries():
    criteria = {chr(97 + i): f"meaning {i}" for i in range(26)}
    request = DecisionRequest(**_base(criteria=criteria))
    assert len(request.criteria) == 26


def test_criteria_keys_must_be_non_empty():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(criteria={" ": "meaning", "b": "other"}))


def test_criteria_values_must_be_non_empty():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(criteria={"a": "  ", "b": "other"}))


def test_temperature_must_be_positive():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(scoring={"temperature": 0.0}))
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(scoring={"temperature": -1.5}))


def test_temperature_must_be_finite():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(scoring={"temperature": float("inf")}))
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(scoring={"temperature": float("nan")}))


def test_strategy_must_be_label_logits():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(scoring={"strategy": "binary_logits"}))


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        DecisionRequest(**_base(unexpected="field"))
