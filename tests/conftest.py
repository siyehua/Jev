from __future__ import annotations

import pytest

from jev.config import DecisionConfig


@pytest.fixture()
def fake_config() -> DecisionConfig:
    return DecisionConfig(
        model_id="fake-model",
        label_mode="letters",
        max_input_chars=1000,
        max_instructions_chars=500,
        max_criteria_value_chars=500,
        max_criteria_key_chars=64,
        max_prompt_tokens=10000,
    )


@pytest.fixture()
def criteria() -> dict[str, str]:
    return {
        "shipping": "A delivery, shipment, or package-arrival problem.",
        "billing": "A payment, invoice, or duplicate-charge problem.",
        "returns": "A return, refund, or exchange request.",
    }


@pytest.fixture()
def decision_request(criteria):
    return {
        "model": "fake-model",
        "input": "My package arrived two weeks late, and my card was charged twice.",
        "instructions": "Choose the category that best matches the customer's main issue.",
        "criteria": criteria,
        "scoring": {"strategy": "label_logits", "temperature": 1.0},
        "request_id": "req-123",
    }
