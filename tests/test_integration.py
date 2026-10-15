"""Integration tests with a real local causal LM (SPEC.md §14).

Skipped automatically when the configured model cannot be loaded (no
network / no cached weights). Default model: Qwen/Qwen2.5-0.5B on CPU.

Covers:
- output label is always among criteria keys;
- no generate() call in direct-logits mode (spy);
- model loads only during startup;
- exact prompt suffix and label token IDs for the configured tokenizer;
- chat template applied for chat models;
- CPU execution;
- multi-token criteria keys rejected in criteria_keys mode.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pytest

from jev.backends.base import DecisionBackend
from jev.backends.transformers_backend import TransformersBackend
from jev.config import DecisionConfig
from jev.schemas import DecisionRequest
from jev.service import DecisionService

MODEL_ID = os.environ.get("JEV_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


def _config(**overrides) -> DecisionConfig:
    defaults = dict(
        model_id=MODEL_ID,
        device="cpu",
        dtype="float32",
        label_mode="letters",
        use_chat_template="auto",
    )
    defaults.update(overrides)
    return DecisionConfig(**defaults)


@lru_cache(maxsize=1)
def _shared_backend() -> TransformersBackend | None:
    backend = TransformersBackend()
    try:
        backend.load(_config())
    except Exception:
        return None
    return backend


@pytest.fixture(scope="module")
def backend() -> TransformersBackend:
    loaded = _shared_backend()
    if loaded is None:
        pytest.skip(f"model {MODEL_ID} not available locally")
    return loaded


@pytest.fixture()
def config() -> DecisionConfig:
    return _config()


@pytest.fixture()
def decision_request() -> dict:
    return {
        "model": MODEL_ID,
        "input": "My package arrived two weeks late, and my card was charged twice.",
        "instructions": (
            "Choose the category that best matches the customer's main issue."
        ),
        "criteria": {
            "shipping": "A delivery, shipment, or package-arrival problem.",
            "billing": "A payment, invoice, or duplicate-charge problem.",
            "returns": "A return, refund, or exchange request.",
        },
        "scoring": {"strategy": "label_logits", "temperature": 1.0},
        "request_id": "it-1",
    }


def test_decision_returns_label_among_criteria(backend, config, decision_request):
    service = DecisionService(config, backend)
    response = service.handle_decision(DecisionRequest(**decision_request))
    assert response.choice in decision_request["criteria"]
    assert set(response.probabilities) == set(decision_request["criteria"])
    assert sum(response.probabilities.values()) == pytest.approx(1.0)
    assert response.usage.generated_tokens == 0
    assert response.usage.prompt_tokens > 0
    assert response.request_id == "it-1"
    assert response.latency.forward_ms >= 0


def test_no_generate_call_in_direct_logits_mode(
    backend, config, decision_request, monkeypatch
):
    model = backend._model

    def spy(*args, **kwargs):
        raise AssertionError("model.generate() must not be called")

    monkeypatch.setattr(model, "generate", spy)
    try:
        service = DecisionService(config, backend)
        response = service.handle_decision(DecisionRequest(**decision_request))
    finally:
        monkeypatch.undo()
    assert response.choice in decision_request["criteria"]


def test_model_loads_only_during_startup(backend):
    """The app calls load() exactly once at startup, never per request."""

    class LoadCounting(DecisionBackend):
        name = "load-counting"

        def __init__(self, inner: DecisionBackend):
            self.inner = inner
            self.load_count = 0

        def load(self, cfg):
            self.load_count += 1  # inner model already loaded (shared)

        def is_ready(self):
            return self.inner.is_ready()

        def health(self):
            return self.inner.health()

        def render_prompt(self, prompt):
            return self.inner.render_prompt(prompt)

        def score_labels(self, prompt, labels):
            return self.inner.score_labels(prompt, labels)

    from fastapi.testclient import TestClient

    from jev.app import create_app

    counting = LoadCounting(backend)
    app = create_app(_config(), counting)
    with TestClient(app) as client:
        assert counting.load_count == 1
        assert client.get("/healthz").json()["model_loaded"] is True
        for _ in range(3):
            payload = {
                "model": MODEL_ID,
                "input": "I want my money back.",
                "instructions": "Pick the issue category.",
                "criteria": {
                    "shipping": "delivery problems",
                    "billing": "payment problems",
                },
            }
            assert client.post("/v1/decision", json=payload).status_code == 200
    assert counting.load_count == 1


def test_exact_prompt_suffix_and_label_token_ids(backend, config, decision_request):
    """SPEC.md §13.7: verify the exact prompt suffix and label token IDs."""
    service = DecisionService(config, backend)
    request = DecisionRequest(**decision_request)
    built = service.prompt_builder.build(
        request.instructions, request.input, request.criteria
    )
    final_prompt = backend.render_prompt(built.prompt)
    # Exact prompt suffix check.
    if backend.uses_chat_template():
        assert final_prompt.endswith("<|im_start|>assistant\n")
        assert "<|im_end|>" in final_prompt
    else:
        assert final_prompt.endswith("Answer:")

    add_special = not backend.uses_chat_template()
    base_ids = backend.tokenize(final_prompt, add_special_tokens=add_special)
    resolved: dict[str, int] = {}
    for fmt in backend.continuation_formats():
        resolved = {}
        ok = True
        for label in built.labels:
            full_ids = backend.tokenize(
                final_prompt + fmt.format(label=label),
                add_special_tokens=add_special,
            )
            added = full_ids[len(base_ids):]
            if full_ids[: len(base_ids)] != base_ids or len(added) != 1:
                ok = False
                break
            resolved[label] = added[0]
        if ok:
            break
    assert ok, "at least one continuation format must resolve every label"
    assert len(set(resolved.values())) == len(built.labels)
    assert resolved == {"A": 32, "B": 33, "C": 34}
    # Round-trip: each selected token decodes back to the label text.
    for label, token_id in resolved.items():
        assert backend._tokenizer.decode([token_id]).strip() == label


def test_chat_template_applied_for_chat_models(backend):
    tokenizer = backend._tokenizer
    has_template = bool(getattr(tokenizer, "chat_template", None))
    assert backend.uses_chat_template() == has_template


def test_multi_token_criteria_key_returns_422(backend, decision_request):
    strict_config = _config(label_mode="criteria_keys")
    decision_request["criteria"] = {
        "billing_issues": "A payment problem.",
        "shipping_problems": "A delivery problem.",
    }
    from jev.errors import UnsupportedLabelTokenization

    service = DecisionService(strict_config, backend)
    with pytest.raises(UnsupportedLabelTokenization) as excinfo:
        service.handle_decision(DecisionRequest(**decision_request))
    assert "billing_issues" in str(excinfo.value)


def test_runs_on_expected_device(backend):
    # CPU execution policy for these tests.
    assert str(backend._device).startswith("cpu")
