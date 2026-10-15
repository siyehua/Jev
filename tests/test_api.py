"""API tests with the FakeBackend (no model download required).

Covers: happy path, all documented error codes (SPEC.md §9), auth,
body-size limits, tie-breaking, and generated_tokens == 0.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jev.app import create_app
from jev.backends.fake import FakeBackend


def make_client(config, backend=None):
    backend = backend or FakeBackend()
    app = create_app(config, backend)
    # Model loads exactly once, during app startup (SPEC.md §7.1).
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def client(fake_config):
    yield from make_client(fake_config)


class TestHappyPath:
    def test_returns_one_supplied_label(self, client, decision_request):
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 200
        body = response.json()
        assert body["choice"] in decision_request["criteria"]
        assert body["strategy"] == "label_logits"
        assert body["model"] == "fake-model"
        assert set(body["probabilities"]) == set(decision_request["criteria"])
        assert set(body["scores"]) == set(decision_request["criteria"])

    def test_probabilities_sum_to_one(self, client, decision_request):
        body = client.post("/v1/decision", json=decision_request).json()
        assert sum(body["probabilities"].values()) == pytest.approx(1.0)

    def test_generated_tokens_is_zero(self, client, decision_request):
        body = client.post("/v1/decision", json=decision_request).json()
        assert body["usage"]["generated_tokens"] == 0
        assert body["usage"]["prompt_tokens"] > 0

    def test_request_id_echoed_when_provided(self, client, decision_request):
        body = client.post("/v1/decision", json=decision_request).json()
        assert body["request_id"] == "req-123"

    def test_request_id_null_when_absent(self, client, decision_request):
        decision_request.pop("request_id")
        body = client.post("/v1/decision", json=decision_request).json()
        assert body["request_id"] is None

    def test_scores_and_probabilities_share_order(self, client, decision_request):
        body = client.post("/v1/decision", json=decision_request).json()
        assert list(body["probabilities"]) == list(body["scores"])
        assert list(body["probabilities"]) == list(decision_request["criteria"])

    def test_latency_metadata_present(self, client, decision_request):
        body = client.post("/v1/decision", json=decision_request).json()
        latency = body["latency"]
        assert {"tokenize_ms", "forward_ms", "postprocess_ms", "total_ms"} <= set(latency)

    def test_choice_follows_highest_logit(self, fake_config, decision_request):
        # Letters mode: label "B" gets the highest logit.
        backend = FakeBackend(
            logit_fn=lambda label: {"A": -1.0, "B": 5.0, "C": -2.0}.get(label, -9.0)
        )
        with TestClient(create_app(fake_config, backend)) as client:
            body = client.post("/v1/decision", json=decision_request).json()
        assert body["choice"] == "billing"  # second criterion = label B

    def test_tie_breaks_to_first_criterion(self, fake_config, decision_request):
        backend = FakeBackend(logit_fn=lambda tid: 1.0)
        with TestClient(create_app(fake_config, backend)) as client:
            body = client.post("/v1/decision", json=decision_request).json()
        assert body["choice"] == "shipping"  # first criterion wins ties
        for probability in body["probabilities"].values():
            assert probability == pytest.approx(1 / 3)


class TestErrors:
    def test_unknown_model_returns_404(self, client, decision_request):
        decision_request["model"] = "other-model"
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"

    def test_missing_fields_return_400(self, client, decision_request):
        decision_request.pop("instructions")
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_bad_json_returns_400(self, client):
        response = client.post(
            "/v1/decision",
            content="{not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_empty_input_returns_400(self, client, decision_request):
        decision_request["input"] = "   "
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_one_criterion_returns_400(self, client, decision_request):
        decision_request["criteria"] = {"only": "one entry"}
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_zero_temperature_returns_400(self, client, decision_request):
        decision_request["scoring"]["temperature"] = 0
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_unknown_strategy_returns_400(self, client, decision_request):
        decision_request["scoring"]["strategy"] = "binary_logits"
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_too_large_input_returns_413(self, client, decision_request):
        decision_request["input"] = "x" * 1001  # limit is 1000
        response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "input_too_large"

    def test_too_many_prompt_tokens_return_413(self, fake_config, decision_request):
        fake_config = fake_config.__class__(
            model_id="fake-model", max_prompt_tokens=10
        )
        with TestClient(create_app(fake_config, FakeBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "input_too_large"

    def test_multi_token_criteria_key_returns_422(self, fake_config, decision_request):
        # criteria_keys mode: keys must themselves be single-token labels;
        # the fake tokenizer is char-level, so "shipping" is multi-token.
        fake_config = fake_config.__class__(
            model_id="fake-model", label_mode="criteria_keys"
        )
        with TestClient(create_app(fake_config, FakeBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "unsupported_label_tokenization"
        assert "shipping" in error["message"]

    def test_model_not_ready_returns_503(self, fake_config, decision_request):
        class NeverReadyBackend(FakeBackend):
            def is_ready(self) -> bool:
                return False

        with TestClient(create_app(fake_config, NeverReadyBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"

    def test_backend_failure_returns_500_without_traceback(
        self, fake_config, decision_request
    ):
        class ExplodingBackend(FakeBackend):
            def score_labels(self, prompt, labels):
                raise RuntimeError("secret internal detail")

        with TestClient(create_app(fake_config, ExplodingBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 500
        error = response.json()["error"]
        assert error["code"] == "inference_error"
        assert "secret internal detail" not in error["message"]

    def test_body_size_limit_returns_413(self, fake_config, decision_request):
        fake_config = fake_config.__class__(
            model_id="fake-model", max_body_bytes=200
        )
        with TestClient(create_app(fake_config, FakeBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "input_too_large"

    def test_error_shape_is_stable(self, client, decision_request):
        decision_request["model"] = "nope"
        response = client.post("/v1/decision", json=decision_request)
        error = response.json()["error"]
        assert set(error) == {"code", "message"}


class TestAuth:
    def test_api_key_required_when_configured(self, fake_config, decision_request):
        fake_config = fake_config.__class__(model_id="fake-model", api_key="s3cret")
        with TestClient(create_app(fake_config, FakeBackend())) as client:
            response = client.post("/v1/decision", json=decision_request)
            assert response.status_code == 401
            response = client.post(
                "/v1/decision",
                json=decision_request,
                headers={"authorization": "Bearer wrong"},
            )
            assert response.status_code == 401
            response = client.post(
                "/v1/decision",
                json=decision_request,
                headers={"authorization": "Bearer s3cret"},
            )
            assert response.status_code == 200

    def test_healthz_open_without_auth(self, fake_config):
        fake_config = fake_config.__class__(model_id="fake-model", api_key="s3cret")
        with TestClient(create_app(fake_config, FakeBackend())) as client:
            assert client.get("/healthz").status_code == 200


class TestHealthz:
    def test_healthz_reports_model_loaded(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["backend"] == "fake"
