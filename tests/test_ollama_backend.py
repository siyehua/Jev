"""Ollama backend tests using a mocked HTTP transport (SPEC.md §8, §14).

Verifies the one-token generation fallback semantics:
- honest strategy reporting (single_token_generation, generated_tokens=1);
- label scoring from top-k logprobs, including leading-space matching;
- missing labels / missing logprobs / HTTP failures raise documented errors;
- full API integration reports the actual strategy and nonzero usage
  (SPEC.md §13.8 for generation fallbacks).
"""

from __future__ import annotations

import math

import httpx
import pytest
from fastapi.testclient import TestClient

from jev.app import create_app
from jev.backends.ollama_backend import OllamaBackend
from jev.config import DecisionConfig
from jev.errors import InferenceError, UnsupportedLabelTokenization
from jev.scoring import softmax

MODEL = "qwen2.5:0.5b"


def make_config(**overrides) -> DecisionConfig:
    defaults = dict(
        model_id=MODEL,
        backend="ollama",
        ollama_base_url="http://ollama.test",
        ollama_mode="chat",
    )
    defaults.update(overrides)
    return DecisionConfig(**defaults)


def make_backend(handler, **config_overrides) -> OllamaBackend:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )
    backend = OllamaBackend(client=client)
    backend.load(make_config(**config_overrides))
    return backend


def default_handler(
    top_logprobs: list[dict] | None = None,
    *,
    generated: str = "A",
    with_logprobs: bool = True,
    status: int = 200,
    models: list[str] | None = None,
    mode: str = "chat",
    prompt_eval_count: int = 134,
):
    if top_logprobs is None:
        top_logprobs = [
            {"token": "A", "logprob": -0.3},
            {"token": "B", "logprob": -1.5},
            {"token": "C", "logprob": -2.5},
        ]
    body: dict = {
        "model": MODEL,
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        # num_predict=2 workaround: server decodes 2 tokens (observed
        # against Ollama v0.12.11; num_predict=1 drops logprobs entirely).
        "eval_count": 2,
    }
    if mode == "chat":
        body["message"] = {"role": "assistant", "content": generated}
    else:
        body["response"] = generated
    if with_logprobs:
        body["logprobs"] = [
            {"token": generated, "logprob": -0.3, "top_logprobs": top_logprobs}
        ]

    def handler_fn(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if path == "/api/tags":
            listed = models if models is not None else [MODEL]
            return httpx.Response(
                200, json={"models": [{"name": m} for m in listed]}
            )
        if path == "/api/show":
            if models is not None and MODEL not in models:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(200, json={"license": "x"})
        if path in ("/api/chat", "/api/generate"):
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "unknown route"})

    return handler_fn


class TestLoad:
    def test_load_succeeds_and_health_reports_details(self):
        backend = make_backend(default_handler())
        assert backend.is_ready()
        health = backend.health()
        assert health["backend"] == "ollama"
        assert health["model"] == MODEL
        assert health["model_loaded"] is True
        assert health["server_version"] == "0.12.0"
        assert health["strategy"] == "single_token_generation"

    def test_load_fails_when_model_missing(self):
        client = httpx.Client(
            transport=httpx.MockTransport(default_handler(models=[])),
            base_url="http://ollama.test",
        )
        backend = OllamaBackend(client=client)
        with pytest.raises(RuntimeError, match="not found on the Ollama server"):
            backend.load(make_config())

    def test_load_fails_on_connection_error(self):
        def failing(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = httpx.Client(
            transport=httpx.MockTransport(failing), base_url="http://ollama.test"
        )
        backend = OllamaBackend(client=client)
        with pytest.raises(RuntimeError, match="cannot reach the Ollama server"):
            backend.load(make_config())


class TestScoring:
    def test_happy_path_scores_from_top_logprobs(self):
        backend = make_backend(default_handler())
        result = backend.score_labels("prompt", ["A", "B", "C"])
        assert result.strategy == "single_token_generation"
        # Honest: eval_count from the server (2 decoded tokens due to the
        # num_predict=2 logprobs workaround).
        assert result.generated_tokens == 2
        assert result.prompt_tokens == 134
        assert result.scores == {"A": -0.3, "B": -1.5, "C": -2.5}

    def test_generate_mode_matches_leading_space_tokens(self):
        top = [
            {"token": " A", "logprob": -0.2},
            {"token": " B", "logprob": -1.1},
            {"token": " C", "logprob": -3.0},
        ]
        backend = make_backend(
            default_handler(top_logprobs=top, mode="generate"),
            ollama_mode="generate",
        )
        result = backend.score_labels("Answer:", ["A", "B", "C"])
        assert result.scores == {"A": -0.2, "B": -1.1, "C": -3.0}

    def test_legacy_content_prob_shape_parsed(self):
        top = [
            {"content": "A", "prob": 0.6},
            {"content": "B", "prob": 0.3},
            {"content": "C", "prob": 0.1},
        ]
        backend = make_backend(default_handler(top_logprobs=top))
        result = backend.score_labels("prompt", ["A", "B", "C"])
        assert result.scores["A"] == pytest.approx(math.log(0.6))
        assert result.scores["B"] == pytest.approx(math.log(0.3))

    def test_missing_label_raises_422(self):
        top = [
            {"token": "A", "logprob": -0.3},
            {"token": "B", "logprob": -1.5},
            # label C absent from top-k
        ]
        backend = make_backend(default_handler(top_logprobs=top))
        with pytest.raises(UnsupportedLabelTokenization, match="C"):
            backend.score_labels("prompt", ["A", "B", "C"])

    def test_missing_logprobs_raises_inference_error(self):
        backend = make_backend(default_handler(with_logprobs=False))
        with pytest.raises(InferenceError, match="logprobs"):
            backend.score_labels("prompt", ["A", "B", "C"])

    def test_http_error_raises_inference_error(self):
        backend = make_backend(default_handler(status=500))
        with pytest.raises(InferenceError, match="HTTP 500"):
            backend.score_labels("prompt", ["A", "B", "C"])

    def test_not_loaded_raises_inference_error(self):
        backend = OllamaBackend()
        with pytest.raises(InferenceError, match="not loaded"):
            backend.score_labels("prompt", ["A", "B"])

    def test_request_payload_shape(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/chat":
                captured["payload"] = request.read()
            return default_handler()(request)

        backend = make_backend(handler)
        backend.score_labels("prompt", ["A", "B", "C"])
        import json

        payload = json.loads(captured["payload"])
        assert payload["stream"] is False
        assert payload["logprobs"] is True
        assert payload["top_logprobs"] >= 3
        # num_predict=2: workaround for the server dropping logprobs
        # when num_predict=1 (observed on Ollama v0.12.11).
        assert payload["options"]["num_predict"] == 2
        assert payload["options"]["temperature"] == 0
        assert payload["messages"] == [{"role": "user", "content": "prompt"}]

    def test_top_logprobs_clamped_to_server_cap(self):
        captured: dict = {}

        # 26 labels would request top_logprobs=26; the server caps at 20.
        labels = [chr(ord("A") + i) for i in range(26)]
        top = [{"token": label, "logprob": -1.0} for label in labels]
        original = default_handler(top_logprobs=top)

        def handler_all(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/chat":
                captured["payload"] = request.read()
            return original(request)

        backend = make_backend(handler_all, ollama_top_logprobs=26)
        backend.score_labels("prompt", labels)
        import json

        payload = json.loads(captured["payload"])
        assert payload["top_logprobs"] == 20

    def test_ollama_error_body_raises_inference_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in ("/api/chat", "/api/generate"):
                # Ollama returns {"error": ...} — sometimes with HTTP 200.
                return httpx.Response(
                    200, json={"error": "top_logprobs must be between 0 and 20"}
                )
            return default_handler()(request)

        backend = make_backend(handler)
        with pytest.raises(InferenceError, match="top_logprobs"):
            backend.score_labels("prompt", ["A", "B", "C"])


class TestMathEquivalence:
    def test_softmax_over_logprobs_equals_restricted_probs(self):
        # softmax(log p) over a label subset equals p_i / sum(p_j over subset)
        probs = {"A": 0.6, "B": 0.3, "C": 0.1}
        logprobs = [math.log(p) for p in probs.values()]
        result = softmax(logprobs)
        total = sum(probs.values())
        for actual, expected in zip(result, probs.values()):
            assert actual == pytest.approx(expected / total)


class TestApiIntegration:
    def test_api_reports_actual_strategy_and_usage(self):
        backend = make_backend(default_handler())
        config = make_config()
        with TestClient(create_app(config, backend)) as client:
            response = client.post(
                "/v1/decision",
                json={
                    "model": MODEL,
                    "input": "My card was charged twice.",
                    "instructions": "Pick the issue category.",
                    "criteria": {
                        "shipping": "delivery problems",
                        "billing": "payment problems",
                        "returns": "refund requests",
                    },
                },
            )
        assert response.status_code == 200
        body = response.json()
        # Honest fallback reporting (SPEC.md §8): strategy and usage must
        # describe what actually executed, not what was requested.
        assert body["strategy"] == "single_token_generation"
        assert body["usage"]["generated_tokens"] == 2  # eval_count
        assert body["usage"]["prompt_tokens"] == 134
        assert body["backend"] == "ollama"
        assert body["choice"] in {"shipping", "billing", "returns"}
        assert sum(body["probabilities"].values()) == pytest.approx(1.0)
        # scores are logprobs here, not raw logits
        assert body["scores"]["shipping"] == pytest.approx(-0.3)

    def test_api_missing_label_returns_422(self):
        top = [
            {"token": "A", "logprob": -0.3},
            {"token": "B", "logprob": -1.5},
        ]
        backend = make_backend(default_handler(top_logprobs=top))
        with TestClient(create_app(make_config(), backend)) as client:
            response = client.post(
                "/v1/decision",
                json={
                    "model": MODEL,
                    "input": "x",
                    "instructions": "pick",
                    "criteria": {
                        "a": "first",
                        "b": "second",
                        "c": "third",
                    },
                },
            )
        assert response.status_code == 422
        assert (
            response.json()["error"]["code"] == "unsupported_label_tokenization"
        )

    def test_api_unknown_model_returns_404(self):
        backend = make_backend(default_handler())
        with TestClient(create_app(make_config(), backend)) as client:
            response = client.post(
                "/v1/decision",
                json={
                    "model": "other-model",
                    "input": "x",
                    "instructions": "pick",
                    "criteria": {"a": "first", "b": "second"},
                },
            )
        assert response.status_code == 404

    def test_healthz_reports_ollama_backend(self):
        backend = make_backend(default_handler())
        with TestClient(create_app(make_config(), backend)) as client:
            body = client.get("/healthz").json()
        assert body["backend"] == "ollama"
        assert body["model_loaded"] is True
        assert body["strategy"] == "single_token_generation"
