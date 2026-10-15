"""Ollama backend — one-token generation fallback (SPEC.md §8).

The public Ollama API does not expose pre-generation full-vocabulary
logits: its ``logprobs`` describe *generated* tokens (plus optional
top-token alternatives at each generated position). True decode-free
direct-logits scoring is therefore NOT possible through this API.

This backend implements the documented fallback mode instead:

1. Send the decision prompt with a small ``num_predict`` budget and
   top-k logprobs enabled, then score each label from the **first**
   generated position's top-k logprobs. Mathematically, softmax over
   log-probabilities equals softmax over the underlying logits
   restricted to the same label subset (softmax is shift-invariant), so
   the normalized probabilities match what a direct-logits backend
   would compute from the same distribution.
2. Report the execution honestly: strategy
   ``single_token_generation`` and ``generated_tokens`` taken from the
   server's ``eval_count``.

Ollama runtime quirks observed against v0.12.11 (handled here):

- ``top_logprobs`` is hard-capped at 20 by the server; requesting more
  returns ``{"error": "top_logprobs must be between 0 and 20"}``. The
  request is clamped to 20; label sets whose tokens fall outside the
  top-20 at the answer position fail with 422.
- With ``num_predict: 1`` the server omits the ``logprobs`` array
  entirely (empty message, ``done_reason: "length"``). Requesting 2
  tokens works around this; only the first logprobs position is used.
- Error responses are JSON bodies shaped ``{"error": "..."}``; both
  non-200 and 200-with-error bodies are surfaced as ``inference_error``.

Label coverage limit: a label whose token is not within the returned
top-k logprobs cannot be scored, and the request fails with
``unsupported_label_tokenization`` instead of silently scoring a wrong
token.

Model compatibility note: this backend assumes the model emits its
answer at the first generated position after the prompt (true for
standard instruct/chat models such as gemma3 or qwen2.5). Reasoning
models with mandatory analysis channels (e.g. gpt-oss harmony format)
start with analysis prose, so the first position does not contain the
answer and label scoring is meaningless — prefer a standard chat model.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

import httpx

from jev.backends.base import (
    STRATEGY_SINGLE_TOKEN_GENERATION,
    DecisionBackend,
    ScoreResult,
)
from jev.config import DecisionConfig
from jev.errors import InferenceError, UnsupportedLabelTokenization

logger = logging.getLogger("jev.backend")


def _entry_token(entry: dict) -> str:
    # Current API uses "token"; some older builds used "content".
    return entry.get("token") or entry.get("content") or ""


def _entry_logprob(entry: dict) -> float:
    if "logprob" in entry:
        return float(entry["logprob"])
    if "prob" in entry:  # legacy field: plain probability, not log
        return math.log(max(float(entry["prob"]), 1e-12))
    raise InferenceError(
        "logprobs entry has neither 'logprob' nor 'prob' field"
    )


class OllamaBackend(DecisionBackend):
    """Scores labels via one-token generation + top-k logprobs."""

    name = "ollama"

    # Ollama server hard cap (observed on v0.12.x).
    TOP_LOGPROBS_CAP = 20
    # num_predict=1 makes the server drop the logprobs array; 2 works.
    NUM_PREDICT = 2

    def __init__(self, client: httpx.Client | None = None):
        self._injected_client = client
        self._client: httpx.Client | None = client
        self._config: DecisionConfig | None = None
        self._ready = False
        self._server_version: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self, config: DecisionConfig) -> None:
        self._config = config
        if self._client is None:
            self._client = httpx.Client(
                base_url=config.ollama_base_url.rstrip("/"),
                timeout=config.request_timeout_seconds,
            )
        try:
            version = self._client.get("/api/version")
            version.raise_for_status()
            self._server_version = version.json().get("version")
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(
                f"cannot reach the Ollama server at "
                f"{config.ollama_base_url}: {exc.__class__.__name__}"
            ) from exc
        if not self._model_exists(config.model_id):
            raise RuntimeError(
                f"model {config.model_id!r} was not found on the Ollama "
                f"server at {config.ollama_base_url}. Pull it first with: "
                f"ollama pull {config.model_id}"
            )
        self._ready = True
        logger.info(
            "ollama backend ready: model=%s base_url=%s mode=%s server_version=%s",
            config.model_id,
            config.ollama_base_url,
            config.ollama_mode,
            self._server_version,
        )

    def _model_exists(self, model_id: str) -> bool:
        try:
            tags = self._client.get("/api/tags")
            if tags.status_code == 200:
                for model in tags.json().get("models", []):
                    if model_id in (model.get("name"), model.get("model")):
                        return True
            show = self._client.post("/api/show", json={"model": model_id})
            return show.status_code == 200
        except (httpx.HTTPError, ValueError):
            return False

    def is_ready(self) -> bool:
        return self._ready

    def health(self) -> dict:
        return {
            "backend": self.name,
            "model": self._config.model_id if self._config else None,
            "model_loaded": self._ready,
            "base_url": self._config.ollama_base_url if self._config else None,
            "mode": self._config.ollama_mode if self._config else None,
            "server_version": self._server_version,
            "strategy": STRATEGY_SINGLE_TOKEN_GENERATION,
        }

    # -- scoring -----------------------------------------------------------

    def render_prompt(self, prompt: str) -> str:
        # The chat template is applied by the Ollama server itself when
        # using /api/chat, so the prompt is passed through unchanged.
        return prompt

    def score_labels(self, prompt: str, labels: Sequence[str]) -> ScoreResult:
        if not self._ready or self._client is None or self._config is None:
            raise InferenceError("ollama backend is not loaded")
        labels = list(labels)
        cfg = self._config
        top_k = min(
            max(cfg.ollama_top_logprobs, len(labels)), self.TOP_LOGPROBS_CAP
        )

        payload: dict = {
            "model": cfg.model_id,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {"num_predict": self.NUM_PREDICT, "temperature": 0},
        }
        if cfg.ollama_mode == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
            endpoint = "/api/chat"
        else:
            payload["prompt"] = prompt
            payload["raw"] = True  # bypass the model card's template
            endpoint = "/api/generate"

        forward_started = time.perf_counter()
        try:
            response = self._client.post(endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise InferenceError(
                f"ollama request timed out after "
                f"{cfg.request_timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"ollama request failed: {exc.__class__.__name__}"
            ) from exc
        forward_ms = (time.perf_counter() - forward_started) * 1000
        if response.status_code != 200:
            raise InferenceError(
                f"ollama returned HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise InferenceError("ollama returned a non-JSON body") from exc
        if isinstance(body, dict) and body.get("error"):
            # Ollama reports some failures as {"error": "..."} — sometimes
            # even with HTTP 200 (e.g. top_logprobs out of range).
            raise InferenceError(f"ollama error: {body['error']}")

        post_started = time.perf_counter()
        logprobs = body.get("logprobs") or []
        if not logprobs:
            raise InferenceError(
                "the Ollama server returned no logprobs; a recent Ollama "
                "version with logprobs support is required"
            )
        top_entries = logprobs[0].get("top_logprobs") or []
        scores = self._match_labels(top_entries, labels, cfg.ollama_mode)
        postprocess_ms = (time.perf_counter() - post_started) * 1000

        # Honest usage: report how many tokens the server actually decoded
        # (eval_count includes any stop/EOS token).
        generated_tokens = int(body.get("eval_count") or len(logprobs))
        prompt_tokens = body.get("prompt_eval_count") or 0

        return ScoreResult(
            scores=scores,
            strategy=STRATEGY_SINGLE_TOKEN_GENERATION,
            generated_tokens=generated_tokens,
            prompt_tokens=int(prompt_tokens),
            tokenize_ms=0.0,
            forward_ms=forward_ms,
            postprocess_ms=postprocess_ms,
        )

    def _match_labels(
        self, top_entries: list[dict], labels: list[str], mode: str
    ) -> dict[str, float]:
        exact: dict[str, float] = {}
        stripped_token: dict[str, str] = {}
        for entry in top_entries:
            token = _entry_token(entry)
            if not token:
                continue
            logprob = _entry_logprob(entry)
            exact.setdefault(token, logprob)
            stripped_token.setdefault(token.strip(), token)

        prefer_space = mode == "generate"
        matched: dict[str, float] = {}
        matched_token: dict[str, str] = {}
        missing: list[str] = []
        for label in labels:
            candidates = (
                (f" {label}", label) if prefer_space else (label, f" {label}")
            )
            token_hit = next((c for c in candidates if c in exact), None)
            if token_hit is None:
                token_hit = stripped_token.get(label.strip())
            if token_hit is None:
                missing.append(label)
            else:
                matched[label] = exact[token_hit]
                matched_token[label] = token_hit

        if missing:
            raise UnsupportedLabelTokenization(
                f"Label {missing[0]!r} is not within the top-{len(top_entries)} "
                "logprobs returned by Ollama at the answer position, so it "
                "cannot be scored. Increase JEV_OLLAMA_TOP_LOGPROBS or use "
                "more compact labels."
            )

        seen: dict[str, str] = {}
        for label, token in matched_token.items():
            other = seen.get(token)
            if other is not None:
                raise UnsupportedLabelTokenization(
                    f"labels {other!r} and {label!r} matched the same token "
                    f"{token!r}; the label set is not distinguishable by "
                    "this model"
                )
            seen[token] = label
        return matched
