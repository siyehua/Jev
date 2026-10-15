"""Deterministic in-process backend for tests and offline development.

Uses a character-level token model: a one-character label is exactly one
token, while multi-character labels such as ``billing`` split into
several tokens and are rejected with unsupported_label_tokenization —
mirroring the real tokenizers' behavior closely enough for tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jev.backends.base import STRATEGY_LABEL_LOGITS, DecisionBackend, ScoreResult
from jev.config import DecisionConfig
from jev.errors import InputTooLarge, UnsupportedLabelTokenization


class FakeBackend(DecisionBackend):
    name = "fake"

    def __init__(self, logit_fn: Callable[[str], float] | None = None):
        self._logit_fn = logit_fn or (lambda label: -1.0)
        self._config: DecisionConfig | None = None
        self._loaded = False
        self.load_count = 0
        self.score_calls = 0

    def load(self, config: DecisionConfig) -> None:
        self.load_count += 1
        self._config = config
        self._loaded = True

    def is_ready(self) -> bool:
        return self._loaded

    def health(self) -> dict:
        return {
            "backend": self.name,
            "model_loaded": self._loaded,
        }

    def score_labels(self, prompt: str, labels: Sequence[str]) -> ScoreResult:
        self.score_calls += 1
        labels = list(labels)
        multi = [label for label in labels if len(label) != 1]
        if multi:
            raise UnsupportedLabelTokenization(
                f"Label {multi[0]!r} does not tokenize as one next token "
                "for this model and prompt template. Use compact labels or "
                "configure a compatible template."
            )
        if self._config is not None and len(prompt) > self._config.max_prompt_tokens:
            raise InputTooLarge(
                f"prompt is {len(prompt)} tokens; the configured limit is "
                f"{self._config.max_prompt_tokens}"
            )
        return ScoreResult(
            scores={label: self._logit_fn(label) for label in labels},
            strategy=STRATEGY_LABEL_LOGITS,
            generated_tokens=0,
            prompt_tokens=len(prompt),
            tokenize_ms=0.0,
            forward_ms=0.0,
            postprocess_ms=0.0,
        )
