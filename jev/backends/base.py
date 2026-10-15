"""Backend contract (SPEC.md §8).

Application logic (validation, prompt building, scoring, response
construction) never touches a model runtime directly; it only talks to
this interface. Alternative backends (Transformers direct-logits, Ollama
generation fallback, SGLang/MLX adapters, ...) implement it.

The interface is string-centric at the scoring boundary: each backend
owns tokenization, label-token validation, and the forward pass, and
returns per-label raw scores plus honest execution metadata.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jev.config import DecisionConfig

STRATEGY_LABEL_LOGITS = "label_logits"
STRATEGY_SINGLE_TOKEN_GENERATION = "single_token_generation"


@dataclass(frozen=True)
class ScoreResult:
    """Scoring outcome for one decision request.

    ``scores`` maps each label to the raw score the scorer will softmax:
    next-token logits for decode-free backends, or token log-probabilities
    for generation-fallback backends. Both are equivalent under the
    label-restricted softmax because softmax is shift-invariant.

    ``strategy`` and ``generated_tokens`` describe what actually executed
    and must be reported honestly in the API response (SPEC.md §8).
    """

    scores: Mapping[str, float]
    strategy: str
    generated_tokens: int
    prompt_tokens: int
    tokenize_ms: float
    forward_ms: float
    postprocess_ms: float


class DecisionBackend(abc.ABC):
    """Owns the model/tokenizer lifecycle and the scoring execution."""

    name: str = "abstract"

    @abc.abstractmethod
    def load(self, config: DecisionConfig) -> None:
        """Initialize the backend. Called once at service startup; must
        fail fast with a clear error when the runtime or model is not
        usable (SPEC.md §8)."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when the backend can serve requests."""

    @abc.abstractmethod
    def health(self) -> dict:
        """Health/status details for GET /healthz."""

    def render_prompt(self, prompt: str) -> str:
        """Final prompt string handed to the model runtime. Backends whose
        runtime applies a chat template server-side (e.g. Ollama /api/chat)
        return the prompt unchanged."""
        return prompt

    @abc.abstractmethod
    def score_labels(self, prompt: str, labels: Sequence[str]) -> ScoreResult:
        """Score every label for the exact prompt.

        Backends validate that the labels are scoreable in this exact
        prompt context (exactly-one-token continuation, or presence in the
        returned top-k logprobs) and raise UnsupportedLabelTokenization
        otherwise. Must never claim decode-free execution when generation
        occurred — report the actual strategy in ScoreResult."""
