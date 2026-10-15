"""Decision orchestration (SPEC.md §5, §7).

Pure application logic: validates limits, builds the prompt, delegates
scoring to the configured backend (direct-logits Transformers or a
generation fallback such as Ollama), normalizes scores, and builds the
response. Runs inside a worker thread; all model runtime specifics stay
behind the DecisionBackend interface.
"""

from __future__ import annotations

import logging
import time

from jev.backends.base import DecisionBackend
from jev.config import DecisionConfig
from jev.errors import InputTooLarge, ModelNotFound, ModelNotReady
from jev.ids import new_decision_id
from jev.prompt import BuiltPrompt, PromptBuilder
from jev.schemas import DecisionRequest, DecisionResponse, Latency, Usage
from jev.scoring import argmax_first, softmax

logger = logging.getLogger("jev.decision")


class DecisionService:
    def __init__(self, config: DecisionConfig, backend: DecisionBackend):
        self.config = config
        self.backend = backend
        self.prompt_builder = PromptBuilder(label_mode=config.label_mode)

    def handle_decision(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()

        if not self.config.model_allowed(request.model):
            raise ModelNotFound(
                f"model {request.model!r} is not configured. This server "
                f"serves {self.config.served_model!r}."
            )
        if not self.backend.is_ready():
            raise ModelNotReady("model is not loaded or is unavailable")

        self._check_limits(request)

        built = self.prompt_builder.build(
            instructions=request.instructions,
            input_text=request.input,
            criteria=request.criteria,
        )
        final_prompt = self.backend.render_prompt(built.prompt)

        result = self.backend.score_labels(final_prompt, list(built.labels))

        label_scores = [result.scores[label] for label in built.labels]
        probabilities = softmax(label_scores, request.scoring.temperature)
        choice_index = argmax_first(probabilities)
        total_ms = (time.perf_counter() - started) * 1000

        keys = built.keys
        return DecisionResponse(
            id=new_decision_id(),
            model=self.config.served_model,
            # Honest execution metadata from the backend (SPEC.md §8):
            strategy=result.strategy,
            choice=keys[choice_index],
            probabilities={key: probabilities[i] for i, key in enumerate(keys)},
            scores={
                key: result.scores[label]
                for key, label in zip(keys, built.labels)
            },
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
            ),
            request_id=request.request_id,
            backend=self.backend.name,
            label_mode=self.config.label_mode,
            prompt_template_version=built.template_version,
            latency=Latency(
                tokenize_ms=round(result.tokenize_ms, 3),
                forward_ms=round(result.forward_ms, 3),
                postprocess_ms=round(result.postprocess_ms, 3),
                total_ms=round(total_ms, 3),
            ),
        )

    def _check_limits(self, request: DecisionRequest) -> None:
        cfg = self.config
        if len(request.input) > cfg.max_input_chars:
            raise InputTooLarge(
                f"input is {len(request.input)} chars; the configured limit "
                f"is {cfg.max_input_chars}"
            )
        if len(request.instructions) > cfg.max_instructions_chars:
            raise InputTooLarge(
                f"instructions is {len(request.instructions)} chars; the "
                f"configured limit is {cfg.max_instructions_chars}"
            )
        for key, value in request.criteria.items():
            if len(key) > cfg.max_criteria_key_chars:
                raise InputTooLarge(
                    f"criteria key {key!r} exceeds the configured limit of "
                    f"{cfg.max_criteria_key_chars} chars"
                )
            if len(value) > cfg.max_criteria_value_chars:
                raise InputTooLarge(
                    f"criteria value for {key!r} exceeds the configured "
                    f"limit of {cfg.max_criteria_value_chars} chars"
                )
