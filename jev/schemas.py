"""Request/response schemas (SPEC.md §6.1).

Structural validation (types, required fields, 2–26 criteria, non-empty
text) happens here and surfaces as 400 invalid_request. Size and token
limits are enforced by the service layer (413 input_too_large) because
they are configurable at runtime.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MIN_CRITERIA = 2
MAX_CRITERIA = 26


class ScoringParams(BaseModel):
    strategy: Literal["label_logits"] = "label_logits"
    temperature: float = Field(default=1.0, gt=0.0)

    @field_validator("temperature")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("temperature must be a finite number")
        return v


class DecisionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    model: str = Field(min_length=1, max_length=200)
    input: str
    instructions: str
    criteria: dict[str, str]
    scoring: ScoringParams = Field(default_factory=ScoringParams)
    request_id: str | None = Field(default=None, max_length=256)

    @field_validator("model")
    @classmethod
    def _model_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be blank")
        return v

    @field_validator("input", "instructions")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty after trimming")
        return v

    @field_validator("criteria")
    @classmethod
    def _criteria_valid(cls, v: dict[str, str]) -> dict[str, str]:
        if not (MIN_CRITERIA <= len(v) <= MAX_CRITERIA):
            raise ValueError(
                f"criteria must contain between {MIN_CRITERIA} and "
                f"{MAX_CRITERIA} entries, got {len(v)}"
            )
        for key, value in v.items():
            if not key.strip():
                raise ValueError("criteria keys must be non-empty")
            if not value.strip():
                raise ValueError(
                    f"criteria value for key {key!r} must be non-empty"
                )
        return v


class Usage(BaseModel):
    prompt_tokens: int
    generated_tokens: int


class Latency(BaseModel):
    tokenize_ms: float
    forward_ms: float
    postprocess_ms: float
    total_ms: float


class DecisionResponse(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    model: str
    # Honest strategy: what actually executed. Direct-logits backends
    # report label_logits; generation fallbacks (e.g. Ollama) report
    # single_token_generation (SPEC.md §8).
    strategy: Literal["label_logits", "single_token_generation"]
    choice: str
    probabilities: dict[str, float]
    scores: dict[str, float]
    usage: Usage
    request_id: str | None = None
    backend: str
    label_mode: str
    prompt_template_version: str
    latency: Latency
