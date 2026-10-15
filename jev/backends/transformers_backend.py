"""Hugging Face Transformers direct-logits backend (SPEC.md §7, §8).

Device/dtype policy:
    device=auto -> CUDA when available, otherwise CPU.
    dtype=auto  -> float32 on CPU; bfloat16 on CUDA when supported,
                   else float16.

The model and tokenizer are loaded exactly once in ``load()`` (service
startup), the model is switched to ``eval()`` and every forward pass runs
under ``torch.inference_mode()``. ``model.generate()`` is never called:
this backend is decode-free and reports ``generated_tokens: 0``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jev.backends.base import (
    STRATEGY_LABEL_LOGITS,
    DecisionBackend,
    ScoreResult,
)
from jev.config import DecisionConfig
from jev.errors import InputTooLarge, UnsupportedLabelTokenization
from jev.tokens import resolve_label_tokens

logger = logging.getLogger("jev.backend")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype in mapping:
        return mapping[dtype]
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class TransformersBackend(DecisionBackend):
    name = "transformers"

    def __init__(self) -> None:
        self._config: DecisionConfig | None = None
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._use_chat_template = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def load(self, config: DecisionConfig) -> None:
        self._config = config
        device = _resolve_device(config.device)
        dtype = _resolve_dtype(config.dtype, device)
        logger.info(
            "loading model=%s device=%s dtype=%s",
            config.model_id,
            device,
            dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        try:
            # transformers >= 4.56 renamed torch_dtype -> dtype
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, dtype=dtype
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=dtype
            )
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype
        if config.use_chat_template == "auto":
            self._use_chat_template = bool(getattr(tokenizer, "chat_template", None))
        else:
            self._use_chat_template = config.use_chat_template == "true"
        self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        """Fail at startup with a clear error for incompatible models."""
        try:
            ids = self._tokenizer("compatibility check", return_tensors="pt")
            ids = {k: v.to(self._device) for k, v in ids.items()}
            with torch.inference_mode():
                outputs = self._model(**ids)
            logits = outputs.logits
            if logits is None or logits.dim() != 3:
                raise RuntimeError(
                    "model output has no causal-LM logits tensor"
                )
            vocab = logits.shape[-1]
            if vocab <= max(self._tokenizer("labels").input_ids):
                raise RuntimeError(
                    "logits vocabulary smaller than tokenizer vocabulary"
                )
        except RuntimeError as exc:
            raise RuntimeError(
                f"model {self._config.model_id!r} is not a supported causal "
                f"language model for the transformers direct-logits backend: "
                f"{exc}"
            ) from exc

    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def health(self) -> dict:
        return {
            "backend": self.name,
            "model": self._config.model_id if self._config else None,
            "model_loaded": self.is_ready(),
            "device": str(self._device) if self._device else None,
            "dtype": str(self._dtype).replace("torch.", "") if self._dtype else None,
            "chat_template": self._use_chat_template,
        }

    # -- prompt handling ---------------------------------------------------

    def render_prompt(self, prompt: str) -> str:
        if not self._use_chat_template:
            return prompt
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def uses_chat_template(self) -> bool:
        return self._use_chat_template

    def continuation_formats(self) -> tuple[str, ...]:
        if self._use_chat_template:
            # After an assistant-turn header the label usually starts
            # without a leading space; try that first, deterministically.
            return ("{label}", " {label}")
        return (" {label}",)

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer(
            text, add_special_tokens=add_special_tokens
        ).input_ids

    # -- scoring -----------------------------------------------------------

    def score_labels(self, prompt: str, labels: Sequence[str]) -> ScoreResult:
        labels = list(labels)
        add_special = not self._use_chat_template

        tokenize_started = time.perf_counter()
        prompt_ids = self._tokenizer(
            prompt, add_special_tokens=add_special
        ).input_ids
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000

        if self._config is not None and len(prompt_ids) > self._config.max_prompt_tokens:
            raise InputTooLarge(
                f"prompt is {len(prompt_ids)} tokens; the configured limit "
                f"is {self._config.max_prompt_tokens}"
            )

        try:
            resolution = resolve_label_tokens(
                tokenize=lambda text: self._tokenizer(
                    text, add_special_tokens=add_special
                ).input_ids,
                prompt=prompt,
                labels=labels,
                continuation_formats=list(self.continuation_formats()),
            )
        except ValueError as exc:
            raise UnsupportedLabelTokenization(
                f"Label set not supported by the loaded tokenizer/template: "
                f"{exc} Use compact labels or configure a compatible template."
            ) from exc
        label_token_ids = [resolution.token_ids[label] for label in labels]

        forward_started = time.perf_counter()
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        # Sequential execution: one forward pass at a time (SPEC.md §10).
        with self._lock, torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids.to(self._device),
                attention_mask=attention_mask.to(self._device),
            )
        next_token_logits = outputs.logits[0, -1, :]
        forward_ms = (time.perf_counter() - forward_started) * 1000

        post_started = time.perf_counter()
        scores = {
            label: float(next_token_logits[int(token_id)])
            for label, token_id in zip(labels, label_token_ids)
        }
        postprocess_ms = (time.perf_counter() - post_started) * 1000

        return ScoreResult(
            scores=scores,
            strategy=STRATEGY_LABEL_LOGITS,
            generated_tokens=0,
            prompt_tokens=len(prompt_ids),
            tokenize_ms=tokenize_ms,
            forward_ms=forward_ms,
            postprocess_ms=postprocess_ms,
        )
