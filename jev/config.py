"""Server configuration, loaded from environment variables.

All limits are enforced server-side; model selection is never taken from
untrusted requests (see SPEC.md §11).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    return float(raw)


VALID_LABEL_MODES = ("letters", "criteria_keys")
VALID_DEVICE = ("auto", "cpu", "cuda")
VALID_DTYPE = ("auto", "float32", "float16", "bfloat16")
VALID_CHAT_TEMPLATE = ("auto", "true", "false")
VALID_BACKENDS = ("transformers", "ollama")
VALID_OLLAMA_MODE = ("chat", "generate")


@dataclass(frozen=True)
class DecisionConfig:
    """Immutable service configuration."""

    model_id: str = _DEFAULT_MODEL
    # Models the server will answer for. MVP serves exactly one loaded model;
    # other names produce 404 model_not_found.
    allowed_models: tuple[str, ...] = ()

    # Backend runtime: "transformers" = in-process direct logits (decode-free);
    # "ollama" = one-token generation fallback via a remote Ollama server.
    backend: str = "transformers"

    # Ollama backend settings. base_url is server-side configuration only —
    # never taken from requests (SPEC.md §11).
    ollama_base_url: str = "http://localhost:11434"
    # chat -> /api/chat (server applies the model's chat template);
    # generate -> /api/generate with raw=True (deterministic raw prompt).
    ollama_mode: str = "chat"
    # Top-k logprobs requested at the answer position. Labels outside the
    # returned top-k cannot be scored -> 422 unsupported_label_tokenization.
    ollama_top_logprobs: int = 20

    # Device/dtype policy (documented per SPEC.md §7.1):
    #   device auto  -> cuda if available else cpu
    #   dtype  auto  -> cpu: float32; cuda: bfloat16 if supported else float16
    device: str = "auto"
    dtype: str = "auto"

    # letters       -> internal compact keys A..Z (spec §7.3), mapped back to
    #                 criteria keys in the response
    # criteria_keys -> criteria keys must themselves be single-token labels
    label_mode: str = "letters"

    # auto -> apply the tokenizer's chat template when one exists
    use_chat_template: str = "auto"

    # Request limits (SPEC.md §9: input limits must be configurable).
    max_input_chars: int = 8000
    max_instructions_chars: int = 4000
    max_criteria_value_chars: int = 2000
    max_criteria_key_chars: int = 128
    max_prompt_tokens: int = 4096
    max_body_bytes: int = 1_048_576
    request_timeout_seconds: float = 120.0
    # SPEC.md §10: start with sequential execution.
    max_concurrency: int = 1

    # Optional bearer-token auth for network-accessible deployments.
    api_key: str | None = None

    log_level: str = "INFO"
    # Only enable for trusted development environments (SPEC.md §12).
    log_inputs: bool = False

    def __post_init__(self) -> None:
        if self.label_mode not in VALID_LABEL_MODES:
            raise ValueError(f"label_mode must be one of {VALID_LABEL_MODES}")
        if self.device not in VALID_DEVICE:
            raise ValueError(f"device must be one of {VALID_DEVICE}")
        if self.dtype not in VALID_DTYPE:
            raise ValueError(f"dtype must be one of {VALID_DTYPE}")
        if self.use_chat_template not in VALID_CHAT_TEMPLATE:
            raise ValueError(
                f"use_chat_template must be one of {VALID_CHAT_TEMPLATE}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}")
        if self.ollama_mode not in VALID_OLLAMA_MODE:
            raise ValueError(
                f"ollama_mode must be one of {VALID_OLLAMA_MODE}"
            )
        if self.ollama_top_logprobs < 1:
            raise ValueError("ollama_top_logprobs must be >= 1")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must be an http(s) URL")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if not self.model_id:
            raise ValueError("model_id must not be empty")

    @property
    def served_model(self) -> str:
        return self.model_id

    def model_allowed(self, model: str) -> bool:
        allowed = self.allowed_models or (self.model_id,)
        return model in allowed

    @classmethod
    def from_env(cls) -> "DecisionConfig":
        allowed_raw = _env("JEV_ALLOWED_MODELS", "")
        allowed = tuple(
            m.strip() for m in allowed_raw.split(",") if m.strip()
        )
        return cls(
            model_id=_env("JEV_MODEL_ID", _DEFAULT_MODEL),
            allowed_models=allowed,
            backend=_env("JEV_BACKEND", "transformers"),
            ollama_base_url=_env("JEV_OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_mode=_env("JEV_OLLAMA_MODE", "chat"),
            ollama_top_logprobs=_env_int("JEV_OLLAMA_TOP_LOGPROBS", 20),
            device=_env("JEV_DEVICE", "auto"),
            dtype=_env("JEV_DTYPE", "auto"),
            label_mode=_env("JEV_LABEL_MODE", "letters"),
            use_chat_template=_env("JEV_USE_CHAT_TEMPLATE", "auto"),
            max_input_chars=_env_int("JEV_MAX_INPUT_CHARS", 8000),
            max_instructions_chars=_env_int("JEV_MAX_INSTRUCTIONS_CHARS", 4000),
            max_criteria_value_chars=_env_int("JEV_MAX_CRITERIA_VALUE_CHARS", 2000),
            max_criteria_key_chars=_env_int("JEV_MAX_CRITERIA_KEY_CHARS", 128),
            max_prompt_tokens=_env_int("JEV_MAX_PROMPT_TOKENS", 4096),
            max_body_bytes=_env_int("JEV_MAX_BODY_BYTES", 1_048_576),
            request_timeout_seconds=_env_float("JEV_REQUEST_TIMEOUT_SECONDS", 120.0),
            max_concurrency=_env_int("JEV_MAX_CONCURRENCY", 1),
            api_key=_env("JEV_API_KEY"),
            log_level=_env("JEV_LOG_LEVEL", "INFO"),
            log_inputs=_env_bool("JEV_LOG_INPUTS", False),
        )
