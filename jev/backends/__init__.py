from jev.backends.base import DecisionBackend, ScoreResult
from jev.backends.fake import FakeBackend
from jev.backends.ollama_backend import OllamaBackend
from jev.backends.transformers_backend import TransformersBackend


def create_backend(config):
    """Build the configured backend (JEV_BACKEND env var)."""
    if config.backend == "ollama":
        return OllamaBackend()
    if config.backend == "transformers":
        return TransformersBackend()
    raise ValueError(f"unknown backend: {config.backend!r}")


__all__ = [
    "DecisionBackend",
    "ScoreResult",
    "FakeBackend",
    "OllamaBackend",
    "TransformersBackend",
    "create_backend",
]
