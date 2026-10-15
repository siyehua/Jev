"""Stable error taxonomy (SPEC.md §9).

Every client-visible failure maps to one JevError with an HTTP status,
a stable machine code, and a human-readable message that never includes
stack traces or raw logits.
"""

from __future__ import annotations


class JevError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class InvalidRequest(JevError):
    status_code = 400
    code = "invalid_request"


class ModelNotFound(JevError):
    status_code = 404
    code = "model_not_found"


class InputTooLarge(JevError):
    status_code = 413
    code = "input_too_large"


class UnsupportedLabelTokenization(JevError):
    status_code = 422
    code = "unsupported_label_tokenization"


class ModelNotReady(JevError):
    status_code = 503
    code = "model_not_ready"


class InferenceError(JevError):
    status_code = 500
    code = "inference_error"


class Unauthorized(JevError):
    status_code = 401
    code = "unauthorized"
