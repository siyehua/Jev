"""FastAPI application wiring (SPEC.md §6, §9, §11, §12)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from jev.backends.base import DecisionBackend
from jev.config import DecisionConfig
from jev.errors import InferenceError, JevError, Unauthorized
from jev.ids import new_decision_id
from jev.schemas import DecisionRequest, DecisionResponse
from jev.service import DecisionService

access_logger = logging.getLogger("jev.access")


def _configure_logging(config: DecisionConfig) -> None:
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    for name in ("jev.access", "jev.decision", "jev.backend"):
        logging.getLogger(name).setLevel(level)


def create_app(config: DecisionConfig, backend: DecisionBackend) -> FastAPI:
    _configure_logging(config)
    service = DecisionService(config, backend)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Model loads exactly once, at startup (SPEC.md §7.1, §13.4).
        backend.load(config)
        logging.getLogger("jev").info(
            "service ready: model=%s backend=%s", config.model_id, backend.name
        )
        yield

    app = FastAPI(
        title="Jev Decision API",
        version="0.1.0",
        description="Logits-based decision API. Returned probabilities are "
        "relative label scores, not calibrated confidence.",
        lifespan=lifespan,
    )

    async def require_auth(request: Request) -> None:
        if not config.api_key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or token.strip() != config.api_key:
            raise Unauthorized("missing or invalid API key")

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > config.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "input_too_large",
                            "message": (
                                f"request body exceeds the configured limit "
                                f"of {config.max_body_bytes} bytes"
                            ),
                        }
                    },
                )
        return await call_next(request)

    @app.exception_handler(JevError)
    async def jev_error_handler(request: Request, exc: JevError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": f"request validation failed: {details}",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": (
                        "not_found"
                        if exc.status_code == 404
                        else "invalid_request"
                    ),
                    "message": str(exc.detail),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        # Never leak stack traces to clients (SPEC.md §9).
        logging.getLogger("jev").exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "inference_error",
                    "message": "unexpected backend failure",
                }
            },
        )

    @app.get("/healthz")
    async def healthz():
        health = backend.health()
        ready = backend.is_ready()
        return {
            "status": "ok" if ready else "degraded",
            "model_loaded": ready,
            "backend": backend.name,
            **health,
        }

    @app.post(
        "/v1/decision",
        response_model=DecisionResponse,
        dependencies=[Depends(require_auth)],
    )
    async def decision(request: DecisionRequest):
        decision_id = new_decision_id()
        started = time.perf_counter()
        error_code: str | None = None
        response: DecisionResponse | None = None
        try:
            async with semaphore:
                try:
                    response = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, service.handle_decision, request
                        ),
                        timeout=config.request_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    raise InferenceError(
                        "inference timed out after "
                        f"{config.request_timeout_seconds}s"
                    )
            return response
        except JevError as exc:
            error_code = exc.code
            raise
        except Exception:
            # Convert to a stable 500 without leaking details; log the
            # traceback server-side only (SPEC.md §9).
            error_code = "inference_error"
            logging.getLogger("jev").exception("backend failure")
            raise InferenceError("unexpected backend failure") from None
        finally:
            # Structured access log without user content (SPEC.md §12).
            record = {
                "id": decision_id,
                "request_id": request.request_id,
                "model": request.model,
                "backend": backend.name,
                "strategy": request.scoring.strategy,
                "temperature": request.scoring.temperature,
                "criteria_count": len(request.criteria),
                "status": "error" if error_code else "ok",
                "error_code": error_code,
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                "prompt_tokens": response.usage.prompt_tokens if response else None,
                "tokenize_ms": response.latency.tokenize_ms if response else None,
                "forward_ms": response.latency.forward_ms if response else None,
                "postprocess_ms": (
                    response.latency.postprocess_ms if response else None
                ),
            }
            if config.log_inputs:
                record["input"] = request.input
                record["instructions"] = request.instructions
            access_logger.info(json.dumps(record, ensure_ascii=False))

    return app
