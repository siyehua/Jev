#!/usr/bin/env python3
"""Jev Decision API entry point.

Usage:
    python run.py [--host 0.0.0.0] [--port 8000]

Configuration comes from environment variables (see jev/config.py).
Backend examples:
    # in-process Transformers direct-logits (default)
    python run.py

    # remote Ollama server, one-token generation fallback
    JEV_BACKEND=ollama \
    JEV_MODEL_ID=qwen2.5:0.5b \
    JEV_OLLAMA_BASE_URL=http://ollama-host:11434 \
    python run.py
"""

from __future__ import annotations

import argparse

import uvicorn

from jev.app import create_app
from jev.backends import create_backend
from jev.config import DecisionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Jev Decision API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = DecisionConfig.from_env()
    backend = create_backend(config)
    app = create_app(config, backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
