"""ULID-style, time-sortable request IDs (no external dependency)."""

from __future__ import annotations

import os
import time

_ENCODING = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32, lowercase


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        chars.append(_ENCODING[rem])
    return "".join(reversed(chars))


def new_ulid() -> str:
    """48-bit millisecond timestamp + 80 bits of randomness, 26 chars."""
    ts_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    return _encode(ts_ms & ((1 << 48) - 1), 10) + _encode(rand, 16)


def new_decision_id() -> str:
    return "dec_" + new_ulid()
