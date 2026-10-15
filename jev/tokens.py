"""Label token resolution (SPEC.md §7.3).

Never assume `tokenizer.encode(label)` matches the token the model would
emit right after the prompt: tokenization depends on preceding text. For
each label we tokenize the exact prompt and the prompt concatenated with
the label in the exact continuation form used by the template, and only
accept the label if the continuation adds exactly one token.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelTokenResolution:
    """Validated label -> next-token mapping for one exact prompt context."""

    token_ids: dict[str, int]
    continuation_format: str  # e.g. " {label}"


def resolve_label_tokens(
    tokenize: Callable[[str], list[int]],
    prompt: str,
    labels: list[str],
    continuation_formats: list[str],
) -> LabelTokenResolution:
    """Resolve each label to exactly one next token.

    ``continuation_formats`` are tried in order (deterministic); the first
    format under which *every* label adds exactly one token wins. Raises
    ``ValueError`` describing the offending labels otherwise.
    """
    base_ids = tokenize(prompt)

    failures: dict[str, str] = {}
    for fmt in continuation_formats:
        token_ids: dict[str, int] = {}
        ok = True
        for label in labels:
            full_ids = tokenize(prompt + fmt.format(label=label))
            added = _added_tokens(base_ids, full_ids)
            if added is None:
                failures[label] = "changes tokenization of the prompt prefix"
                ok = False
            elif len(added) != 1:
                failures[label] = (
                    f"continuation adds {len(added)} tokens "
                    f"instead of exactly one"
                )
                ok = False
            else:
                token_ids[label] = added[0]
        if not ok:
            continue
        # Distinct labels must read distinct tokens, otherwise two labels
        # would be scored with identical logits.
        seen: dict[int, str] = {}
        for label, tid in token_ids.items():
            other = seen.get(tid)
            if other is not None:
                raise ValueError(
                    f"labels {other!r} and {label!r} map to the same token; "
                    "the label set is not distinguishable by this tokenizer"
                )
            seen[tid] = label
        return LabelTokenResolution(token_ids=token_ids, continuation_format=fmt)

    bad = "; ".join(f"{label!r}: {reason}" for label, reason in failures.items())
    raise ValueError(
        "labels do not tokenize as exactly one next token in the exact "
        f"continuation context ({bad}). Use compact label keys or switch "
        "label_mode to 'letters'."
    )


def _added_tokens(base_ids: list[int], full_ids: list[int]) -> list[int] | None:
    """Tokens added by the continuation, or None if the prefix re-tokenized."""
    if len(full_ids) < len(base_ids):
        return None
    if full_ids[: len(base_ids)] != base_ids:
        return None
    return full_ids[len(base_ids) :]
