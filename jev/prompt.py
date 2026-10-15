"""Deterministic, versioned prompt construction (SPEC.md §7.2).

The template is plain application code — no LLM is used to write prompts.
User-supplied content is wrapped in explicit delimiters to reduce prompt
injection risk; the scorer only ever returns one of the supplied labels,
so injected instructions cannot widen the output space.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_TEMPLATE_VERSION = "v2"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_TEMPLATE = """You are a classification engine. Select exactly one label from the available labels.
Do not explain your answer.

Task:
<instructions>
{instructions}
</instructions>

Context:
<context>
{input}
</context>

Candidate labels:
{candidate_descriptions}

Respond with the letter of the best label only:
Answer:"""


def _single_line(text: str, max_len: int | None = None) -> str:
    """Collapse whitespace so user content cannot break the template layout."""
    cleaned = " ".join(text.strip().split())
    if max_len is not None and len(cleaned) > max_len:
        # Length limits are enforced at the service layer; this is a guard.
        cleaned = cleaned[:max_len]
    return cleaned


@dataclass(frozen=True)
class BuiltPrompt:
    prompt: str
    template_version: str
    label_mode: str
    # Ordered business keys (criteria order as received).
    keys: tuple[str, ...]
    # Compact labels aligned with keys by position (letters mode: A, B, ...;
    # criteria_keys mode: identical to keys).
    labels: tuple[str, ...]


class PromptBuilder:
    """Renders the deterministic decision prompt."""

    def __init__(self, label_mode: str = "letters"):
        if label_mode not in ("letters", "criteria_keys"):
            raise ValueError(f"unknown label_mode: {label_mode}")
        self.label_mode = label_mode

    def build(self, instructions: str, input_text: str, criteria: dict[str, str]) -> BuiltPrompt:
        keys = tuple(criteria.keys())
        descriptions = []
        labels: list[str]
        if self.label_mode == "letters":
            labels = [LETTERS[i] for i in range(len(keys))]
            for label, key in zip(labels, keys):
                descriptions.append(
                    f"{label}. {_single_line(key)} — {_single_line(criteria[key])}"
                )
        else:
            labels = list(keys)
            for key in keys:
                descriptions.append(
                    f"{_single_line(key)} — {_single_line(criteria[key])}"
                )
        prompt = _TEMPLATE.format(
            instructions=_single_line(instructions),
            input=_single_line(input_text),
            candidate_descriptions="\n".join(descriptions),
        )
        return BuiltPrompt(
            prompt=prompt,
            template_version=PROMPT_TEMPLATE_VERSION,
            label_mode=self.label_mode,
            keys=keys,
            labels=tuple(labels),
        )
