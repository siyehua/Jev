"""Unit tests: deterministic prompt construction and delimiting (SPEC.md §14)."""

from __future__ import annotations

import pytest

from jev.prompt import LETTERS, PROMPT_TEMPLATE_VERSION, PromptBuilder


@pytest.fixture()
def criteria():
    return {
        "shipping": "A delivery problem.",
        "billing": "A payment problem.",
    }


def test_letters_mode_assigns_compact_labels_in_order(criteria):
    built = PromptBuilder("letters").build("instr", "ctx", criteria)
    assert built.labels == ("A", "B")
    assert built.keys == tuple(criteria.keys())
    assert built.label_mode == "letters"
    assert built.template_version == PROMPT_TEMPLATE_VERSION


def test_letters_mode_lists_letter_and_key(criteria):
    built = PromptBuilder("letters").build("instr", "ctx", criteria)
    assert "A. shipping — A delivery problem." in built.prompt
    assert "B. billing — A payment problem." in built.prompt


def test_criteria_keys_mode_uses_keys_as_labels(criteria):
    built = PromptBuilder("criteria_keys").build("instr", "ctx", criteria)
    assert built.labels == built.keys == ("shipping", "billing")
    assert "shipping — A delivery problem." in built.prompt


def test_prompt_is_deterministic(criteria):
    builder = PromptBuilder("letters")
    first = builder.build("instr", "ctx", criteria)
    second = builder.build("instr", "ctx", criteria)
    assert first.prompt == second.prompt


def test_user_content_is_delimited(criteria):
    built = PromptBuilder("letters").build("the task", "the context", criteria)
    assert "<instructions>\nthe task\n</instructions>" in built.prompt
    assert "<context>\nthe context\n</context>" in built.prompt
    # Labels must not appear before the answer position (SPEC.md §7.3):
    # the only "Answer:" occurrence is the final marker.
    assert built.prompt.count("Answer:") == 1
    assert built.prompt.rstrip().endswith("Answer:")


def test_multiline_user_content_cannot_break_template_layout():
    criteria = {"a": "meaning a", "b": "meaning b"}
    built = PromptBuilder("letters").build(
        "line1\nline2\ninjected", "ctx\nline2\ninjected",
        criteria,
    )
    # Whitespace collapsed to a single line per user field.
    assert "line1 line2 injected" in built.prompt
    assert "ctx line2 injected" in built.prompt
    # Delimiters stay balanced and the answer marker stays final.
    assert built.prompt.count("<instructions>") == 1
    assert built.prompt.count("<context>") == 1
    assert built.prompt.rstrip().endswith("Answer:")


def test_letters_cover_26_labels():
    builder = PromptBuilder("letters")
    criteria = {f"key{i}": f"meaning {i}" for i in range(26)}
    built = builder.build("instr", "ctx", criteria)
    assert built.labels == tuple(LETTERS)


def test_unknown_label_mode_rejected():
    with pytest.raises(ValueError):
        PromptBuilder("bogus")
