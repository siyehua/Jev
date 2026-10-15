"""Unit tests: label token resolution, including multi-token labels
(SPEC.md §7.3, §14). Uses a fake character-level tokenizer."""

from __future__ import annotations

import pytest

from jev.tokens import resolve_label_tokens


def char_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]


PROMPT = "Answer:"


def test_leading_space_format_adds_two_tokens_for_char_tokenizer():
    # "Answer:" + " A" adds chars ' ' and 'A' -> two tokens for a char
    # tokenizer, which must be rejected.
    with pytest.raises(ValueError, match="instead of exactly one"):
        resolve_label_tokens(char_tokenize, PROMPT, ["A"], [" {label}"])


def test_bare_label_format_adds_one_token_for_char_tokenizer():
    resolution = resolve_label_tokens(char_tokenize, PROMPT, ["A", "B"], ["{label}"])
    assert resolution.token_ids == {"A": ord("A"), "B": ord("B")}
    assert resolution.continuation_format == "{label}"


def test_multi_token_label_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        resolve_label_tokens(char_tokenize, PROMPT, ["billing"], ["{label}"])
    assert "billing" in str(excinfo.value)
    assert "7 tokens" in str(excinfo.value)


def test_error_lists_all_offending_labels():
    with pytest.raises(ValueError) as excinfo:
        resolve_label_tokens(char_tokenize, PROMPT, ["billing", "A"], ["{label}"])
    message = str(excinfo.value)
    assert "billing" in message
    # "A" is fine and not blamed.
    assert "'A'" not in message


def test_prompt_prefix_retokenization_is_rejected():
    # A tokenizer whose continuation changes the prefix encoding must be
    # rejected even if it adds exactly one token overall.
    class WeirdTokenizer:
        def __call__(self, text: str) -> list[int]:
            # "X:" at the end retokenizes when a continuation appears.
            if text == "Go:":
                return [10, 11]
            if text == "Go:A":
                return [10, 99, 12]  # prefix changed (11 -> 99)
            return [ord(c) for c in text]

    with pytest.raises(ValueError, match="prefix"):
        resolve_label_tokens(WeirdTokenizer(), "Go:", ["A"], ["{label}"])


def test_first_fully_valid_format_wins_deterministically():
    resolution = resolve_label_tokens(
        char_tokenize, PROMPT, ["A"], [" {label}", "{label}"]
    )
    # " A" adds two char tokens, so the second format must be selected.
    assert resolution.continuation_format == "{label}"


def test_duplicate_token_ids_are_rejected():
    # Two different labels mapping to the same token id.
    def tokenize(text: str) -> list[int]:
        base = [ord(c) for c in text[:-1]]
        if text == PROMPT + "X" or text == PROMPT + "Y":
            return base + [42]  # both labels collapse onto token 42
        return [ord(c) for c in text]

    with pytest.raises(ValueError, match="same token"):
        resolve_label_tokens(tokenize, PROMPT, ["X", "Y"], ["{label}"])


def test_letters_a_to_z_all_resolve_for_char_tokenizer():
    labels = [chr(ord("A") + i) for i in range(26)]
    resolution = resolve_label_tokens(char_tokenize, PROMPT, labels, ["{label}"])
    assert set(resolution.token_ids.values()) == {ord(label) for label in labels}
