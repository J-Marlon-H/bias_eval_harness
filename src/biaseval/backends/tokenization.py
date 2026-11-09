"""Tokenizer helpers for scoring option-letter completions."""

from __future__ import annotations

from typing import Any


def encode_answer_candidates(
    tokenizer: Any,
    prompt: str,
    n_options: int,
) -> tuple[list[int], list[list[int]]]:
    """Encode the exact continuation token sequence for each answer letter."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Cannot score an empty prompt")
    prefix = "" if prompt[-1].isspace() else " "
    candidates = []
    for index in range(n_options):
        label = chr(ord("A") + index)
        full_ids = tokenizer.encode(prompt + prefix + label, add_special_tokens=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "Tokenizer changes prompt tokens at the answer boundary; "
                f"cannot score letter {label!r} as a continuation"
            )
        continuation = full_ids[len(prompt_ids):]
        if not continuation:
            raise RuntimeError(f"Tokenizer produced no answer tokens for letter {label!r}")
        candidates.append(continuation)
    return prompt_ids, candidates
