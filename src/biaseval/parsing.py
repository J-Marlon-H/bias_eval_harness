"""Robust parsing for generated multiple-choice answers."""

from __future__ import annotations

import re
from typing import Sequence

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_LETTER = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove complete reasoning blocks emitted by models such as Qwen."""
    return _THINK_BLOCK.sub("", text).strip()


def parse_answer(text: str, options: Sequence[str]) -> int | str:
    """Return a zero-based option index, or `invalid` if no answer can be extracted."""
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    cleaned = strip_think_blocks(text)
    valid_letters = {chr(ord("A") + index): index for index in range(len(options))}
    for match in _LETTER.finditer(cleaned):
        letter = match.group(1).upper()
        if letter in valid_letters:
            return valid_letters[letter]

    matches = []
    for index, option in enumerate(options):
        if not option:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(option.strip())}(?!\w)",
            re.IGNORECASE,
        )
        match = pattern.search(cleaned)
        if match:
            matches.append((match.start(), index))
    return min(matches)[1] if matches else "invalid"
