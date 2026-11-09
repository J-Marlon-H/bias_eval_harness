"""Shared prompt construction for BBQ and MMLU."""

from __future__ import annotations

from typing import Any, Sequence


def build_prompt(
    item: dict[str, Any],
    *,
    system_prompt: str | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Build the common multiple-choice prompt, using a model chat template if available."""
    options: Sequence[str] = item["options"]
    if not 2 <= len(options) <= 26:
        raise ValueError(f"Expected 2-26 answer options, received {len(options)}")
    labels = [chr(ord("A") + index) for index in range(len(options))]
    question = []
    if item.get("context"):
        question.append(f"Context: {item['context']}")
    question.append(f"Question: {item['question']}")
    question.extend(f"{label}. {option}" for label, option in zip(labels, options))
    question.append("Answer with the letter of the correct option.")
    user_message = "\n".join(question)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        model_name = str(getattr(tokenizer, "name_or_path", "")).casefold()
        if "qwen3" in model_name:
            template_kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    if system_prompt:
        return f"System: {system_prompt}\n\n{user_message}"
    return user_message
