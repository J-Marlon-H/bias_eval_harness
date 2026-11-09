"""Load and normalize the MMLU test split."""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

MMLU_DATASET_ID = "cais/mmlu"


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    required = {"subject", "question", "choices", "answer"}
    missing = required.difference(row)
    if missing:
        raise ValueError(f"MMLU item is missing required fields: {sorted(missing)}")
    choices = list(row["choices"])
    if len(choices) != 4:
        raise ValueError(f"MMLU items must have four options, received {len(choices)}")
    answer = int(row["answer"])
    if answer not in range(4):
        raise ValueError(f"MMLU answer must be 0, 1, 2, or 3; received {answer}")
    subject = str(row["subject"])
    return {
        "id": f"{subject}:{index}",
        "subject": subject,
        "question_index": str(index),
        "context": "",
        "question": str(row["question"]),
        "options": [str(choice) for choice in choices],
        "label": answer,
        "source": "mmlu",
    }


def load_mmlu(*, token: str | None = None) -> pd.DataFrame:
    """Load MMLU's public test split; Hugging Face caches the downloaded files."""
    dataset = load_dataset(MMLU_DATASET_ID, "all", split="test", token=token)
    return pd.DataFrame(_normalize_row(row, index) for index, row in enumerate(dataset))
