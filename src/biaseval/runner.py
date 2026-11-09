"""Small BBQ evaluation loop with saved predictions and metrics."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biaseval.metrics import compute_bbq_metrics
from biaseval.parsing import parse_answer
from biaseval.prompts import build_prompt
from biaseval.results import create_run_id, write_json


def run_evaluation(
    data: pd.DataFrame,
    backend: Any,
    *,
    results_dir: str | Path = "results",
    model_id: str | None = None,
    prompt_condition: str = "plain",
    system_prompt: str | None = None,
    scoring_mode: str = "auto",
    max_new_tokens: int = 8,
    seed: int = 42,
    invalid_rate_warning_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Evaluate BBQ rows and save one self-contained run directory."""
    required = {
        "id", "category", "example_id", "question_index", "condition",
        "context", "question", "options", "label", "target_loc", "unknown_loc",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"BBQ evaluation data is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot evaluate an empty dataset")
    if data["id"].astype(str).duplicated().any():
        raise ValueError("Evaluation item IDs must be unique")
    if scoring_mode not in {"auto", "loglik", "generation"}:
        raise ValueError("scoring_mode must be auto, loglik, or generation")
    if not 0 <= invalid_rate_warning_threshold <= 1:
        raise ValueError("invalid_rate_warning_threshold must be between 0 and 1")

    tokenizer = getattr(backend, "tokenizer", None)
    first_item = data.iloc[0].to_dict()
    probe_prompt = build_prompt(first_item, system_prompt=system_prompt, tokenizer=tokenizer)
    if scoring_mode == "auto" and hasattr(backend, "probe_logprobs"):
        backend.probe_logprobs(probe_prompt)
    actual_mode = scoring_mode
    if actual_mode == "auto":
        actual_mode = "loglik" if backend.capabilities["loglik"] else "generation"
    if actual_mode == "loglik" and not backend.capabilities["loglik"]:
        raise ValueError("The selected backend does not support log-likelihood scoring")
    if actual_mode == "generation" and not backend.capabilities["generate"]:
        raise ValueError("The selected backend does not support generation")

    identity = model_id or getattr(backend, "model_id", backend.__class__.__name__)
    run_id = create_run_id(identity, "bbq", prompt_condition)
    run_dir = Path(results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(data.iterrows(), total=len(data), desc="Evaluating BBQ")
    except ImportError:
        iterator = data.iterrows()

    predictions = []
    for _, series in iterator:
        item = series.to_dict()
        prompt = build_prompt(item, system_prompt=system_prompt, tokenizer=tokenizer)
        if actual_mode == "loglik":
            scores = backend.score_options([prompt], len(item["options"]))[0]
            prediction: int | str = int(np.argmax(scores))
            raw_text = None
        else:
            raw_text = backend.generate([prompt], max_new_tokens=max_new_tokens)[0]
            prediction = parse_answer(raw_text, item["options"])
            scores = None

        record = {
            "id": str(item["id"]),
            "category": str(item["category"]),
            "example_id": str(item["example_id"]),
            "question_index": str(item["question_index"]),
            "condition": item["condition"],
            "label_type": item.get("label_type"),
            "label": int(item["label"]),
            "target_loc": int(item["target_loc"]),
            "unknown_loc": int(item["unknown_loc"]),
            "prediction": prediction,
            "option_scores": scores,
            "raw_text": raw_text,
            "correct": prediction == int(item["label"]) if isinstance(prediction, int) else None,
        }
        predictions.append(record)

    predictions_frame = pd.DataFrame(predictions)
    metrics = compute_bbq_metrics(predictions_frame)
    if metrics["overall"]["invalid_rate"] > invalid_rate_warning_threshold:
        warnings.warn(
            f"Invalid-answer rate is above {invalid_rate_warning_threshold:.0%}; "
            "review raw generated answers.",
            RuntimeWarning,
            stacklevel=2,
        )

    config = {
        "run_id": run_id,
        "model_id": identity,
        "backend": getattr(backend, "metadata", {"backend": backend.__class__.__name__}),
        "dataset": "bbq",
        "prompt_condition": prompt_condition,
        "system_prompt": system_prompt,
        "scoring_mode": actual_mode,
        "scoring_method": (
            backend.scoring_method if actual_mode == "loglik" else "generation_greedy"
        ),
        "sample_size": len(data),
        "seed": seed,
        "invalid_rate_warning_threshold": invalid_rate_warning_threshold,
    }
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    return predictions_frame, metrics, run_dir
