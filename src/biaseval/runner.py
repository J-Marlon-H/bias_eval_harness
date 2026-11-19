"""Evaluation loops with per-item persistence and resumable run folders."""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from biaseval.metrics import compute_bbq_metrics
from biaseval.parsing import parse_answer
from biaseval.prompts import build_answer_fallback_prompt, build_prompt
from biaseval.results import create_run_id, update_run_index, write_json
from biaseval.stats import bootstrap_bbq_metrics

PREDICTION_PARSER_VERSION = "thinking-block-v2"
PROMPT_BUILDER_VERSION = "thinking-completion-v1"


def _read_saved_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Invalid prediction record in {path} at line {line_number}")
            records.append(record)
    return records


def _evaluation_signature(config: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "dataset",
        "model_id",
        "backend",
        "prompt_condition",
        "system_prompt",
        "enable_thinking",
        "scoring_mode",
        "scoring_method",
        "sample_size",
        "sample_sha256",
        "seed",
        "max_new_tokens",
        "invalid_rate_warning_threshold",
        "bootstrap_resamples",
        "prediction_parser_version",
        "prompt_builder_version",
    )
    return {
        field: (
            config.get(field, False)
            if field == "enable_thinking"
            else (
                None
                if (
                    field == "prompt_builder_version"
                    and not config.get("enable_thinking", False)
                )
                else (
                    None
                    if (
                        field == "prediction_parser_version"
                        and config.get("scoring_mode") != "generation"
                    )
                    else config.get(field)
                )
            )
        )
        for field in fields
    }


def _find_compatible_run(
    results_dir: Path,
    config: dict[str, Any],
) -> str | None:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return None
    expected_signature = _evaluation_signature(config)
    compatible = []
    for run_dir in runs_dir.iterdir():
        config_path = run_dir / "config.json"
        predictions_path = run_dir / "predictions.jsonl"
        if not run_dir.is_dir() or not config_path.is_file() or not predictions_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as stream:
            previous_config = json.load(stream)
        if _evaluation_signature(previous_config) == expected_signature:
            compatible.append(
                (
                    previous_config.get("status") == "complete",
                    str(previous_config.get("created_at", "")),
                    run_dir.name,
                )
            )
    if not compatible:
        return None
    return max(compatible)[2]


def _run_evaluation(
    data: pd.DataFrame,
    backend: Any,
    *,
    dataset: str,
    metric_function: Callable[[pd.DataFrame], dict[str, dict[str, Any]]],
    required: set[str],
    results_dir: str | Path,
    model_id: str | None,
    prompt_condition: str,
    system_prompt: str | None,
    enable_thinking: bool,
    scoring_mode: str,
    max_new_tokens: int,
    seed: int,
    resume_run_id: str | None,
    invalid_rate_warning_threshold: float | None = None,
    bootstrap_resamples: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Path]:
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{dataset} evaluation data is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot evaluate an empty dataset")
    if data["id"].astype(str).duplicated().any():
        raise ValueError("Evaluation item IDs must be unique")
    if scoring_mode not in {"auto", "loglik", "generation"}:
        raise ValueError("scoring_mode must be auto, loglik, or generation")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples cannot be negative")
    if invalid_rate_warning_threshold is not None and not (
        0 <= invalid_rate_warning_threshold <= 1
    ):
        raise ValueError("invalid_rate_warning_threshold must be between 0 and 1")

    tokenizer = getattr(backend, "tokenizer", None)
    first_item = {str(key): value for key, value in data.iloc[0].to_dict().items()}
    probe_prompt = build_prompt(
        first_item,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )
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
    scoring_method = (
        backend.scoring_method if actual_mode == "loglik" else "generation_greedy"
    )
    item_ids = data["id"].astype(str).tolist()
    item_ids_sha256 = hashlib.sha256(
        json.dumps(item_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sample_sha256 = hashlib.sha256(
        data.to_json(orient="records", force_ascii=False).encode("utf-8")
    ).hexdigest()
    results_dir = Path(results_dir)
    run_id = (
        resume_run_id
        if resume_run_id is not None and resume_run_id != "auto"
        else create_run_id(identity, dataset, prompt_condition)
    )
    run_dir = results_dir / "runs" / run_id
    predictions_path = run_dir / "predictions.jsonl"

    config = {
        "run_id": run_id,
        "model_id": identity,
        "backend": getattr(backend, "metadata", {"backend": backend.__class__.__name__}),
        "dataset": dataset,
        "prompt_condition": prompt_condition,
        "system_prompt": system_prompt,
        "enable_thinking": enable_thinking,
        "scoring_mode": actual_mode,
        "scoring_method": scoring_method,
        "sample_size": len(data),
        "item_ids_sha256": item_ids_sha256,
        "sample_sha256": sample_sha256,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invalid_rate_warning_threshold": invalid_rate_warning_threshold,
        "bootstrap_resamples": bootstrap_resamples,
        "prediction_parser_version": PREDICTION_PARSER_VERSION,
        "prompt_builder_version": PROMPT_BUILDER_VERSION,
    }

    if resume_run_id == "auto":
        resume_run_id = _find_compatible_run(results_dir, config)
        if resume_run_id:
            print(f"Auto-resuming compatible local run: {resume_run_id}")
        else:
            print("No compatible local run found; creating a new run.")
        run_id = resume_run_id or create_run_id(identity, dataset, prompt_condition)
        run_dir = results_dir / "runs" / run_id
        predictions_path = run_dir / "predictions.jsonl"
        config["run_id"] = run_id

    if resume_run_id:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot resume run without config: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            previous_config = json.load(stream)
        if _evaluation_signature(previous_config) != _evaluation_signature(config):
            raise ValueError(
                f"Run {resume_run_id} does not match the model, prompts, scoring mode, "
                "seed, or evaluation items requested"
            )
        previous_config.setdefault("enable_thinking", False)
        config = previous_config
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    saved_records = _read_saved_predictions(predictions_path)
    saved_by_id: dict[str, dict[str, Any]] = {}
    expected_item_ids = set(item_ids)
    for record in saved_records:
        item_id = str(record["id"])
        if item_id in saved_by_id:
            raise ValueError(f"Duplicate saved prediction for item {item_id}")
        if item_id not in expected_item_ids:
            raise ValueError(f"Saved prediction {item_id} is not in this evaluation sample")
        saved_by_id[item_id] = record

    config["status"] = "running"
    write_json(run_dir / "config.json", config)
    update_run_index(
        results_dir, config, status="running", completed_items=len(saved_by_id)
    )
    print(f"Run ID: {run_id}")

    pending = data.loc[~data["id"].astype(str).isin(saved_by_id)].iterrows()
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(pending, total=len(data) - len(saved_by_id), desc=f"Evaluating {dataset}")
    except ImportError:
        iterator = pending
    try:
        with predictions_path.open("a", encoding="utf-8") as stream:
            for _, series in iterator:
                item = {str(key): value for key, value in series.to_dict().items()}
                prompt = build_prompt(
                    item,
                    system_prompt=system_prompt,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                )
                if actual_mode == "loglik":
                    scores = backend.score_options([prompt], len(item["options"]))[0]
                    prediction: int | str = int(np.argmax(scores))
                    raw_text = None
                else:
                    raw_text = backend.generate([prompt], max_new_tokens=max_new_tokens)[0]
                    prediction = parse_answer(raw_text, item["options"])
                    fallback_raw_text = None
                    fallback_used = False
                    if enable_thinking and prediction == "invalid":
                        fallback_prompt = build_answer_fallback_prompt(
                            item,
                            raw_text,
                            tokenizer=tokenizer,
                        )
                        fallback_raw_text = backend.generate(
                            [fallback_prompt], max_new_tokens=8
                        )[0]
                        fallback_prediction = parse_answer(
                            fallback_raw_text, item["options"]
                        )
                        if fallback_prediction != "invalid":
                            prediction = fallback_prediction
                        fallback_used = True
                    scores = None

                record = {
                    "id": str(item["id"]),
                    "label": int(item["label"]),
                    "prediction": prediction,
                    "option_scores": scores,
                    "raw_text": raw_text,
                    "fallback_used": fallback_used if actual_mode == "generation" else False,
                    "fallback_raw_text": (
                        fallback_raw_text if actual_mode == "generation" else None
                    ),
                    "correct": prediction == int(item["label"])
                    if isinstance(prediction, int)
                    else None,
                }
                for field in (
                    "category",
                    "example_id",
                    "question_index",
                    "condition",
                    "polarity",
                    "target_loc",
                    "unknown_loc",
                    "label_type",
                    "subject",
                ):
                    if field in item:
                        value = item[field]
                        record[field] = value.item() if hasattr(value, "item") else value
                stream.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                stream.flush()
                saved_by_id[record["id"]] = record
                update_run_index(
                    results_dir,
                    config,
                    status="running",
                    completed_items=len(saved_by_id),
                )

        ordered_records = [saved_by_id[item_id] for item_id in item_ids]
        predictions = pd.DataFrame(ordered_records)
        metrics = metric_function(predictions)
        write_json(run_dir / "metrics.json", metrics)
        if bootstrap_resamples:
            if dataset == "bbq":
                intervals = bootstrap_bbq_metrics(
                    predictions, n_resamples=bootstrap_resamples, seed=seed
                )
                write_json(run_dir / "confidence_intervals.json", intervals)
        config["status"] = "complete"
        write_json(run_dir / "config.json", config)
        update_run_index(
            results_dir,
            config,
            status="complete",
            completed_items=len(saved_by_id),
            metrics=metrics,
        )
    except Exception:
        config["status"] = "interrupted"
        write_json(run_dir / "config.json", config)
        update_run_index(
            results_dir,
            config,
            status="interrupted",
            completed_items=len(saved_by_id),
        )
        raise

    if invalid_rate_warning_threshold is not None and (
        metrics["overall"]["invalid_rate"] > invalid_rate_warning_threshold
    ):
        warnings.warn(
            f"Invalid-answer rate is above {invalid_rate_warning_threshold:.0%}; "
            "review raw generated answers.",
            RuntimeWarning,
            stacklevel=2,
        )
    return predictions, metrics, run_dir


def run_evaluation(
    data: pd.DataFrame,
    backend: Any,
    *,
    results_dir: str | Path = "results",
    model_id: str | None = None,
    prompt_condition: str = "plain",
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    scoring_mode: str = "auto",
    max_new_tokens: int = 8,
    seed: int = 42,
    invalid_rate_warning_threshold: float = 0.05,
    bootstrap_resamples: int = 200,
    resume_run_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Path]:
    """Evaluate BBQ rows, saving each prediction so an interrupted run can resume."""
    required = {
        "id", "category", "example_id", "question_index", "condition",
        "context", "question", "options", "label", "target_loc", "unknown_loc",
    }
    return _run_evaluation(
        data,
        backend,
        dataset="bbq",
        metric_function=compute_bbq_metrics,
        required=required,
        results_dir=results_dir,
        model_id=model_id,
        prompt_condition=prompt_condition,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        scoring_mode=scoring_mode,
        max_new_tokens=max_new_tokens,
        seed=seed,
        resume_run_id=resume_run_id,
        invalid_rate_warning_threshold=invalid_rate_warning_threshold,
        bootstrap_resamples=bootstrap_resamples,
    )


def run_mmlu_evaluation(
    data: pd.DataFrame,
    backend: Any,
    *,
    results_dir: str | Path = "results",
    model_id: str | None = None,
    prompt_condition: str = "plain",
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    scoring_mode: str = "auto",
    max_new_tokens: int = 8,
    seed: int = 42,
    resume_run_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Path]:
    """Evaluate MMLU rows with the same resumable runner used by BBQ."""
    required = {"id", "subject", "question", "options", "label"}

    def mmlu_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
        valid = predictions.loc[
            predictions["prediction"].map(
                lambda value: isinstance(value, (int, np.integer))
                and not isinstance(value, bool)
            )
        ]
        overall = {
            "n_items": len(predictions),
            "n_valid": len(valid),
            "n_invalid": len(predictions) - len(valid),
            "invalid_rate": (
                float((len(predictions) - len(valid)) / len(predictions))
                if len(predictions)
                else float("nan")
            ),
            "accuracy": (
                float((valid["prediction"] == valid["label"]).mean())
                if len(valid)
                else float("nan")
            ),
        }
        result = {"overall": overall}
        for subject, frame in predictions.groupby("subject", sort=True):
            subject_valid = frame.loc[
                frame["prediction"].map(
                    lambda value: isinstance(value, (int, np.integer))
                    and not isinstance(value, bool)
                )
            ]
            result[f"subject|{subject}"] = {
                "n_items": len(frame),
                "n_valid": len(subject_valid),
                "n_invalid": len(frame) - len(subject_valid),
                "invalid_rate": (
                    float((len(frame) - len(subject_valid)) / len(frame))
                    if len(frame)
                    else float("nan")
                ),
                "accuracy": (
                    float((subject_valid["prediction"] == subject_valid["label"]).mean())
                    if len(subject_valid)
                    else float("nan")
                ),
            }
        return result

    return _run_evaluation(
        data,
        backend,
        dataset="mmlu",
        metric_function=mmlu_metrics,
        required=required,
        results_dir=results_dir,
        model_id=model_id,
        prompt_condition=prompt_condition,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        scoring_mode=scoring_mode,
        max_new_tokens=max_new_tokens,
        seed=seed,
        resume_run_id=resume_run_id,
        bootstrap_resamples=0,
    )
