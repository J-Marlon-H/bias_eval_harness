"""Cluster bootstrap intervals and paired comparisons for BBQ runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biaseval.metrics import compute_bbq_metrics
from biaseval.results import write_json

_BOOTSTRAP_METRICS = (
    "invalid_rate",
    "accuracy_ambiguous_valid",
    "accuracy_disambiguated_valid",
    "bias_score_ambiguous",
    "bias_score_disambiguated",
    "unknown_rate_disambiguated_valid",
)
_PAIRED_METRICS = tuple(
    metric for metric in _BOOTSTRAP_METRICS if metric != "invalid_rate"
)


def _cluster_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    required = {"category", "question_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cluster bootstrap data is missing columns: {sorted(missing)}")
    return [
        group
        for _, group in frame.groupby(
            ["category", "question_index"], dropna=False, sort=False
        )
    ]


def _bootstrap_samples(
    groups: list[pd.DataFrame],
    n_resamples: int,
    rng: np.random.Generator,
):
    for _ in range(n_resamples):
        indexes = rng.integers(0, len(groups), size=len(groups))
        yield pd.concat([groups[index] for index in indexes], ignore_index=True)


def _summarize_draws(draws: list[float], n_resamples: int) -> dict[str, Any]:
    values = np.asarray(draws, dtype=float)
    defined = values[np.isfinite(values)]
    if len(defined):
        low, high = np.percentile(defined, [2.5, 97.5])
        low_value, high_value = float(low), float(high)
    else:
        low_value = high_value = float("nan")
    return {
        "low": low_value,
        "high": high_value,
        "defined_draws": int(len(defined)),
        "undefined_draws": int(n_resamples - len(defined)),
    }


def bootstrap_bbq_metrics(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return percentile confidence intervals from template-cluster resamples."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if predictions.empty:
        raise ValueError("Cannot bootstrap an empty prediction table")
    groups = _cluster_groups(predictions)
    rng = np.random.default_rng(seed)
    point_metrics = compute_bbq_metrics(predictions)
    draws = {
        scope: {name: [] for name in _BOOTSTRAP_METRICS}
        for scope in point_metrics
    }
    for sample in _bootstrap_samples(groups, n_resamples, rng):
        sample_metrics = compute_bbq_metrics(sample)
        for scope, scope_draws in draws.items():
            values = sample_metrics.get(scope, {})
            for name in _BOOTSTRAP_METRICS:
                scope_draws[name].append(float(values.get(name, float("nan"))))
    return {
        scope: {
            name: _summarize_draws(values, n_resamples)
            for name, values in metrics.items()
        }
        for scope, metrics in draws.items()
    }


def _read_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.jsonl"
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return pd.DataFrame(records)


def _valid(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def compare_bbq_runs(
    run_a_dir: str | Path,
    run_b_dir: str | Path,
    *,
    n_resamples: int = 200,
    seed: int = 42,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two saved BBQ runs on paired items with valid answers in both."""
    run_a_dir, run_b_dir = Path(run_a_dir), Path(run_b_dir)
    with (run_a_dir / "config.json").open(encoding="utf-8") as stream:
        config_a = json.load(stream)
    with (run_b_dir / "config.json").open(encoding="utf-8") as stream:
        config_b = json.load(stream)
    if config_a.get("dataset") != "bbq" or config_b.get("dataset") != "bbq":
        raise ValueError("Paired BBQ comparison requires two BBQ runs")
    method_a, method_b = config_a.get("scoring_method"), config_b.get("scoring_method")
    if not method_a or not method_b or method_a != method_b:
        raise ValueError(
            f"Cannot compare scoring methods {method_a!r} and {method_b!r}"
        )
    frame_a, frame_b = _read_predictions(run_a_dir), _read_predictions(run_b_dir)
    required = {"category", "example_id", "question_index", "prediction"}
    for label, frame in (("first", frame_a), ("second", frame_b)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"The {label} run is missing fields: {sorted(missing)}")
    paired = frame_a.merge(
        frame_b,
        on=["category", "example_id"],
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["prediction_a"].map(_valid) & paired["prediction_b"].map(_valid)
    ].copy()
    if paired.empty:
        raise ValueError("The runs have no paired items with valid predictions")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    left = paired.rename(
        columns={
            f"{column}_a": column
            for column in ("question_index", "condition", "label", "target_loc", "unknown_loc")
            if f"{column}_a" in paired
        } | {"prediction_a": "prediction"}
    )
    right = paired.rename(
        columns={
            f"{column}_b": column
            for column in ("question_index", "condition", "label", "target_loc", "unknown_loc")
            if f"{column}_b" in paired
        } | {"prediction_b": "prediction"}
    )
    point_a, point_b = compute_bbq_metrics(left), compute_bbq_metrics(right)
    groups = _cluster_groups(paired.rename(columns={"question_index_a": "question_index"}))
    rng = np.random.default_rng(seed)
    common_scopes = [scope for scope in point_a if scope in point_b]
    draws = {
        scope: {name: [] for name in _PAIRED_METRICS}
        for scope in common_scopes
    }
    for sampled in _bootstrap_samples(groups, n_resamples, rng):
        sample_a = sampled.rename(
            columns={
                f"{column}_a": column
                for column in ("question_index", "condition", "label", "target_loc", "unknown_loc")
                if f"{column}_a" in sampled
            } | {"prediction_a": "prediction"}
        )
        sample_b = sampled.rename(
            columns={
                f"{column}_b": column
                for column in ("question_index", "condition", "label", "target_loc", "unknown_loc")
                if f"{column}_b" in sampled
            } | {"prediction_b": "prediction"}
        )
        metrics_a, metrics_b = compute_bbq_metrics(sample_a), compute_bbq_metrics(sample_b)
        for scope, scope_draws in draws.items():
            for name in _PAIRED_METRICS:
                if scope not in metrics_a or scope not in metrics_b:
                    scope_draws[name].append(float("nan"))
                    continue
                delta = float(metrics_a[scope][name]) - float(metrics_b[scope][name])
                scope_draws[name].append(delta)

    comparison: dict[str, Any] = {
        "run_a": config_a["run_id"],
        "run_b": config_b["run_id"],
        "scoring_method": method_a,
        "paired_valid_items": len(paired),
        "metrics": {},
    }
    for scope, metrics in draws.items():
        comparison["metrics"][scope] = {}
        for name, delta_draws in metrics.items():
            delta = float(point_a[scope][name]) - float(point_b[scope][name])
            interval = _summarize_draws(delta_draws, n_resamples)
            comparison["metrics"][scope][name] = {
                "run_a": float(point_a[scope][name]),
                "run_b": float(point_b[scope][name]),
                "difference_a_minus_b": delta,
                **interval,
            }

    destination = Path(results_dir) if results_dir else run_a_dir.parent.parent
    comparison_dir = destination / "comparisons"
    comparison_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(comparison_dir / f"{comparison_id}.json", comparison)
    return comparison
