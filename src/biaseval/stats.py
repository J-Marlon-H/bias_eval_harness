"""Cluster bootstrap intervals for BBQ metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biaseval.metrics import compute_bbq_metrics

_BOOTSTRAP_METRICS = (
    "invalid_rate",
    "accuracy_ambiguous_valid",
    "accuracy_disambiguated_valid",
    "bias_score_ambiguous",
    "bias_score_disambiguated",
    "unknown_rate_disambiguated_valid",
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
