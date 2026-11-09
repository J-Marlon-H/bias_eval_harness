"""Reproducible stratified sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_bbq(
    data: pd.DataFrame,
    n_per_category: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Take a deterministic per-category sample balanced over condition and polarity."""
    if n_per_category < 1:
        raise ValueError("n_per_category must be positive")
    required = {"category", "condition", "polarity"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"BBQ sample input is missing columns: {sorted(missing)}")
    data = data.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    samples = []
    for category, category_frame in data.groupby("category", sort=True):
        strata = [
            group for _, group in category_frame.groupby(
                ["condition", "polarity"], sort=True, dropna=False
            )
        ]
        target = min(n_per_category, len(category_frame))
        quota = target // len(strata)
        remainder = target % len(strata)
        selected: list[pd.DataFrame] = []
        remaining: list[pd.DataFrame] = []
        for index, stratum in enumerate(strata):
            desired = quota + (1 if index < remainder else 0)
            count = min(desired, len(stratum))
            if count:
                chosen = rng.choice(stratum.index.to_numpy(), size=count, replace=False)
                selected.append(stratum.loc[chosen])
            if count < len(stratum):
                remaining.append(stratum.drop(index=chosen if count else []))
        left = target - sum(len(sample) for sample in selected)
        while left > 0 and remaining:
            next_remaining = []
            progressed = False
            for stratum in remaining:
                if left == 0:
                    next_remaining.append(stratum)
                    continue
                chosen_index = rng.choice(stratum.index.to_numpy())
                selected.append(stratum.loc[[chosen_index]])
                stratum = stratum.drop(index=chosen_index)
                left -= 1
                progressed = True
                if not stratum.empty:
                    next_remaining.append(stratum)
            remaining = next_remaining
            if not progressed:
                break
        samples.extend(selected)
    if not samples:
        return data.iloc[0:0].copy()
    result = pd.concat(samples).sort_index()
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
