# Evaluation summary

These are metrics-only summaries of the completed local runs. Per-item predictions, question text, downloaded datasets, model files, and the interactive wrong-answer report are not included in this file.

## Run setup

| Dataset | Model | Sample | Scoring | Device |
|---|---|---:|---|---|
| BBQ | Qwen/Qwen3-1.7B | 180 (20 per category) | Sequence log-probability | CPU |
| MMLU | Qwen/Qwen3-1.7B | 114 (2 per subject, 57 subjects) | Sequence log-probability | CPU |

Both runs produced valid answers for every sampled item (0% invalid answers).

## BBQ

The BBQ sample covers nine categories, with 10 ambiguous and 10 disambiguated questions per category. Intervals are 95% cluster-bootstrap percentile intervals, resampling by `(category, question_index)`; all 200 bootstrap draws were defined for these overall metrics.

| Measure | Result | 95% interval |
|---|---:|---:|
| Ambiguous accuracy (chose “unknown”) | 54.4% | 44.6%–65.1% |
| Disambiguated accuracy | 75.6% | 64.4%–84.0% |
| Ambiguous bias score | +0.100 | −0.046–+0.214 |
| Disambiguated bias score | −0.038 | −0.251–+0.183 |
| “Unknown” on answerable questions | 12.2% | 5.4%–20.8% |

The intervals for both overall bias scores cross zero, so these results do not establish a clear overall directional tilt in this sample. The per-category sample is small; use the category estimates as exploratory signals rather than firm rankings.

## MMLU

| Measure | Result |
|---|---:|
| Correct | 53 / 114 (46.5%) |
| Four-choice random-choice reference | 25% |
| Invalid answers | 0 / 114 (0%) |

The random-choice figure is only a rough reference, not a significance test. With two questions per subject, this is a small capability check, not a reliable subject-by-subject ranking. MMLU is public and may have appeared in model training data.
