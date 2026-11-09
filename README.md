# BBQ Bias Evaluation Harness

A notebook-driven harness for evaluating local Hugging Face causal language models on BBQ bias questions and MMLU general capability. It supports log-likelihood scoring (preferred) or answer generation, resumable per-item evaluation, clustered bootstrap intervals, and paired comparisons for BBQ runs. API models, training, data splits, and custom datasets are out of scope.

## Setup

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The default model is `Qwen/Qwen3-1.7B`; its files are cached by Hugging Face outside the repository. Qwen3 inference disables thinking in its chat template. To use a gated model, accept its Hugging Face license, put a read token in the local `.env`, and change `MODEL_ID` in the notebook.

Choose a PyTorch wheel appropriate for the available hardware if the pinned default is not suitable. Device selection is automatic (`CUDA`, `MPS`, then CPU), or can be set explicitly in the notebook.

## Run the smoke evaluation

Open `notebooks/bias_eval.ipynb` and run cells in order. The default BBQ run takes 20 items per category across all nine categories (180 items total), balanced across condition and question polarity. On CPU, use fewer items per category if the run would exceed your preferred time.

Each completed item is appended to its run's `predictions.jsonl`. If a run is interrupted, copy the printed run ID into `RESUME_RUN_ID` and rerun the evaluation cell; already completed items are skipped. Resume is refused if the model, scoring method, prompt, seed, or sample differs. The local `results/index.csv` tracks running, interrupted, and completed runs.

The notebook reports 95% percentile bootstrap intervals for BBQ metrics, resampling clusters by `(category, question_index)`. Undefined draws are skipped and counted. A simple category plot and optional plain-versus-debias paired comparison are included. Paired comparisons use only items with valid predictions in both runs and require the same scoring method.

MMLU is an optional capability check, disabled by default so it does not add another model run. Enable `RUN_MMLU` to load the cached `cais/mmlu` test split and evaluate a deterministic sample of two questions per subject with the plain prompt. An optional debias-prompt side-check can be enabled separately. The MMLU files are cached by Hugging Face.

MMLU is a fixed public benchmark and models may have encountered it during pretraining; use it as a relative capability-preservation check, not as a contamination-free absolute score.

Run the BBQ loader count test with:

```powershell
$env:RUN_BBQ_DATA_TEST = "1"
python -m unittest tests.test_bbq_loader -v
```

## Metrics

- Ambiguous accuracy: valid predictions selecting the unknown option.
- Disambiguated accuracy: valid predictions selecting the gold answer.
- Disambiguated and ambiguous bias scores use BBQ `target_loc`; invalid and unknown predictions are excluded from the bias denominator.
- Unknown rate and invalid-answer rate are reported separately. An invalid rate above 5% triggers a warning. Undefined metrics are NaN.

The nine single-category BBQ files contain 31,356 usable items after metadata deduplication and dropping rows without `target_loc`; `Race_ethnicity` has 6,880. Raw files are cached under `data/raw/bbq/` and are not committed.

Each run is saved under `results/runs/` with predictions, metrics, confidence intervals, and its resolved configuration. Run outputs, the local results index, model caches, and downloaded data are gitignored.

### Inspect incorrect answers

The final notebook cell writes `results/wrong_answers.html`, a local page showing only incorrect or invalid BBQ and MMLU answers, with their context/questions, choices, and the model's answer beside the correct answer. Green marks the correct choice; red marks the model's choice. For BBQ, the report also shows per-category/condition totals and accuracy. Search or filter by dataset, BBQ category/condition, or MMLU subject; a live summary updates for combined filters. Open it using the notebook's clickable report link.
