# BBQ Bias Evaluation Harness — MVP

The first working version evaluates BBQ `Race_ethnicity` questions with a local Hugging Face causal language model. It includes a notebook smoke run, log-likelihood scoring (default) or generation parsing, core BBQ metrics, and saved predictions/configuration. API models, MMLU, bootstrap statistics, paired comparisons, data splits, and custom datasets are deferred to later steps.

## Setup

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The default model is `Qwen/Qwen3-1.7B`. Qwen3 inference disables thinking in its chat template. To use gated Llama 3.2, accept its Hugging Face license, set `HF_TOKEN` in `.env`, and change `MODEL_ID` in the notebook.

Choose a PyTorch wheel appropriate for the available hardware if the pinned default is not suitable. Device selection is automatic (`CUDA`, `MPS`, then CPU), or can be set explicitly in the notebook.

## Run the smoke evaluation

Open `notebooks/bias_eval.ipynb`, keep `RUN_FULL = False`, then run all cells. The notebook loads only BBQ `Race_ethnicity`, takes a deterministic sample balanced across condition and question polarity, checks one prompt, evaluates the sample, and displays the core metrics.

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

Each run is saved under `results/runs/` with predictions, metrics, and its resolved configuration. Run outputs and downloaded data are gitignored.
