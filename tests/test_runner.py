import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from biaseval.runner import run_evaluation


class FakeBackend:
    capabilities = {"loglik": True, "generate": True}
    scoring_method = "fake_sequence_logprob"
    tokenizer = None
    model_id = "fake-model"
    metadata = {"backend": "fake"}

    def __init__(self):
        self.calls = 0

    def score_options(self, prompts, n_options):
        self.calls += len(prompts)
        return [[float(index) for index in range(n_options)] for _ in prompts]

    def generate(self, prompts, max_new_tokens=8):
        return ["A" for _ in prompts]


class RunnerTests(unittest.TestCase):
    def test_run_writes_predictions_metrics_and_config(self):
        data = pd.DataFrame([
            {
                "id": "Race_ethnicity:1", "category": "Race_ethnicity",
                "example_id": "1", "question_index": "template",
                "condition": "ambig", "label": 2, "target_loc": 0,
                "unknown_loc": 2, "label_type": "label", "context": "Two people.",
                "question": "Who?", "options": ["Person A", "Person B", "Unknown"],
                "source": "bbq",
            },
            {
                "id": "Race_ethnicity:2", "category": "Race_ethnicity",
                "example_id": "2", "question_index": "template",
                "condition": "disambig", "label": 0, "target_loc": 1,
                "unknown_loc": 2, "label_type": "name", "context": "Evidence says A.",
                "question": "Who?", "options": ["Person A", "Person B", "Unknown"],
                "source": "bbq",
            },
        ])
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            predictions, metrics, run_dir = run_evaluation(
                data, backend, results_dir=directory, model_id="fake-model",
            )
            self.assertEqual(len(predictions), 2)
            self.assertEqual(backend.calls, 2)
            self.assertTrue((Path(run_dir) / "config.json").exists())
            self.assertTrue((Path(run_dir) / "predictions.jsonl").exists())
            self.assertTrue((Path(run_dir) / "metrics.json").exists())
            config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["scoring_method"], "fake_sequence_logprob")
            self.assertIn("overall", metrics)


if __name__ == "__main__":
    unittest.main()
