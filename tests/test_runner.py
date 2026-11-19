import tempfile
import unittest
from pathlib import Path
import json
import contextlib
import io
import re

import pandas as pd

from biaseval.runner import run_evaluation, run_mmlu_evaluation


class FakeBackend:
    capabilities = {"loglik": True, "generate": True}
    scoring_method = "fake_sequence_logprob"
    tokenizer = None
    model_id = "fake-model"
    metadata = {"backend": "fake"}

    def __init__(self, fail_on_call=None, generated=None):
        self.calls = 0
        self.generate_calls = 0
        self.fail_on_call = fail_on_call
        self.generated = list(generated or ["A"])

    def score_options(self, prompts, n_options):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return [[float(index) for index in range(n_options)] for _ in prompts]

    def generate(self, prompts, max_new_tokens=8):
        self.generate_calls += len(prompts)
        result = self.generated[:len(prompts)]
        del self.generated[:len(prompts)]
        return result


class RunnerTests(unittest.TestCase):
    def test_thinking_invalid_answer_uses_short_fallback(self):
        data = pd.DataFrame([{
            "id": "Race_ethnicity:1", "category": "Race_ethnicity",
            "example_id": "1", "question_index": "template",
            "condition": "ambig", "label": 2, "target_loc": 0,
            "unknown_loc": 2, "label_type": "label", "context": "Two people.",
            "question": "Who?", "options": ["Person A", "Person B", "Unknown"],
            "source": "bbq",
        }])
        backend = FakeBackend(generated=["I need more time.", "C"])
        with tempfile.TemporaryDirectory() as directory:
            predictions, _, _ = run_evaluation(
                data,
                backend,
                results_dir=directory,
                enable_thinking=True,
                scoring_mode="generation",
                max_new_tokens=512,
                bootstrap_resamples=0,
            )

        self.assertEqual(predictions.iloc[0]["prediction"], 2)
        self.assertTrue(predictions.iloc[0]["fallback_used"])
        self.assertEqual(predictions.iloc[0]["raw_text"], "I need more time.")
        self.assertEqual(predictions.iloc[0]["fallback_raw_text"], "C")
        self.assertEqual(backend.generate_calls, 2)

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
                bootstrap_resamples=5,
            )
            self.assertEqual(len(predictions), 2)
            self.assertEqual(backend.calls, 2)
            self.assertTrue((Path(run_dir) / "config.json").exists())
            self.assertTrue((Path(run_dir) / "predictions.jsonl").exists())
            self.assertTrue((Path(run_dir) / "metrics.json").exists())
            self.assertTrue((Path(run_dir) / "confidence_intervals.json").exists())
            config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["scoring_method"], "fake_sequence_logprob")
            self.assertIn("overall", metrics)
            index = Path(directory) / "index.csv"
            self.assertTrue(index.exists())
            self.assertIn("complete", index.read_text(encoding="utf-8"))

    def test_interrupted_run_resumes_without_repeating_saved_items(self):
        data = pd.DataFrame([
            {
                "id": f"Race_ethnicity:{index}", "category": "Race_ethnicity",
                "example_id": str(index), "question_index": f"template-{index}",
                "condition": "disambig", "label": 0, "target_loc": 0,
                "unknown_loc": 2, "label_type": "label", "context": "Evidence.",
                "question": "Who?", "options": ["Person A", "Person B", "Unknown"],
                "source": "bbq",
            }
            for index in range(3)
        ])
        with tempfile.TemporaryDirectory() as directory:
            interrupted_backend = FakeBackend(fail_on_call=2)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_evaluation(
                        data,
                        interrupted_backend,
                        results_dir=directory,
                        bootstrap_resamples=5,
                    )
            run_id = re.search(r"Run ID: (\S+)", captured.getvalue()).group(1)
            run_dir = Path(directory) / "runs" / run_id
            saved = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(saved), 1)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["status"], "interrupted")

            resumed_backend = FakeBackend()
            predictions, _, _ = run_evaluation(
                data,
                resumed_backend,
                results_dir=directory,
                resume_run_id=run_id,
                bootstrap_resamples=5,
            )
            self.assertEqual(len(predictions), 3)
            self.assertEqual(resumed_backend.calls, 2)
            self.assertEqual(
                len((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()),
                3,
            )
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["status"], "complete")

    def test_auto_resume_reuses_compatible_local_run_and_creates_first_run(self):
        data = pd.DataFrame([
            {
                "id": "Race_ethnicity:1", "category": "Race_ethnicity",
                "example_id": "1", "question_index": "template",
                "condition": "ambig", "label": 2, "target_loc": 0,
                "unknown_loc": 2, "label_type": "label", "context": "Two people.",
                "question": "Who?", "options": ["Person A", "Person B", "Unknown"],
                "source": "bbq",
            },
        ])
        with tempfile.TemporaryDirectory() as directory:
            first_backend = FakeBackend()
            _, _, first_run_dir = run_evaluation(
                data,
                first_backend,
                results_dir=directory,
                model_id="fake-model",
                bootstrap_resamples=1,
                resume_run_id="auto",
            )
            self.assertEqual(first_backend.calls, 1)
            config_path = Path(first_run_dir) / "config.json"
            legacy_config = json.loads(config_path.read_text(encoding="utf-8"))
            legacy_config.pop("enable_thinking")
            config_path.write_text(json.dumps(legacy_config), encoding="utf-8")

            second_backend = FakeBackend()
            _, _, second_run_dir = run_evaluation(
                data,
                second_backend,
                results_dir=directory,
                model_id="fake-model",
                bootstrap_resamples=1,
                resume_run_id="auto",
            )
            second_config = json.loads(
                (Path(second_run_dir) / "config.json").read_text(encoding="utf-8")
            )
            other_model_backend = FakeBackend()
            _, _, other_model_run_dir = run_evaluation(
                data,
                other_model_backend,
                results_dir=directory,
                model_id="fine-tuned-model",
                bootstrap_resamples=1,
                resume_run_id="auto",
            )
            prompted_backend = FakeBackend()
            _, _, prompted_run_dir = run_evaluation(
                data,
                prompted_backend,
                results_dir=directory,
                model_id="fake-model",
                prompt_condition="template",
                system_prompt="Use only explicit evidence.",
                bootstrap_resamples=1,
                resume_run_id="auto",
            )
            thinking_backend = FakeBackend()
            _, _, thinking_run_dir = run_evaluation(
                data,
                thinking_backend,
                results_dir=directory,
                model_id="fake-model",
                prompt_condition="plain_thinking",
                enable_thinking=True,
                scoring_mode="generation",
                max_new_tokens=512,
                bootstrap_resamples=1,
                resume_run_id="auto",
            )
            thinking_config = json.loads(
                (Path(thinking_run_dir) / "config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(first_run_dir, second_run_dir)
        self.assertEqual(second_backend.calls, 0)
        self.assertFalse(second_config["enable_thinking"])
        self.assertNotEqual(first_run_dir, other_model_run_dir)
        self.assertEqual(other_model_backend.calls, 1)
        self.assertNotEqual(first_run_dir, prompted_run_dir)
        self.assertEqual(prompted_backend.calls, 1)
        self.assertNotEqual(first_run_dir, thinking_run_dir)
        self.assertEqual(thinking_config["prompt_condition"], "plain_thinking")
        self.assertTrue(thinking_config["enable_thinking"])
        self.assertEqual(thinking_config["scoring_mode"], "generation")
        self.assertEqual(thinking_config["max_new_tokens"], 512)

    def test_resume_rejects_changed_sample(self):
        data = pd.DataFrame([{
            "id": "Race_ethnicity:1", "category": "Race_ethnicity",
            "example_id": "1", "question_index": "template",
            "condition": "disambig", "label": 0, "target_loc": 0,
            "unknown_loc": 2, "context": "", "question": "Who?",
            "options": ["A", "B", "Unknown"],
        }])
        with tempfile.TemporaryDirectory() as directory:
            _, _, run_dir = run_evaluation(
                data, FakeBackend(), results_dir=directory,
                bootstrap_resamples=1,
            )
            changed = data.copy()
            changed.loc[0, "question"] = "Changed question?"
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_evaluation(
                    changed, FakeBackend(), results_dir=directory,
                    resume_run_id=run_dir.name, bootstrap_resamples=1,
                )

    def test_mmlu_runner_saves_overall_and_subject_accuracy(self):
        data = pd.DataFrame([{
            "id": "math:1",
            "subject": "math",
            "question_index": "1",
            "context": "",
            "question": "Choose the answer.",
            "options": ["A", "B", "C", "D"],
            "label": 3,
        }])
        with tempfile.TemporaryDirectory() as directory:
            predictions, metrics, run_dir = run_mmlu_evaluation(
                data,
                FakeBackend(),
                results_dir=directory,
                scoring_mode="loglik",
            )
            self.assertEqual(len(predictions), 1)
            self.assertEqual(metrics["overall"]["accuracy"], 1.0)
            self.assertEqual(metrics["subject|math"]["accuracy"], 1.0)
            self.assertTrue((Path(run_dir) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
