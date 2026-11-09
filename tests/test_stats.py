import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biaseval.stats import bootstrap_bbq_metrics, compare_bbq_runs


class StatsTests(unittest.TestCase):
    def test_bootstrap_reports_defined_and_undefined_draws(self):
        frame = pd.DataFrame([
            {
                "id": "a",
                "category": "Race_ethnicity",
                "question_index": "template-a",
                "condition": "ambig",
                "prediction": 2,
                "label": 2,
                "target_loc": 0,
                "unknown_loc": 2,
            },
            {
                "id": "b",
                "category": "Race_ethnicity",
                "question_index": "template-b",
                "condition": "disambig",
                "prediction": 0,
                "label": 0,
                "target_loc": 0,
                "unknown_loc": 2,
            },
        ])
        intervals = bootstrap_bbq_metrics(frame, n_resamples=20, seed=3)
        ambiguous = intervals["overall"]["accuracy_ambiguous_valid"]
        self.assertEqual(ambiguous["defined_draws"] + ambiguous["undefined_draws"], 20)
        self.assertGreater(ambiguous["defined_draws"], 0)
        self.assertGreater(ambiguous["undefined_draws"], 0)

    def test_comparison_uses_paired_valid_items_and_saves_summary(self):
        records_a = [
            {"id": "A:a", "category": "A", "example_id": "a",
             "question_index": "q1", "condition": "disambig",
             "prediction": 0, "label": 0, "target_loc": 0, "unknown_loc": 2},
            {"id": "A:b", "category": "A", "example_id": "b",
             "question_index": "q1", "condition": "disambig",
             "prediction": 1, "label": 0, "target_loc": 0, "unknown_loc": 2},
            {"id": "A:c", "category": "A", "example_id": "c",
             "question_index": "q2", "condition": "disambig",
             "prediction": "invalid", "label": 0, "target_loc": 0, "unknown_loc": 2},
        ]
        records_b = [
            {**records_a[0], "prediction": 1},
            {**records_a[1], "prediction": 0},
            {**records_a[2], "prediction": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, records in (("a", records_a), ("b", records_b)):
                run_dir = root / "runs" / name
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(json.dumps({
                    "run_id": name, "dataset": "bbq", "scoring_method": "same",
                }), encoding="utf-8")
                (run_dir / "predictions.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
            result = compare_bbq_runs(
                root / "runs" / "a", root / "runs" / "b",
                n_resamples=20, seed=4,
            )
            self.assertEqual(result["paired_valid_items"], 2)
            self.assertEqual(
                result["metrics"]["overall"]["accuracy_disambiguated_valid"]
                ["difference_a_minus_b"],
                0.0,
            )
            self.assertEqual(len(list((root / "comparisons").glob("*.json"))), 1)

    def test_comparison_rejects_different_scoring_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a", root / "b"]
            for path, method in zip(paths, ("loglik", "generation")):
                path.mkdir()
                (path / "config.json").write_text(json.dumps({
                    "dataset": "bbq", "scoring_method": method,
                }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot compare scoring methods"):
                compare_bbq_runs(*paths)


if __name__ == "__main__":
    unittest.main()
