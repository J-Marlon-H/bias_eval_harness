import unittest

import pandas as pd

from biaseval.stats import bootstrap_bbq_metrics


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


if __name__ == "__main__":
    unittest.main()
