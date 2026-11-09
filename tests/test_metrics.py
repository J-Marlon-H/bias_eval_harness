import unittest

import pandas as pd

from biaseval.metrics import compute_bbq_metrics


class MetricsTests(unittest.TestCase):
    def test_bbq_hand_calculated_metrics_exclude_invalid_from_bias(self):
        frame = pd.DataFrame([
            {"id": "a1", "category": "Race_ethnicity", "condition": "ambig",
             "label_type": "label", "prediction": 2, "label": 2, "target_loc": 0,
             "unknown_loc": 2},
            {"id": "a2", "category": "Race_ethnicity", "condition": "ambig",
             "label_type": "label", "prediction": 0, "label": 2, "target_loc": 0,
             "unknown_loc": 2},
            {"id": "d1", "category": "Race_ethnicity", "condition": "disambig",
             "label_type": "label", "prediction": 0, "label": 0, "target_loc": 0,
             "unknown_loc": 2},
            {"id": "d2", "category": "Race_ethnicity", "condition": "disambig",
             "label_type": "name", "prediction": 1, "label": 1, "target_loc": 0,
             "unknown_loc": 2},
            {"id": "d3", "category": "Race_ethnicity", "condition": "disambig",
             "label_type": "name", "prediction": 2, "label": 0, "target_loc": 0,
             "unknown_loc": 2},
            {"id": "d4", "category": "Race_ethnicity", "condition": "disambig",
             "label_type": "name", "prediction": "invalid", "label": 1, "target_loc": 0,
             "unknown_loc": 2},
        ])
        metrics = compute_bbq_metrics(frame)["overall"]
        self.assertEqual(metrics["n_items"], 6)
        self.assertEqual(metrics["n_valid"], 5)
        self.assertAlmostEqual(metrics["invalid_rate"], 1 / 6)
        self.assertAlmostEqual(metrics["accuracy_ambiguous_valid"], 0.5)
        self.assertAlmostEqual(metrics["bias_score_ambiguous"], 0.5)
        self.assertAlmostEqual(metrics["accuracy_disambiguated_valid"], 2 / 3)
        self.assertEqual(metrics["bias_score_disambiguated"], 0.0)
        self.assertAlmostEqual(metrics["unknown_rate_disambiguated_valid"], 1 / 3)

    def test_undefined_bias_is_nan(self):
        frame = pd.DataFrame([{
            "id": "one", "category": "A", "condition": "disambig", "label_type": "x",
            "prediction": 2, "label": 0, "target_loc": 0, "unknown_loc": 2,
        }])
        metrics = compute_bbq_metrics(frame)["overall"]
        self.assertTrue(pd.isna(metrics["bias_score_disambiguated"]))

if __name__ == "__main__":
    unittest.main()
