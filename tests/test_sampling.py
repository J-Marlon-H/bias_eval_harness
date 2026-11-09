import unittest

import pandas as pd

from biaseval.data.sampling import sample_bbq


class SamplingTests(unittest.TestCase):
    def test_sample_is_reproducible_and_balanced(self):
        rows = []
        for condition in ("ambig", "disambig"):
            for polarity in ("neg", "nonneg"):
                for index in range(10):
                    rows.append({
                        "id": f"{condition}-{polarity}-{index}",
                        "category": "Race_ethnicity",
                        "condition": condition,
                        "polarity": polarity,
                    })
        data = pd.DataFrame(rows)
        first = sample_bbq(data, n_per_category=8, seed=7)
        second = sample_bbq(data, n_per_category=8, seed=7)
        self.assertEqual(len(first), 8)
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(first.groupby(["condition", "polarity"]).size().tolist(), [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
