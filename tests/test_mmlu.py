import unittest
from unittest.mock import patch

from biaseval.data.mmlu import _normalize_row, load_mmlu
from biaseval.data.sampling import sample_mmlu
import pandas as pd


class MMLUTests(unittest.TestCase):
    def test_normalize_row_uses_four_choices_and_stable_id(self):
        row = {
            "subject": "abstract_algebra",
            "question": "Which?",
            "choices": ["A", "B", "C", "D"],
            "answer": 2,
        }
        normalized = _normalize_row(row, 7)
        self.assertEqual(normalized["id"], "abstract_algebra:7")
        self.assertEqual(normalized["options"], row["choices"])
        self.assertEqual(normalized["label"], 2)
        self.assertEqual(normalized["context"], "")

    def test_loader_requests_all_subjects_test_split(self):
        rows = [{
            "subject": "math",
            "question": "Q",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
        }]
        with patch("biaseval.data.mmlu.load_dataset", return_value=rows) as loader:
            data = load_mmlu(token="read-token")
        loader.assert_called_once_with(
            "cais/mmlu", "all", split="test", token="read-token"
        )
        self.assertEqual(len(data), 1)

    def test_sample_is_reproducible_and_capped_per_subject(self):
        data = pd.DataFrame([
            {"id": f"{subject}-{index}", "subject": subject}
            for subject in ("history", "math")
            for index in range(5)
        ])
        first = sample_mmlu(data, n_per_subject=2, seed=10)
        second = sample_mmlu(data, n_per_subject=2, seed=10)
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(first.groupby("subject").size().to_dict(), {
            "history": 2, "math": 2,
        })


if __name__ == "__main__":
    unittest.main()
