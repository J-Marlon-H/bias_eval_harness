import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from biaseval.data.bbq import BBQ_CATEGORIES, load_bbq


class BBQLoaderIntegrationTests(unittest.TestCase):
    def test_metadata_column_order_is_stable(self):
        raw_item = {
            "example_id": "1",
            "question_index": "template-1",
            "question_polarity": "neg",
            "context_condition": "ambig",
            "category": "Age",
            "context": "Two people are talking.",
            "question": "Who?",
            "label": 2,
            "answer_info": {
                "ans0": "Person A",
                "ans1": "Person B",
                "ans2": "unknown",
            },
            "ans0": "Person A",
            "ans1": "Person B",
            "ans2": "Unknown",
        }
        metadata = pd.DataFrame(
            [{"target_loc": 0, "label_type": "name",
              "example_id": "1", "category": "Age"}]
        )
        with (
            patch("biaseval.data.bbq._download", side_effect=[
                Path("Age.jsonl"), Path("additional_metadata.csv"),
            ]),
            patch("biaseval.data.bbq._read_jsonl", return_value=[raw_item]),
            patch("biaseval.data.bbq.pd.read_csv", return_value=metadata),
        ):
            frame = load_bbq(categories=("Age",))

        self.assertLess(
            frame.columns.get_loc("label_type"),
            frame.columns.get_loc("target_loc"),
        )

    @unittest.skipUnless(
        __import__("os").environ.get("RUN_BBQ_DATA_TEST") == "1",
        "Set RUN_BBQ_DATA_TEST=1 to download and validate the official BBQ files",
    )
    def test_official_counts_and_category_scoped_ids(self):
        frame = load_bbq(BBQ_CATEGORIES)
        expected = {
            "Age": 3680,
            "Disability_status": 1556,
            "Gender_identity": 5656,
            "Nationality": 3080,
            "Physical_appearance": 1576,
            "Race_ethnicity": 6880,
            "Religion": 1200,
            "SES": 6864,
            "Sexual_orientation": 864,
        }
        self.assertEqual(len(frame), 31356)
        self.assertEqual(frame["category"].value_counts().to_dict(), expected)
        self.assertFalse(frame.duplicated(["category", "example_id"]).any())


if __name__ == "__main__":
    unittest.main()
