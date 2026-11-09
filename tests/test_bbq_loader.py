import unittest

from biaseval.data.bbq import BBQ_CATEGORIES, load_bbq


class BBQLoaderIntegrationTests(unittest.TestCase):
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
