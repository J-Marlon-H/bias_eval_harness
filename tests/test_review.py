import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biaseval.review import (
    build_review_frame,
    create_wrong_answers_report,
    write_wrong_answers_html,
)


class ReviewTests(unittest.TestCase):
    def test_build_review_includes_details_for_wrong_answers_only(self):
        predictions = pd.DataFrame(
            [
                {
                    "id": "math:4",
                    "label": 2,
                    "prediction": 0,
                    "raw_text": None,
                    "subject": "math",
                }
            ]
        )
        source = pd.DataFrame(
            [
                {
                    "id": "math:4",
                    "subject": "math",
                    "question": "Which is prime?",
                    "options": ["4", "6", "7", "8"],
                    "label": 2,
                    "context": "",
                }
            ]
        )

        review = build_review_frame(predictions, source, dataset="mmlu")

        self.assertEqual(review.loc[0, "question"], "Which is prime?")
        self.assertEqual(review.loc[0, "option_C"], "7")
        self.assertEqual(review.loc[0, "correct_option"], "C")
        self.assertEqual(review.loc[0, "model_option"], "A")
        self.assertEqual(review.loc[0, "model_answer"], "4")
        self.assertFalse(review.loc[0, "is_correct"])
        self.assertEqual(len(review), 1)

    def test_html_report_shows_wrong_answers_and_escapes_question_text(self):
        review = pd.DataFrame(
            [
                {
                    "item_id": "math:4",
                    "subject": "math",
                    "condition": "",
                    "category": "",
                    "context": "",
                    "question": "<script>alert(1)</script>",
                    "option_A": "4",
                    "option_B": "6",
                    "option_C": "7",
                    "option_D": "8",
                    "correct_option": "C",
                    "correct_answer": "7",
                    "model_option": "A",
                    "model_answer": "4",
                    "answer_valid": True,
                    "target_option": "",
                    "unknown_option": "",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_wrong_answers_html(
                [("MMLU", review)], Path(temp_dir) / "wrong_answers.html"
            )
            content = path.read_text(encoding="utf-8")
            self.assertTrue(path.exists())
            self.assertIn("Correct answer", content)
            self.assertIn("Model selected", content)
            self.assertIn("data-filter=\"subject\"", content)
            self.assertIn("placeholder=\"Search questions and answers\"", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertNotIn("<script>alert(1)</script>", content)

    def test_create_report_joins_predictions_and_source_data(self):
        predictions = pd.DataFrame(
            [{"id": "math:4", "label": 2, "prediction": 0, "raw_text": None}]
        )
        source = pd.DataFrame(
            [{
                "id": "math:4",
                "subject": "math",
                "question": "Which is prime?",
                "options": ["4", "6", "7", "8"],
                "label": 2,
            }]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions = pd.concat(
                [
                    predictions,
                    pd.DataFrame(
                        [{"id": "math:5", "label": 0, "prediction": 0, "raw_text": None}]
                    ),
                ],
                ignore_index=True,
            )
            source = pd.concat(
                [
                    source,
                    pd.DataFrame(
                        [{
                            "id": "math:5",
                            "subject": "math",
                            "question": "Which is even?",
                            "options": ["4", "6", "7", "8"],
                            "label": 0,
                        }]
                    ),
                ],
                ignore_index=True,
            )
            path = create_wrong_answers_report(
                [("MMLU", predictions, source, "mmlu")],
                Path(temp_dir) / "wrong_answers.html",
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("Which is prime?", content)
            self.assertIn("Correct answer:</strong> C - 7", content)
            self.assertIn("1 incorrect or invalid out of 2", content)
            self.assertIn("<td>1 / 2</td>", content)
            self.assertIn("Filtered results:", content)
            self.assertIn('data-dataset="MMLU" data-category="" data-condition="" data-subject="math"', content)
            self.assertIn('"total": 2, "correct": 1, "incorrect": 1', content)

    def test_missing_source_item_is_an_explicit_error(self):
        predictions = pd.DataFrame(
            [{"id": "math:missing", "label": 0, "prediction": 0}]
        )
        source = pd.DataFrame(
            [
                {
                    "id": "math:present",
                    "question": "Question",
                    "options": ["A", "B", "C", "D"],
                    "label": 0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "Could not find"):
            build_review_frame(predictions, source, dataset="mmlu")


if __name__ == "__main__":
    unittest.main()
