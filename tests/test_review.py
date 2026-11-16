import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biaseval.review import (
    build_review_frame,
    create_saved_runs_report,
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

    def test_report_has_filterable_bbq_and_mmlu_performance_overview(self):
        predictions = pd.DataFrame(
            [
                {"id": "Age:ambig-correct", "category": "Age", "condition": "ambig",
                 "label": 2, "prediction": 2, "target_loc": 1, "unknown_loc": 2},
                {"id": "Age:ambig-wrong", "category": "Age", "condition": "ambig",
                 "label": 2, "prediction": 1, "target_loc": 1, "unknown_loc": 2},
                {"id": "Age:disambig-correct", "category": "Age", "condition": "disambig",
                 "label": 0, "prediction": 0, "target_loc": 1, "unknown_loc": 2},
                {"id": "Age:disambig-unknown", "category": "Age", "condition": "disambig",
                 "label": 0, "prediction": 2, "target_loc": 1, "unknown_loc": 2},
            ]
        )
        source = pd.DataFrame(
            [
                {"id": row["id"], "category": row["category"],
                 "condition": row["condition"], "label": row["label"],
                 "target_loc": row["target_loc"], "unknown_loc": row["unknown_loc"],
                 "question": row["id"], "options": ["A", "B", "Unknown"]}
                for row in predictions.to_dict(orient="records")
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = create_wrong_answers_report(
                [("BBQ", predictions, source, "bbq")],
                Path(temp_dir) / "performance.html",
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn('data-tab="overview"', content)
            self.assertIn('data-tab="bbq"', content)
            self.assertIn('data-tab="mmlu"', content)
            self.assertIn('id="ambiguous-chart"', content)
            self.assertIn('id="disambiguated-chart"', content)
            self.assertIn('"condition": "disambig"', content)
            self.assertIn('"correct": 1, "incorrect": 1, "invalid": 0, "unknown": 1', content)
            self.assertIn("Click a category to filter the report", content)

    def test_saved_runs_report_groups_models_and_selects_one_run(self):
        source = pd.DataFrame(
            [
                {"id": "Age:1", "category": "Age", "condition": "ambig",
                 "label": 2, "target_loc": 1, "unknown_loc": 2,
                 "question": "Who?", "options": ["A", "B", "Unknown"]}
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runs_dir = results_dir / "runs"
            for run_id, model_id, prediction in (
                ("run-base", "org/base-model", 2),
                ("run-finetuned", "org/fine-tuned-model", 1),
            ):
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(
                    json.dumps({
                        "run_id": run_id,
                        "model_id": model_id,
                        "dataset": "bbq",
                        "prompt_condition": "plain",
                        "scoring_mode": "loglik",
                        "scoring_method": "sequence_logprob",
                        "created_at": run_id,
                        "status": "complete",
                    }),
                    encoding="utf-8",
                )
                (run_dir / "predictions.jsonl").write_text(
                    json.dumps({
                        "id": "Age:1",
                        "label": 2,
                        "prediction": prediction,
                        "category": "Age",
                        "condition": "ambig",
                        "target_loc": 1,
                        "unknown_loc": 2,
                    }) + "\n",
                    encoding="utf-8",
                )

            path = create_saved_runs_report(
                results_dir,
                {"bbq": source},
                results_dir / "performance.html",
            )
            content = path.read_text(encoding="utf-8")

        self.assertIn('id="run-selector"', content)
        self.assertIn('<optgroup label="org/base-model">', content)
        self.assertIn('<optgroup label="org/fine-tuned-model">', content)
        self.assertIn('data-run-id="run-base"', content)
        self.assertIn('data-run-id="run-finetuned"', content)
        self.assertIn('group.selection_id === selectionId', content)

    def test_saved_report_combines_datasets_and_ignores_superseded_run(self):
        bbq_source = pd.DataFrame(
            [
                {"id": "Age:1", "category": "Age", "condition": "ambig",
                 "label": 2, "target_loc": 1, "unknown_loc": 2,
                 "question": "BBQ question?", "options": ["A", "B", "Unknown"]}
            ]
        )
        mmlu_source = pd.DataFrame(
            [
                {"id": "math:1", "subject": "math", "label": 0,
                 "question": "MMLU question?", "options": ["A", "B", "C", "D"]}
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            runs_dir = results_dir / "runs"
            run_specs = (
                ("old-bbq", "bbq", 2, 2, "interrupted"),
                ("current-bbq", "bbq", 2, 1, "complete"),
                ("current-mmlu", "mmlu", 0, 1, "complete"),
            )
            for run_id, dataset, label, prediction, status in run_specs:
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(
                    json.dumps({
                        "run_id": run_id,
                        "model_id": "org/vanilla-model",
                        "dataset": dataset,
                        "prompt_condition": "plain",
                        "system_prompt": None,
                        "scoring_mode": "loglik",
                        "scoring_method": "sequence_logprob",
                        "created_at": run_id,
                        "status": status,
                    }),
                    encoding="utf-8",
                )
                record = {
                    "id": "Age:1" if dataset == "bbq" else "math:1",
                    "label": label,
                    "prediction": prediction,
                }
                if dataset == "bbq":
                    record.update({
                        "category": "Age",
                        "condition": "ambig",
                        "target_loc": 1,
                        "unknown_loc": 2,
                    })
                else:
                    record["subject"] = "math"
                (run_dir / "predictions.jsonl").write_text(
                    json.dumps(record) + "\n",
                    encoding="utf-8",
                )

            path = create_saved_runs_report(
                results_dir,
                {"bbq": bbq_source, "mmlu": mmlu_source},
                results_dir / "performance.html",
            )
            content = path.read_text(encoding="utf-8")

        run_options = content.split('id="run-selector">', 1)[1].split("</select>", 1)[0]
        self.assertEqual(run_options.count("<option value="), 1)
        self.assertIn("plain prompt · BBQ, MMLU", content)
        self.assertIn('"BBQ": "current-bbq"', content)
        self.assertIn('"MMLU": "current-mmlu"', content)
        self.assertIn('data-run-id="current-bbq"', content)
        self.assertIn('data-run-id="current-mmlu"', content)
        self.assertNotIn('data-run-id="old-bbq"', content)

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
