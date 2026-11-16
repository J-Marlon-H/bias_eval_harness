import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biaseval.results import update_run_index


class ResultsTests(unittest.TestCase):
    def test_locked_run_index_warns_without_aborting_saved_run(self):
        config = {
            "run_id": "test-run",
            "dataset": "bbq",
            "model_id": "test-model",
            "prompt_condition": "plain",
            "scoring_mode": "loglik",
            "scoring_method": "sequence_logprob",
            "sample_size": 1,
            "created_at": "2026-11-16T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "biaseval.results.os.replace",
                    side_effect=PermissionError("file is locked"),
                ),
                self.assertWarnsRegex(RuntimeWarning, "Predictions and run configuration remain saved"),
            ):
                update_run_index(
                    temp_dir,
                    config,
                    status="running",
                    completed_items=1,
                )

            self.assertFalse((Path(temp_dir) / "index.csv").exists())
            self.assertEqual(list(Path(temp_dir).glob(".index-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
