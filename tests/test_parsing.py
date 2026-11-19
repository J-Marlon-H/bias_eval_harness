import unittest

from biaseval.parsing import parse_answer, strip_think_blocks


class ParsingTests(unittest.TestCase):
    def test_strips_think_block(self):
        self.assertEqual(strip_think_blocks("<think>reasoning</think>\nB"), "B")

    def test_unterminated_think_block_is_not_treated_as_an_answer(self):
        self.assertEqual(
            parse_answer("<think>Reasoning about A, B, and C", ["one", "two", "three"]),
            "invalid",
        )

    def test_parses_first_valid_letter(self):
        self.assertEqual(parse_answer("The answer is option C.", ["one", "two", "three"]), 2)

    def test_prefers_final_answer_after_reasoning(self):
        self.assertEqual(
            parse_answer(
                "I considered A first, but the context supports C. Final answer: C",
                ["one", "two", "three"],
            ),
            2,
        )

    def test_accepts_option_text(self):
        self.assertEqual(parse_answer("I choose water damage.", ["A broken pipe", "Water damage"]), 1)

    def test_unparseable_answer_is_invalid(self):
        self.assertEqual(parse_answer("I cannot decide.", ["yes", "no", "unknown"]), "invalid")


if __name__ == "__main__":
    unittest.main()
