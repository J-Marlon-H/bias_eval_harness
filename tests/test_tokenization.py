import unittest

from biaseval.backends.tokenization import encode_answer_candidates


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "prompt.":
            return [10]
        if text == "prompt. A":
            return [10, 11, 12]
        if text == "prompt. B":
            return [10, 13]
        if text == "prompt. C":
            return [10, 14, 15, 16]
        raise AssertionError(f"Unexpected tokenization input: {text}")


class TokenizationTests(unittest.TestCase):
    def test_scoring_supports_multi_token_letters_and_prefix_space(self):
        prompt_ids, candidates = encode_answer_candidates(FakeTokenizer(), "prompt.", 3)
        self.assertEqual(prompt_ids, [10])
        self.assertEqual(candidates, [[11, 12], [13], [14, 15, 16]])


if __name__ == "__main__":
    unittest.main()
