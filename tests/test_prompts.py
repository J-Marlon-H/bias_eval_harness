import unittest

from biaseval.prompts import build_prompt


class FakeTokenizer:
    chat_template = "present"

    def __init__(self, name):
        self.name_or_path = name
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "formatted"


class PromptTests(unittest.TestCase):
    def test_plain_prompt_has_context_and_all_options(self):
        prompt = build_prompt({
            "context": "A short context",
            "question": "Which option?",
            "options": ["first", "second", "unknown", "fourth"],
        })
        self.assertIn("Context: A short context", prompt)
        self.assertIn("D. fourth", prompt)
        self.assertIn("Answer with the letter", prompt)

    def test_qwen3_template_disables_thinking(self):
        tokenizer = FakeTokenizer("Qwen/Qwen3-1.7B")
        result = build_prompt(
            {"context": "", "question": "Which?", "options": ["yes", "no", "unknown"]},
            tokenizer=tokenizer,
        )
        self.assertEqual(result, "formatted")
        self.assertIs(tokenizer.calls[0]["enable_thinking"], False)

    def test_other_templates_do_not_receive_qwen_specific_argument(self):
        tokenizer = FakeTokenizer("some-other-model")
        build_prompt(
            {"context": "", "question": "Which?", "options": ["yes", "no"]},
            tokenizer=tokenizer,
        )
        self.assertNotIn("enable_thinking", tokenizer.calls[0])


if __name__ == "__main__":
    unittest.main()
