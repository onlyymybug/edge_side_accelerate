from __future__ import annotations

import sys
import unittest
from pathlib import Path


BITNET_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "QNN" / "bitnet" / "scripts"
)
sys.path.insert(0, str(BITNET_SCRIPTS))

import run_math_benchmarks  # noqa: E402


class BitNetRawPromptTest(unittest.TestCase):
    def test_native_chat_prompt_has_no_llama_headers(self) -> None:
        rendered = run_math_benchmarks.native_chat_prompt(
            "What is 2 + 3?", "You are concise."
        )
        self.assertEqual(
            rendered,
            "System: You are concise.<|eot_id|>"
            "User: What is 2 + 3?<|eot_id|>Assistant: ",
        )
        self.assertNotIn("<|start_header_id|>", rendered)

    def test_extracts_only_native_assistant_completion(self) -> None:
        raw = (
            "User: Problem: 2 + 3<|eot_id|>Assistant: "
            "The answer is 5.<|eot_id|>"
        )
        self.assertEqual(
            run_math_benchmarks.assistant_text(raw), "The answer is 5."
        )

if __name__ == "__main__":
    unittest.main()
