from __future__ import annotations

import sys
import unittest
from pathlib import Path


TMAC_CODE = Path(__file__).resolve().parents[1] / "T-MAC"
sys.path.insert(0, str(TMAC_CODE))

from runtime import clean_completion, is_transport_error, parse_perf, qwen_chatml  # noqa: E402


class TmacRuntimeTest(unittest.TestCase):
    def test_qwen_chatml_has_exact_role_boundaries(self) -> None:
        rendered = qwen_chatml("What is 2+3?")
        self.assertIn("<|im_start|>system\n", rendered)
        self.assertIn("<|im_start|>user\nWhat is 2+3?<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_clean_completion_removes_only_runner_noise_and_eos(self) -> None:
        raw = (
            "[13:45:52] /src/ggml-tmac.cpp:38: ggml_tmac_init\n"
            "Paris [end of text]\n"
        )
        self.assertEqual(clean_completion(raw), ("Paris", True))

    def test_parse_llama_perf(self) -> None:
        log = """
llama_perf_context_print:        load time =    1458.90 ms
llama_perf_context_print: prompt eval time =     797.48 ms /    33 tokens (   24.17 ms per token,    41.38 tokens per second)
llama_perf_context_print:        eval time =      38.73 ms /     1 runs   (   38.73 ms per token,    25.82 tokens per second)
llama_perf_context_print:       total time =     837.42 ms /    34 tokens
"""
        parsed = parse_perf(log)
        self.assertEqual(parsed["prompt_tokens"], 33)
        self.assertEqual(parsed["decode_eval_runs"], 1)
        self.assertEqual(parsed["llama_total_tokens"], 34)
        self.assertAlmostEqual(parsed["llama_decode_token_per_sec"], 25.82)

    def test_adb_transport_error_detection(self) -> None:
        self.assertTrue(is_transport_error("RuntimeError: adb: device offline"))
        self.assertTrue(is_transport_error("error: no devices/emulators found"))
        self.assertTrue(is_transport_error("error: device '10.0.0.2:5555' not found"))
        self.assertFalse(is_transport_error("llama-cli returned exit status 1"))


if __name__ == "__main__":
    unittest.main()
