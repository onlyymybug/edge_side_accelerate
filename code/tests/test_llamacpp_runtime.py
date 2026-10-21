from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUNTIME_PATH = Path(__file__).resolve().parents[1] / "llama.cpp" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("llamacpp_runtime", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


class LlamaCppRuntimeTest(unittest.TestCase):
    def test_qwen_chatml_uses_qwen_default_system_and_role_boundaries(self) -> None:
        rendered = runtime.qwen_chatml("What is 2+3?")
        self.assertIn("You are Qwen, created by Alibaba Cloud.", rendered)
        self.assertIn("<|im_start|>user\nWhat is 2+3?<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_clean_output_file_extracts_only_assistant(self) -> None:
        prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        raw = f"User:\n{prompt}\nAssistant:\n<tool_call>\n{{\"name\":\"x\"}}\n</tool_call>\n"
        completion, eos = runtime.clean_completion(raw, prompt)
        self.assertEqual(completion, '<tool_call>\n{"name":"x"}\n</tool_call>')
        self.assertFalse(eos)

    def test_parse_compact_upstream_timing(self) -> None:
        parsed = runtime.parse_perf("[ Prompt: 28.3 t/s | Generation: 18.1 t/s ]")
        self.assertEqual(parsed["llama_prefill_token_per_sec"], 28.3)
        self.assertEqual(parsed["llama_decode_token_per_sec"], 18.1)

    def test_transport_error_detection(self) -> None:
        self.assertTrue(runtime.is_transport_error("adb: device offline"))
        self.assertFalse(runtime.is_transport_error("llama-cli returned exit status 1"))

    def test_configuration_defaults(self) -> None:
        config = runtime.LlamaCppConfig(serial="test")
        self.assertEqual(config.threads, 6)
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.ubatch_size, 64)
        self.assertFalse(config.allow_active_runner)

    def test_phone_check_rejects_an_existing_llama_cli(self) -> None:
        runner = runtime.LlamaCppPhoneRunner.__new__(runtime.LlamaCppPhoneRunner)
        runner.config = runtime.LlamaCppConfig(serial="test")
        replies = iter(
            (
                runtime.subprocess.CompletedProcess([], 0, "device\n", ""),
                runtime.subprocess.CompletedProcess(
                    [], 0, "pid=42 command=./llama-cli -m another.gguf\n", ""
                ),
            )
        )
        runner.adb = lambda *args, **kwargs: next(replies)
        with self.assertRaisesRegex(RuntimeError, "another llama-cli workload"):
            runner._check_phone()


if __name__ == "__main__":
    unittest.main()
