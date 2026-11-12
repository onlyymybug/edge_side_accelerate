from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUNTIME_PATH = (
    Path(__file__).resolve().parents[1] / "llama.cpp-NPU" / "runtime.py"
)
SPEC = importlib.util.spec_from_file_location("llamacpp_npu_runtime", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


class LlamaCppNpuRuntimeTest(unittest.TestCase):
    def test_qwen_chatml_has_exact_role_boundaries(self) -> None:
        rendered = runtime.qwen_chatml("What is 2+3?")
        self.assertIn("<|im_start|>system\n", rendered)
        self.assertIn("<|im_start|>user\nWhat is 2+3?<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_clean_completion_stops_at_qwen_eos(self) -> None:
        self.assertEqual(runtime.clean_completion("5<|im_end|> [end of text]\n"), ("5", True))

    def test_parse_llama_perf_and_htp_evidence(self) -> None:
        log = """
load_backend: failed to find ggml_backend_init in ./libggml-htp.so
llm_load_tensors:       RPCMEM model buffer size =  1885.30 MiB
Initializing HTP backend... (You should see this once)
llama_new_context_with_model: graph splits = 3
llama_perf_context_print:        load time =    4429.48 ms
llama_perf_context_print: prompt eval time =     507.33 ms /    32 tokens (   15.85 ms per token,    63.08 tokens per second)
llama_perf_context_print:        eval time =    1573.19 ms /     1 runs   ( 1573.19 ms per token,     0.64 tokens per second)
llama_perf_context_print:       total time =    2084.05 ms /    33 tokens
"""
        perf = runtime.parse_perf(log)
        self.assertEqual(perf["prompt_tokens"], 32)
        self.assertEqual(perf["decode_eval_runs"], 1)
        self.assertAlmostEqual(perf["llama_decode_token_per_sec"], 0.64)
        htp = runtime.parse_htp(log)
        self.assertTrue(htp["htp_execution_verified"])
        self.assertAlmostEqual(htp["htp_rpcmem_model_mib"], 1885.30)
        self.assertEqual(htp["htp_graph_splits"], 3)

    def test_missing_htp_evidence_is_not_verified(self) -> None:
        htp = runtime.parse_htp("ordinary llama.cpp CPU log")
        self.assertFalse(htp["htp_execution_verified"])

    def test_parse_zero_time_prefill_keeps_token_count(self) -> None:
        perf = runtime.parse_perf(
            "llama_perf_context_print: prompt eval time = 0.00 ms / 1 tokens "
            "(0.00 ms per token, inf tokens per second)"
        )
        self.assertEqual(perf["prompt_tokens"], 1)
        self.assertEqual(perf["prompt_eval_ms"], 0.0)

    def test_adb_transport_error_detection(self) -> None:
        self.assertTrue(runtime.is_transport_error("adb: device offline"))
        self.assertFalse(runtime.is_transport_error("llama-cli returned exit status 1"))

    def test_concurrent_runner_guard_defaults_to_enabled(self) -> None:
        config = runtime.LlamaCppNpuConfig(serial="test")
        self.assertFalse(config.allow_active_runner)
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.ubatch_size, 64)
        self.assertEqual(
            config.required_dsp_skel_sha256,
            "e808e9333a1efdc601bb87588e27ff6348d165b0f1b64bf17a8354a754051a9e",
        )

    def test_phone_check_rejects_an_existing_llama_cli(self) -> None:
        runner = runtime.LlamaCppNpuPhoneRunner.__new__(runtime.LlamaCppNpuPhoneRunner)
        runner.config = runtime.LlamaCppNpuConfig(serial="test")
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
