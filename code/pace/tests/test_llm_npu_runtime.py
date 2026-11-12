from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUNTIME_PATH = Path(__file__).resolve().parents[1] / "llm.npu" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("llm_npu_runtime", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


SAMPLE_LOG = """
\x1b[32m[INFO]\x1b[0m QNN HTP performance mode: balanced
[INFO] Successfully created QNNModel for graph: model.0.s32
[INFO] Successfully created QNNModel for graph: model.0.s1
[INFO] QNN context loaded successfully from context.bin
[INFO] PromptProcessor.cpp:129 num_tokens: 36
部署成功<|im_end|>\x1b[36m
============== Performance Summary ===============
Prefill time        :  107884.00 μs (333.69 tokens/s)
Decode time         :   66675.00 μs ( 30.00 tokens/s)
Prefill tokens      :         36
Decode tokens       :          2
"""


class LlmNpuRuntimeTest(unittest.TestCase):
    def test_qwen_chatml_matches_qnn_default_system_prompt(self) -> None:
        rendered = runtime.qwen_chatml("What is 2+3?")
        self.assertTrue(
            rendered.startswith(
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n"
            )
        )
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_clean_completion_removes_mllm_diagnostics_and_eos(self) -> None:
        self.assertEqual(runtime.clean_completion(SAMPLE_LOG), ("部署成功", True))

    def test_parse_mllm_perf(self) -> None:
        perf = runtime.parse_perf(SAMPLE_LOG)
        self.assertEqual(perf["prompt_tokens"], 36)
        self.assertEqual(perf["generated_tokens"], 2)
        self.assertAlmostEqual(perf["prompt_eval_ms"], 107.884)
        self.assertAlmostEqual(perf["decode_eval_ms"], 66.675)
        self.assertAlmostEqual(perf["inference_ms"], 174.559)
        self.assertAlmostEqual(perf["runner_decode_token_per_sec"], 30.0)

    def test_htp_mode_and_graphs_are_verified(self) -> None:
        htp = runtime.parse_htp(SAMPLE_LOG, "balanced")
        self.assertTrue(htp["htp_execution_verified"])
        self.assertEqual(htp["htp_performance_actual"], "balanced")
        self.assertEqual(htp["qnn_graphs_loaded"], "model.0.s1,model.0.s32")

    def test_wrong_htp_mode_is_rejected(self) -> None:
        self.assertFalse(
            runtime.parse_htp(SAMPLE_LOG, "burst")["htp_execution_verified"]
        )

    def test_aot_defaults(self) -> None:
        config = runtime.LlmNpuConfig(serial="test")
        self.assertEqual(config.max_seq_len, 1024)
        self.assertEqual(config.ar_len, 32)
        self.assertEqual(config.htp_performance, "burst")
        self.assertFalse(config.allow_active_runner)

    def test_adb_transport_error_detection(self) -> None:
        self.assertTrue(runtime.is_transport_error("adb: device offline"))
        self.assertFalse(runtime.is_transport_error("QNN runner returned status 1"))


if __name__ == "__main__":
    unittest.main()
