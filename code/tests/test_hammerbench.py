from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


QWEN_CODE = Path(__file__).resolve().parents[1] / "QNN" / "qwen"
sys.path.insert(0, str(QWEN_CODE))

import run_hammerbench as hammer  # noqa: E402
import run_qnn_math_benchmarks as pace_runner  # noqa: E402


def sample(*, kind: str, split: str = "single-turn") -> dict[str, object]:
    return {
        "hammer_type": kind,
        "language": "en",
        "split": split,
        "source_index": 0,
        "original_id": kind,
        "conversation_id": kind,
        "turn_index": 1,
        "label_name": "Navigation.Map.search",
        "label_parameters": {"city": "Shanghai"},
    }


class HammerBenchScoringTest(unittest.TestCase):
    def test_inference_export_accepts_failed_row_without_output_path(self) -> None:
        benchmark_sample = {
            "sample_id": "failed_0001",
            "original_id": "failed_0001",
            "visible_input": "Call the test tool.",
            "label_name": "Diagnostics.Test",
            "label_parameters": {},
        }
        row = {
            "sample_id": "failed_0001",
            "runner_returncode": -1,
            "error": "ADB command timed out",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "inference_results.json"
            hammer.write_inference_results(
                output_path, [row], {"failed_0001": benchmark_sample}
            )
            exported = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exported[0]["predict"], "")

    def test_bitnet_completion_uses_outer_llama_assistant_boundary(self) -> None:
        prompt = (
            "<|im_start|>system\nRules<|im_end|>\n"
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        raw = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            + '{"name":"Diagnostics.Test","parameters":{}}<|eot_id|>'
        )
        original = pace_runner.RUNNER_MODEL_PROFILE
        pace_runner.RUNNER_MODEL_PROFILE = "bitnet"
        try:
            completion = pace_runner.completion_from_output(raw, prompt)
        finally:
            pace_runner.RUNNER_MODEL_PROFILE = original
        self.assertEqual(
            '{"name":"Diagnostics.Test","parameters":{}}', completion
        )

    def test_bitnet_raw_prompt_counting_has_no_outer_wrapper(self) -> None:
        original_profile = pace_runner.RUNNER_MODEL_PROFILE
        original_raw = pace_runner.BITNET_RAW_PROMPT
        pace_runner.RUNNER_MODEL_PROFILE = "bitnet"
        pace_runner.BITNET_RAW_PROMPT = True
        try:
            rendered = pace_runner.runner_prompt_for_counting(
                "System: rules<|eot_id|>User: question<|eot_id|>Assistant: "
            )
        finally:
            pace_runner.RUNNER_MODEL_PROFILE = original_profile
            pace_runner.BITNET_RAW_PROMPT = original_raw
        self.assertEqual(
            "System: rules<|eot_id|>User: question<|eot_id|>Assistant: ",
            rendered,
        )

    def test_bitnet_separates_system_policy_from_user_request(self) -> None:
        value = sample(kind="ST-Perfect")
        value["tools"] = [
            {
                "name": "Navigation.Map.search",
                "description": "Search a map",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        value["visible_messages"] = [
            {"role": "user", "content": "Find Shanghai."}
        ]
        original = hammer.MODEL_PROFILE
        original_layout = hammer.BITNET_HAMMER_PROMPT_LAYOUT
        hammer.MODEL_PROFILE = "bitnet"
        hammer.BITNET_HAMMER_PROMPT_LAYOUT = "llama3_system"
        try:
            system_prompt = hammer.build_system_prompt(hammer.TASK_NAME, value, {})
            user_prompt = hammer.build_prompt(hammer.TASK_NAME, value, {})
        finally:
            hammer.MODEL_PROFILE = original
            hammer.BITNET_HAMMER_PROMPT_LAYOUT = original_layout
        self.assertIn("<tools>", system_prompt)
        self.assertIn("Navigation.Map.search", system_prompt)
        self.assertEqual("Find Shanghai.", user_prompt)
        self.assertNotIn("<tools>", user_prompt)

    def test_bitnet_native_template_can_be_nested_for_legacy_runner(self) -> None:
        value = sample(kind="ST-Perfect")
        value["tools"] = [
            {
                "name": "Navigation.Map.search",
                "description": "Search a map",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        value["visible_messages"] = [
            {"role": "user", "content": "Find Shanghai."}
        ]
        original = hammer.MODEL_PROFILE
        original_layout = hammer.BITNET_HAMMER_PROMPT_LAYOUT
        hammer.MODEL_PROFILE = "bitnet"
        hammer.BITNET_HAMMER_PROMPT_LAYOUT = "native_nested"
        try:
            prompt = hammer.build_prompt(hammer.TASK_NAME, value, {})
            runner_system = hammer.build_runner_system_prompt(
                hammer.TASK_NAME, value, {}
            )
        finally:
            hammer.MODEL_PROFILE = original
            hammer.BITNET_HAMMER_PROMPT_LAYOUT = original_layout
        self.assertTrue(prompt.startswith("System: "))
        self.assertIn("<|eot_id|>User: Find Shanghai.<|eot_id|>Assistant: ", prompt)
        self.assertEqual("", runner_system)

    def test_qwen_template_can_be_nested_for_bitnet_diagnostic(self) -> None:
        value = sample(kind="ST-Perfect")
        value["tools"] = [
            {
                "name": "Navigation.Map.search",
                "description": "Search a map",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        value["visible_messages"] = [
            {"role": "user", "content": "Find Shanghai."}
        ]
        original = hammer.MODEL_PROFILE
        original_layout = hammer.BITNET_HAMMER_PROMPT_LAYOUT
        hammer.MODEL_PROFILE = "bitnet"
        hammer.BITNET_HAMMER_PROMPT_LAYOUT = "qwen_nested"
        try:
            prompt = hammer.build_prompt(hammer.TASK_NAME, value, {})
            runner_system = hammer.build_runner_system_prompt(
                hammer.TASK_NAME, value, {}
            )
        finally:
            hammer.MODEL_PROFILE = original
            hammer.BITNET_HAMMER_PROMPT_LAYOUT = original_layout
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_start|>user\nFind Shanghai.<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual("", runner_system)

    def test_relevant_call_requires_function_and_arguments(self) -> None:
        result = hammer.score_output(
            hammer.TASK_NAME,
            sample(kind="ST-Perfect"),
            '{"name":"Navigation.Map.search","parameters":{"city":"Shanghai"}}',
            {},
        )
        self.assertTrue(result["relevant_end_to_end_correct"])
        self.assertFalse(result["irrelevant_expected"])
        self.assertTrue(result["score_correct"])

    def test_irrelevant_rejection_is_benchmark_correct(self) -> None:
        result = hammer.score_output(
            hammer.TASK_NAME,
            sample(kind="ir-ST-Perfect"),
            hammer.REJECTION_TEXT,
            {},
        )
        self.assertTrue(result["irrelevant_expected"])
        self.assertTrue(result["irrelevant_correct"])
        self.assertFalse(result["function_correct"])
        self.assertTrue(result["score_correct"])

    def test_irrelevant_tool_call_is_incorrect(self) -> None:
        result = hammer.score_output(
            hammer.TASK_NAME,
            sample(kind="ir-ST-Perfect"),
            '{"name":"Navigation.Map.search","parameters":{"city":"Shanghai"}}',
            {},
        )
        self.assertFalse(result["irrelevant_correct"])
        self.assertFalse(result["score_correct"])

    def test_multi_turn_progress_and_success(self) -> None:
        rows = [
            {
                "split": "multi-turn",
                "conversation_id": "Based_0",
                "turn_index": 1,
                "hammer_type": "Based",
                "score_correct": True,
            },
            {
                "split": "multi-turn",
                "conversation_id": "Based_0",
                "turn_index": 2,
                "hammer_type": "Based",
                "score_correct": False,
            },
            {
                "split": "multi-turn",
                "conversation_id": "Based_0",
                "turn_index": 3,
                "hammer_type": "Based",
                "score_correct": True,
            },
            {
                "split": "multi-turn",
                "conversation_id": "Based_1",
                "turn_index": 1,
                "hammer_type": "Based",
                "score_correct": True,
            },
        ]
        metrics = hammer.conversation_metrics(rows)
        self.assertEqual(2, len(metrics))
        self.assertAlmostEqual(1 / 3, metrics[0]["progress_rate"])
        self.assertFalse(metrics[0]["success"])
        self.assertEqual(1.0, metrics[1]["progress_rate"])
        self.assertTrue(metrics[1]["success"])

    def test_summary_uses_irrelevant_aware_accuracy(self) -> None:
        relevant = hammer.score_output(
            hammer.TASK_NAME,
            sample(kind="ST-Perfect"),
            '{"name":"Navigation.Map.search","parameters":{"city":"Shanghai"}}',
            {},
        )
        irrelevant = hammer.score_output(
            hammer.TASK_NAME,
            sample(kind="ir-ST-Perfect"),
            hammer.REJECTION_TEXT,
            {},
        )
        relevant["runner_returncode"] = 0
        irrelevant["runner_returncode"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation_summary.csv"
            hammer.write_evaluation_summary(output, [relevant, irrelevant])
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        overall = rows[0]
        self.assertEqual(1.0, float(overall["end_to_end_acc"]))
        self.assertEqual(1.0, float(overall["function_irrelevant_accuracy"]))
        self.assertEqual(1.0, float(overall["relevant_end_to_end_acc"]))


if __name__ == "__main__":
    unittest.main()
