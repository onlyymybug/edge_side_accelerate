#!/usr/bin/env python3
"""Protocol and accounting tests; these do not run a model or a phone."""

import json
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from agent_closed_loop import (
    DEFAULT_BUNDLE,
    DEFAULT_TASKS,
    MockBackend,
    clean_generation,
    parse_tool_call,
    run_closed_loop,
)


class AgentClosedLoopTests(unittest.TestCase):
    def test_tool_call_parser(self) -> None:
        self.assertEqual(
            parse_tool_call('{"name":"lookup_item","parameters":{"item":"x"}}<|im_end|>'),
            "x",
        )
        self.assertEqual(clean_generation("done<|im_end|>ignored"), "done")
        for value in ('{"name":"other","parameters":{"item":"x"}}', "not JSON"):
            with self.assertRaises(ValueError):
                parse_tool_call(value)

    def test_two_tasks_complete_with_real_tool_wait(self) -> None:
        fixture = json.loads(Path(DEFAULT_TASKS).read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(str(DEFAULT_BUNDLE / "device" / "tokenizer.json"))
        backend = MockBackend(fixture["tasks"])
        events, requests, results = run_closed_loop(
            backend, tokenizer, fixture["tasks"], fixture["catalog"],
            tool_delay_ms=20, max_new_tokens=96, context_tokens=6144,
        )
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(sum(event["event"] == "tool_end" for event in events), 2)
        self.assertEqual(sum(event["event"] == "session_release" for event in events), 2)
        self.assertTrue(all(request["queue_ms"] >= 0 for request in requests))
        self.assertTrue(all(request["prompt_prepare_ms"] >= 0 for request in requests))
        self.assertEqual({request["task_id"] for request in requests},
                         {"smoke_case", "smoke_cable"})


if __name__ == "__main__":
    unittest.main()
