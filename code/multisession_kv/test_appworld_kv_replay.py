import unittest

from appworld_kv_replay import common_prefix, simulate


class ReplayTests(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix([], [1]), 0)

    def test_ready_signal_changes_victim_without_future_events(self):
        trace = {
            "events": [
                {"event": "llm_ready", "task_id": "a"},
                {"event": "inference_start", "task_id": "a", "request_id": "a1"},
                {"event": "llm_ready", "task_id": "b"},
                {"event": "inference_start", "task_id": "b", "request_id": "b1"},
                {"event": "llm_ready", "task_id": "c"},
                {"event": "llm_ready", "task_id": "a"},
                {"event": "inference_start", "task_id": "c", "request_id": "c1"},
                {"event": "inference_start", "task_id": "a", "request_id": "a2"},
            ],
            "requests": {
                "a1": {"task_id": "a", "input_token_ids": [1, 2],
                       "generated_token_ids": [3], "final_generated_token_cached": True},
                "b1": {"task_id": "b", "input_token_ids": [5],
                       "generated_token_ids": [6], "final_generated_token_cached": True},
                "c1": {"task_id": "c", "input_token_ids": [7],
                       "generated_token_ids": [8], "final_generated_token_cached": True},
                "a2": {"task_id": "a", "input_token_ids": [1, 2, 3, 4],
                       "generated_token_ids": [9], "final_generated_token_cached": True},
            },
        }
        plain = simulate(trace, policy="lru", kv_slots=2, release_on_done=True)
        ready = simulate(trace, policy="ready_lru", kv_slots=2, release_on_done=True)
        self.assertEqual(plain["evictions"][0]["victim"], "a")
        self.assertEqual(ready["evictions"][0]["victim"], "b")
        self.assertEqual(plain["requests"][-1]["reused_tokens"], 0)
        self.assertEqual(ready["requests"][-1]["reused_tokens"], 3)

    def test_close_signal_is_separate_ablation(self):
        requests = {name: {"task_id": name, "input_token_ids": [index],
                           "generated_token_ids": [],
                           "final_generated_token_cached": True}
                    for index, name in enumerate(("a", "b", "c"), start=1)}
        events = []
        for name in ("a", "b"):
            events.extend(({"event": "llm_ready", "task_id": name},
                           {"event": "inference_start", "task_id": name,
                            "request_id": name}))
        events.append({"event": "task_end", "task_id": "a"})
        events.extend(({"event": "llm_ready", "task_id": "c"},
                       {"event": "inference_start", "task_id": "c", "request_id": "c"}))
        trace = {"events": events, "requests": requests}
        no_close = simulate(trace, policy="lru", kv_slots=2, release_on_done=False)
        close = simulate(trace, policy="lru", kv_slots=2, release_on_done=True)
        self.assertEqual(no_close["capacity_evictions"], 1)
        self.assertEqual(close["capacity_evictions"], 0)


if __name__ == "__main__":
    unittest.main()
