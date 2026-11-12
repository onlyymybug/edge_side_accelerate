import queue
import threading
import time
import unittest

from collections import OrderedDict

from appworld_multisession_smoke import choose_lru_victim, run_dispatcher


class FakeBackend:
    def __init__(self):
        self.commands = []

    def request(self, payload):
        self.commands.append(payload["command"])
        if payload["command"] == "get_capabilities":
            return {"status": "ok", "server_pid": 7, "model_load_count": 1}
        if payload["command"] == "run_request":
            time.sleep(0.03)
            return {"status": "ok", "server_pid": 7, "model_load_count": 1,
                    "generated_text": "ok"}
        return {"status": "ok"}


class DispatcherTests(unittest.TestCase):
    def test_ready_lru_protects_only_observed_ready_sessions(self):
        resident = OrderedDict((task_id, None) for task_id in ("a", "b"))
        self.assertEqual(choose_lru_victim(resident, {"a"}, "lru"), "a")
        self.assertEqual(choose_lru_victim(resident, {"a"}, "ready_lru"), "b")
        self.assertEqual(choose_lru_victim(resident, {"a", "b"}, "ready_lru"), "a")

    def test_no_release_on_done_keeps_kv_until_capacity_eviction(self):
        outbox = queue.Queue()
        inbox = queue.Queue()

        def worker():
            outbox.put({"type": "ready", "task_id": "a", "step": 1,
                        "ready_ns": time.monotonic_ns(),
                        "token_ids": [1], "prompt_prepare_ms": 0.0})
            inbox.get(timeout=5)
            outbox.put({"type": "done", "task_id": "a",
                        "summary": {"task_id": "a", "success": True,
                                    "steps": 1, "error": None}})

        thread = threading.Thread(target=worker)
        thread.start()
        backend = FakeBackend()
        events, requests, summaries = run_dispatcher(
            backend, ["a"], {"a": thread}, {"a": inbox}, outbox,
            max_new_tokens=8, worker_timeout=5, kv_slots=1,
            release_on_done=False)
        thread.join(timeout=5)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(backend.commands.count("evict_session"), 0)
        self.assertFalse(any(row["event"] == "session_release" for row in events))

    def test_two_workers_share_one_serialized_backend(self):
        outbox = queue.Queue()
        inboxes = {name: queue.Queue() for name in ("a", "b")}
        barrier = threading.Barrier(2)

        def worker(task_id):
            barrier.wait()
            for step in (1, 2):
                outbox.put({"type": "ready", "task_id": task_id, "step": step,
                            "ready_ns": time.monotonic_ns(),
                            "token_ids": [1, 2, 3], "prompt_prepare_ms": 0.1})
                self.assertEqual(inboxes[task_id].get(timeout=5)["status"], "ok")
                time.sleep(0.01)
            outbox.put({"type": "done", "task_id": task_id,
                        "summary": {"task_id": task_id, "success": True,
                                    "steps": 2, "error": None}})

        workers = {task_id: threading.Thread(target=worker, args=(task_id,))
                   for task_id in inboxes}
        for item in workers.values():
            item.start()
        backend = FakeBackend()
        events, requests, summaries = run_dispatcher(
            backend, list(inboxes), workers, inboxes, outbox,
            max_new_tokens=8, worker_timeout=5, kv_slots=2)
        for item in workers.values():
            item.join(timeout=5)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(backend.commands.count("create_session"), 2)
        self.assertEqual(backend.commands.count("evict_session"), 2)
        self.assertTrue(any(row["queue_ms"] > 1 for row in requests))
        self.assertEqual(sum(row["event"] == "task_end" for row in events), 2)

    def test_three_tasks_two_kv_slots_evict_without_recreating_session(self):
        outbox = queue.Queue()
        names = ("a", "b", "c")
        inboxes = {name: queue.Queue() for name in names}
        barrier = threading.Barrier(len(names))

        def worker(task_id):
            barrier.wait()
            for step in (1, 2):
                outbox.put({"type": "ready", "task_id": task_id, "step": step,
                            "ready_ns": time.monotonic_ns(),
                            "token_ids": [1, 2, 3], "prompt_prepare_ms": 0.1})
                self.assertEqual(inboxes[task_id].get(timeout=5)["status"], "ok")
                time.sleep(0.01)
            outbox.put({"type": "done", "task_id": task_id,
                        "summary": {"task_id": task_id, "success": True,
                                    "steps": 2, "error": None}})

        workers = {name: threading.Thread(target=worker, args=(name,)) for name in names}
        for item in workers.values():
            item.start()
        backend = FakeBackend()
        events, requests, summaries = run_dispatcher(
            backend, list(names), workers, inboxes, outbox,
            max_new_tokens=8, worker_timeout=5, kv_slots=2)
        for item in workers.values():
            item.join(timeout=5)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(len(requests), 6)
        self.assertEqual(backend.commands.count("create_session"), 3)
        self.assertTrue(any(row["event"] == "session_evict" and
                            row["reason"] == "lru_capacity" for row in events))
        self.assertFalse(all(row["policy_kv_hit"] for row in requests))

if __name__ == "__main__":
    unittest.main()
