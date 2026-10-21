from pathlib import Path
import tempfile
import unittest

from appendix_common import DEFAULT_CONFIG, read_config
from prepare import budget_plan, graph_plan, threads_plan


class PlansTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_config(DEFAULT_CONFIG)

    def test_graph_count_and_balance(self):
        rows = graph_plan(self.cfg)
        self.assertEqual(len(rows), 90)
        for runtime in ("qnn", "llm.npu"):
            subset = [row for row in rows if row["runtime"] == runtime]
            self.assertEqual(len(subset), 45)
            self.assertEqual({row["repeat"] for row in subset}, set(range(1, 6)))

    def test_thread_count_and_settings(self):
        rows = threads_plan(self.cfg)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({row["sample_id"] for row in rows}), 12)
        self.assertEqual({(row["batch_threads"], row["decode_threads"]) for row in rows}, {(6, 6), (8, 6), (6, 8), (8, 8)})

    def test_budget_count_and_categories(self):
        rows = budget_plan(self.cfg)
        self.assertEqual(len(rows), 288)
        self.assertEqual({row["historical_category"] for row in rows}, {"eos_unchanged", "truncated_improved", "truncated_still_wrong"})
        for task in ("GSM8K", "MATH500"):
            self.assertEqual(len({row["sample_id"] for row in rows if row["task"] == task}), 24)


if __name__ == "__main__":
    unittest.main()
