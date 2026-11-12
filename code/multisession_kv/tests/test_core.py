from __future__ import annotations

import unittest

from build_trace import lcp, parse_snapshot_id, stable_fixed_tools
from replay import schedule_rows
from run_matrix import build_jobs, expand_policy
from analyze_oracle import simulate_close_aware_lru, simulate_policy, token_optimal_reuse
from analyze_lifecycle import simulate_lifecycle_policy, validate_online_source


class TraceCoreTest(unittest.TestCase):
    def test_snapshot_id_uses_right_split(self) -> None:
        self.assertEqual(
            parse_snapshot_id("SO-case1_123_4"), ("SO-case1", "123", 4)
        )

    def test_lcp(self) -> None:
        self.assertEqual(lcp([1, 2, 3], [1, 2, 9]), 2)

    def test_fixed_tools_are_sorted_and_deduplicated(self) -> None:
        a = {"name": "a", "parameters": {}}
        b = {"name": "b", "parameters": {}}
        self.assertEqual(stable_fixed_tools([{"tools": [b, a]}, {"tools": [a]}]), [a, b])

    def test_round_robin_skips_finished_sessions(self) -> None:
        grouped = {"A": [{"id": "A1"}, {"id": "A2"}], "B": [{"id": "B1"}]}
        self.assertEqual(
            [row["id"] for row in schedule_rows(grouped, "round_robin")],
            ["A1", "B1", "A2"],
        )

    def test_lru_ratios_that_map_to_same_slots_are_collapsed(self) -> None:
        self.assertEqual(
            expand_policy({"name": "lru", "budget_ratios": [0.25, 0.5, 0.75]}, 2),
            [
                {
                    "policy": "lru",
                    "lru_slots": 1,
                    "nominal_budget_ratios": [0.25, 0.5],
                    "actual_slot_ratio": 0.5,
                },
                {
                    "policy": "lru",
                    "lru_slots": 2,
                    "nominal_budget_ratios": [0.75],
                    "actual_slot_ratio": 1.0,
                },
            ],
        )

    def test_lifecycle_policy_expands_to_physical_slots(self) -> None:
        self.assertEqual(
            expand_policy(
                {"name": "lifecycle_lru", "budget_ratios": [0.25, 0.5]}, 8
            ),
            [
                {
                    "policy": "lifecycle_lru",
                    "lru_slots": 2,
                    "nominal_budget_ratios": [0.25],
                    "actual_slot_ratio": 0.25,
                },
                {
                    "policy": "lifecycle_lru",
                    "lru_slots": 4,
                    "nominal_budget_ratios": [0.5],
                    "actual_slot_ratio": 0.5,
                },
            ],
        )

    def test_matrix_order_is_reproducible(self) -> None:
        spec = {
            "seed": 42,
            "session_counts": [2],
            "max_groups_per_session_count": 1,
            "repeats": 1,
            "schedules": ["grouped", "round_robin"],
            "policies": [{"name": "all_keep"}, {"name": "no_reuse"}],
        }
        first = build_jobs(spec, ["a", "b", "c"])
        second = build_jobs(spec, ["a", "b", "c"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_matrix_experiment_blocks_limit_cross_products(self) -> None:
        spec = {
            "seed": 42,
            "repeats": 1,
            "max_groups_per_session_count": None,
            "experiments": [
                {
                    "name": "grouped_controls",
                    "session_counts": [2],
                    "max_groups_per_session_count": 1,
                    "schedules": ["grouped"],
                    "policies": [{"name": "all_keep"}, {"name": "no_reuse"}],
                },
                {
                    "name": "round_robin_pressure",
                    "session_counts": [2],
                    "max_groups_per_session_count": 1,
                    "schedules": ["round_robin"],
                    "policies": [
                        {"name": "all_keep"},
                        {"name": "lru", "budget_ratios": [0.25, 0.5, 0.75]},
                    ],
                },
            ],
        }
        jobs = build_jobs(spec, ["a", "b", "c", "d"])
        self.assertEqual(len(jobs), 5)
        self.assertEqual(
            {job["experiment"] for job in jobs},
            {"grouped_controls", "round_robin_pressure"},
        )

    def test_oracle_dominates_online_lru(self) -> None:
        sessions = ["A", "B", "C", "A", "B", "C"]
        potential = [0, 0, 0, 10, 20, 30]
        lru = simulate_policy(sessions, potential, 2, "lru")
        belady = simulate_policy(sessions, potential, 2, "belady")
        optimal = token_optimal_reuse(sessions, potential, 2)
        self.assertEqual(lru, 0)
        self.assertGreaterEqual(belady, lru)
        self.assertGreaterEqual(optimal, belady)

    def test_one_slot_strict_alternation_has_no_oracle_gap(self) -> None:
        sessions = ["A", "B", "A", "B"]
        potential = [0, 0, 10, 20]
        self.assertEqual(simulate_policy(sessions, potential, 1, "lru"), 0)
        self.assertEqual(simulate_policy(sessions, potential, 1, "belady"), 0)
        self.assertEqual(token_optimal_reuse(sessions, potential, 1), 0)

    def test_terminal_signal_releases_capacity(self) -> None:
        sessions = ["A", "B", "C", "A", "B"]
        potential = [0, 0, 0, 10, 20]
        self.assertGreater(
            simulate_close_aware_lru(sessions, potential, 2),
            simulate_policy(sessions, potential, 2, "lru"),
        )

    def test_online_lifecycle_rejects_future_oracle_source(self) -> None:
        with self.assertRaises(ValueError):
            validate_online_source("future_oracle")

    def test_lifecycle_lru_matches_lru_without_lifecycle_rules(self) -> None:
        sessions = ["A", "B", "C", "A", "B", "C"]
        potential = [0, 0, 0, 10, 20, 30]
        inputs = [10, 20, 30, 40, 50, 60]
        result = simulate_lifecycle_policy(
            sessions,
            potential,
            inputs,
            2,
            "lru",
            "serial_arrival",
        )
        self.assertEqual(result["reused_tokens"], simulate_policy(sessions, potential, 2, "lru"))

    def test_queued_ready_signal_is_separate_from_serial_arrival(self) -> None:
        sessions = ["A", "B", "C", "A", "B", "C"]
        potential = [0, 0, 0, 10, 20, 30]
        inputs = [10, 20, 30, 40, 50, 60]
        serial = simulate_lifecycle_policy(
            sessions,
            potential,
            inputs,
            2,
            "ready_aware_lru",
            "serial_arrival",
        )
        queued = simulate_lifecycle_policy(
            sessions,
            potential,
            inputs,
            2,
            "ready_aware_lru",
            "queued_next_turn",
        )
        self.assertGreater(queued["reused_tokens"], serial["reused_tokens"])
        self.assertGreater(queued["evicted_ready"], 0)


if __name__ == "__main__":
    unittest.main()
