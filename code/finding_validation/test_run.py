from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "run.py"
SPEC = importlib.util.spec_from_file_location("pace_select_run", MODULE_PATH)
run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run)


def row(correct: bool, latency_ms: float, energy_j: float, tps: float = 10.0, jpt: float = 1.0):
    return {
        "score_correct": correct,
        "e2e_latency_ms": latency_ms,
        "energy_j": energy_j,
        "decode_token_per_sec": tps,
        "energy_per_token_j": jpt,
    }


class MetricTests(unittest.TestCase):
    def test_four_row_hand_check(self):
        rows = [
            row(True, 5_000, 2.0),
            row(True, 10_000, 2.0),
            row(True, 11_000, 2.0),
            row(False, 4_000, 2.0),
        ]
        values = run.metrics(rows, 10)
        self.assertEqual(values["final_accuracy"], 0.75)
        self.assertEqual(values["timely_success_rate"], 0.50)
        self.assertEqual(values["mean_request_energy_j"], 2.0)
        self.assertEqual(values["energy_per_timely_correct_j"], 4.0)

    def test_zero_timely_correct_has_na_epc(self):
        values = run.metrics([row(False, 1_000, 2.0)], 10)
        self.assertEqual(values["timely_success_rate"], 0.0)
        self.assertIsNone(values["energy_per_timely_correct_j"])


class SelectionTests(unittest.TestCase):
    def test_pace_filters_then_minimizes_energy(self):
        candidates = {
            "cheap_failure": {"timely_success_rate": 0.5, "mean_request_energy_j": 1.0},
            "expensive_success": {"timely_success_rate": 0.75, "mean_request_energy_j": 3.0},
            "cheap_success": {"timely_success_rate": 0.75, "mean_request_energy_j": 2.0},
        }
        self.assertEqual(run.select_pace(candidates, 0.6), "cheap_success")

    def test_no_feasible_candidate(self):
        candidates = {"candidate": {"timely_success_rate": 0.5, "mean_request_energy_j": 1.0}}
        self.assertIsNone(run.select_pace(candidates, 0.6))

    def test_candidate_id_breaks_exact_tie(self):
        candidates = {
            "b": {"timely_success_rate": 0.75, "mean_request_energy_j": 2.0},
            "a": {"timely_success_rate": 0.75, "mean_request_energy_j": 2.0},
        }
        self.assertEqual(run.select_pace(candidates, 0.6), "a")

    def test_evaluation_mutation_does_not_change_frozen_choice(self):
        calibration = {
            "a": [row(True, 5_000, 2.0), row(True, 5_000, 2.0)],
            "b": [row(True, 5_000, 3.0), row(True, 5_000, 3.0)],
        }
        evaluation = copy.deepcopy(calibration)
        cal_metrics = {cid: run.metrics(rows, 10) for cid, rows in calibration.items()}
        frozen = run.select_pace(cal_metrics, 0.6)
        evaluation["a"][0].update(score_correct=False, e2e_latency_ms=999_000, energy_j=999.0)
        self.assertEqual(run.select_pace(cal_metrics, 0.6), frozen)
        self.assertEqual(frozen, "a")

    def test_fixed_strategy_uses_its_only_candidate(self):
        candidates = {
            "fixed": run.metrics([row(False, 20_000, 2.0)], 10),
            "other": run.metrics([row(True, 1_000, 1.0)], 10),
        }
        strategy = {"name": "fixed", "kind": "fixed", "candidate_ids": ["fixed"]}
        self.assertEqual(run.choose_strategy(strategy, candidates, 0.6), "fixed")


if __name__ == "__main__":
    unittest.main()
