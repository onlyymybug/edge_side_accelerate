#!/usr/bin/env python3
"""Integrity checks for the generated Section 9 replay."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "result/finding_validation/optional_calibration"


def rows(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


audits = rows("pairing_audit.csv")
assert len(audits) == 6
assert all(int(r["common_n"]) == 1319 for r in audits)
assert all(int(r["completion_hash_match"]) == 1319 for r in audits)
assert all(int(r["original_token_id_hash_available_n"]) == 0 for r in audits)
assert all(r["sharing_status"] == "passed" for r in audits)
assert all(r["phone_needed_under_adopted_definition"] == "False" for r in audits)

budget = rows("request_budget_audit.csv")
assert len(budget) == 20 * 4 * 4
assert all(r["budget_matches"] == "True" for r in budget)
assert all(float(r["calibration_inference_seconds"]) > 0 for r in budget)
assert all(float(r["calibration_recorded_energy_j"]) > 0 for r in budget)
assert all(r["load_switch_cooling_overhead"] == "not_recorded_excluded" for r in budget)

decisions = rows("decisions.csv")
assert len(decisions) == 20 * 4 * 4 * 3 * 2
assert {r["method"] for r in decisions} == {
    "direct_lookup", "shared_accuracy", "paired_mapping", "workload_model"
}

splits = rows("split_ids.csv")
for seed in range(20):
    selected = [r for r in splits if int(r["seed"]) == seed]
    cal = {r["sample_id"] for r in selected if r["split"] == "calibration_pool"}
    val = {r["sample_id"] for r in selected if r["split"] == "evaluation"}
    assert len(cal) == 512 and len(val) == 806 and cal.isdisjoint(val)

phone_plan = rows("phone_request_plan.csv")
assert len(phone_plan) == 144
assert {r["status"] for r in phone_plan} == {"not_required_under_text_identity_definition"}
assert len({r["sample_id"] for r in phone_plan}) == 16
assert {r["runtime"] for r in phone_plan} == {"QNN", "llm.npu", "T-MAC"}

curve = rows("efficiency_curve.csv")
assert len(curve) == 4 * 4
assert all(int(r["evaluation_decisions"]) == 120 for r in curve)
assert all(float(r["mean_calibration_inference_seconds"]) > 0 for r in curve)
assert all(float(r["mean_calibration_recorded_energy_j"]) > 0 for r in curve)

print("optional calibration checks passed")
