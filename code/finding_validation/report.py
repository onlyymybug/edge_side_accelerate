#!/usr/bin/env python3
"""Generate the Day 3 evidence table and RESULTS.md from CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here / "config.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    return float(value)


def evidence_row(
    rows: list[dict[str, str]], experiment: str, deadline: int, target: float, baseline: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["experiment_id"] == experiment
        and as_float(row["deadline_s"]) == deadline
        and as_float(row["target_success"]) == target
        and row["baseline_strategy"] == baseline
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence row, found {len(matches)} for {(experiment, deadline, target, baseline)}")
    return matches[0]


def pct(row: dict[str, str]) -> str:
    value = row["paired_mean_energy_saving_percent"]
    return "NA" if not value else f"{as_float(value):.1f}%"


def build_evidence_summary(evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    f1_llm = evidence_row(evidence, "f1_llm_npu_gsm8k_b512", 20, 0.60, "fastest_mode")
    f1_tmac = evidence_row(evidence, "f1_t_mac_gsm8k_b512", 30, 0.60, "fastest_mode")
    f2_tmac = evidence_row(evidence, "f2_t_mac_gsm8k_threads_6_budgets", 20, 0.60, "fixed_budget_512")
    f2_qnn_math = evidence_row(evidence, "f2_qnn_math500_balanced_budgets", 60, 0.40, "fixed_budget_512")
    f2_qnn_max = evidence_row(evidence, "f2_qnn_math500_balanced_budgets", 60, 0.40, "max_measured_budget")
    f3_gsm = evidence_row(evidence, "f3_gsm8k_b512_reference_deployments", 10, 0.60, "lowest_j_per_token")
    f3_math = evidence_row(evidence, "f3_math500_b512_reference_deployments", 60, 0.40, "lowest_mean_request_energy")
    return [
        {
            "finding": "F1",
            "design_action": "Select mode by deadline, then minimize request energy among feasible modes.",
            "comparison": "Calibration-fastest mode.",
            "evidence": (
                f"llm.npu at D=20 s/q=60%: both met 20/20 and PACE saved {pct(f1_llm)} paired energy. "
                f"T-MAC at D=30 s/q=60%: both met 20/20 and PACE saved {pct(f1_tmac)}."
            ),
            "limits": "QNN showed no mode-selection energy gain; tight/high-target cells often had no feasible candidate.",
        },
        {
            "finding": "F2",
            "design_action": "Select the real output budget using the task target.",
            "comparison": "Fixed 512 and maximum measured budget.",
            "evidence": (
                f"T-MAC/GSM8K at D=20 s/q=60%: adaptive budget met {f2_tmac['adaptive_target_met_count']}/20 vs "
                f"{f2_tmac['baseline_target_met_count']}/20 for fixed 512. QNN/MATH at D=60 s/q=40%: adaptive met "
                f"{f2_qnn_math['adaptive_target_met_count']}/20 vs {f2_qnn_math['baseline_target_met_count']}/20 for fixed 512; "
                f"maximum budget met {f2_qnn_max['baseline_target_met_count']}/20."
            ),
            "limits": "Near-boundary calibration sometimes chose an insufficient budget; adaptive selection did not dominate fixed policies in every cell.",
        },
        {
            "finding": "F3",
            "design_action": "Constrain deployment choice by timely correctness before comparing request energy.",
            "comparison": "Decode tokens/s, J/token, and request energy alone.",
            "evidence": (
                f"GSM8K at D=10 s/q=60%: PACE met {f3_gsm['adaptive_target_met_count']}/20 vs "
                f"{f3_gsm['baseline_target_met_count']}/20 for lowest J/token. MATH at D=60 s/q=40%: PACE met "
                f"{f3_math['adaptive_target_met_count']}/20 vs {f3_math['baseline_target_met_count']}/20 for energy-only selection."
            ),
            "limits": "PACE still had no-selection and evaluation-violation cases because calibration feasibility is empirical, not guaranteed.",
        },
    ]


def summary_row(
    rows: list[dict[str, str]], experiment: str, deadline: int, target: float, strategy: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["experiment_id"] == experiment
        and as_float(row["deadline_s"]) == deadline
        and as_float(row["target_success"]) == target
        and row["strategy"] == strategy
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for {(experiment, deadline, target, strategy)}")
    return matches[0]


def write_results(
    config: dict[str, Any],
    manifest: list[dict[str, str]],
    coverage: list[dict[str, str]],
    summary: list[dict[str, str]],
    evidence_summary: list[dict[str, str]],
    output: Path,
) -> None:
    total_records = sum(int(row["row_count"]) for row in manifest)
    fully_scored = sum(row["fully_scoring_audited"] == "True" for row in manifest)
    fully_costed = sum(row["fully_cost_complete"] == "True" for row in manifest)
    negative_energy = sum(int(row["negative_energy_count"]) for row in manifest)
    joint_coverage = {
        row["task"]: int(row["common_complete_rows"])
        for row in coverage
        if row["finding"] == "JOINT"
    }
    gsm_joint = summary_row(summary, "joint_gsm8k_all_candidates", 30, 0.75, "full_joint_select")
    gsm_fixed = summary_row(summary, "joint_gsm8k_all_candidates", 30, 0.75, "fixed_budget_512")
    gsm_energy = summary_row(summary, "joint_gsm8k_all_candidates", 30, 0.75, "lowest_mean_request_energy")
    math_joint = summary_row(summary, "joint_math500_all_candidates", 60, 0.40, "full_joint_select")
    math_fixed = summary_row(summary, "joint_math500_all_candidates", 60, 0.40, "fixed_budget_512")
    math_energy = summary_row(summary, "joint_math500_all_candidates", 60, 0.40, "lowest_mean_request_energy")

    lines = [
        "# PACE-Select retrospective design validation",
        "",
        "## Scope",
        "",
        "This is an offline retrospective validation using previously collected OnePlus per-sample measurements. "
        "No phone inference was rerun for this analysis. The selector chooses one configuration per task batch; it does not route individual samples.",
        "",
        f"The inventory contains {len(manifest)} measured configurations and {total_records:,} records. All {fully_scored} configurations use "
        f"`{config['target_scorer_version']}` after offline rescoring. {fully_costed}/{len(manifest)} configurations have complete costs for every row; "
        f"the experiments use common complete IDs where needed. The joint subsets contain {joint_coverage['gsm8k']} GSM8K and "
        f"{joint_coverage['math500']} MATH-500 samples. {negative_energy} signed battery-energy observations are negative and were retained.",
        "",
        "## Protocol",
        "",
        "For each fixed candidate set, IDs are sorted and shuffled with NumPy `default_rng(seed)` for seeds 0–19. "
        "The first floor(20%) form calibration and the remainder evaluation. A candidate is feasible when calibration timely correctness "
        "is at least q; PACE-Select minimizes mean complete-request energy among feasible candidates. Evaluation fields never affect the frozen decision.",
        "",
        "## Finding evidence",
        "",
        "| Finding | Result | Boundary |",
        "|---|---|---|",
    ]
    for row in evidence_summary:
        lines.append(f"| {row['finding']} | {row['evidence']} | {row['limits']} |")
    lines.extend(
        [
            "",
            "## Joint selection",
            "",
            f"For GSM8K at D=30 s/q=75%, full joint selection met the evaluation target in {gsm_joint['evaluation_target_met_count']}/20 splits, "
            f"versus {gsm_fixed['evaluation_target_met_count']}/20 for fixed budget 512 and {gsm_energy['evaluation_target_met_count']}/20 for energy-only selection.",
            "",
            f"For MATH-500 at D=60 s/q=40%, full joint selection met {math_joint['evaluation_target_met_count']}/20, "
            f"fixed budget 512 met {math_fixed['evaluation_target_met_count']}/20, and energy-only selection met {math_energy['evaluation_target_met_count']}/20. "
            "Full joint selection tied the fixed-reference-mode design in these highlighted cells, so the added mode freedom did not provide an observed increment there.",
            "",
            "All 12 target cells, including no-selection and negative results, are retained in `summary.csv`, `design_evidence.csv`, and `joint_comparisons.csv`.",
            "",
            "## Interpretation and limits",
            "",
            "The results support using deadline-aware modes in some runtime/target conditions, using task-dependent budgets, and avoiding deployment ranking by token proxies alone. "
            "They do not show universal improvement: simple fixed policies tie or outperform PACE-Select in several cells, especially near feasibility boundaries.",
            "",
            "The 20 splits measure sensitivity to calibration composition; they are not independent phone repetitions. Energy is signed net battery energy, not external-meter total device energy. "
            "Historical run order, temperature drift, and missing measurements limit causal and deployment-wide claims. New-task generalization or physical deployment savings require frozen-rule validation on new phone runs.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m pip install -r code/finding_validation/requirements.txt",
            "python code/finding_validation/run.py",
            "python code/finding_validation/report.py",
            "python code/finding_validation/test_run.py",
            "```",
            "",
            "The scripts read `results/` and regenerate all derived artifacts in `result/finding_validation/`.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    output = Path(config["output_root"])
    manifest = read_csv(output / "manifest.csv")
    coverage = read_csv(output / "coverage.csv")
    summary = read_csv(output / "summary.csv")
    evidence = read_csv(output / "design_evidence.csv")
    evidence_summary = build_evidence_summary(evidence)
    write_csv(output / "design_evidence_summary.csv", evidence_summary)
    write_results(config, manifest, coverage, summary, evidence_summary, output)
    print(f"Wrote {output / 'design_evidence_summary.csv'}")
    print(f"Wrote {output / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
