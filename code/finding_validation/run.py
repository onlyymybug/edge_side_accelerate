#!/usr/bin/env python3
"""Build the Day 1 PACE-Select inventory and retrospective validation outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NORMALIZED_FIELDS = [
    "sample_id",
    "device",
    "task",
    "runtime",
    "mode",
    "output_budget",
    "candidate_id",
    "score_correct",
    "scorer_version",
    "e2e_latency_ms",
    "energy_j",
    "generated_tokens",
    "decode_token_per_sec",
    "energy_per_token_j",
    "prompt_tokens",
    "returncode",
    "error",
    "energy_measurement_status",
    "external_powered_start",
    "external_powered_end",
    "source_file",
    "scoring_audited",
    "core_record_complete",
    "proxy_record_complete",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = str(path.resolve())
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_score(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    return None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def candidate_id(device: str, task: str, runtime: str, mode: str, budget: str) -> str:
    safe_runtime = runtime.lower().replace(".", "_").replace("-", "_")
    return f"{device}__{task}__{safe_runtime}__{mode}__b{budget}"


def preferred_summary(directory: Path) -> Path | None:
    rescored = directory / "summary.rescored_v2.csv"
    regular = directory / "summary.csv"
    if rescored.exists():
        return rescored
    return regular if regular.exists() else None


def discover_sources(config: dict[str, Any]) -> list[dict[str, str]]:
    data_root = Path(config["data_root"])
    device = config["device"]
    allowed = set(config["allowed_budgets"])
    found: list[dict[str, str]] = []
    for runtime, runtime_config in config["runtimes"].items():
        model = runtime_config["model_dir"]
        for task in config["tasks"]:
            task_root = data_root / runtime / device / task / model
            if not task_root.exists():
                continue
            for budget_dir in sorted(task_root.glob("*/output_budget_*")):
                if not budget_dir.is_dir():
                    continue
                budget = budget_dir.name.removeprefix("output_budget_")
                if budget not in allowed:
                    continue
                source = preferred_summary(budget_dir)
                if source is None:
                    continue
                mode = budget_dir.parent.name
                found.append(
                    {
                        "device": device,
                        "task": task,
                        "runtime": runtime,
                        "model_dir": model,
                        "mode": mode,
                        "output_budget": budget,
                        "candidate_id": candidate_id(device, task, runtime, mode, budget),
                        "source_file": str(source.resolve()),
                    }
                )
    return sorted(found, key=lambda item: item["candidate_id"])


def read_source(spec: dict[str, str], target_scorer: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(spec["source_file"])
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
        source_fields = set(reader.fieldnames or [])

    ids = [str(row.get("sample_id", "")) for row in raw_rows]
    duplicate_count = len(ids) - len(set(ids))
    scorer_counts = Counter((row.get("scorer_version") or "<missing>") for row in raw_rows)
    score_valid_count = sum(parse_score(row.get("score_correct")) is not None for row in raw_rows)
    time_valid_count = sum(finite_float(row.get("e2e_latency_ms")) is not None for row in raw_rows)
    energy_valid_count = sum(finite_float(row.get("energy_j")) is not None for row in raw_rows)
    decode_valid_count = sum(finite_float(row.get("decode_token_per_sec")) is not None for row in raw_rows)
    jpt_valid_count = sum(finite_float(row.get("energy_per_token_j")) is not None for row in raw_rows)
    negative_energy_count = sum(
        (energy := finite_float(row.get("energy_j"))) is not None and energy < 0
        for row in raw_rows
    )
    scorer_ready_count = sum(
        row.get("scorer_version") == target_scorer
        and parse_score(row.get("score_correct")) is not None
        for row in raw_rows
    )

    manifest = {
        **spec,
        "source_name": source.name,
        "sha256": sha256(source),
        "row_count": len(raw_rows),
        "unique_sample_ids": len(set(ids)),
        "duplicate_sample_ids": duplicate_count,
        "scorer_versions": json.dumps(dict(sorted(scorer_counts.items())), ensure_ascii=False),
        "score_valid_count": score_valid_count,
        "time_valid_count": time_valid_count,
        "energy_valid_count": energy_valid_count,
        "decode_tps_valid_count": decode_valid_count,
        "j_per_token_valid_count": jpt_valid_count,
        "negative_energy_count": negative_energy_count,
        "scorer_ready_count": scorer_ready_count,
        "fully_scoring_audited": scorer_ready_count == len(raw_rows),
        "fully_cost_complete": time_valid_count == len(raw_rows) and energy_valid_count == len(raw_rows),
        "preferred_rescored_file": source.name == "summary.rescored_v2.csv",
        "source_has_scorer_column": "scorer_version" in source_fields,
    }

    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        score = parse_score(raw.get("score_correct"))
        time_ms = finite_float(raw.get("e2e_latency_ms"))
        energy = finite_float(raw.get("energy_j"))
        decode_tps = finite_float(raw.get("decode_token_per_sec"))
        j_per_token = finite_float(raw.get("energy_per_token_j"))
        scorer = raw.get("scorer_version") or ""
        scoring_audited = scorer == target_scorer and score is not None
        core_complete = scoring_audited and time_ms is not None and energy is not None
        proxy_complete = core_complete and decode_tps is not None and j_per_token is not None
        normalized.append(
            {
                "sample_id": str(raw.get("sample_id", "")),
                "device": spec["device"],
                "task": spec["task"],
                "runtime": spec["runtime"],
                "mode": spec["mode"],
                "output_budget": spec["output_budget"],
                "candidate_id": spec["candidate_id"],
                "score_correct": score,
                "scorer_version": scorer,
                "e2e_latency_ms": time_ms,
                "energy_j": energy,
                "generated_tokens": finite_float(raw.get("generated_tokens")),
                "decode_token_per_sec": decode_tps,
                "energy_per_token_j": j_per_token,
                "prompt_tokens": finite_float(raw.get("prompt_tokens")),
                "returncode": raw.get("returncode", ""),
                "error": raw.get("error", ""),
                "energy_measurement_status": raw.get("energy_measurement_status", ""),
                "external_powered_start": raw.get("external_powered_start", ""),
                "external_powered_end": raw.get("external_powered_end", ""),
                "source_file": str(source.resolve()),
                "scoring_audited": scoring_audited,
                "core_record_complete": core_complete,
                "proxy_record_complete": proxy_complete,
            }
        )
    return manifest, normalized


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def metrics(rows: list[dict[str, Any]], deadline_s: float) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "final_accuracy": None,
            "timely_correct_count": 0,
            "timely_success_rate": None,
            "mean_request_energy_j": None,
            "energy_per_timely_correct_j": None,
            "mean_e2e_latency_s": None,
            "mean_decode_token_per_sec": None,
            "mean_energy_per_token_j": None,
        }
    timely_correct = sum(
        row["score_correct"] is True and float(row["e2e_latency_ms"]) <= deadline_s * 1000.0
        for row in rows
    )
    total_energy = sum(float(row["energy_j"]) for row in rows)
    return {
        "n": len(rows),
        "final_accuracy": sum(row["score_correct"] is True for row in rows) / len(rows),
        "timely_correct_count": timely_correct,
        "timely_success_rate": timely_correct / len(rows),
        "mean_request_energy_j": total_energy / len(rows),
        "energy_per_timely_correct_j": total_energy / timely_correct if timely_correct else None,
        "mean_e2e_latency_s": sum(float(row["e2e_latency_ms"]) for row in rows) / len(rows) / 1000.0,
        "mean_decode_token_per_sec": (
            sum(float(row["decode_token_per_sec"]) for row in rows) / len(rows)
            if all(row.get("decode_token_per_sec") is not None for row in rows)
            else None
        ),
        "mean_energy_per_token_j": (
            sum(float(row["energy_per_token_j"]) for row in rows) / len(rows)
            if all(row.get("energy_per_token_j") is not None for row in rows)
            else None
        ),
    }


def select_pace(candidate_metrics: dict[str, dict[str, Any]], target: float) -> str | None:
    feasible = [
        (values["mean_request_energy_j"], cid)
        for cid, values in candidate_metrics.items()
        if values["timely_success_rate"] is not None
        and values["timely_success_rate"] >= target
    ]
    return min(feasible)[1] if feasible else None


def select_fastest(candidate_metrics: dict[str, dict[str, Any]]) -> str:
    return min(
        (values["mean_e2e_latency_s"], values["mean_request_energy_j"], cid)
        for cid, values in candidate_metrics.items()
    )[2]


def select_proxy(candidate_metrics: dict[str, dict[str, Any]], strategy: str) -> str:
    if strategy == "highest_decode_tokens_per_s":
        return min((-values["mean_decode_token_per_sec"], cid) for cid, values in candidate_metrics.items())[1]
    if strategy == "lowest_j_per_token":
        return min((values["mean_energy_per_token_j"], cid) for cid, values in candidate_metrics.items())[1]
    if strategy == "lowest_mean_request_energy":
        return min((values["mean_request_energy_j"], cid) for cid, values in candidate_metrics.items())[1]
    raise ValueError(f"Unknown proxy strategy: {strategy}")


def common_ids(
    records_by_candidate: dict[str, dict[str, dict[str, Any]]],
    candidate_ids: list[str],
    *,
    require_proxies: bool,
) -> set[str]:
    flag = "proxy_record_complete" if require_proxies else "core_record_complete"
    sets = []
    for cid in candidate_ids:
        if cid not in records_by_candidate:
            raise KeyError(f"Missing candidate {cid}")
        sets.append({sid for sid, row in records_by_candidate[cid].items() if row[flag]})
    return set.intersection(*sets) if sets else set()


def runtime_slug(runtime: str) -> str:
    return runtime.lower().replace(".", "_").replace("-", "_")


def build_experiments(
    config: dict[str, Any], all_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    device = config["device"]
    experiments: list[dict[str, Any]] = []
    metadata = {}
    for row in all_records:
        metadata.setdefault(
            row["candidate_id"],
            {
                "task": row["task"],
                "runtime": row["runtime"],
                "mode": row["mode"],
                "output_budget": row["output_budget"],
            },
        )

    # F1: within-runtime mode selection at a fixed 512-token budget.
    for runtime, runtime_config in config["runtimes"].items():
        candidates = [
            candidate_id(device, "gsm8k", runtime, mode, "512")
            for mode in runtime_config["f1_modes"]
        ]
        experiments.append(
            {
                "experiment_id": f"f1_{runtime_slug(runtime)}_gsm8k_b512",
                "finding": "F1",
                "task": "gsm8k",
                "candidate_ids": candidates,
                "strategies": [
                    {"name": "pace_select", "kind": "pace", "candidate_ids": candidates},
                    {"name": "fastest_mode", "kind": "fastest", "candidate_ids": candidates},
                ],
                "require_proxies": False,
            }
        )

    # F2: within-runtime budget selection at each runtime's reference mode.
    for task in config["tasks"]:
        for runtime, runtime_config in config["runtimes"].items():
            reference_mode = runtime_config["reference_mode"]
            candidates = sorted(
                [
                    cid
                    for cid, item in metadata.items()
                    if item["task"] == task
                    and item["runtime"] == runtime
                    and item["mode"] == reference_mode
                ]
            )
            if len(candidates) < 2:
                continue
            budget_to_candidate = {metadata[cid]["output_budget"]: cid for cid in candidates}
            if "512" not in budget_to_candidate:
                continue
            largest_budget = "no_budget" if "no_budget" in budget_to_candidate else max(
                budget_to_candidate, key=lambda value: int(value)
            )
            experiments.append(
                {
                    "experiment_id": f"f2_{runtime_slug(runtime)}_{task}_{reference_mode}_budgets",
                    "finding": "F2",
                    "task": task,
                    "candidate_ids": candidates,
                    "strategies": [
                        {"name": "adaptive_budget", "kind": "pace", "candidate_ids": candidates},
                        {
                            "name": "fixed_budget_512",
                            "kind": "fixed",
                            "candidate_ids": [budget_to_candidate["512"]],
                        },
                        {
                            "name": "max_measured_budget",
                            "kind": "fixed",
                            "candidate_ids": [budget_to_candidate[largest_budget]],
                        },
                    ],
                    "require_proxies": False,
                }
            )

    # F3: deployment selection among the three fixed reference configurations.
    for task in config["tasks"]:
        reference_candidates = [
            candidate_id(device, task, runtime, runtime_config["reference_mode"], "512")
            for runtime, runtime_config in config["runtimes"].items()
        ]
        experiments.append(
            {
                "experiment_id": f"f3_{task}_b512_reference_deployments",
                "finding": "F3",
                "task": task,
                "candidate_ids": reference_candidates,
                "strategies": [
                    {"name": "pace_select", "kind": "pace", "candidate_ids": reference_candidates},
                    {
                        "name": "highest_decode_tokens_per_s",
                        "kind": "highest_decode_tokens_per_s",
                        "candidate_ids": reference_candidates,
                    },
                    {"name": "lowest_j_per_token", "kind": "lowest_j_per_token", "candidate_ids": reference_candidates},
                    {
                        "name": "lowest_mean_request_energy",
                        "kind": "lowest_mean_request_energy",
                        "candidate_ids": reference_candidates,
                    },
                ],
                "require_proxies": True,
            }
        )

    # Joint PACE-Select and the three prescribed design ablations.
    for task in config["tasks"]:
        candidates = sorted(cid for cid, item in metadata.items() if item["task"] == task)
        fixed_512 = [cid for cid in candidates if metadata[cid]["output_budget"] == "512"]
        fixed_reference = [
            cid
            for cid in candidates
            if metadata[cid]["mode"] == config["runtimes"][metadata[cid]["runtime"]]["reference_mode"]
        ]
        experiments.append(
            {
                "experiment_id": f"joint_{task}_all_candidates",
                "finding": "JOINT",
                "task": task,
                "candidate_ids": candidates,
                "strategies": [
                    {"name": "full_joint_select", "kind": "pace", "candidate_ids": candidates},
                    {"name": "fixed_budget_512", "kind": "pace", "candidate_ids": fixed_512},
                    {"name": "fixed_reference_mode", "kind": "pace", "candidate_ids": fixed_reference},
                    {
                        "name": "lowest_mean_request_energy",
                        "kind": "lowest_mean_request_energy",
                        "candidate_ids": candidates,
                    },
                ],
                "require_proxies": False,
            }
        )
    return experiments


def split_ids(sample_ids: set[str], seed: int, calibration_fraction: float) -> tuple[list[str], list[str]]:
    ordered = np.asarray(sorted(sample_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    calibration_n = math.floor(calibration_fraction * len(ordered))
    return ordered[:calibration_n].tolist(), ordered[calibration_n:].tolist()


def choose_strategy(
    strategy: dict[str, Any],
    calibration_metrics: dict[str, dict[str, Any]],
    target: float,
) -> str | None:
    eligible = {cid: calibration_metrics[cid] for cid in strategy["candidate_ids"]}
    kind = strategy["kind"]
    if kind == "pace":
        return select_pace(eligible, target)
    if kind == "fastest":
        return select_fastest(eligible)
    if kind == "fixed":
        if len(eligible) != 1:
            raise ValueError(f"Fixed strategy requires one candidate: {strategy}")
        return next(iter(eligible))
    return select_proxy(eligible, kind)


def run_experiments(
    config: dict[str, Any],
    all_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_records:
        cid = row["candidate_id"]
        sid = row["sample_id"]
        if sid in by_candidate[cid]:
            raise ValueError(f"Duplicate sample_id {sid} in {cid}")
        by_candidate[cid][sid] = row

    split_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for experiment in build_experiments(config, all_records):
        cids = experiment["candidate_ids"]
        grid = config["target_grid"][experiment["task"]]
        shared = common_ids(by_candidate, cids, require_proxies=experiment["require_proxies"])
        for cid in cids:
            flag = "proxy_record_complete" if experiment["require_proxies"] else "core_record_complete"
            valid = {sid for sid, row in by_candidate[cid].items() if row[flag]}
            coverage_rows.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "finding": experiment["finding"],
                    "task": experiment["task"],
                    "candidate_id": cid,
                    "available_rows": len(by_candidate[cid]),
                    "valid_rows": len(valid),
                    "common_complete_rows": len(shared),
                    "require_proxies": experiment["require_proxies"],
                }
            )
            for sid in sorted(set(by_candidate[cid]) - shared):
                row = by_candidate[cid][sid]
                reasons = []
                if not row["scoring_audited"]:
                    reasons.append("scoring_not_audited")
                if row["e2e_latency_ms"] is None:
                    reasons.append("missing_e2e_latency")
                if row["energy_j"] is None:
                    reasons.append("missing_energy")
                if experiment["require_proxies"] and row["decode_token_per_sec"] is None:
                    reasons.append("missing_decode_tokens_per_s")
                if experiment["require_proxies"] and row["energy_per_token_j"] is None:
                    reasons.append("missing_j_per_token")
                if not reasons:
                    reasons.append("not_common_to_all_candidates")
                exclusion_rows.append(
                    {
                                "experiment_id": experiment["experiment_id"],
                                "candidate_id": cid,
                        "sample_id": sid,
                        "reasons": ";".join(reasons),
                        "returncode": row["returncode"],
                        "energy_measurement_status": row["energy_measurement_status"],
                        "error": row["error"],
                    }
                )

        if not shared:
            raise ValueError(f"No common complete records for {experiment['experiment_id']}")
        for seed in config["seeds"]:
            calibration_ids, evaluation_ids = split_ids(shared, seed, config["calibration_fraction"])
            for subset, ids in (("calibration", calibration_ids), ("evaluation", evaluation_ids)):
                split_rows.extend(
                    {
                        "experiment_id": experiment["experiment_id"],
                        "seed": seed,
                        "subset": subset,
                        "sample_id": sid,
                    }
                    for sid in ids
                )

            for deadline_s in grid["deadlines_s"]:
                calibration_metrics = {
                    cid: metrics([by_candidate[cid][sid] for sid in calibration_ids], deadline_s)
                    for cid in cids
                }
                for target in grid["target_success"]:
                    for strategy in experiment["strategies"]:
                        selected = choose_strategy(strategy, calibration_metrics, target)
                        status = "selected" if selected is not None else "no_feasible_candidate"
                        selected_cal = calibration_metrics.get(selected) if selected else None
                        decision_rows.append(
                            {
                                "experiment_id": experiment["experiment_id"],
                                "finding": experiment["finding"],
                                "task": experiment["task"],
                                "seed": seed,
                                "deadline_s": deadline_s,
                                "target_success": target,
                                "strategy": strategy["name"],
                                "strategy_candidate_count": len(strategy["candidate_ids"]),
                                "selection_status": status,
                                "candidate_id": selected,
                                "calibration_n": len(calibration_ids),
                                "calibration_final_accuracy": selected_cal["final_accuracy"] if selected_cal else None,
                                "calibration_timely_success_rate": selected_cal["timely_success_rate"] if selected_cal else None,
                                "calibration_mean_request_energy_j": selected_cal["mean_request_energy_j"] if selected_cal else None,
                                "calibration_mean_e2e_latency_s": selected_cal["mean_e2e_latency_s"] if selected_cal else None,
                                "calibration_mean_decode_token_per_sec": selected_cal["mean_decode_token_per_sec"] if selected_cal else None,
                                "calibration_mean_energy_per_token_j": selected_cal["mean_energy_per_token_j"] if selected_cal else None,
                            }
                        )
                        evaluated = (
                            metrics([by_candidate[selected][sid] for sid in evaluation_ids], deadline_s)
                            if selected
                            else None
                        )
                        evaluation_rows.append(
                            {
                                "experiment_id": experiment["experiment_id"],
                                "finding": experiment["finding"],
                                "task": experiment["task"],
                                "seed": seed,
                                "deadline_s": deadline_s,
                                "target_success": target,
                                "strategy": strategy["name"],
                                "strategy_candidate_count": len(strategy["candidate_ids"]),
                                "selection_status": status,
                                "candidate_id": selected,
                                "evaluation_n": len(evaluation_ids),
                                "evaluation_final_accuracy": evaluated["final_accuracy"] if evaluated else None,
                                "evaluation_timely_correct_count": evaluated["timely_correct_count"] if evaluated else None,
                                "evaluation_timely_success_rate": evaluated["timely_success_rate"] if evaluated else None,
                                "evaluation_mean_request_energy_j": evaluated["mean_request_energy_j"] if evaluated else None,
                                "evaluation_energy_per_timely_correct_j": evaluated["energy_per_timely_correct_j"] if evaluated else None,
                                "evaluation_target_met": evaluated["timely_success_rate"] >= target if evaluated else None,
                            }
                        )
    return split_rows, decision_rows, evaluation_rows, coverage_rows, exclusion_rows


def mean_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def summarize(evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        key = (
            row["experiment_id"],
            row["finding"],
            row["task"],
            row["deadline_s"],
            row["target_success"],
            row["strategy"],
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        selected = [row for row in rows if row["selection_status"] == "selected"]
        selection_counts = Counter(row["candidate_id"] for row in selected)
        met_count = sum(row["evaluation_target_met"] is True for row in selected)
        output.append(
            {
                "experiment_id": key[0],
                "finding": key[1],
                "task": key[2],
                "deadline_s": key[3],
                "target_success": key[4],
                "strategy": key[5],
                "repetitions": len(rows),
                "selection_count": len(selected),
                "no_selection_count": len(rows) - len(selected),
                "evaluation_target_met_count": met_count,
                "selected_but_violation_count": len(selected) - met_count,
                "selected_candidate_counts": json.dumps(dict(sorted(selection_counts.items()))),
                "mean_evaluation_timely_success_rate": mean_or_none([row["evaluation_timely_success_rate"] for row in selected]),
                "mean_evaluation_request_energy_j": mean_or_none([row["evaluation_mean_request_energy_j"] for row in selected]),
                "mean_evaluation_epc_j": mean_or_none([row["evaluation_energy_per_timely_correct_j"] for row in selected]),
            }
        )
    return output


def evidence_comparisons(evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create paired, per-condition comparisons without treating failure energy as a win."""
    comparison_strategies = {
        "F1": ("pace_select", ["fastest_mode"]),
        "F2": ("adaptive_budget", ["fixed_budget_512", "max_measured_budget"]),
        "F3": (
            "pace_select",
            ["highest_decode_tokens_per_s", "lowest_j_per_token", "lowest_mean_request_energy"],
        ),
        "JOINT": (
            "full_joint_select",
            ["fixed_budget_512", "fixed_reference_mode", "lowest_mean_request_energy"],
        ),
    }
    indexed = {
        (
            row["experiment_id"], row["seed"], row["deadline_s"],
            row["target_success"], row["strategy"],
        ): row
        for row in evaluation_rows
    }
    conditions = sorted(
        {
            (row["experiment_id"], row["finding"], row["task"], row["deadline_s"], row["target_success"])
            for row in evaluation_rows
        }
    )
    output = []
    for experiment_id, finding, task, deadline_s, target in conditions:
        adaptive_name, baselines = comparison_strategies[finding]
        seeds = sorted(
            {
                row["seed"]
                for row in evaluation_rows
                if row["experiment_id"] == experiment_id
                and row["deadline_s"] == deadline_s
                and row["target_success"] == target
            }
        )
        for baseline_name in baselines:
            pairs = [
                (
                    indexed[(experiment_id, seed, deadline_s, target, adaptive_name)],
                    indexed[(experiment_id, seed, deadline_s, target, baseline_name)],
                )
                for seed in seeds
            ]
            adaptive_selected = sum(a["selection_status"] == "selected" for a, _ in pairs)
            baseline_selected = sum(b["selection_status"] == "selected" for _, b in pairs)
            adaptive_candidates = Counter(a["candidate_id"] for a, _ in pairs if a["candidate_id"])
            baseline_candidates = Counter(b["candidate_id"] for _, b in pairs if b["candidate_id"])
            adaptive_met = sum(a["evaluation_target_met"] is True for a, _ in pairs)
            baseline_met = sum(b["evaluation_target_met"] is True for _, b in pairs)
            both_met = [(a, b) for a, b in pairs if a["evaluation_target_met"] is True and b["evaluation_target_met"] is True]
            differences = [
                float(b["evaluation_mean_request_energy_j"]) - float(a["evaluation_mean_request_energy_j"])
                for a, b in both_met
            ]
            percentages = [
                100.0
                * (float(b["evaluation_mean_request_energy_j"]) - float(a["evaluation_mean_request_energy_j"]))
                / float(b["evaluation_mean_request_energy_j"])
                for a, b in both_met
                if float(b["evaluation_mean_request_energy_j"]) != 0.0
            ]
            output.append(
                {
                    "finding": finding,
                    "experiment_id": experiment_id,
                    "task": task,
                    "deadline_s": deadline_s,
                    "target_success": target,
                    "adaptive_strategy": adaptive_name,
                    "baseline_strategy": baseline_name,
                    "repetitions": len(pairs),
                    "adaptive_selection_count": adaptive_selected,
                    "adaptive_no_selection_count": len(pairs) - adaptive_selected,
                    "baseline_selection_count": baseline_selected,
                    "adaptive_candidate_counts": json.dumps(dict(sorted(adaptive_candidates.items()))),
                    "baseline_candidate_counts": json.dumps(dict(sorted(baseline_candidates.items()))),
                    "adaptive_target_met_count": adaptive_met,
                    "baseline_target_met_count": baseline_met,
                    "adaptive_only_met_count": sum(
                        a["evaluation_target_met"] is True and b["evaluation_target_met"] is not True
                        for a, b in pairs
                    ),
                    "baseline_only_met_count": sum(
                        b["evaluation_target_met"] is True and a["evaluation_target_met"] is not True
                        for a, b in pairs
                    ),
                    "both_met_count": len(both_met),
                    "paired_mean_energy_saving_j": mean_or_none(differences),
                    "paired_mean_energy_saving_percent": mean_or_none(percentages),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    sources = discover_sources(config)
    if not sources:
        raise SystemExit(f"No scoped summaries found below {config['data_root']}")

    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for spec in sources:
        manifest, normalized = read_source(spec, config["target_scorer_version"])
        manifests.append(manifest)
        records.extend(normalized)

    manifest_fields = list(manifests[0])
    write_csv(output_root / "manifest.csv", manifests, manifest_fields)
    write_csv(output_root / "records.csv", records, NORMALIZED_FIELDS)
    with (output_root / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if args.inventory_only:
        print(f"Inventory: {len(manifests)} candidates, {len(records)} records")
        print(f"Wrote {output_root}")
        return

    split_rows, decisions, evaluations, coverage, exclusions = run_experiments(config, records)
    write_csv(output_root / "split_ids.csv", split_rows, ["experiment_id", "seed", "subset", "sample_id"])
    write_csv(
        output_root / "decisions.csv",
        decisions,
        [
            "experiment_id", "finding", "task", "seed", "deadline_s", "target_success", "strategy",
            "strategy_candidate_count",
            "selection_status", "candidate_id", "calibration_n", "calibration_final_accuracy",
            "calibration_timely_success_rate", "calibration_mean_request_energy_j",
            "calibration_mean_e2e_latency_s", "calibration_mean_decode_token_per_sec",
            "calibration_mean_energy_per_token_j",
        ],
    )
    write_csv(
        output_root / "metrics.csv",
        evaluations,
        [
            "experiment_id", "finding", "task", "seed", "deadline_s", "target_success", "strategy",
            "strategy_candidate_count",
            "selection_status", "candidate_id", "evaluation_n", "evaluation_final_accuracy",
            "evaluation_timely_correct_count", "evaluation_timely_success_rate",
            "evaluation_mean_request_energy_j", "evaluation_energy_per_timely_correct_j",
            "evaluation_target_met",
        ],
    )
    summaries = summarize(evaluations)
    write_csv(output_root / "summary.csv", summaries, list(summaries[0]))
    evidence = evidence_comparisons(evaluations)
    finding_evidence = [row for row in evidence if row["finding"] != "JOINT"]
    joint_evidence = [row for row in evidence if row["finding"] == "JOINT"]
    write_csv(output_root / "design_evidence.csv", finding_evidence, list(finding_evidence[0]))
    write_csv(output_root / "joint_comparisons.csv", joint_evidence, list(joint_evidence[0]))
    write_csv(
        output_root / "coverage.csv",
        coverage,
        ["experiment_id", "finding", "task", "candidate_id", "available_rows", "valid_rows", "common_complete_rows", "require_proxies"],
    )
    write_csv(
        output_root / "exclusions.csv",
        exclusions,
        ["experiment_id", "candidate_id", "sample_id", "reasons", "returncode", "energy_measurement_status", "error"],
    )
    print(f"Inventory: {len(manifests)} candidates, {len(records)} records")
    print(f"Day 1+2: {len(decisions)} decisions, {len(evaluations)} evaluations")
    print(f"Wrote {output_root}")


if __name__ == "__main__":
    main()
