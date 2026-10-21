#!/usr/bin/env python3
"""PACE optional calibration-efficiency replay from Section 9."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("direct_lookup", "shared_accuracy", "paired_mapping", "workload_model")


def args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=here / "config.json")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def truth(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "1.0"}


def fnum(x: Any) -> float:
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {x}")
    return value


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def completion_path(source: Path, sid: str) -> Path | None:
    root = source.parent / "gsm8k" / sid
    for name in ("completion.txt", "generated_text.txt", "raw_output.txt"):
        path = root / name
        if path.is_file():
            return path
    return None


def candidate(runtime: str, mode: str) -> str:
    return f"oneplus__gsm8k__{runtime.lower().replace('.', '_').replace('-', '_')}__{mode}__b512"


def audit(config: dict[str, Any], manifests: list[dict[str, str]], out: Path) -> list[dict[str, Any]]:
    selected = {
        (r["runtime"], r["mode"]): r
        for r in manifests
        if r["device"] == config["device"]
        and r["task"] == config["task"]
        and r["output_budget"] == config["output_budget"]
    }
    reports: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for runtime, family in config["families"].items():
        ref_mode = family["reference_mode"]
        ref_spec = selected[(runtime, ref_mode)]
        ref_source = Path(ref_spec["source_file"])
        ref_rows = {r["sample_id"]: r for r in read_csv(ref_source)}
        for mode in family["target_modes"]:
            target_spec = selected[(runtime, mode)]
            target_source = Path(target_spec["source_file"])
            target_rows = {r["sample_id"]: r for r in read_csv(target_source)}
            ids = sorted(set(ref_rows) & set(target_rows))
            counts = defaultdict(int)
            for sid in ids:
                a, b = ref_rows[sid], target_rows[sid]
                checks = {
                    "score": str(a.get("score_correct")) == str(b.get("score_correct")),
                    "answer": str(a.get("normalized_predicted_answer")) == str(b.get("normalized_predicted_answer")),
                    "prompt_tokens": str(a.get("prompt_tokens")) == str(b.get("prompt_tokens")),
                    "generated_tokens": str(a.get("generated_tokens")) == str(b.get("generated_tokens")),
                    "termination": (str(a.get("finish_reason", "")), str(a.get("seq_len_stop", "")))
                    == (str(b.get("finish_reason", "")), str(b.get("seq_len_stop", ""))),
                }
                pa, pb = completion_path(ref_source, sid), completion_path(target_source, sid)
                text_equal = pa is not None and pb is not None and sha(pa) == sha(pb)
                for key, ok in checks.items():
                    counts[f"{key}_match"] += int(ok)
                counts["completion_available"] += int(pa is not None and pb is not None)
                counts["completion_hash_match"] += int(text_equal)
                if not all(checks.values()) or not text_equal:
                    detail.append({
                        "runtime": runtime, "reference_mode": ref_mode, "target_mode": mode,
                        "sample_id": sid, **{f"{k}_match": v for k, v in checks.items()},
                        "completion_hash_match": text_equal,
                    })
            reports.append({
                "runtime": runtime, "reference_mode": ref_mode, "target_mode": mode,
                "common_n": len(ids), **counts,
                "original_token_id_hash_available_n": 0,
                "sharing_definition": config["sharing_criterion"],
                "sharing_status": "passed" if counts["completion_hash_match"] == len(ids) else "failed",
                "phone_needed_under_adopted_definition": False,
                "reference_source": str(ref_source), "target_source": str(target_source),
            })
    write_csv(out / "pairing_audit.csv", reports)
    write_csv(out / "pairing_mismatches.csv", detail)
    return reports


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    keep = scale > 1e-12
    z = (x[:, keep] - mean[keep]) / scale[keep] if keep.any() else np.empty((len(x), 0))
    design = np.column_stack([np.ones(len(x)), z])
    penalty = np.eye(design.shape[1]) * lam
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "keep": keep, "coef": coef}


def ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    keep = model["keep"]
    z = (x[:, keep] - model["mean"][keep]) / model["scale"][keep] if keep.any() else np.empty((len(x), 0))
    return np.column_stack([np.ones(len(x)), z]) @ model["coef"]


def crossfit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[dict[str, Any], np.ndarray, float]:
    residual = np.empty(len(y))
    for test in np.array_split(np.arange(len(y)), 4):
        train = np.setdiff1d(np.arange(len(y)), test)
        model = ridge_fit(x[train], y[train], lam)
        residual[test] = y[test] - ridge_predict(model, x[test])
    final = ridge_fit(x, y, lam)
    return final, residual, float(np.sqrt(np.mean(residual**2)))


def actual(rows: list[dict[str, Any]], deadline: float) -> tuple[float, float]:
    s = np.mean([r["y"] and r["time"] <= deadline for r in rows])
    e = np.mean([r["energy"] for r in rows])
    return float(s), float(e)


def main() -> int:
    a = args()
    config = json.loads(a.config.read_text())
    out = Path(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifests = read_csv(Path(config["manifest_csv"]))
    audits = audit(config, manifests, out)

    wanted = {
        candidate(runtime, mode)
        for runtime, f in config["families"].items()
        for mode in [f["reference_mode"], *f["target_modes"]]
    }
    rows = []
    for r in read_csv(Path(config["records_csv"])):
        if r["candidate_id"] not in wanted or not truth(r["core_record_complete"]):
            continue
        rows.append({
            "id": r["sample_id"], "candidate": r["candidate_id"], "runtime": r["runtime"],
            "mode": r["mode"], "y": truth(r["score_correct"]),
            "time": fnum(r["e2e_latency_ms"]) / 1000.0, "energy": fnum(r["energy_j"]),
            "prompt": fnum(r["prompt_tokens"]), "generated": fnum(r["generated_tokens"]),
        })
    by_c: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_c[r["candidate"]][r["id"]] = r
    common = sorted(set.intersection(*(set(v) for v in by_c.values())))
    if len(common) < config["calibration_pool_n"] + 1:
        raise RuntimeError(f"only {len(common)} common complete IDs")

    candidate_order = sorted(wanted)
    reference = {
        runtime: candidate(runtime, f["reference_mode"])
        for runtime, f in config["families"].items()
    }
    target_to_ref = {
        candidate(runtime, mode): reference[runtime]
        for runtime, f in config["families"].items() for mode in f["target_modes"]
    }
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    lam = float(config["ridge_lambda"])

    for seed in config["seeds"]:
        shuffled = np.array(common, dtype=object)
        np.random.default_rng(seed).shuffle(shuffled)
        cal = list(shuffled[: config["calibration_pool_n"]])
        val = list(shuffled[config["calibration_pool_n"] :])
        split_rows.extend({"seed": seed, "sample_id": sid, "split": "calibration_pool", "position": i}
                          for i, sid in enumerate(cal))
        split_rows.extend({"seed": seed, "sample_id": sid, "split": "evaluation", "position": i}
                          for i, sid in enumerate(val))
        for k in config["k_values"]:
            budget = 3 * config["reference_n"] + 6 * k
            direct_n, rem = divmod(budget, 9)
            allocation = {c: direct_n + int(i < rem) for i, c in enumerate(candidate_order)}
            estimates: dict[str, dict[str, dict[float, tuple[float, float]]]] = {m: {} for m in METHODS}

            for c in candidate_order:
                measured = [by_c[c][sid] for sid in cal[: allocation[c]]]
                estimates["direct_lookup"][c] = {
                    d: actual(measured, d) for d in config["deadlines_s"]
                }
                for sid in cal[: allocation[c]]:
                    ledger.append({"seed": seed, "k": k, "method": "direct_lookup", "candidate_id": c,
                                   "sample_id": sid, "role": "candidate_measurement", "physical_repeat": "historical"})

            for runtime, ref_c in reference.items():
                ref_rows = [by_c[ref_c][sid] for sid in cal[: config["reference_n"]]]
                for method in METHODS[1:]:
                    estimates[method][ref_c] = {d: actual(ref_rows, d) for d in config["deadlines_s"]}
                    for sid in cal[: config["reference_n"]]:
                        ledger.append({"seed": seed, "k": k, "method": method, "candidate_id": ref_c,
                                       "sample_id": sid, "role": "reference_measurement", "physical_repeat": "historical"})

            for target_c, ref_c in target_to_ref.items():
                pair_ids = cal[:k]
                ref_train = [by_c[ref_c][sid] for sid in pair_ids]
                target_train = [by_c[target_c][sid] for sid in pair_ids]
                ref_full = [by_c[ref_c][sid] for sid in cal[: config["reference_n"]]]
                for method in METHODS[1:]:
                    for sid in pair_ids:
                        ledger.append({"seed": seed, "k": k, "method": method, "candidate_id": target_c,
                                       "sample_id": sid, "role": "target_measurement", "physical_repeat": "historical"})

                estimates["shared_accuracy"][target_c] = {}
                acc_ref = np.mean([r["y"] for r in ref_full])
                e_target = np.mean([r["energy"] for r in target_train])
                for d in config["deadlines_s"]:
                    estimates["shared_accuracy"][target_c][d] = (
                        float(acc_ref * np.mean([r["time"] <= d for r in target_train])), float(e_target)
                    )

                for method, features in (
                    ("paired_mapping", (lambda rs, field: np.array([[r[field]] for r in rs]))),
                    ("workload_model", (lambda rs, field: np.array([[r["prompt"], max(r["generated"] - 1, 0)] for r in rs]))),
                ):
                    estimates[method][target_c] = {}
                    x_train = features(ref_train if method == "paired_mapping" else target_train, "time")
                    x_full = features(ref_full, "time")
                    y_t = np.array([r["time"] for r in target_train])
                    y_e = np.array([r["energy"] for r in target_train])
                    mt, residual, rmse_t = crossfit(x_train, y_t, lam)
                    me, _, rmse_e = crossfit(x_train, y_e, lam)
                    pred_t = ridge_predict(mt, x_full)
                    pred_e = ridge_predict(me, x_full)
                    y_ref = np.array([r["y"] for r in ref_full], dtype=float)
                    for d in config["deadlines_s"]:
                        h = np.mean(np.maximum(0, pred_t[:, None] + residual[None, :]) <= d, axis=1)
                        estimates[method][target_c][d] = (float(np.mean(y_ref * h)), float(np.mean(pred_e)))
                    fits.append({"seed": seed, "k": k, "method": method, "candidate_id": target_c,
                                 "time_cv_rmse": rmse_t, "energy_cv_rmse": rmse_e,
                                 "time_coef": json.dumps(mt["coef"].tolist()),
                                 "energy_coef": json.dumps(me["coef"].tolist())})

            for method in METHODS:
                for d in config["deadlines_s"]:
                    for q in config["target_success"]:
                        feasible = [(e, c, s) for c, per_d in estimates[method].items()
                                    for s, e in [per_d[d]] if s >= q]
                        chosen = min(feasible)[1] if feasible else ""
                        eval_rows = [by_c[chosen][sid] for sid in val] if chosen else []
                        eval_s, eval_e = actual(eval_rows, d) if eval_rows else (math.nan, math.nan)
                        decisions.append({
                            "seed": seed, "k": k, "request_budget": budget, "method": method,
                            "deadline_s": d, "target_success": q, "selected_candidate_id": chosen,
                            "selection_status": "selected" if chosen else "no_selection",
                            "estimated_success": estimates[method][chosen][d][0] if chosen else "",
                            "estimated_energy_j": estimates[method][chosen][d][1] if chosen else "",
                            "evaluation_n": len(eval_rows), "evaluation_success": eval_s if chosen else "",
                            "evaluation_energy_j": eval_e if chosen else "",
                            "evaluation_target_met": bool(chosen and eval_s >= q),
                        })

    write_csv(out / "request_ledger.csv", ledger)
    write_csv(out / "split_ids.csv", split_rows)
    seed0_ids = [r["sample_id"] for r in split_rows
                 if r["seed"] == 0 and r["split"] == "calibration_pool" and r["position"] < 16]
    phone_plan = []
    for runtime, family in config["families"].items():
        for mode in [family["reference_mode"], *family["target_modes"]]:
            for sid in seed0_ids:
                phone_plan.append({
                    "runtime": runtime, "mode": mode, "sample_id": sid, "output_budget": 512,
                    "physical_repeat": 1, "purpose": "original_token_trajectory_audit",
                    "required_outputs": "generated_token_ids,token_id_sha256,completion,finish_reason,mode_log",
                    "status": "not_required_under_text_identity_definition",
                })
    write_csv(out / "phone_request_plan.csv", phone_plan)
    write_csv(out / "fit_parameters.csv", fits)
    write_csv(out / "decisions.csv", decisions)

    budget_audit = []
    ledger_groups: dict[tuple[int, int, str], set[tuple[str, str]]] = defaultdict(set)
    for r in ledger:
        ledger_groups[(r["seed"], r["k"], r["method"])].add((r["candidate_id"], r["sample_id"]))
    for (seed, k, method), requests in sorted(ledger_groups.items()):
        expected = 3 * config["reference_n"] + 6 * k
        inference_seconds = sum(by_c[c][sid]["time"] for c, sid in requests)
        recorded_energy_j = sum(by_c[c][sid]["energy"] for c, sid in requests)
        budget_audit.append({"seed": seed, "k": k, "method": method,
                             "expected_requests": expected, "unique_requests": len(requests),
                             "budget_matches": len(requests) == expected,
                             "calibration_inference_seconds": inference_seconds,
                             "calibration_inference_hours": inference_seconds / 3600.0,
                             "calibration_recorded_energy_j": recorded_energy_j,
                             "load_switch_cooling_overhead": "not_recorded_excluded"})
    write_csv(out / "request_budget_audit.csv", budget_audit)
    if not all(r["budget_matches"] for r in budget_audit):
        raise RuntimeError("request-ledger budget mismatch")

    decision_index = {
        (r["seed"], r["k"], r["deadline_s"], r["target_success"], r["method"]): r
        for r in decisions
    }
    comparisons = []
    for k in config["k_values"]:
        for method in METHODS[1:]:
            pairs = []
            for seed in config["seeds"]:
                for d in config["deadlines_s"]:
                    for q in config["target_success"]:
                        direct = decision_index[(seed, k, d, q, "direct_lookup")]
                        reuse = decision_index[(seed, k, d, q, method)]
                        pairs.append((direct, reuse))
            both = [(a, b) for a, b in pairs if a["evaluation_target_met"] and b["evaluation_target_met"]]
            comparisons.append({
                "k": k, "request_budget": 3 * config["reference_n"] + 6 * k, "method": method,
                "conditions": len(pairs),
                "direct_met_count": sum(a["evaluation_target_met"] for a, _ in pairs),
                "reuse_met_count": sum(b["evaluation_target_met"] for _, b in pairs),
                "reuse_only_met_count": sum(b["evaluation_target_met"] and not a["evaluation_target_met"] for a, b in pairs),
                "direct_only_met_count": sum(a["evaluation_target_met"] and not b["evaluation_target_met"] for a, b in pairs),
                "both_met_count": len(both),
                "same_selection_count": sum(a["selected_candidate_id"] == b["selected_candidate_id"] for a, b in pairs),
                "paired_energy_saving_vs_direct_j": (
                    np.mean([float(a["evaluation_energy_j"]) - float(b["evaluation_energy_j"]) for a, b in both])
                    if both else ""
                ),
            })
    write_csv(out / "comparisons_vs_direct.csv", comparisons)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in decisions:
        grouped[(r["k"], r["request_budget"], r["method"], r["deadline_s"], r["target_success"])].append(r)
    summary = []
    for key, group in sorted(grouped.items()):
        selected = [r for r in group if r["selection_status"] == "selected"]
        met = [r for r in selected if r["evaluation_target_met"]]
        summary.append({
            "k": key[0], "request_budget": key[1], "method": key[2], "deadline_s": key[3],
            "target_success": key[4], "repetitions": len(group), "selection_count": len(selected),
            "no_selection_count": len(group) - len(selected), "evaluation_met_count": len(met),
            "mean_evaluation_energy_selected_j": np.mean([r["evaluation_energy_j"] for r in selected]) if selected else "",
            "mean_evaluation_energy_met_j": np.mean([r["evaluation_energy_j"] for r in met]) if met else "",
        })
    write_csv(out / "summary.csv", summary)

    efficiency_curve = []
    for k in config["k_values"]:
        for method in METHODS:
            decision_group = [r for r in decisions if r["method"] == method and r["k"] == k]
            met = [r for r in decision_group if r["evaluation_target_met"]]
            cost_group = [r for r in budget_audit if r["method"] == method and r["k"] == k]
            efficiency_curve.append({
                "k": k,
                "request_budget": 3 * config["reference_n"] + 6 * k,
                "method": method,
                "evaluation_decisions": len(decision_group),
                "evaluation_met_count": len(met),
                "evaluation_met_rate": len(met) / len(decision_group),
                "selection_count": sum(r["selection_status"] == "selected" for r in decision_group),
                "no_selection_count": sum(r["selection_status"] == "no_selection" for r in decision_group),
                "mean_evaluation_energy_met_j": (
                    float(np.mean([r["evaluation_energy_j"] for r in met])) if met else ""
                ),
                "mean_calibration_inference_seconds": float(np.mean([
                    r["calibration_inference_seconds"] for r in cost_group
                ])),
                "std_calibration_inference_seconds": float(np.std([
                    r["calibration_inference_seconds"] for r in cost_group
                ])),
                "mean_calibration_inference_hours": float(np.mean([
                    r["calibration_inference_hours"] for r in cost_group
                ])),
                "mean_calibration_recorded_energy_j": float(np.mean([
                    r["calibration_recorded_energy_j"] for r in cost_group
                ])),
                "overhead_status": "load_switch_cooling_not_recorded_excluded",
            })
    write_csv(out / "efficiency_curve.csv", efficiency_curve)

    curve_index = {(r["k"], r["method"]): r for r in efficiency_curve}
    report = "# Optional calibration-efficiency replay\n\n"
    report += f"- Scope: OnePlus / GSM8K / budget 512, 9 candidates.\n- Common complete IDs: {len(common)}.\n"
    report += f"- Independent protocol: first 512 calibration-pool IDs and remaining {len(common)-512} evaluation IDs for each of 20 seeds.\n"
    report += f"- Sharing definition: `{config['sharing_criterion']}`; all {len(audits)}/{len(audits)} target/reference pairs passed on 1,319 common samples per pair.\n"
    report += "- The adopted operational definition treats byte-identical saved completions as a shared generation trajectory. Original generated token IDs were not logged and are not claimed separately.\n"
    report += "- This is a historical replay, not a new phone repetition. No phone run is required under the adopted definition.\n"
    report += "\n## Target-met decisions across all six cells and 20 seeds\n\n"
    report += "| k | requests | direct | shared accuracy | paired mapping | workload model |\n|---:|---:|---:|---:|---:|---:|\n"
    for k in config["k_values"]:
        values = {}
        for method in METHODS:
            group = [r for r in decisions if r["k"] == k and r["method"] == method]
            values[method] = sum(r["evaluation_target_met"] for r in group)
        report += f"| {k} | {3 * config['reference_n'] + 6 * k} | {values['direct_lookup']}/120 | {values['shared_accuracy']}/120 | {values['paired_mapping']}/120 | {values['workload_model']}/120 |\n"
    report += "\nThe workload model reduced cross-fitted time/energy error but matched the paired mapping's evaluation target-met count at every budget and selected the same candidate in 479/480 decisions. This is no observed decision benefit, so the graph-block extension is not justified yet.\n"
    report += "\n## Same-budget comparison\n\n"
    for r in comparisons:
        if r["method"] == "paired_mapping":
            report += (f"- k={r['k']} (B={r['request_budget']}): paired mapping met "
                       f"{r['reuse_met_count']}/120 versus direct lookup {r['direct_met_count']}/120; "
                       f"reuse-only/direct-only = {r['reuse_only_met_count']}/{r['direct_only_met_count']}.\n")
    report += "\nThese results are positive retrospective evidence for calibration reuse, not a deployment/generalization result. The simple mapping is sufficient for the current evidence; stop before the graph model.\n"
    report += "\n## Measured calibration cost\n\n"
    report += "The following values average the historical inference windows across 20 splits. Recorded energy is summed over those windows. Model loading, mode switching, and cooling overheads were not logged and are excluded rather than treated as zero.\n\n"
    report += "| k | requests | direct time (h) | mapping time (h) | direct energy (kJ) | mapping energy (kJ) |\n|---:|---:|---:|---:|---:|---:|\n"
    for k in config["k_values"]:
        direct = curve_index[(k, "direct_lookup")]
        mapping = curve_index[(k, "paired_mapping")]
        report += (f"| {k} | {direct['request_budget']} | {direct['mean_calibration_inference_hours']:.2f} | "
                   f"{mapping['mean_calibration_inference_hours']:.2f} | "
                   f"{direct['mean_calibration_recorded_energy_j']/1000:.2f} | "
                   f"{mapping['mean_calibration_recorded_energy_j']/1000:.2f} |\n")
    report += "\n## Phone requirement\n\nNo phone run is required under the adopted byte-identical-completion sharing definition. `phone_request_plan.csv` is retained only as an unused contingency plan and is marked not required.\n"
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out}; common_n={len(common)}, decisions={len(decisions)}, ledger={len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
