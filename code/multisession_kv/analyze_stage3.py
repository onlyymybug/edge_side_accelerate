#!/usr/bin/env python3
"""Analyze the completed all-group HammerBench Stage-3 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.mean(generator.choice(values) for _ in values))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def label(row: dict[str, Any]) -> str:
    if row["policy"] == "lru":
        return f"LRU-{row['lru_slots']}"
    return {"all_keep": "AllKeep", "single_slot": "SingleSlot", "no_reuse": "NoReuse"}[row["policy"]]


def load_runs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    if any(job["status"] != "pass" for job in manifest["jobs"]):
        raise RuntimeError("Stage-3 matrix is incomplete")
    runs = []
    for job in manifest["jobs"]:
        directory = Path(job["run_dir"])
        requests = read_jsonl(directory / "requests.jsonl")
        telemetry = read_jsonl(directory / "telemetry.jsonl")
        input_tokens = sum(int(row["input_tokens"]) for row in requests)
        prefill_tokens = sum(int(row["prefill_tokens"]) for row in requests)
        prefill_ms = sum(float(row["prefill_ms"]) for row in requests)
        decode_rows = [row for row in requests if int(row["output_tokens"]) > 1]
        decode_tokens = sum(int(row["output_tokens"]) - 1 for row in decode_rows)
        decode_ms = sum(float(row["decode_ms"]) for row in decode_rows)
        rss = [int(row["rss_kb"]) for row in telemetry if row.get("rss_kb") is not None]
        temperatures = [float(row["skin_temp_c"]) for row in telemetry if row.get("skin_temp_c") is not None]
        runs.append(
            {
                "job_id": job["job_id"],
                "experiment": job["experiment"],
                "nominal_session_count": int(job["nominal_session_count"]),
                "actual_session_count": int(job["session_count"]),
                "session_offset": int(job["session_offset"]),
                "tail_group": int(job["session_count"]) != int(job["nominal_session_count"]),
                "schedule": job["schedule"],
                "policy": job["policy"],
                "policy_label": label(job),
                "lru_slots": job.get("lru_slots"),
                "actual_slot_ratio": job.get("actual_slot_ratio"),
                "request_count": len(requests),
                "input_tokens": input_tokens,
                "prefill_tokens": prefill_tokens,
                "reused_tokens": sum(int(row["reused_tokens"]) for row in requests),
                "reuse_rate": sum(int(row["reused_tokens"]) for row in requests) / input_tokens,
                "prefill_ms": prefill_ms,
                "prefill_tok_s": prefill_tokens * 1000 / prefill_ms,
                "decode_tok_s": decode_tokens * 1000 / decode_ms,
                "ttft_ms_p50": statistics.median(float(row["ttft_ms"]) for row in requests),
                "makespan_ms": sum(float(row["request_ms"]) for row in requests),
                "rss_mib_peak": max(rss) / 1024,
                "skin_start_c": temperatures[0],
                "skin_peak_c": max(temperatures),
                "model_load_values": sorted({int(row["model_load_count"]) for row in requests}),
            }
        )
    return manifest, runs


def summarize(runs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["tail_group"]:
            continue
        groups[(run["nominal_session_count"], run["schedule"], run["policy"], run["lru_slots"])].append(run)
    output = []
    for index, (key, rows) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
        total_input = sum(row["input_tokens"] for row in rows)
        total_reused = sum(row["reused_tokens"] for row in rows)
        ttft = [row["ttft_ms_p50"] for row in rows]
        ttft_low, ttft_high = bootstrap_mean_ci(ttft, seed + index)
        output.append(
            {
                "nominal_session_count": key[0],
                "schedule": key[1],
                "policy": key[2],
                "policy_label": rows[0]["policy_label"],
                "lru_slots": key[3],
                "actual_slot_ratio": rows[0]["actual_slot_ratio"],
                "session_groups": len(rows),
                "requests": sum(row["request_count"] for row in rows),
                "input_tokens": total_input,
                "prefill_tokens": sum(row["prefill_tokens"] for row in rows),
                "token_weighted_reuse_rate": total_reused / total_input,
                "group_mean_ttft_ms_p50": statistics.mean(ttft),
                "group_mean_ttft_ms_p50_ci_low": ttft_low,
                "group_mean_ttft_ms_p50_ci_high": ttft_high,
                "group_mean_prefill_tok_s": statistics.mean(row["prefill_tok_s"] for row in rows),
                "group_mean_decode_tok_s": statistics.mean(row["decode_tok_s"] for row in rows),
                "group_mean_makespan_ms": statistics.mean(row["makespan_ms"] for row in rows),
                "group_mean_rss_mib_peak": statistics.mean(row["rss_mib_peak"] for row in rows),
                "maximum_skin_c": max(row["skin_peak_c"] for row in rows),
            }
        )
    return output


def paired_summary(runs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    full = [run for run in runs if not run["tail_group"]]
    lookup = {
        (run["nominal_session_count"], run["schedule"], run["session_offset"], run["policy"], run["lru_slots"]): run
        for run in full
    }
    rows_by_config: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for run in full:
        if run["policy"] == "all_keep":
            continue
        baseline = lookup[(run["nominal_session_count"], run["schedule"], run["session_offset"], "all_keep", None)]
        rows_by_config[(run["nominal_session_count"], run["schedule"], run["policy"], run["lru_slots"])].append(
            {
                "prefill_delta_pct": (run["prefill_tokens"] / baseline["prefill_tokens"] - 1) * 100,
                "ttft_delta_pct": (run["ttft_ms_p50"] / baseline["ttft_ms_p50"] - 1) * 100,
                "makespan_delta_pct": (run["makespan_ms"] / baseline["makespan_ms"] - 1) * 100,
                "rss_delta_mib": run["rss_mib_peak"] - baseline["rss_mib_peak"],
            }
        )
    output = []
    for index, (key, pairs) in enumerate(sorted(rows_by_config.items(), key=lambda item: str(item[0]))):
        row: dict[str, Any] = {
            "nominal_session_count": key[0],
            "schedule": key[1],
            "policy": key[2],
            "policy_label": f"LRU-{key[3]}" if key[2] == "lru" else {"single_slot": "SingleSlot", "no_reuse": "NoReuse"}[key[2]],
            "lru_slots": key[3],
            "paired_groups": len(pairs),
        }
        for metric in ("prefill_delta_pct", "ttft_delta_pct", "makespan_delta_pct", "rss_delta_mib"):
            values = [pair[metric] for pair in pairs]
            low, high = bootstrap_mean_ci(values, seed + index * 10 + len(metric))
            row[f"mean_{metric}"] = statistics.mean(values)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        output.append(row)
    return output


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(summary: list[dict[str, Any]], paired: list[dict[str, Any]], output: Path) -> None:
    rr = [row for row in summary if row["schedule"] == "round_robin"]
    counts = [2, 4, 8]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for column, count in enumerate(counts):
        rows = [row for row in rr if row["nominal_session_count"] == count]
        labels = [row["policy_label"] for row in rows]
        axes[0][column].bar(labels, [row["token_weighted_reuse_rate"] * 100 for row in rows])
        axes[1][column].bar(labels, [row["group_mean_ttft_ms_p50"] for row in rows])
        axes[0][column].set_title(f"{count} Sessions")
        for axis in (axes[0][column], axes[1][column]):
            axis.tick_params(axis="x", rotation=45, labelsize=8)
            axis.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Token-weighted reuse rate (%)")
    axes[1][0].set_ylabel("Mean group median TTFT (ms)")
    fig.suptitle("All HammerBench groups: RoundRobin cache pressure")
    fig.tight_layout()
    fig.savefig(output / "round_robin_population.png", dpi=180)
    plt.close(fig)

    rows = [row for row in paired if row["schedule"] == "round_robin"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    for column, count in enumerate(counts):
        subset = [row for row in rows if row["nominal_session_count"] == count]
        labels = [row["policy_label"] for row in subset]
        values = [row["mean_ttft_delta_pct"] for row in subset]
        low = [value - row["ttft_delta_pct_ci_low"] for value, row in zip(values, subset)]
        high = [row["ttft_delta_pct_ci_high"] - value for value, row in zip(values, subset)]
        axes[0][column].bar(labels, values, yerr=[low, high], capsize=3)
        axes[0][column].axhline(0, color="black", linewidth=0.8)
        axes[0][column].set_title(f"{count} Sessions")
        axes[0][column].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0][column].grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("TTFT change vs matched AllKeep (%)\nmean and group-bootstrap 95% CI")
    fig.suptitle("RoundRobin paired TTFT penalty")
    fig.tight_layout()
    fig.savefig(output / "round_robin_paired_ttft.png", dpi=180)
    plt.close(fig)

    slot_points: dict[int, list[float]] = defaultdict(list)
    for run in rr:
        slots = run["nominal_session_count"] if run["policy"] == "all_keep" else (run["lru_slots"] or 1)
        slot_points[int(slots)].append(float(run["group_mean_rss_mib_peak"]))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = sorted(slot_points)
    axis.plot(x, [statistics.median(slot_points[value]) for value in x], marker="o")
    axis.set_xlabel("Resident Session slots")
    axis.set_ylabel("Mean group peak runner RSS (MiB)")
    axis.set_title("Stage-3 resident-state memory scaling")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "rss_population.png", dpi=180)
    plt.close(fig)


def report(manifest: dict[str, Any], runs: list[dict[str, Any]], summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    rr = [row for row in summary if row["schedule"] == "round_robin"]
    lines = [
        "# HammerBench fixed-tools Stage-3 population report",
        "",
        "## Integrity and scope",
        "",
        f"- Completed jobs: {len(runs)}/{len(manifest['jobs'])}; failures/retries: 0/0.",
        f"- Full Session groups in primary analysis: {sum(not run['tail_group'] for run in runs)} runs; tail-group runs retained separately: {sum(run['tail_group'] for run in runs)}.",
        "- Bootstrap resampling unit is the fixed Session group, never an individual turn.",
        "- This is a fixed-tools system replay and is not an official HammerBench accuracy score.",
        "",
        "## RoundRobin population summary",
        "",
        "| Sessions | Policy | Groups | Reuse | Prefill tokens | Mean group TTFT p50 | RSS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rr, key=lambda item: (item["nominal_session_count"], item["policy_label"])):
        lines.append(
            f"| {row['nominal_session_count']} | {row['policy_label']} | {row['session_groups']} | "
            f"{row['token_weighted_reuse_rate'] * 100:.1f}% | {row['prefill_tokens']} | "
            f"{row['group_mean_ttft_ms_p50']:.1f} ms | {row['group_mean_rss_mib_peak']:.1f} MiB |"
        )
    lines += [
        "",
        "## Paired effect versus AllKeep",
        "",
        "| Sessions | Schedule | Policy | Groups | Prefill change | TTFT change (95% CI) | Makespan change | RSS change |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired:
        lines.append(
            f"| {row['nominal_session_count']} | {row['schedule']} | {row['policy_label']} | {row['paired_groups']} | "
            f"{row['mean_prefill_delta_pct']:+.1f}% | {row['mean_ttft_delta_pct']:+.1f}% "
            f"[{row['ttft_delta_pct_ci_low']:+.1f}, {row['ttft_delta_pct_ci_high']:+.1f}] | "
            f"{row['mean_makespan_delta_pct']:+.1f}% | {row['mean_rss_delta_mib']:+.1f} MiB |"
        )
    lines += [
        "",
        "## Runtime invariants and limitations",
        "",
        f"- Model load count values: {sorted(set().union(*(run['model_load_values'] for run in runs)))}.",
        f"- Decode throughput across population configurations: {min(row['group_mean_decode_tok_s'] for row in summary):.2f}--{max(row['group_mean_decode_tok_s'] for row in summary):.2f} token/s.",
        f"- Maximum observed skin temperature: {max(run['skin_peak_c'] for run in runs):.3f} C.",
        "- Nominal 4- and 8-Session partitions each have a final two-Session tail group. Those runs remain in the manifest but are excluded from primary nominal-count estimates.",
        "- Natural EOS means Decode can dominate makespan; TTFT and Prefill changes need not translate proportionally to total trace time.",
        "- Native-tools and BFCL generalization have not yet been tested.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, runs = load_runs(args.matrix_root)
    seed = int(manifest["randomization_seed"])
    summary = summarize(runs, seed)
    paired = paired_summary(runs, seed)
    write_rows(args.output_dir / "population_summary.csv", summary)
    write_rows(args.output_dir / "paired_effects.csv", paired)
    (args.output_dir / "population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "paired_effects.json").write_text(json.dumps(paired, indent=2) + "\n")
    make_figures(summary, paired, args.output_dir)
    (args.output_dir / "report.md").write_text(report(manifest, runs, summary, paired))
    print(json.dumps({"status": "pass", "runs": len(runs), "summary_rows": len(summary), "paired_rows": len(paired), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
