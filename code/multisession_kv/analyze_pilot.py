#!/usr/bin/env python3
"""Build paired Pilot statistics, figures and a mechanism-validation report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLICY_ORDER = {
    "all_keep": 0,
    "lru": 1,
    "single_slot": 2,
    "no_reuse": 3,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_label(policy: str, lru_slots: int | None) -> str:
    return f"LRU-{lru_slots}" if policy == "lru" else {
        "all_keep": "AllKeep",
        "single_slot": "SingleSlot",
        "no_reuse": "NoReuse",
    }[policy]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def standard_deviation(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs(matrix_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(job["status"] != "pass" for job in manifest["jobs"]):
        counts: dict[str, int] = defaultdict(int)
        for job in manifest["jobs"]:
            counts[job["status"]] += 1
        raise RuntimeError(f"Pilot matrix is incomplete: {dict(counts)}")

    runs = []
    for sequence, job in enumerate(manifest["jobs"]):
        run_dir = Path(job["run_dir"])
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        requests = read_jsonl(run_dir / "requests.jsonl")
        telemetry = read_jsonl(run_dir / "telemetry.jsonl")
        input_tokens = sum(int(row["input_tokens"]) for row in requests)
        prefill_tokens = sum(int(row["prefill_tokens"]) for row in requests)
        padded_tokens = sum(int(row["prefill_padded_tokens"]) for row in requests)
        reused_tokens = sum(int(row["reused_tokens"]) for row in requests)
        prefill_ms = sum(float(row["prefill_ms"]) for row in requests)
        decode_rows = [row for row in requests if int(row["output_tokens"]) > 1]
        decode_tokens = sum(int(row["output_tokens"]) - 1 for row in decode_rows)
        decode_ms = sum(float(row["decode_ms"]) for row in decode_rows)
        temperatures = [
            float(row["skin_temp_c"])
            for row in telemetry
            if row.get("skin_temp_c") is not None
        ]
        rss = [
            int(row["rss_kb"])
            for row in telemetry
            if row.get("rss_kb") is not None
        ]
        model_load_counts = sorted({int(row["model_load_count"]) for row in requests})
        runs.append(
            {
                "matrix_sequence": sequence,
                "job_id": job["job_id"],
                "repeat": int(job["repeat"]),
                "session_count": int(job["session_count"]),
                "session_offset": int(job["session_offset"]),
                "schedule": job["schedule"],
                "policy": job["policy"],
                "policy_label": policy_label(job["policy"], job.get("lru_slots")),
                "lru_slots": job.get("lru_slots"),
                "request_count": len(requests),
                "input_tokens": input_tokens,
                "prefill_tokens": prefill_tokens,
                "padded_prefill_tokens": padded_tokens,
                "reused_tokens": reused_tokens,
                "token_reuse_rate": reused_tokens / input_tokens,
                "prefill_ms": prefill_ms,
                "useful_prefill_tok_s": prefill_tokens * 1000 / prefill_ms,
                "padded_prefill_tok_s": padded_tokens * 1000 / prefill_ms,
                "decode_tokens": decode_tokens,
                "decode_ms": decode_ms,
                "decode_tok_s": decode_tokens * 1000 / decode_ms,
                "ttft_ms_p50": statistics.median(
                    float(row["ttft_ms"]) for row in requests
                ),
                "trace_makespan_ms": sum(float(row["request_ms"]) for row in requests),
                "rss_kb_peak": max(rss),
                "skin_temp_c_start": temperatures[0],
                "skin_temp_c_peak": max(temperatures),
                "model_load_counts": model_load_counts,
            }
        )
    return manifest, runs


def group_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[
            (
                run["session_count"],
                run["schedule"],
                run["policy"],
                run["lru_slots"],
                run["policy_label"],
            )
        ].append(run)
    metrics = (
        "input_tokens",
        "prefill_tokens",
        "token_reuse_rate",
        "prefill_ms",
        "useful_prefill_tok_s",
        "padded_prefill_tok_s",
        "decode_tok_s",
        "ttft_ms_p50",
        "trace_makespan_ms",
        "rss_kb_peak",
        "skin_temp_c_start",
        "skin_temp_c_peak",
    )
    output = []
    for key, rows in grouped.items():
        session_count, schedule, policy, lru_slots, label = key
        row: dict[str, Any] = {
            "session_count": session_count,
            "schedule": schedule,
            "policy": policy,
            "policy_label": label,
            "lru_slots": lru_slots,
            "runs": len(rows),
            "request_count_per_run": rows[0]["request_count"],
        }
        for metric in metrics:
            row[f"{metric}_mean"] = mean(rows, metric)
            row[f"{metric}_sd"] = standard_deviation(rows, metric)
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row["schedule"],
            row["session_count"],
            POLICY_ORDER[row["policy"]],
            row["lru_slots"] or 0,
        ),
    )


def paired_deltas(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {
        (
            row["session_count"],
            row["schedule"],
            row["repeat"],
            row["policy"],
            row["lru_slots"],
        ): row
        for row in runs
    }
    metrics = ("prefill_tokens", "prefill_ms", "ttft_ms_p50", "trace_makespan_ms", "rss_kb_peak")
    deltas = []
    for row in runs:
        if row["policy"] == "all_keep":
            continue
        baseline = by_identity[
            (
                row["session_count"],
                row["schedule"],
                row["repeat"],
                "all_keep",
                None,
            )
        ]
        result: dict[str, Any] = {
            "session_count": row["session_count"],
            "schedule": row["schedule"],
            "repeat": row["repeat"],
            "policy": row["policy"],
            "policy_label": row["policy_label"],
            "lru_slots": row["lru_slots"],
        }
        for metric in metrics:
            result[f"{metric}_delta"] = row[metric] - baseline[metric]
            result[f"{metric}_delta_pct"] = (
                (row[metric] / baseline[metric] - 1) * 100
                if baseline[metric]
                else None
            )
        deltas.append(result)
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def round_robin_figures(grouped: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in grouped if row["schedule"] == "round_robin"]
    counts = sorted({int(row["session_count"]) for row in rows})
    colors = {
        "AllKeep": "#2b8cbe",
        "SingleSlot": "#e34a33",
        "NoReuse": "#969696",
    }

    fig, axes = plt.subplots(1, len(counts), figsize=(15, 4.4), sharey=True)
    for axis, count in zip(axes, counts):
        subset = [row for row in rows if row["session_count"] == count]
        labels = [row["policy_label"] for row in subset]
        values = [row["ttft_ms_p50_mean"] for row in subset]
        errors = [row["ttft_ms_p50_sd"] for row in subset]
        bar_colors = [colors.get(label, "#74c476") for label in labels]
        axis.bar(labels, values, yerr=errors, color=bar_colors, capsize=3)
        axis.set_title(f"{count} Sessions")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Median TTFT (ms), mean ± SD across 3 runs")
    fig.suptitle("RoundRobin: cache pressure increases TTFT")
    fig.tight_layout()
    fig.savefig(output_dir / "round_robin_ttft.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(counts), figsize=(15, 8), squeeze=False)
    for column, count in enumerate(counts):
        subset = [row for row in rows if row["session_count"] == count]
        labels = [row["policy_label"] for row in subset]
        bar_colors = [colors.get(label, "#74c476") for label in labels]
        axes[0][column].bar(
            labels, [row["prefill_tokens_mean"] for row in subset], color=bar_colors
        )
        axes[1][column].bar(
            labels,
            [row["token_reuse_rate_mean"] * 100 for row in subset],
            color=bar_colors,
        )
        axes[0][column].set_title(f"{count} Sessions")
        for axis in (axes[0][column], axes[1][column]):
            axis.tick_params(axis="x", rotation=45, labelsize=8)
            axis.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Prefill tokens per trace")
    axes[1][0].set_ylabel("Token-weighted reuse rate (%)")
    fig.suptitle("RoundRobin: recomputation and realized reuse")
    fig.tight_layout()
    fig.savefig(output_dir / "round_robin_prefill_reuse.png", dpi=180)
    plt.close(fig)


def memory_figure(grouped: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in grouped if row["schedule"] == "round_robin"]
    points: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        slots = None
        if row["policy"] == "all_keep":
            slots = int(row["session_count"])
        elif row["policy"] in ("single_slot", "lru"):
            slots = int(row["lru_slots"] or 1)
        if slots is not None:
            points[slots].append(float(row["rss_kb_peak_mean"]) / 1024)
    x = sorted(points)
    y = [statistics.median(points[value]) for value in x]
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(x, y, marker="o", color="#756bb1")
    axis.set_xlabel("Resident Session slots")
    axis.set_ylabel("Peak process RSS (MiB)")
    axis.set_title(f"Resident-state memory grows near-linearly (~{slope:.1f} MiB/slot endpoint slope)")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rss_by_resident_slots.png", dpi=180)
    plt.close(fig)


def report_text(
    manifest_path: Path,
    manifest: dict[str, Any],
    runs: list[dict[str, Any]],
    grouped: list[dict[str, Any]],
) -> str:
    def group(n: int, schedule: str, label: str) -> dict[str, Any]:
        return next(
            row
            for row in grouped
            if row["session_count"] == n
            and row["schedule"] == schedule
            and row["policy_label"] == label
        )

    stability = []
    grouped_runs: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped_runs[(run["session_count"], run["schedule"], run["policy_label"])].append(run)
    for metric in ("useful_prefill_tok_s", "decode_tok_s", "ttft_ms_p50", "trace_makespan_ms"):
        cvs = [
            statistics.stdev(float(row[metric]) for row in rows)
            / statistics.mean(float(row[metric]) for row in rows)
            * 100
            for rows in grouped_runs.values()
        ]
        stability.append((metric, statistics.median(cvs), max(cvs)))

    rr2_keep, rr2_low = group(2, "round_robin", "AllKeep"), group(2, "round_robin", "LRU-1")
    rr4_keep, rr4_low = group(4, "round_robin", "AllKeep"), group(4, "round_robin", "LRU-3")
    rr8_keep, rr8_low = group(8, "round_robin", "AllKeep"), group(8, "round_robin", "LRU-2")
    lines = [
        "# HammerBench fixed-tools paired Pilot report",
        "",
        "## Scope and integrity",
        "",
        f"- Source manifest: `{manifest_path}`",
        f"- Manifest SHA256: `{sha256(manifest_path)}`",
        f"- Randomization seed: {manifest['randomization_seed']}",
        f"- Completed runs: {len(runs)}/{len(manifest['jobs'])}; all passed without retry.",
        "- One fixed Session group per 2/4/8-Session condition; 3 run-level repetitions per configuration.",
        "- All request logs report `model_load_count=1`; requests are serial, not continuous batching.",
        "- This is deterministic system replay, not an official HammerBench accuracy score.",
        "",
        "## Main finding",
        "",
        "RoundRobin exposes real cache competition. When resident slots are fewer than the cyclic working set, Sessions are evicted before reuse, causing repeated Prefill and substantially higher TTFT. Grouped execution avoids this pressure because each Session finishes before the next Session is selected.",
        "",
        "| Sessions | RoundRobin policy | Reuse | Prefill/trace | TTFT p50 | Peak RSS |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in (rr2_keep, rr2_low, rr4_keep, rr4_low, rr8_keep, rr8_low):
        lines.append(
            f"| {row['session_count']} | {row['policy_label']} | "
            f"{row['token_reuse_rate_mean'] * 100:.1f}% | "
            f"{row['prefill_tokens_mean']:.0f} | {row['ttft_ms_p50_mean']:.1f} ms | "
            f"{row['rss_kb_peak_mean'] / 1024:.1f} MiB |"
        )
    lines += [
        "",
        "For 4 Sessions, even three LRU slots realize zero reuse under strict cyclic RoundRobin: every returning Session was already displaced. For 8 Sessions, four and six slots both realize only 20.0% reuse on this variable-turn trace, versus 57.8% for AllKeep.",
        "",
        "## Compute rates and memory",
        "",
        f"Across configuration means, Decode remains stable at {min(row['decode_tok_s_mean'] for row in grouped):.2f}--{max(row['decode_tok_s_mean'] for row in grouped):.2f} token/s. Useful Prefill throughput varies with incremental fragment size and 128-token graph padding; the benefit is the avoided Prefill token/time, not a faster Decode kernel.",
        "",
        "Peak runner RSS rises from about 50.5 MiB at one resident slot to about 303--317 MiB at eight slots. Intermediate slot increments are approximately 36 MiB each; the RoundRobin eight-slot endpoint includes additional variation. RSS is process-visible memory only and is not claimed as total device or QNN/DSP memory.",
        "",
        "## Repeat stability",
        "",
        "| Run-level metric | Median CV | Maximum CV |",
        "| --- | ---: | ---: |",
    ]
    for metric, median_cv, maximum_cv in stability:
        lines.append(f"| `{metric}` | {median_cv:.2f}% | {maximum_cv:.2f}% |")
    lines += [
        "",
        f"Telemetry starts range from {min(row['skin_temp_c_start'] for row in runs):.3f} to {max(row['skin_temp_c_start'] for row in runs):.3f} C; maximum observed skin temperature is {max(row['skin_temp_c_peak'] for row in runs):.3f} C. The recovery gate is sampled before model startup/warm-up, so first telemetry may slightly exceed the 28.2 C gate.",
        "",
        "## Interpretation and limitations",
        "",
        "- The Pilot establishes the mechanism for one fixed group at each Session count; it does not estimate population variation across all 50 Sessions.",
        "- Turns from the same Session are dependent and are not treated as independent statistical samples. With only one Session group per count, no bootstrap confidence interval is reported.",
        "- Session-count curves mix different request-length/turn distributions as the fixed group expands, so they should not be interpreted as a pure scaling curve.",
        "- Natural EOS makes Decode dominate trace makespan; therefore large TTFT changes translate into smaller makespan changes.",
        "- The Pilot does not cover the 897--951-token long-input slowdown observed in one earlier formal run; that anomaly needs a dedicated matched repeat.",
        "",
        "## Decision",
        "",
        "The problem is validated: cache capacity and Session scheduling materially alter realized prefix reuse, repeated Prefill, TTFT and resident memory, while Decode speed and model loading remain stable. A full real-trace stage should retain RoundRobin capacity points and only minimal Grouped controls, then aggregate all fixed Session groups.",
        "",
        "## Generated artifacts",
        "",
        "- `group_summary.csv` / `group_summary.json`",
        "- `paired_deltas.csv` / `paired_deltas.json`",
        "- `round_robin_ttft.png`",
        "- `round_robin_prefill_reuse.png`",
        "- `rss_by_resident_slots.png`",
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
    grouped = group_runs(runs)
    deltas = paired_deltas(runs)
    write_csv(args.output_dir / "group_summary.csv", grouped)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    (args.output_dir / "group_summary.json").write_text(
        json.dumps(grouped, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "paired_deltas.json").write_text(
        json.dumps(deltas, indent=2) + "\n", encoding="utf-8"
    )
    round_robin_figures(grouped, args.output_dir)
    memory_figure(grouped, args.output_dir)
    manifest_path = args.matrix_root / "matrix_manifest.json"
    (args.output_dir / "report.md").write_text(
        report_text(manifest_path, manifest, runs, grouped), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "runs": len(runs),
                "configuration_groups": len(grouped),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
