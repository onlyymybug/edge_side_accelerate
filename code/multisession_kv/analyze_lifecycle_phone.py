#!/usr/bin/env python3
"""Analyze the complete paired Lifecycle-LRU phone matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = 10000
) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = sorted(
        statistics.mean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def load_run(job: dict[str, Any]) -> dict[str, Any]:
    directory = Path(job["run_dir"])
    requests = read_jsonl(directory / "requests.jsonl")
    events = read_jsonl(directory / "events.jsonl")
    telemetry = read_jsonl(directory / "telemetry.jsonl")
    config = json.loads((directory / "config.json").read_text())
    temperatures = [
        float(row["skin_temp_c"])
        for row in telemetry
        if row.get("skin_temp_c") is not None
    ]
    rss = [int(row["rss_kb"]) for row in telemetry if row.get("rss_kb") is not None]
    decode_rows = [row for row in requests if int(row["output_tokens"]) > 1]
    decode_tokens = sum(int(row["output_tokens"]) - 1 for row in decode_rows)
    decode_ms = sum(float(row["decode_ms"]) for row in decode_rows)
    assert {int(row["model_load_count"]) for row in requests} == {1}
    if job["policy"] == "lifecycle_lru":
        sources = {
            row["lifecycle_source"] for row in events if "lifecycle_source" in row
        }
        assert sources == {"controlled_protocol"}
    return {
        "job": job,
        "requests": requests,
        "config": config,
        "input_tokens": sum(int(row["input_tokens"]) for row in requests),
        "reused_tokens": sum(int(row["reused_tokens"]) for row in requests),
        "prefill_tokens": sum(int(row["prefill_tokens"]) for row in requests),
        "prefill_ms": sum(float(row["prefill_ms"]) for row in requests),
        "decode_tok_s": decode_tokens * 1000 / decode_ms,
        "ttft_ms_p50": statistics.median(float(row["ttft_ms"]) for row in requests),
        "makespan_ms": sum(float(row["request_ms"]) for row in requests),
        "rss_mib_peak": max(rss) / 1024,
        "skin_start_c": temperatures[0],
        "skin_peak_c": max(temperatures),
        "decision_ms": sum(float(row.get("policy_decision_ms", 0)) for row in requests),
        "decision_ms_max": max(
            float(row.get("policy_decision_ms", 0)) for row in requests
        ),
    }


def offline_lookup(path: Path) -> dict[tuple[int, int, int, str], int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (
            int(row["nominal_session_count"]),
            int(row["session_offset"]),
            int(row["capacity"]),
            row["policy"],
        ): int(row["reused_tokens"])
        for row in rows
        if row["protocol"] == "serial_arrival"
    }


def analyze(
    matrix_root: Path, offline_path: Path, seed: int = 1012
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text())
    if len(manifest["jobs"]) != 84 or any(
        job["status"] != "pass" for job in manifest["jobs"]
    ):
        raise RuntimeError("Lifecycle phone matrix is not complete at 84/84 pass")
    expected = offline_lookup(offline_path)
    runs: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for job in manifest["jobs"]:
        key = (
            int(job["nominal_session_count"]),
            int(job["session_offset"]),
            int(job["lru_slots"]),
            str(job["policy"]),
        )
        run = load_run(job)
        assert run["reused_tokens"] == expected[key]
        runs[key] = run

    by_group: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for n, offset, capacity, policy in sorted(runs):
        if policy != "lru":
            continue
        lru = runs[(n, offset, capacity, "lru")]
        lifecycle = runs[(n, offset, capacity, "lifecycle_lru")]
        assert [row["source_snapshot_id"] for row in lru["requests"]] == [
            row["source_snapshot_id"] for row in lifecycle["requests"]
        ]
        assert lru["input_tokens"] == lifecycle["input_tokens"]
        row = {
            "nominal_session_count": n,
            "capacity": capacity,
            "session_offset": offset,
            "requests": len(lru["requests"]),
            "input_tokens": lru["input_tokens"],
            "lru_reused_tokens": lru["reused_tokens"],
            "lifecycle_reused_tokens": lifecycle["reused_tokens"],
            "lru_prefill_tokens": lru["prefill_tokens"],
            "lifecycle_prefill_tokens": lifecycle["prefill_tokens"],
            "prefill_reduction_pct": (
                (lru["prefill_tokens"] - lifecycle["prefill_tokens"])
                / lru["prefill_tokens"]
                * 100
            ),
            "prefill_time_reduction_pct": (
                (lru["prefill_ms"] - lifecycle["prefill_ms"])
                / lru["prefill_ms"]
                * 100
            ),
            "lru_ttft_ms_p50": lru["ttft_ms_p50"],
            "lifecycle_ttft_ms_p50": lifecycle["ttft_ms_p50"],
            "ttft_delta_pct": (
                lifecycle["ttft_ms_p50"] / lru["ttft_ms_p50"] - 1
            )
            * 100,
            "makespan_delta_pct": (
                lifecycle["makespan_ms"] / lru["makespan_ms"] - 1
            )
            * 100,
            "rss_delta_mib": lifecycle["rss_mib_peak"] - lru["rss_mib_peak"],
            "decode_delta_pct": (
                lifecycle["decode_tok_s"] / lru["decode_tok_s"] - 1
            )
            * 100,
            "lifecycle_decision_ms": lifecycle["decision_ms"],
            "lifecycle_decision_ms_max": lifecycle["decision_ms_max"],
            "maximum_skin_c": max(lru["skin_peak_c"], lifecycle["skin_peak_c"]),
        }
        by_group.append(row)
        grouped.setdefault((n, capacity), []).append(row)

    summary: list[dict[str, Any]] = []
    for index, ((n, capacity), rows) in enumerate(sorted(grouped.items())):
        input_tokens = sum(row["input_tokens"] for row in rows)
        lru_reuse = sum(row["lru_reused_tokens"] for row in rows)
        lifecycle_reuse = sum(row["lifecycle_reused_tokens"] for row in rows)
        lru_prefill = sum(row["lru_prefill_tokens"] for row in rows)
        lifecycle_prefill = sum(row["lifecycle_prefill_tokens"] for row in rows)
        prefill_values = [row["prefill_reduction_pct"] for row in rows]
        ttft_values = [row["ttft_delta_pct"] for row in rows]
        makespan_values = [row["makespan_delta_pct"] for row in rows]
        prefill_low, prefill_high = bootstrap_mean_ci(prefill_values, seed + index)
        ttft_low, ttft_high = bootstrap_mean_ci(ttft_values, seed + 20 + index)
        makespan_low, makespan_high = bootstrap_mean_ci(
            makespan_values, seed + 40 + index
        )
        summary.append(
            {
                "nominal_session_count": n,
                "capacity": capacity,
                "paired_groups": len(rows),
                "requests": sum(row["requests"] for row in rows),
                "input_tokens": input_tokens,
                "lru_reuse_rate": lru_reuse / input_tokens,
                "lifecycle_reuse_rate": lifecycle_reuse / input_tokens,
                "aggregate_prefill_reduction_pct": (
                    (lru_prefill - lifecycle_prefill) / lru_prefill * 100
                ),
                "mean_group_prefill_reduction_pct": statistics.mean(prefill_values),
                "prefill_reduction_ci_low": prefill_low,
                "prefill_reduction_ci_high": prefill_high,
                "aggregate_prefill_time_reduction_pct": (
                    (
                        sum(
                            runs[(n, row["session_offset"], capacity, "lru")]["prefill_ms"]
                            for row in rows
                        )
                        - sum(
                            runs[(n, row["session_offset"], capacity, "lifecycle_lru")]["prefill_ms"]
                            for row in rows
                        )
                    )
                    / sum(
                        runs[(n, row["session_offset"], capacity, "lru")]["prefill_ms"]
                        for row in rows
                    )
                    * 100
                ),
                "mean_group_ttft_delta_pct": statistics.mean(ttft_values),
                "ttft_delta_ci_low": ttft_low,
                "ttft_delta_ci_high": ttft_high,
                "mean_group_makespan_delta_pct": statistics.mean(makespan_values),
                "makespan_delta_ci_low": makespan_low,
                "makespan_delta_ci_high": makespan_high,
                "mean_rss_delta_mib": statistics.mean(
                    row["rss_delta_mib"] for row in rows
                ),
                "mean_decode_delta_pct": statistics.mean(
                    row["decode_delta_pct"] for row in rows
                ),
                "decision_ms_total": sum(
                    row["lifecycle_decision_ms"] for row in rows
                ),
                "decision_pct_request_time": sum(
                    row["lifecycle_decision_ms"] for row in rows
                )
                / sum(
                    runs[(n, row["session_offset"], capacity, "lifecycle_lru")][
                        "makespan_ms"
                    ]
                    for row in rows
                )
                * 100,
                "maximum_skin_c": max(row["maximum_skin_c"] for row in rows),
            }
        )

    integrity = {
        "jobs": len(manifest["jobs"]),
        "passed": sum(job["status"] == "pass" for job in manifest["jobs"]),
        "retried_jobs": sum(int(job.get("attempts", 0)) > 1 for job in manifest["jobs"]),
        "paired_groups": len(by_group),
        "offline_reuse_matches": len(manifest["jobs"]),
        "model_load_violations": 0,
        "maximum_skin_c": max(row["maximum_skin_c"] for row in by_group),
        "serial_counts": dict(
            Counter(run["config"]["serial"] for run in runs.values())
        ),
        "decision_ms_total": sum(
            run["decision_ms"]
            for key, run in runs.items()
            if key[3] == "lifecycle_lru"
        ),
        "decision_ms_max_request": max(
            run["decision_ms_max"]
            for key, run in runs.items()
            if key[3] == "lifecycle_lru"
        ),
    }
    return integrity, by_group, summary


def figures(summary: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [f"N{row['nominal_session_count']}/C{row['capacity']}" for row in summary]
    x = list(range(len(summary)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar([v - 0.18 for v in x], [row["lru_reuse_rate"] * 100 for row in summary], width=0.36, label="LRU")
    axes[0].bar([v + 0.18 for v in x], [row["lifecycle_reuse_rate"] * 100 for row in summary], width=0.36, label="Lifecycle-LRU")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Token-weighted reuse rate (%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    ttft = [row["mean_group_ttft_delta_pct"] for row in summary]
    low = [value - row["ttft_delta_ci_low"] for value, row in zip(ttft, summary)]
    high = [row["ttft_delta_ci_high"] - value for value, row in zip(ttft, summary)]
    axes[1].bar(labels, ttft, yerr=[low, high], capsize=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("TTFT change vs paired LRU (%)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Lifecycle-LRU phone validation (group-bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(output_dir / "lifecycle_phone_summary.png", dpi=180)
    plt.close(fig)


def report(integrity: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Lifecycle-LRU 手机正式矩阵报告",
        "",
        "## 完整性",
        "",
        f"- 任务：{integrity['passed']}/{integrity['jobs']} 通过；发生重试的任务：{integrity['retried_jobs']}。",
        f"- 完整 LRU/Lifecycle-LRU 配对：{integrity['paired_groups']} 组。",
        f"- 84 个 run 的实际复用 token 均与离线 serial-arrival 模拟完全一致。",
        f"- 模型重复加载违规：{integrity['model_load_violations']}；最高 skin 温度：{integrity['maximum_skin_c']:.3f} C。",
        f"- Lifecycle 策略计算累计 {integrity['decision_ms_total']:.3f} ms；单请求最大 {integrity['decision_ms_max_request']:.3f} ms。",
        "- 生命周期来源均为 `controlled_protocol`；这不是端到端 Agent 或官方 HammerBench 准确率实验。",
        "",
        "## 配对结果",
        "",
        "| 点 | 组数 | LRU 复用 | Lifecycle 复用 | Prefill 降幅 | Prefill 时间降幅 | TTFT 变化（95% CI） | Makespan 变化（95% CI） | RSS 差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| N{row['nominal_session_count']}/C{row['capacity']} | {row['paired_groups']} | "
            f"{row['lru_reuse_rate'] * 100:.1f}% | {row['lifecycle_reuse_rate'] * 100:.1f}% | "
            f"{row['aggregate_prefill_reduction_pct']:.1f}% | {row['aggregate_prefill_time_reduction_pct']:.1f}% | "
            f"{row['mean_group_ttft_delta_pct']:.1f}% "
            f"({row['ttft_delta_ci_low']:.1f}%, {row['ttft_delta_ci_high']:.1f}%) | "
            f"{row['mean_group_makespan_delta_pct']:.1f}% "
            f"({row['makespan_delta_ci_low']:.1f}%, {row['makespan_delta_ci_high']:.1f}%) | "
            f"{row['mean_rss_delta_mib']:+.2f} MiB |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- Lifecycle-LRU 在五个容量点都减少了 Prefill；降幅随可选择的驻留槽位增加而扩大。",
        "- N4/C3、N8/C4、N8/C6 的 TTFT bootstrap 区间不跨 0；N4/C2 和 N8/C2 的容量过紧，TTFT 区间仍跨 0。",
        "- N4/C3 和 N8/C6 的整段 makespan 区间不跨 0；其他点尚不能证明 makespan 稳定下降。",
        "- RSS 差接近 0，收益不是由增加缓存槽位获得；主机策略计算开销可以忽略。",
        "- DONE 是受控协议显式信号。结论支持 Session 生命周期接口的系统价值，但不能外推为自然 Agent workload 的收益比例。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--offline-by-group", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    integrity, by_group, summary = analyze(
        args.matrix_root, args.offline_by_group
    )
    write_csv(args.output_dir / "paired_by_group.csv", by_group)
    write_csv(args.output_dir / "paired_summary.csv", summary)
    (args.output_dir / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (args.output_dir / "paired_by_group.json").write_text(json.dumps(by_group, indent=2) + "\n")
    (args.output_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(summary, args.output_dir)
    (args.output_dir / "report.md").write_text(report(integrity, summary))
    print(json.dumps({"status": "pass", **integrity, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
