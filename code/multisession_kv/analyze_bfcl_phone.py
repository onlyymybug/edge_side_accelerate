#!/usr/bin/env python3
"""Analyze completed BFCL V4 LRU/Lifecycle-LRU phone pairs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_lifecycle_phone import bootstrap_mean_ci, load_run, write_csv


def analyze(matrix_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text())
    jobs = manifest["jobs"]
    if not jobs or any(job["status"] != "pass" for job in jobs):
        raise RuntimeError("matrix is not complete; every job must pass before paired analysis")
    by_key: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    model_hashes: set[str] = set()
    trace_hashes: set[str] = set()
    for job in jobs:
        key = (
            int(job["nominal_session_count"]),
            int(job["session_offset"]),
            int(job["lru_slots"]),
            int(job["repeat"]),
        )
        if job["policy"] in by_key[key]:
            raise ValueError(f"duplicate policy for pair {key}")
        run = load_run(job)
        by_key[key][job["policy"]] = run
        model_hashes.add(run["config"]["model_sha256"])
        trace_hashes.add(run["config"]["trace_sha256"])
    if len(model_hashes) != 1 or len(trace_hashes) != 1:
        raise ValueError("paired jobs used different model or trace hashes")

    pairs: list[dict[str, Any]] = []
    for (n, offset, capacity, repeat), policies in sorted(by_key.items()):
        if set(policies) != {"lru", "lifecycle_lru"}:
            raise ValueError(f"missing LRU/Lifecycle-LRU pair for {(n, offset, capacity, repeat)}")
        lru = policies["lru"]
        lifecycle = policies["lifecycle_lru"]
        if [r["source_snapshot_id"] for r in lru["requests"]] != [
            r["source_snapshot_id"] for r in lifecycle["requests"]
        ]:
            raise ValueError(f"paired request sequences differ for {(n, offset, capacity, repeat)}")
        if lru["input_tokens"] != lifecycle["input_tokens"]:
            raise ValueError(f"paired input tokens differ for {(n, offset, capacity, repeat)}")
        pairs.append(
            {
                "nominal_session_count": n,
                "session_offset": offset,
                "capacity": capacity,
                "repeat": repeat,
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
                "decision_ms": lifecycle["decision_ms"],
                "maximum_skin_c": max(lru["skin_peak_c"], lifecycle["skin_peak_c"]),
            }
        )

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(pair["nominal_session_count"], pair["capacity"])].append(pair)
    summary: list[dict[str, Any]] = []
    for index, ((n, capacity), rows) in enumerate(sorted(groups.items())):
        input_tokens = sum(r["input_tokens"] for r in rows)
        lru_prefill = sum(r["lru_prefill_tokens"] for r in rows)
        lifecycle_prefill = sum(r["lifecycle_prefill_tokens"] for r in rows)
        prefill = [r["prefill_reduction_pct"] for r in rows]
        ttft = [r["ttft_delta_pct"] for r in rows]
        makespan = [r["makespan_delta_pct"] for r in rows]
        prefill_ci = bootstrap_mean_ci(prefill, 6144 + index) if len(rows) > 1 else (None, None)
        ttft_ci = bootstrap_mean_ci(ttft, 6244 + index) if len(rows) > 1 else (None, None)
        makespan_ci = bootstrap_mean_ci(makespan, 6344 + index) if len(rows) > 1 else (None, None)
        summary.append(
            {
                "nominal_session_count": n,
                "capacity": capacity,
                "paired_groups": len(rows),
                "input_tokens": input_tokens,
                "lru_reuse_rate": sum(r["lru_reused_tokens"] for r in rows) / input_tokens,
                "lifecycle_reuse_rate": sum(r["lifecycle_reused_tokens"] for r in rows) / input_tokens,
                "aggregate_prefill_reduction_pct": (lru_prefill - lifecycle_prefill) / lru_prefill * 100,
                "mean_group_prefill_reduction_pct": statistics.mean(prefill),
                "prefill_ci_low": prefill_ci[0],
                "prefill_ci_high": prefill_ci[1],
                "mean_group_ttft_delta_pct": statistics.mean(ttft),
                "ttft_ci_low": ttft_ci[0],
                "ttft_ci_high": ttft_ci[1],
                "mean_group_makespan_delta_pct": statistics.mean(makespan),
                "makespan_ci_low": makespan_ci[0],
                "makespan_ci_high": makespan_ci[1],
                "mean_rss_delta_mib": statistics.mean(r["rss_delta_mib"] for r in rows),
                "decision_ms_total": sum(r["decision_ms"] for r in rows),
                "maximum_skin_c": max(r["maximum_skin_c"] for r in rows),
            }
        )
    integrity = {
        "jobs": len(jobs),
        "passed": len(jobs),
        "paired_groups": len(pairs),
        "model_sha256": next(iter(model_hashes)),
        "trace_sha256": next(iter(trace_hashes)),
        "bootstrap_unit": "fixed session group",
        "bootstrap_samples": 10000,
    }
    return integrity, pairs, summary


def report(integrity: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    def interval(row: dict[str, Any], metric: str) -> str:
        value = row[f"mean_group_{metric}_delta_pct"]
        low = row[f"{metric}_ci_low"]
        high = row[f"{metric}_ci_high"]
        return (
            f"{value:.1f}%（单组，无区间）"
            if low is None or high is None
            else f"{value:.1f}% ({low:.1f}%, {high:.1f}%)"
        )

    lines = [
        "# BFCL V4 手机配对分析",
        "",
        f"- 已通过 {integrity['passed']}/{integrity['jobs']} 个任务，形成 {integrity['paired_groups']} 组配对。",
        "- 本实验使用 BFCL V4 的完整工具定义、官方 ground truth 调用及模拟工具返回构造固定参考历史；结果不是官方 BFCL 准确率。",
        "- 策略生命周期信号来自受控协议。单组 Pilot 不计算置信区间。",
        "",
        "| 点 | 组数 | LRU 复用率 | Lifecycle 复用率 | Prefill 降幅 | TTFT 变化（95% CI） | Makespan 变化（95% CI） | RSS 差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| N{row['nominal_session_count']}/C{row['capacity']} | {row['paired_groups']} | "
            f"{row['lru_reuse_rate'] * 100:.1f}% | {row['lifecycle_reuse_rate'] * 100:.1f}% | "
            f"{row['aggregate_prefill_reduction_pct']:.1f}% | "
            f"{interval(row, 'ttft')} | "
            f"{interval(row, 'makespan')} | "
            f"{row['mean_rss_delta_mib']:+.2f} MiB |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    integrity, pairs, summary = analyze(args.matrix_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_by_group.csv", pairs)
    write_csv(args.output_dir / "paired_summary.csv", summary)
    (args.output_dir / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report(integrity, summary))
    print(json.dumps({"status": "pass", **integrity, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
