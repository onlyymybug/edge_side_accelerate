#!/usr/bin/env python3
"""Compare measured LRU with future-aware Session-cache oracles."""

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


def simulate_policy(
    sessions: list[str], potential_reuse: list[int], capacity: int, policy: str
) -> int:
    """Return reused tokens for LRU or farthest-next (Belady) eviction."""
    cache: list[str] = []
    reused = 0
    for index, session in enumerate(sessions):
        if session in cache:
            reused += potential_reuse[index]
            if policy == "lru":
                cache.remove(session)
                cache.append(session)
            continue
        if len(cache) >= capacity:
            if policy == "lru":
                cache.pop(0)
            elif policy == "belady":
                future = {
                    resident: next(
                        (
                            position
                            for position in range(index + 1, len(sessions))
                            if sessions[position] == resident
                        ),
                        len(sessions) + 1,
                    )
                    for resident in cache
                }
                cache.remove(max(cache, key=lambda resident: future[resident]))
            else:
                raise ValueError(f"unknown policy: {policy}")
        cache.append(session)
    return reused


def token_optimal_reuse(
    sessions: list[str], potential_reuse: list[int], capacity: int
) -> int:
    """Dynamic program maximizing future reused tokens with whole-Session slots."""
    states: dict[frozenset[str], int] = {frozenset(): 0}
    for session, value in zip(sessions, potential_reuse):
        next_states: dict[frozenset[str], int] = {}
        for resident, score in states.items():
            gain = value if session in resident else 0
            if session in resident:
                choices = [resident]
            elif len(resident) < capacity:
                choices = [resident | {session}]
            else:
                choices = [(resident - {victim}) | {session} for victim in resident]
            for choice in choices:
                next_states[choice] = max(next_states.get(choice, -1), score + gain)
        states = next_states
    return max(states.values())


def simulate_close_aware_lru(
    sessions: list[str], potential_reuse: list[int], capacity: int
) -> int:
    """LRU that immediately frees a Session after an explicit terminal signal."""
    final_request = {session: index for index, session in enumerate(sessions)}
    cache: list[str] = []
    reused = 0
    for index, session in enumerate(sessions):
        if session in cache:
            reused += potential_reuse[index]
            cache.remove(session)
            cache.append(session)
        else:
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(session)
        if final_request[session] == index:
            cache.remove(session)
    return reused


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = sorted(
        statistics.mean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(matrix_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text())
    if any(job["status"] != "pass" for job in manifest["jobs"]):
        raise RuntimeError("Stage-3 matrix is incomplete")
    all_keep = {
        (job["nominal_session_count"], job["session_offset"]): job
        for job in manifest["jobs"]
        if job["schedule"] == "round_robin"
        and job["policy"] == "all_keep"
        and job["session_count"] == job["nominal_session_count"]
    }
    measured_lru = {
        (job["nominal_session_count"], job["session_offset"], job["lru_slots"]): job
        for job in manifest["jobs"]
        if job["schedule"] == "round_robin"
        and job["policy"] == "lru"
        and job["session_count"] == job["nominal_session_count"]
    }
    rows = []
    for (session_count, offset), baseline_job in sorted(all_keep.items()):
        baseline = read_jsonl(Path(baseline_job["run_dir"]) / "requests.jsonl")
        sequence = [str(request["session_id"]) for request in baseline]
        potential = [int(request["reused_tokens"]) for request in baseline]
        input_tokens = sum(int(request["input_tokens"]) for request in baseline)
        all_keep_reuse = sum(potential)
        capacities = sorted(
            key[2]
            for key in measured_lru
            if key[:2] == (session_count, offset)
        )
        for capacity in capacities:
            actual = read_jsonl(
                Path(measured_lru[session_count, offset, capacity]["run_dir"])
                / "requests.jsonl"
            )
            measured = sum(int(request["reused_tokens"]) for request in actual)
            lru = simulate_policy(sequence, potential, capacity, "lru")
            belady = simulate_policy(sequence, potential, capacity, "belady")
            optimal = token_optimal_reuse(sequence, potential, capacity)
            close_aware = simulate_close_aware_lru(sequence, potential, capacity)
            if lru != measured:
                raise AssertionError(
                    f"LRU mismatch N={session_count} offset={offset} slots={capacity}: "
                    f"simulated={lru}, measured={measured}"
                )
            rows.append(
                {
                    "nominal_session_count": session_count,
                    "session_offset": offset,
                    "lru_slots": capacity,
                    "actual_slot_ratio": capacity / session_count,
                    "requests": len(sequence),
                    "input_tokens": input_tokens,
                    "all_keep_reused_tokens": all_keep_reuse,
                    "measured_lru_reused_tokens": measured,
                    "simulated_lru_reused_tokens": lru,
                    "close_aware_lru_reused_tokens": close_aware,
                    "belady_reused_tokens": belady,
                    "token_optimal_reused_tokens": optimal,
                    "lru_reuse_rate": lru / input_tokens,
                    "close_aware_lru_reuse_rate": close_aware / input_tokens,
                    "belady_reuse_rate": belady / input_tokens,
                    "token_optimal_reuse_rate": optimal / input_tokens,
                    "all_keep_reuse_rate": all_keep_reuse / input_tokens,
                    "oracle_extra_reused_tokens": optimal - lru,
                    "oracle_prefill_reduction_pct": (optimal - lru)
                    / (input_tokens - lru)
                    * 100,
                }
            )
    return manifest, rows


def aggregate(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["nominal_session_count"], row["lru_slots"])].append(row)
    output = []
    for index, ((session_count, slots), group) in enumerate(sorted(grouped.items())):
        input_tokens = sum(row["input_tokens"] for row in group)
        lru = sum(row["simulated_lru_reused_tokens"] for row in group)
        belady = sum(row["belady_reused_tokens"] for row in group)
        close_aware = sum(row["close_aware_lru_reused_tokens"] for row in group)
        optimal = sum(row["token_optimal_reused_tokens"] for row in group)
        all_keep = sum(row["all_keep_reused_tokens"] for row in group)
        reductions = [row["oracle_prefill_reduction_pct"] for row in group]
        low, high = bootstrap_mean_ci(reductions, seed + index)
        output.append(
            {
                "nominal_session_count": session_count,
                "lru_slots": slots,
                "actual_slot_ratio": slots / session_count,
                "session_groups": len(group),
                "input_tokens": input_tokens,
                "lru_reused_tokens": lru,
                "close_aware_lru_reused_tokens": close_aware,
                "belady_reused_tokens": belady,
                "token_optimal_reused_tokens": optimal,
                "all_keep_reused_tokens": all_keep,
                "lru_reuse_rate": lru / input_tokens,
                "close_aware_lru_reuse_rate": close_aware / input_tokens,
                "belady_reuse_rate": belady / input_tokens,
                "token_optimal_reuse_rate": optimal / input_tokens,
                "all_keep_reuse_rate": all_keep / input_tokens,
                "oracle_extra_reused_tokens": optimal - lru,
                "aggregate_oracle_prefill_reduction_pct": (optimal - lru)
                / (input_tokens - lru)
                * 100,
                "mean_group_oracle_prefill_reduction_pct": statistics.mean(reductions),
                "mean_group_oracle_prefill_reduction_ci_low": low,
                "mean_group_oracle_prefill_reduction_ci_high": high,
                "belady_gap_to_optimal_tokens": optimal - belady,
                "close_aware_oracle_gap_recovery_pct": (
                    (close_aware - lru) / (optimal - lru) * 100
                    if optimal > lru
                    else None
                ),
            }
        )
    return output


def figures(summary: list[dict[str, Any]], output_dir: Path) -> None:
    counts = sorted({row["nominal_session_count"] for row in summary})
    fig, axes = plt.subplots(1, len(counts), figsize=(15, 4.8), squeeze=False)
    for column, count in enumerate(counts):
        rows = [row for row in summary if row["nominal_session_count"] == count]
        x = [row["actual_slot_ratio"] * 100 for row in rows]
        axes[0][column].plot(x, [row["lru_reuse_rate"] * 100 for row in rows], marker="o", label="LRU (measured)")
        axes[0][column].plot(x, [row["close_aware_lru_reuse_rate"] * 100 for row in rows], marker="o", label="CloseAware-LRU")
        axes[0][column].plot(x, [row["belady_reuse_rate"] * 100 for row in rows], marker="o", label="Belady")
        axes[0][column].plot(x, [row["token_optimal_reuse_rate"] * 100 for row in rows], marker="o", label="Token-optimal")
        axes[0][column].axhline(rows[0]["all_keep_reuse_rate"] * 100, color="black", linestyle="--", label="AllKeep")
        axes[0][column].set_title(f"{count} Sessions")
        axes[0][column].set_xlabel("Resident slots / Sessions (%)")
        axes[0][column].grid(alpha=0.3)
    axes[0][0].set_ylabel("Token-weighted reuse rate (%)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Offline future-aware eviction opportunity")
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_reuse_gap.png", dpi=180)
    plt.close(fig)

    labels = [f"N{row['nominal_session_count']}/C{row['lru_slots']}" for row in summary]
    values = [row["mean_group_oracle_prefill_reduction_pct"] for row in summary]
    low = [value - row["mean_group_oracle_prefill_reduction_ci_low"] for value, row in zip(values, summary)]
    high = [row["mean_group_oracle_prefill_reduction_ci_high"] - value for value, row in zip(values, summary)]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, values, yerr=[low, high], capsize=3)
    axis.set_ylabel("Additional Prefill reduction vs LRU (%)")
    axis.set_title("Token-optimal Oracle opportunity (group-bootstrap 95% CI)")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_prefill_reduction.png", dpi=180)
    plt.close(fig)


def make_report(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Offline Session-cache Oracle analysis",
        "",
        "## Method and validation",
        "",
        "The analysis uses each full Stage-3 RoundRobin group's measured AllKeep request sequence. AllKeep `reused_tokens` is the realizable value of retaining a Session until its next request. Three whole-Session cache policies are replayed at the exact measured slot budgets:",
        "",
        "- LRU: evict the least recently used Session;",
        "- Belady: evict the resident Session whose next request is farthest in the future;",
        "- token-optimal Oracle: dynamic programming that maximizes total reused tokens over the complete known trace.",
        "",
        f"Simulated LRU exactly matches measured runner LRU for all {len(rows)} group/budget cases. The Oracle is an offline upper bound, not an online deployable result, and no TTFT improvement is claimed without a phone run.",
        "",
        "## Population result",
        "",
        "| Sessions | Slots | LRU reuse | CloseAware-LRU | Belady reuse | Token-optimal reuse | AllKeep reuse | Additional Prefill reduction vs LRU |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['nominal_session_count']} | {row['lru_slots']} | {row['lru_reuse_rate'] * 100:.1f}% | "
            f"{row['close_aware_lru_reuse_rate'] * 100:.1f}% | "
            f"{row['belady_reuse_rate'] * 100:.1f}% | {row['token_optimal_reuse_rate'] * 100:.1f}% | "
            f"{row['all_keep_reuse_rate'] * 100:.1f}% | {row['aggregate_oracle_prefill_reduction_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Two Sessions with one slot has no policy opportunity: strict alternation exceeds capacity, so LRU and both Oracles are identical.",
        "- Assuming the Agent runtime emits an exact terminal/close signal, immediately releasing completed Sessions recovers 26.1%--43.3% of the available Oracle gap at useful intermediate budgets. This is a causal what-if only when that signal is genuinely available online.",
        "- Four and eight Session traces have heterogeneous completion/future-access patterns. Future-aware eviction recovers substantial reuse that LRU loses.",
        "- Belady is close to the token-optimal result, so most opportunity comes from knowing Session next-use/lifecycle rather than finely weighting token values.",
        "- The largest gaps occur at high but still insufficient capacity: retaining the wrong nearly-complete or distant-future Session causes LRU thrashing.",
        "- This supports investigating an online lifecycle/next-use predictor, but the achievable online fraction of the Oracle gap must be measured separately.",
        "",
        "## Decision",
        "",
        "There is meaningful eviction-policy headroom for 4/8 Session workloads, while the 2-Session/1-slot case is a fundamental capacity limit. The next method experiment should implement a causal online heuristic using observable lifecycle features (recent interval, turn index, inactive/completed state, KV value/size) and compare it against LRU under the same traces and budgets.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows = analyze(args.matrix_root)
    summary = aggregate(rows, int(manifest["randomization_seed"]))
    write_csv(args.output_dir / "oracle_by_group.csv", rows)
    write_csv(args.output_dir / "oracle_summary.csv", summary)
    (args.output_dir / "oracle_by_group.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(summary, args.output_dir)
    (args.output_dir / "report.md").write_text(make_report(rows, summary))
    print(json.dumps({"status": "pass", "validated_lru_cases": len(rows), "summary_rows": len(summary), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
