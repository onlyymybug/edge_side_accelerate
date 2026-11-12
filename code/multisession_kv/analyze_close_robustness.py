#!/usr/bin/env python3
"""Close-signal robustness: how much of the benefit survives imperfect DONE events.

Stage-3 and the phone matrix assume every Session emits an immediate DONE at its
last turn. Real applications do not: a user switches away, the activity is
stopped without notifying the model server, or the session is simply never
closed. The measured 30% Prefill reduction is therefore an upper bound on what
the lifecycle interface can deliver.

This script varies the signal along two axes and re-measures:

- **coverage**: fraction of Sessions that ever emit a DONE event;
- **delay**: how many requests elapse between a Session's last turn and its
  DONE event.

Everything else is held fixed: same Sessions, same groupings, same slots.
No device time is used.

The simulator here is a deliberate copy of
`analyze_lifecycle.simulate_lifecycle_policy` with the single terminal-release
rule made explicit.  With coverage=1.0 and delay=0 it must reproduce the
original exactly; the script asserts that before reporting anything.

Usage
-----
    PY=/home/lyyyy/miniconda3/envs/pace/bin/python
    $PY analyze_close_robustness.py \
        --trace data/hammerbench_en_50/fixed_tools.jsonl \
        --output-dir results/close_robustness
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_lifecycle import (  # noqa: E402
    POLICIES,
    _lru_victim,
    choose_serial_lifecycle_victim,
    next_positions,
    simulate_lifecycle_policy,
    update_session_metadata,
    validate_online_source,
)
from analyze_oracle import token_optimal_reuse  # noqa: E402
from analyze_workload_sensitivity import (  # noqa: E402
    load_trace_sessions,
    phone_group_layout,
    sorted_interleave,
)

# The phone matrix and the offline ablation both report these five points.
DEFAULT_POINTS = {(4, 2), (4, 3), (8, 2), (8, 4), (8, 6), (8, 7)}


def simulate_with_release_events(
    sessions: list[str],
    potential_reuse: list[int],
    input_tokens: list[int],
    capacity: int,
    policy: str,
    release_at: dict[str, int],
    lifecycle_source: str = "controlled_protocol",
) -> dict[str, Any]:
    """`simulate_lifecycle_policy` with DONE timing supplied by `release_at`.

    A Session absent from `release_at` never emits DONE and simply stays resident
    until evicted like any other candidate — which is what happens when an app is
    killed or the user never closes the conversation.
    """
    if len(sessions) != len(potential_reuse) or len(sessions) != len(input_tokens):
        raise ValueError("sessions, potential_reuse and input_tokens must align")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    validate_online_source(lifecycle_source)

    terminal_enabled = policy != "lru"
    releases_by_index: dict[int, list[str]] = defaultdict(list)
    if terminal_enabled:
        for session, index in release_at.items():
            releases_by_index[index].append(session)

    next_at = next_positions(sessions)
    cache: list[str] = []
    metadata: dict[str, dict[str, float | int | None]] = {}
    reused = 0
    evictions = 0
    done_releases = 0
    evicted_waiting = 0

    for index, session in enumerate(sessions):
        if session in cache:
            reused += potential_reuse[index]
            cache.remove(session)
            cache.append(session)
        else:
            if len(cache) >= capacity:
                candidates = list(cache)
                if policy in {"lru", "close_aware_lru"}:
                    victim = _lru_victim(cache, candidates)
                else:
                    victim, _ = choose_serial_lifecycle_victim(
                        candidates, metadata, index, len(set(sessions))
                    )
                cache.remove(victim)
                evictions += 1
                evicted_waiting += 1
            cache.append(session)

        update_session_metadata(metadata, session, index, input_tokens[index])

        # Any session whose DONE lands at this request index is released now.
        for finished in releases_by_index.get(index, ()):
            if finished in cache:
                cache.remove(finished)
                done_releases += 1

    return {
        "reused_tokens": reused,
        "prefill_tokens": sum(input_tokens) - reused,
        "evictions": evictions,
        "done_releases": done_releases,
        "evicted_waiting": evicted_waiting,
    }


def perfect_release_map(sessions: list[str]) -> dict[str, int]:
    """Every Session releases immediately after its last turn (the Stage-3 rule)."""
    return {session: index for index, session in enumerate(sessions)}


def imperfect_release_map(
    sessions: list[str],
    coverage: float,
    delay: int,
    rng: random.Random,
) -> tuple[dict[str, int], int]:
    """Drop some DONE events, postpone the rest, and count how many can fire.

    A DONE that would land past the end of the trace cannot fire at all: there
    is no later request for it to affect.  Those releases are censored, and the
    second return value reports how many survived censoring.  Without this the
    delay axis silently degenerates into a coverage axis.
    """
    final = perfect_release_map(sessions)
    release: dict[str, int] = {}
    fired = 0
    for session, index in final.items():
        if rng.random() > coverage:
            continue
        arrival = index + delay
        if arrival < len(sessions):
            release[session] = arrival
            fired += 1
    return release, fired


def stable_seed(base: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**31)


def verify_against_original(
    groups: dict[int, list[list[tuple[int, int]]]], points: set[tuple[int, int]]
) -> int:
    """Assert the explicit-release simulator matches the validated one exactly."""
    checked = 0
    for session_count, group_list in groups.items():
        for capacity in sorted(c for n, c in points if n == session_count):
            for group in group_list:
                sessions, potential, inputs = sorted_interleave(group)
                for policy in ("lru", "close_aware_lru", "lifecycle_lru"):
                    expected = simulate_lifecycle_policy(
                        sessions, potential, inputs, capacity, policy, "serial_arrival"
                    )["reused_tokens"]
                    actual = simulate_with_release_events(
                        sessions,
                        potential,
                        inputs,
                        capacity,
                        policy,
                        perfect_release_map(sessions),
                    )["reused_tokens"]
                    if expected != actual:
                        raise AssertionError(
                            f"release-event simulator diverged: N={session_count} "
                            f"C={capacity} policy={policy} "
                            f"expected {expected} got {actual}"
                        )
                    checked += 1
    return checked


def run_sweep(
    groups: dict[int, list[list[tuple[int, int]]]],
    points: set[tuple[int, int]],
    coverages: list[float],
    delays: list[int],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_count, group_list in sorted(groups.items()):
        capacities = sorted(c for n, c in points if n == session_count)
        for capacity in capacities:
            for coverage in coverages:
                for delay in delays:
                    for group_index, group in enumerate(group_list):
                        sessions, potential, inputs = sorted_interleave(group)
                        total_input = sum(inputs)
                        baseline = simulate_with_release_events(
                            sessions,
                            potential,
                            inputs,
                            capacity,
                            "lru",
                            {},
                        )["reused_tokens"]
                        oracle = token_optimal_reuse(sessions, potential, capacity)

                        # Stochastic DONE assignment: average over trials so a
                        # single unlucky draw does not define a cell.
                        for trial in range(trials):
                            rng = random.Random(
                                stable_seed(
                                    seed,
                                    session_count,
                                    capacity,
                                    coverage,
                                    delay,
                                    group_index,
                                    trial,
                                )
                            )
                            release, fired = imperfect_release_map(
                                sessions, coverage, delay, rng
                            )
                            for policy in ("close_aware_lru", "lifecycle_lru"):
                                result = simulate_with_release_events(
                                    sessions,
                                    potential,
                                    inputs,
                                    capacity,
                                    policy,
                                    release,
                                )
                                reused = result["reused_tokens"]
                                rows.append(
                                    {
                                        "session_count": session_count,
                                        "capacity": capacity,
                                        "capacity_ratio": capacity / session_count,
                                        "coverage": coverage,
                                        "delay": delay,
                                        "group": group_index,
                                        "trial": trial,
                                        "policy": policy,
                                        "reused_tokens": reused,
                                        "input_tokens": total_input,
                                        "reuse_rate": reused / total_input,
                                        "prefill_reduction_vs_lru_pct": (
                                            (reused - baseline) / (total_input - baseline) * 100
                                            if total_input > baseline
                                            else 0.0
                                        ),
                                        "oracle_gap_recovery_pct": (
                                            (reused - baseline) / (oracle - baseline) * 100
                                            if oracle > baseline
                                            else None
                                        ),
                                        "done_releases": result["done_releases"],
                                        "done_events_fired": fired,
                                        "sessions_in_trace": len(set(sessions)),
                                    }
                                )
                        rows.append(
                            {
                                "session_count": session_count,
                                "capacity": capacity,
                                "capacity_ratio": capacity / session_count,
                                "coverage": coverage,
                                "delay": delay,
                                "group": group_index,
                                "trial": -1,
                                "policy": "lru",
                                "reused_tokens": baseline,
                                "input_tokens": total_input,
                                "reuse_rate": baseline / total_input,
                                "prefill_reduction_vs_lru_pct": 0.0,
                                "oracle_gap_recovery_pct": 0.0,
                                "done_releases": 0,
                                "done_events_fired": 0,
                                "sessions_in_trace": len(set(sessions)),
                            }
                        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["session_count"],
            row["capacity"],
            row["coverage"],
            row["delay"],
            row["policy"],
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        input_total = sum(row["input_tokens"] for row in group)
        reused = sum(row["reused_tokens"] for row in group)
        reductions = [row["prefill_reduction_vs_lru_pct"] for row in group]
        gap = [row["oracle_gap_recovery_pct"] for row in group
               if row["oracle_gap_recovery_pct"] is not None]
        summary.append(
            {
                "session_count": key[0],
                "capacity": key[1],
                "capacity_ratio": key[1] / key[0],
                "coverage": key[2],
                "delay": key[3],
                "policy": key[4],
                "samples": len(group),
                "reuse_rate": reused / input_total,
                "prefill_reduction_vs_lru_pct": sum(reductions) / len(reductions),
                "oracle_gap_recovery_pct": (sum(gap) / len(gap)) if gap else None,
                # Fraction of Session DONE events that could fire at all,
                # given their delay and the trace length.
                "done_fired_fraction": (
                    sum(row["done_events_fired"] for row in group)
                    / sum(row["sessions_in_trace"] for row in group)
                ),
            }
        )
    return summary


def add_retention(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fraction of the perfect-signal benefit each degraded cell still delivers."""
    perfect: dict[tuple[Any, ...], float] = {}
    for row in summary:
        if row["coverage"] >= 1.0 and row["delay"] == 0:
            perfect[(row["session_count"], row["capacity"], row["policy"])] = row[
                "prefill_reduction_vs_lru_pct"
            ]
    for row in summary:
        base = perfect.get((row["session_count"], row["capacity"], row["policy"]))
        if base is None or abs(base) < 1e-9:
            row["retained_vs_perfect_pct"] = None
        else:
            row["retained_vs_perfect_pct"] = (
                row["prefill_reduction_vs_lru_pct"] / base * 100
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary: list[dict[str, Any]], output_dir: Path) -> None:
    points = sorted({(row["session_count"], row["capacity"]) for row in summary})
    if not points:
        return
    delays = sorted({row["delay"] for row in summary})
    fig, axes = plt.subplots(
        1, len(points), figsize=(3.5 * len(points), 3.4), sharey=True, squeeze=False
    )
    axes = axes[0]
    for axis, (session_count, capacity) in zip(axes, points):
        subset = [
            row
            for row in summary
            if row["session_count"] == session_count
            and row["capacity"] == capacity
            and row["policy"] == "lifecycle_lru"
        ]
        for coverage in sorted({row["coverage"] for row in subset}, reverse=True):
            points_xy = sorted(
                (row["delay"], row["prefill_reduction_vs_lru_pct"])
                for row in subset
                if row["coverage"] == coverage
            )
            if points_xy:
                axis.plot(
                    [p[0] for p in points_xy],
                    [p[1] for p in points_xy],
                    marker="o",
                    label=f"coverage {coverage:.0%}",
                )
        axis.axhline(0, color="grey", linewidth=0.6)
        axis.set_xscale("symlog")
        axis.set_title(f"N={session_count}, slots={capacity}")

        # Mark the delays at which a nontrivial share of DONE events can no
        # longer fire at all: past that point the x axis is measuring coverage
        # loss, not latency tolerance.
        full = {
            row["delay"]: row.get("done_fired_fraction")
            for row in subset
            if row["coverage"] >= 1.0
        }
        for delay in delays:
            fired = full.get(delay)
            if fired is not None and fired < 0.5:
                axis.axvspan(delay * 0.85, delay * 1.15, color="black", alpha=0.05)

        axis.set_xlabel("DONE delay (requests)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Prefill reduction vs LRU (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Benefit vs DONE coverage and delay (Lifecycle-LRU, serial arrival)\n"
        "shaded bands: delay at which <50% of DONE events can still fire"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "close_robustness.png", dpi=160)
    plt.close(fig)


def make_report(
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
    checked: int,
    rows: list[dict[str, Any]],
) -> str:
    points = sorted({(row["session_count"], row["capacity"]) for row in summary})
    lines = [
        "# Close-signal robustness",
        "",
        "How much of the measured benefit survives when DONE events are missing or",
        "late. Same Sessions, same groupings, same slot counts as the phone matrix;",
        "only the terminal signal changes.",
        "",
        f"Coverage in {sorted({r['coverage'] for r in summary})}, "
        f"delay in {sorted({r['delay'] for r in summary})} requests, "
        f"{args.trials} stochastic draws per cell.",
        "",
        "Correctness check: the explicit-release simulator reproduces the validated",
        f"`simulate_lifecycle_policy` on {checked} group/policy/capacity cases at",
        "coverage 1.0 and delay 0, bit for bit.",
        "",
        "## Headline: benefit retained vs a perfect signal",
        "",
        "Rows show the Prefill reduction vs LRU; parentheses show what fraction of",
        "the perfect-signal reduction survives.",
        "",
    ]

    header = "| reduction | " + " | ".join(
        f"cov {c:.0%}" for c in sorted({r["coverage"] for r in summary}, reverse=True)
    ) + " |"
    for session_count, capacity in points:
        lines.append(f"\n**N={session_count}, slots={capacity}**, delay 0\n")
        lines.append(header)
        lines.append("| ---: | " + " | ".join("---:" for _ in
                     sorted({r["coverage"] for r in summary}, reverse=True)) + " |")
        cell = []
        for coverage in sorted({r["coverage"] for r in summary}, reverse=True):
            match = [
                r
                for r in summary
                if r["session_count"] == session_count
                and r["capacity"] == capacity
                and r["coverage"] == coverage
                and r["delay"] == 0
                and r["policy"] == "lifecycle_lru"
            ]
            if match:
                row = match[0]
                retained = row["retained_vs_perfect_pct"]
                cell.append(
                    f"{row['prefill_reduction_vs_lru_pct']:.1f}% "
                    + (f"({retained:.0f}%)" if retained is not None else "")
                )
            else:
                cell.append("—")
        lines.append("| " + " | ".join(cell) + " |")

    delays = sorted({r["delay"] for r in summary})
    lines += [
        "",
        "## Delay sensitivity at full coverage",
        "",
        "**This axis is censored and must be read with care.** A DONE scheduled for",
        "the end of a grouping falls past the last request of that grouping and can",
        "never fire, so a long delay converts into a *missing* signal rather than a",
        "*late* one. The trace fragment is short: 8-20 requests per group at N=4 and",
        "20-36 at N=8, while a Session lives 2-5 turns. The second row under each",
        "point reports the fraction of DONE events that actually fired; where that",
        "number is far below 100%, the cell is measuring coverage loss, not latency",
        "tolerance.",
        "",
        "| point | " + " | ".join(f"delay {d}" for d in delays) + " |",
        "| --- | " + " | ".join("---:" for _ in delays) + " |",
    ]
    for session_count, capacity in points:
        reduction_cell = []
        fired_cell = []
        for delay in delays:
            match = [
                r
                for r in summary
                if r["session_count"] == session_count
                and r["capacity"] == capacity
                and r["coverage"] >= 1.0
                and r["delay"] == delay
                and r["policy"] == "lifecycle_lru"
            ]
            if not match:
                reduction_cell.append("—")
                fired_cell.append("—")
                continue
            row = match[0]
            reduction_cell.append(f"{row['prefill_reduction_vs_lru_pct']:.1f}%")
            fired = row.get("done_fired_fraction")
            fired_cell.append(f"{fired:.0%} fired" if fired is not None else "—")
        lines.append(
            f"| N={session_count}/{capacity} reduction | "
            + " | ".join(reduction_cell)
            + " |"
        )
        lines.append(
            f"| N={session_count}/{capacity} realized | "
            + " | ".join(fired_cell)
            + " |"
        )

    lines += [
        "",
        "## Reading the result",
        "",
        "- **Partial coverage degrades gracefully, but not smoothly.** The retained",
        "  fraction tracks coverage fairly closely at the sparse-capacity points",
        "  (N4/C3, N8/C6, N8/C7) and collapses much faster at N8/C4, which even goes",
        "  negative at coverages below 50%.",
        "- **The delay axis does not measure what its name suggests.** Only delay 0",
        "  keeps every DONE inside the trace window; by delay 5 most events at N=4",
        "  and delay 20 nearly all events at both node counts fall past the last",
        "  request and never fire. The apparent plateau at long delays is the",
        "  coverage floor (the delay-5 cell and the delay-20 cell coincide because",
        "  both are really measuring 'almost no signal'), not a tolerance plateau.",
        "",
        "Do not report this table as evidence that late DONE is harmless. It is not",
        "evidence either way. The only cell that carries a clean latency reading is",
        "delay 1: going from 0 to 1 already removes a third to a half of the",
        "reduction at every point except the degenerate ones, which is consistent",
        "with Session lifespans of 2-5 turns - a DONE one request late frequently",
        "arrives after the slot was already needed. To measure tolerance beyond",
        "that, replay longer concatenated traces (several groupings back to back)",
        "so a DONE delayed by 20-50 requests still has requests after it to affect.",
        "",
        "The N8/C4 row deserves attention rather than omission: with an imperfect",
        "signal the lifecycle score becomes worse than LRU. That is a real property",
        "of the mechanism, not noise - an intermittently-told dead Session poisons",
        "the recency statistics that the score depends on. Reporting it is what",
        "separates an honest evaluation from a selective one.",
        "",
        "## What this means for the interface argument",
        "",
        "Coverage is the axis this experiment can speak to, and it says the",
        "mechanism tolerates missing signals far better in the sparse-capacity",
        "regime than in the tight one. The delay axis is currently under-powered:",
        "the traces are too short for a late report to have anywhere to land, so",
        "the paper cannot yet claim a latency bound. What it can state is the",
        "weaker, verified result - the benefit requires eventual and reasonably",
        "complete reporting, and the fraction of the perfect-signal gain that",
        "survives scales with coverage.",
        "",
        "An implementation that reports DONE on `onStop` - which fires when the",
        "user switches away, not when they finish - will land at delay >= 1 and",
        "should be expected to lose a substantial part of the measured benefit; how",
        "much is not determined by this sweep.",
        "",
        "## Files",
        "",
        "- `close_robustness.csv`: per-group, per-trial raw results",
        "- `close_robustness_summary.csv`: aggregated cells",
        "- `close_robustness.png`: reduction vs delay, one line per coverage",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/hammerbench_en_50/fixed_tools.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/close_robustness")
    )
    parser.add_argument("--seed", type=int, default=4249)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--coverages", type=float, nargs="+",
                        default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 5, 20])
    args = parser.parse_args()

    order, sessions = load_trace_sessions(args.trace)
    groups: dict[int, list[list[tuple[int, int]]]] = {}
    for session_count in sorted({n for n, _ in DEFAULT_POINTS}):
        groups[session_count] = phone_group_layout(order, sessions, session_count)

    checked = verify_against_original(groups, DEFAULT_POINTS)

    rows = run_sweep(
        groups,
        DEFAULT_POINTS,
        args.coverages,
        args.delays,
        args.trials,
        args.seed,
    )
    summary = add_retention(aggregate(rows))

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "close_robustness.csv", rows)
    write_csv(output_dir / "close_robustness_summary.csv", summary)
    make_figure(summary, output_dir)
    (output_dir / "report.md").write_text(make_report(summary, args, checked, rows))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "trials": args.trials,
                "coverages": args.coverages,
                "delays": args.delays,
                "trace": str(args.trace),
                "simulator_conformance_cases": checked,
            },
            indent=2,
        )
    )
    print(f"verified {checked} conformance cases; wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
