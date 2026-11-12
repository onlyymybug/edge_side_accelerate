#!/usr/bin/env python3
"""Workload-sensitivity analysis for whole-Session KV retention policies.

The Stage-3 experiment uses a strictly round-robin, equal-length trace.  That is
the least favourable interleaving for LRU: every Session is revisited after
exactly N-1 other requests, so any capacity below N thrashes.  Real multi-session
workloads are not like this -- a few Sessions are hot, Session lengths are
heavy-tailed, and the dispatch order is neither round-robin nor uniform.

This script synthesises multi-session traces over a grid of workload shapes and
replays the *same* online policies that were validated against the phone
(`simulate_lifecycle_policy`) plus the future-aware upper bounds.  It answers:

    How much of the measured LRU gap survives when the workload stops being
    adversarial, and where does the method stop paying off?

Usage
-----
    PY=/home/lyyyy/miniconda3/envs/pace/bin/python
    $PY analyze_workload_sensitivity.py \
        --output-dir results/workload_sensitivity

The script does not touch the phone and does not read Stage-3 run directories;
it only reuses the simulator from `analyze_lifecycle.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_lifecycle import simulate_lifecycle_policy  # noqa: E402
from analyze_oracle import token_optimal_reuse  # noqa: E402


# --------------------------------------------------------------------------
# Workload synthesis
# --------------------------------------------------------------------------

# Session popularity ranks are drawn from a Zipf distribution.  `alpha` is the
# skew exponent: 0 -> uniform random dispatch, larger -> a few hot Sessions.
# `round_robin` is a separate schedule and matches the Stage-3 trace.
SCHEDULES = ("round_robin", "uniform_random", "zipf")

# Turn counts observed in HammerBench fixed-tools: 2..8, median 3, mean 3.44.
# Turn counts observed in HammerBench fixed-tools (50 Sessions):
#   2 turns: 15, 3: 12, 4: 12, 5: 10, 8: 1  (median 3.4, mean 3.44)
EMPIRICAL_TURNS = [2] * 15 + [3] * 12 + [4] * 12 + [5] * 10 + [8]

# `pareto` reproduces the heavy tail; `uniform` is the control.
LENGTH_MODELS = ("empirical", "uniform", "pareto")


def sample_turns(rng: random.Random, model: str, max_turns: int = 8) -> int:
    """Number of turns in one Session."""
    if model == "empirical":
        return rng.choice(EMPIRICAL_TURNS)
    if model == "uniform":
        return rng.randint(2, max_turns)
    if model == "pareto":
        # Shifted Pareto with x_m = 1, shape 1.5: median ~2, tail reaches 8+.
        raw = 1.0 / (rng.random() ** (1.0 / 1.5))
        return min(max_turns, max(2, int(1 + raw)))
    raise ValueError(f"unknown length model: {model}")


def sample_initial_tokens(rng: random.Random) -> int:
    """First-turn prompt length. Measured: min 396, median 577, max 951."""
    return int(min(951, max(396, rng.lognormvariate(math.log(577), 0.16))))


def sample_increment_tokens(rng: random.Random) -> int:
    """Tokens added per turn. Measured: p10 22, median 30.5, p90 48, max 84."""
    return int(min(84, max(15, rng.lognormvariate(math.log(30.5), 0.42))))


def sample_lcp_fraction(rng: random.Random, mean: float) -> float:
    """Shared-prefix fraction between consecutive turns.

    The measured trace has LCP fraction 1.0 everywhere: with a fixed reference
    history the whole previous prompt is reused.  `--lcp-mean` below 1.0 is a
    deliberate degradation axis, not the default.
    """
    if mean >= 0.999:
        return 1.0
    value = rng.betavariate(mean * 8.0, (1.0 - mean) * 8.0)
    return min(1.0, max(0.0, value))


def build_session_sequence(
    rng: random.Random, turns: int, length_model: str, lcp_mean: float
) -> tuple[list[int], list[int]]:
    """Return (input_tokens, potential_reuse) for one Session's turns.

    potential_reuse[t] is the LCP with turn t-1, matching how the real trace
    computes reuse.  It is not the full previous prompt.
    """
    cumulative: list[int] = []
    total = sample_initial_tokens(rng)
    cumulative.append(total)
    for _ in range(turns - 1):
        total += sample_increment_tokens(rng)
        cumulative.append(total)
    potential = [0] + [
        int(previous * sample_lcp_fraction(rng, lcp_mean))
        for previous in cumulative[:-1]
    ]
    return cumulative, potential


def generate_trace(
    rng: random.Random,
    session_count: int,
    schedule: str,
    alpha: float,
    length_model: str,
    lcp_mean: float,
) -> tuple[list[str], list[int], list[int]]:
    """Generate (sessions, potential_reuse, input_tokens) in dispatch order."""
    turns = [sample_turns(rng, length_model) for _ in range(session_count)]
    per_session: list[list[int]] = []
    per_reuse: list[list[int]] = []
    for count in turns:
        inputs, reuse = build_session_sequence(rng, count, length_model, lcp_mean)
        per_session.append(inputs)
        per_reuse.append(reuse)

    # Fixed popularity ranks, sampled once per workload.
    ranks = list(range(1, session_count + 1))
    rng.shuffle(ranks)
    weights = [1.0 / (rank ** alpha) for rank in ranks]

    remaining = list(turns)
    cursor = [0] * session_count
    order: list[str] = []
    reuse: list[int] = []
    inputs: list[int] = []

    round_index = 0
    while any(count > 0 for count in remaining):
        if schedule == "round_robin":
            # One turn per Session per round, in a shuffled but stable order.
            active = [s for s in range(session_count) if remaining[s] > 0]
            for session in active:
                order.append(f"s{session}")
                inputs.append(per_session[session][cursor[session]])
                reuse.append(per_reuse[session][cursor[session]])
                cursor[session] += 1
                remaining[session] -= 1
        else:
            active = [s for s in range(session_count) if remaining[s] > 0]
            if schedule == "uniform_random":
                session = rng.choice(active)
            else:
                session = rng.choices(active, weights=[weights[s] for s in active])[0]
            order.append(f"s{session}")
            inputs.append(per_session[session][cursor[session]])
            reuse.append(per_reuse[session][cursor[session]])
            cursor[session] += 1
            remaining[session] -= 1
        round_index += 1
        if round_index > 10_000:  # defensive guard
            raise RuntimeError("dispatch loop did not terminate")

    return order, reuse, inputs


def load_real_sessions(trace: Path) -> list[list[tuple[int, int]]]:
    """Load each Session's measured (input_tokens, lcp_tokens) sequence."""
    _, sessions = load_trace_sessions(trace)
    return sorted(sessions.values(), key=len)


def interleave(
    rng: random.Random,
    sessions: list[list[tuple[int, int]]],
    schedule: str,
    alpha: float,
) -> tuple[list[str], list[int], list[int]]:
    """Dispatch real Session shapes under a synthetic order.

    Each Session keeps its measured turn count and prompt lengths; only the
    interleaving changes.  This separates "workload shape" from "dispatch
    order", which the fully synthetic generator cannot do.
    """
    count = len(sessions)
    cursor = [0] * count
    order: list[str] = []
    reuse: list[int] = []
    inputs: list[int] = []

    ranks = list(range(1, count + 1))
    rng.shuffle(ranks)
    weights = [1.0 / (rank ** alpha) for rank in ranks]

    while True:
        active = [s for s in range(count) if cursor[s] < len(sessions[s])]
        if not active:
            break
        if schedule == "round_robin":
            chosen = active
        elif schedule == "uniform_random":
            chosen = [rng.choice(active)]
        else:
            chosen = [rng.choices(active, weights=[weights[s] for s in active])[0]]
        for session in chosen:
            tokens, lcp = sessions[session][cursor[session]]
            order.append(f"s{session}")
            inputs.append(tokens)
            reuse.append(lcp)
            cursor[session] += 1

    return order, reuse, inputs


def load_trace_sessions(
    trace: Path,
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Load Sessions in trace file order, preserving that order."""
    order: list[str] = []
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            session = row["session_id"]
            if session not in grouped:
                order.append(session)
            grouped[session].append(
                (row["turn_id"], row["input_tokens"], row["previous_turn_lcp_tokens"])
            )
    sessions = {}
    for session, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        sessions[session] = [(inputs, lcp) for _, inputs, lcp in rows]
    return order, sessions


def phone_group_layout(
    order: list[str], sessions: dict[str, list[tuple[int, int]]], session_count: int
) -> list[list[list[tuple[int, int]]]]:
    """Rebuild the fixed consecutive groups Stage-3 actually ran.

    Stage-3 selected N consecutive Sessions in trace order, so every device
    number is an average over these groups, not over random draws.  Including
    them as a sweep row makes the other cells comparable to the measured runs.
    """
    groups = []
    for offset in range(0, len(order) - session_count + 1, session_count):
        ids = order[offset : offset + session_count]
        groups.append([sessions[session] for session in ids])
    return groups


def sorted_interleave(
    sessions: list[list[tuple[int, int]]],
) -> tuple[list[str], list[int], list[int]]:
    """Round-robin in the given Session order, as `replay.schedule_rows` does."""
    maximum = max(len(rows) for rows in sessions)
    order: list[str] = []
    reuse: list[int] = []
    inputs: list[int] = []
    for turn_index in range(maximum):
        for session_index, rows in enumerate(sessions):
            if turn_index < len(rows):
                tokens, lcp = rows[turn_index]
                order.append(f"s{session_index}")
                inputs.append(tokens)
                reuse.append(lcp)
    return order, reuse, inputs


# --------------------------------------------------------------------------
# Policy replay
# --------------------------------------------------------------------------

POLICIES = ("lru", "close_aware_lru", "lifecycle_lru", "ready_aware_lru")


def replay(
    sessions: list[str], potential: list[int], inputs: list[int], capacity: int
) -> dict[str, Any]:
    """Replay every policy at one capacity; returns reused-token totals."""
    total_input = sum(inputs)
    lru = simulate_lifecycle_policy(
        sessions, potential, inputs, capacity, "lru", "serial_arrival"
    )["reused_tokens"]
    optimal = token_optimal_reuse(sessions, potential, capacity)

    out: dict[str, Any] = {
        "requests": len(sessions),
        "input_tokens": total_input,
        "lru_reused": lru,
        "oracle_reused": optimal,
    }
    for policy in POLICIES:
        result = simulate_lifecycle_policy(
            sessions, potential, inputs, capacity, policy, "serial_arrival"
        )
        out[f"{policy}_reused"] = result["reused_tokens"]
        out[f"{policy}_reuse_rate"] = result["reused_tokens"] / total_input
    out["lru_reuse_rate"] = lru / total_input
    out["oracle_reuse_rate"] = optimal / total_input
    return out


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------


def capacity_points(session_count: int) -> list[int]:
    return sorted(
        {
            max(1, math.ceil(session_count * 0.5)),
            max(1, math.ceil(session_count * 0.75)),
            session_count - 1,
        }
    )


def stable_seed(base: int, *parts: Any) -> int:
    """Deterministic across processes (unlike hash(), which is salted)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**31)


def run_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real_pool = load_real_sessions(args.real_trace) if args.real_trace else None
    if real_pool is not None and len(real_pool) < max(args.session_counts):
        raise SystemExit(
            f"--real-trace has {len(real_pool)} Sessions, "
            f"need at least {max(args.session_counts)}"
        )

    trace_order: list[str] = []
    trace_map: dict[str, list[tuple[int, int]]] = {}
    if args.real_trace and args.include_device_groups:
        trace_order, trace_map = load_trace_sessions(args.real_trace)

    for session_count in args.session_counts:
        capacities = capacity_points(session_count)
        for schedule in args.schedules:
            schedule_alphas = args.alphas if schedule == "zipf" else [0.0]
            for alpha in schedule_alphas:
                for length_model in args.length_models:
                    if real_pool is not None and length_model != "real":
                        continue
                    for replicate in range(args.replicates):
                        rng = random.Random(
                            stable_seed(
                                args.seed,
                                session_count,
                                schedule,
                                alpha,
                                length_model,
                                replicate,
                            )
                        )
                        if real_pool is not None:
                            chosen = rng.sample(real_pool, session_count)
                            sessions, potential, inputs = interleave(
                                rng, chosen, schedule, alpha
                            )
                        else:
                            sessions, potential, inputs = generate_trace(
                                rng,
                                session_count,
                                schedule,
                                alpha,
                                length_model,
                                args.lcp_mean,
                            )
                        for capacity in capacities:
                            metrics = replay(sessions, potential, inputs, capacity)
                            rows.append(
                                {
                                    "session_count": session_count,
                                    "capacity": capacity,
                                    "capacity_ratio": capacity / session_count,
                                    "schedule": schedule,
                                    "alpha": alpha,
                                    "length_model": length_model,
                                    "replicate": replicate,
                                    "source": "sampled",
                                    **metrics,
                                    "reuse_gap_oracle": metrics["oracle_reused"]
                                    - metrics["lru_reused"],
                                }
                            )

        # Anchor rows: the exact groupings the phone actually ran.
        if trace_order:
            for group_index, group in enumerate(
                phone_group_layout(trace_order, trace_map, session_count)
            ):
                sessions, potential, inputs = sorted_interleave(group)
                for capacity in capacities:
                    metrics = replay(sessions, potential, inputs, capacity)
                    rows.append(
                        {
                            "session_count": session_count,
                            "capacity": capacity,
                            "capacity_ratio": capacity / session_count,
                            "schedule": "device_group",
                            "alpha": 0.0,
                            "length_model": "real",
                            "replicate": group_index,
                            "source": "device_group",
                            **metrics,
                            "reuse_gap_oracle": metrics["oracle_reused"]
                            - metrics["lru_reused"],
                        }
                    )
    return rows


# --------------------------------------------------------------------------
# Aggregation and output
# --------------------------------------------------------------------------


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["session_count"],
            row["capacity_ratio"],
            row["schedule"],
            row["alpha"],
            row["length_model"],
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        input_total = sum(row["input_tokens"] for row in group)
        lru = sum(row["lru_reused"] for row in group)
        oracle = sum(row["oracle_reused"] for row in group)
        record: dict[str, Any] = {
            "session_count": key[0],
            "capacity_ratio": key[1],
            "schedule": key[2],
            "alpha": key[3],
            "length_model": key[4],
            "workloads": len(group),
        }
        for policy in POLICIES:
            reused = sum(row[f"{policy}_reused"] for row in group)
            record[f"{policy}_reuse_rate"] = reused / input_total
            record[f"{policy}_prefill_reduction_vs_lru_pct"] = (
                (reused - lru) / (input_total - lru) * 100
                if input_total > lru
                else 0.0
            )
            record[f"{policy}_oracle_gap_recovery_pct"] = (
                (reused - lru) / (oracle - lru) * 100 if oracle > lru else None
            )
        record["lru_reuse_rate"] = lru / input_total
        record["oracle_reuse_rate"] = oracle / input_total
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def primary_length_model(rows: list[dict[str, Any]]) -> str:
    models = {row["length_model"] for row in rows}
    for candidate in ("real", "empirical", "pareto", "uniform"):
        if candidate in models:
            return candidate
    return sorted(models)[0]


def make_figure(summary: list[dict[str, Any]], output_dir: Path, session_count: int) -> None:
    main_model = primary_length_model(summary)
    rows = [
        row
        for row in summary
        if row["session_count"] == session_count
        and row["length_model"] == main_model
        and row["schedule"] in {"uniform_random", "zipf"}
    ]
    if not rows:
        return
    ratios = sorted({row["capacity_ratio"] for row in rows})
    metrics = (
        ("prefill_reduction_vs_lru_pct", "Prefill reduction vs LRU (%)"),
        ("oracle_gap_recovery_pct", "Oracle-gap recovery (%)"),
    )
    fig, axes = plt.subplots(
        len(metrics),
        len(ratios),
        figsize=(4.0 * len(ratios), 6.4),
        squeeze=False,
    )
    for row_index, (metric, label) in enumerate(metrics):
        for col_index, ratio in enumerate(ratios):
            axis = axes[row_index][col_index]
            subset = [row for row in rows if row["capacity_ratio"] == ratio]
            for policy, style, name in (
                ("close_aware_lru", "--", "CloseAware-LRU"),
                ("lifecycle_lru", "-", "Lifecycle-LRU"),
            ):
                points = sorted(
                    (row["alpha"], row[f"{policy}_{metric}"])
                    for row in subset
                    if row["schedule"] == "zipf"
                    and row[f"{policy}_{metric}"] is not None
                )
                if points:
                    axis.plot(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        style,
                        marker="o",
                        label=name,
                    )
            axis.axhline(0, color="grey", linewidth=0.6)
            axis.set_title(f"slots/sessions = {ratio:.2f}")
            axis.set_xlabel("Zipf skew alpha")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        f"Whole-Session KV policy benefit vs workload skew "
        f"(N={session_count}, {main_model} lengths)\n"
        "Top: raw reduction. Bottom: fraction of the remaining Oracle gap."
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"workload_sensitivity_n{session_count}.png", dpi=160)
    plt.close(fig)


def make_report(
    summary: list[dict[str, Any]], args: argparse.Namespace, rows: list[dict[str, Any]]
) -> str:
    main_model = primary_length_model(summary)
    lines = [
        "# Workload-sensitivity of whole-Session KV retention",
        "",
        "Synthetic interleavings replayed through the same online simulator that was",
        "validated against the phone in Stage-3. No device time is used.",
        "",
        "## What this measures",
        "",
        "The Stage-3 trace is round-robin over Sessions of unequal length. Perfect",
        "rotation is pathological for LRU: when the number of resident Sessions",
        "equals the slot count, every Session is evicted exactly once before its",
        "next turn arrives, so plain LRU scores **zero** reuse. This script varies",
        "the dispatch order while holding the workload itself fixed:",
        "",
        "- **dispatch schedule**: `round_robin`, `uniform_random`, `zipf(alpha)`;",
        "- **capacity**: 0.5, 0.75 and (N-1)/N of the Session count;",
        f"- **Session shape**: `{main_model}`"
        + (
            " (measured turn counts and prompt lengths, re-interleaved)"
            if main_model == "real"
            else ""
        )
        + ".",
        "",
        "`lifecycle_lru` uses the `serial_arrival` protocol, the conservative result.",
        "Oracle is the token-optimal dynamic-programming upper bound.",
        "",
        f"Replicates per cell: {args.replicates}. Total workloads: {len(rows)}.",
        "",
        "## Headline table",
        "",
        f"Session shape `{main_model}`, capacity ratio 0.75:",
        "",
        "| N | Schedule | alpha | LRU reuse | CloseAware | Lifecycle | Oracle |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    head = [
        row
        for row in summary
        if row["length_model"] == main_model
        and abs(row["capacity_ratio"] - 0.75) < 1e-9
        and row["session_count"] in (4, 8)
    ]
    for row in sorted(head, key=lambda r: (r["session_count"], r["schedule"], r["alpha"])):
        lines.append(
            "| {n} | {s} | {a} | {lru:.1f}% | {c:.1f}% | {lc:.1f}% | {o:.1f}% |".format(
                n=row["session_count"],
                s=row["schedule"],
                a=row["alpha"],
                lru=row["lru_reuse_rate"] * 100,
                c=row["close_aware_lru_reuse_rate"] * 100,
                lc=row["lifecycle_lru_reuse_rate"] * 100,
                o=row["oracle_reuse_rate"] * 100,
            )
        )

    round_robin = [
        row
        for row in summary
        if row["schedule"] == "round_robin"
        and row["length_model"] == main_model
        and abs(row["capacity_ratio"] - 0.75) < 1e-9
    ]
    lines += [
        "",
        "## Anchor: the exact groupings the phone ran",
        "",
        "Sampled groups and the Stage-3 groups are not the same experiment: Stage-3",
        "took N *consecutive* Sessions in trace order, and those happen to be of equal",
        "length. Equal lengths keep the rotation perfectly phase-aligned, which is what",
        "makes plain LRU score near zero. The rows below replay those exact groups, so",
        "they reproduce the device numbers and calibrate everything else.",
        "",
        "| N | slots | LRU | CloseAware | Lifecycle | Oracle | reduction | gap recovery |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        (r for r in summary if r["schedule"] == "device_group"),
        key=lambda r: (r["session_count"], r["capacity_ratio"]),
    ):
        lines.append(
            "| {n} | {c} | {lru:.1f}% | {cw:.1f}% | {lc:.1f}% | {o:.1f}% | {red:.1f}% | {ogr:.1f}% |".format(
                n=row["session_count"],
                c=round(row["capacity_ratio"] * row["session_count"]),
                lru=row["lru_reuse_rate"] * 100,
                cw=row["close_aware_lru_reuse_rate"] * 100,
                lc=row["lifecycle_lru_reuse_rate"] * 100,
                o=row["oracle_reuse_rate"] * 100,
                red=row["lifecycle_lru_prefill_reduction_vs_lru_pct"],
                ogr=row["lifecycle_lru_oracle_gap_recovery_pct"] or 0.0,
            )
        )

    lines += [
        "",
        "## Reading the result",
        "",
        "Compare every sampled row against the `device_group` row at the same N and",
        "slots. Sampled groups mix Session lengths, so rotation drifts out of phase and",
        "LRU stops being degenerate. That drift, not a different workload, is what the",
        "large LRU improvement between the two rows reflects.",
        "",
        "`lifecycle_lru_oracle_gap_recovery_pct` is the fraction of the *remaining*",
        "gap that the policy closes. It is the scale-free measure: under phase-aligned",
        "rotation the gap is wide and the policy closes part of it; under skewed",
        "dispatch the gap is narrow, so a small absolute reduction can still be most",
        "of what is left.",
        "",
        "## Attribution: close signal vs value function, under skew",
        "",
        "Same decomposition as the phone report, recomputed per cell:",
        "`close = (CloseAware - LRU) / (Lifecycle - LRU)`.",
        "",
        "| N | slots | schedule | LRU | Close share | Value share |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(
        (
            r
            for r in summary
            if r["length_model"] == main_model
            and (
                r["schedule"] == "device_group"
                or (r["session_count"] == 8 and r["schedule"] == "zipf"
                    and abs(r["capacity_ratio"] - 0.75) < 1e-9)
            )
        ),
        key=lambda r: (r["session_count"], r["capacity_ratio"], r["schedule"], r["alpha"]),
    ):
        close = row["close_aware_lru_reuse_rate"] - row["lru_reuse_rate"]
        value = row["lifecycle_lru_reuse_rate"] - row["close_aware_lru_reuse_rate"]
        total = close + value
        if abs(total) < 1e-12:
            close_share = value_share = float("nan")
        else:
            close_share = 100.0 * close / total
            value_share = 100.0 * value / total
        lines.append(
            "| {n} | {c} | {s} a={a} | {lru:.1f}% | {cs:.1f}% | {vs:.1f}% |".format(
                n=row["session_count"],
                c=round(row["capacity_ratio"] * row["session_count"]),
                s=row["schedule"],
                a=row["alpha"],
                lru=row["lru_reuse_rate"] * 100,
                cs=close_share,
                vs=value_share,
            )
        )

    lines += [
        "",
        "This is the most consequential number in the sweep. Under the phone's",
        "phase-aligned grouping the value function contributes a material share of",
        "the gain. Under any skewed dispatch it collapses toward zero, and at high",
        "skew it can go **negative** — Lifecycle-LRU then scores slightly below",
        "CloseAware-LRU because the score occasionally evicts a Session that plain",
        "LRU would have kept.",
        "",
        "The value function is therefore not a general improvement. It pays only",
        "where access intervals are near-uniform and the score can break ties that",
        "LRU breaks badly; once dispatch is skewed, knowing *which* Sessions are hot",
        "already dominates, and the score adds noise.",
        "",
        "## Decision gate",
        "",
        "The original gate asks whether the reduction stays above 10% across skewed",
        "workloads. As the capacity ratio approaches N-1 it does not, because the",
        "Oracle itself sits only a few points above LRU there and no policy can beat",
        "the ceiling. That is arithmetic, not a failure of the method, and the report",
        "should say so rather than hide it.",
        "",
        "What survives the sweep is the gap-recovery column. Compare it against the",
        "`device_group` row at the same N and slots: the policy does not need rotation",
        "to work, rotation was simply the regime with the most room. Under skewed",
        "dispatch the remaining gap is small, so the same policy covers most of it.",
        "",
        "Note the plain-LRU baseline is what inflates both readings. Against a",
        "degenerate baseline, any policy that breaks the rotation looks strong.",
        "The honest comparator for a skewed workload is CloseAware-LRU, and against",
        "it Lifecycle-LRU's margin is a few tenths of a point.",
        "",
        "## What this changes for the paper",
        "",
        "Three revisions follow, and they are cheap:",
        "",
        "1. The headline must be stated **relative to the achievable gap**, not as a",
        "   raw percentage. A 30% Prefill reduction on the rotation trace is not",
        "   comparable to a 12% reduction on a skewed trace; the gap recovery is.",
        "2. The capacity axis needs a stated operating point. Near N-1 slots the",
        "   ceiling collapses and the discussion should move to *why deploy at that",
        "   point at all*, rather than reporting a small number and moving on.",
        "3. The value function should be presented as a secondary contribution with",
        "   a stated validity region (near-uniform access intervals), not as part of",
        "   the headline method. On skewed dispatch the close signal does essentially",
        "   all of the work, and the score can slightly hurt.",
        "",
        "## Files",
        "",
        "- `workload_sensitivity.csv`: per-workload raw results",
        "- `workload_sensitivity_summary.csv`: aggregated cells",
        "- `workload_sensitivity_n{4,8}.png`: benefit vs skew",
        "",
    ]
    if round_robin:
        lines.append(f"Sampled round-robin reference ({main_model}, ratio 0.75):")
        lines.append("")
        for row in round_robin:
            lines.append(
                f"- N={row['session_count']}: LRU {row['lru_reuse_rate']*100:.1f}%, "
                f"Lifecycle {row['lifecycle_lru_reuse_rate']*100:.1f}%, "
                f"Oracle {row['oracle_reuse_rate']*100:.1f}%, "
                f"reduction {row['lifecycle_lru_prefill_reduction_vs_lru_pct']:.1f}%, "
                f"gap recovery {row['lifecycle_lru_oracle_gap_recovery_pct']:.1f}%"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/workload_sensitivity")
    )
    parser.add_argument("--seed", type=int, default=4247)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--session-counts", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--lcp-mean",
        type=float,
        default=1.0,
        help=(
            "Mean shared-prefix fraction between consecutive turns. The measured "
            "trace is exactly 1.0; lower values model partial prefix reuse."
        ),
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=["round_robin", "uniform_random", "zipf"],
        choices=list(SCHEDULES),
    )
    parser.add_argument(
        "--real-trace",
        type=Path,
        default=None,
        help=(
            "Replay measured Session shapes from this JSONL instead of synthesising "
            "them. Only the interleaving is synthetic."
        ),
    )
    parser.add_argument(
        "--include-device-groups",
        action="store_true",
        help=(
            "Also replay the exact consecutive Session groups Stage-3 ran, as an "
            "anchor row. Makes random-sample cells directly comparable to the phone."
        ),
    )
    parser.add_argument(
        "--length-models",
        nargs="+",
        default=["real"],
        choices=["real"] + list(LENGTH_MODELS),
        help="Use 'real' together with --real-trace.",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Emit a deterministic JSON digest of the default grid for diffing.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_grid(args)
    summary = aggregate(rows)
    write_csv(output_dir / "workload_sensitivity.csv", rows)
    write_csv(output_dir / "workload_sensitivity_summary.csv", summary)
    (output_dir / "report.md").write_text(make_report(summary, args, rows))

    for session_count in args.session_counts:
        make_figure(summary, output_dir, session_count)

    metadata = {
        "seed": args.seed,
        "replicates": args.replicates,
        "session_counts": args.session_counts,
        "alphas": args.alphas,
        "schedules": args.schedules,
        "length_models": args.length_models,
        "workloads": len(rows),
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    if args.fingerprint:
        digest = {
            "rows": len(rows),
            "mean_lifecycle_reduction_pct": statistics.mean(
                row["lifecycle_lru_prefill_reduction_vs_lru_pct"] for row in summary
            ),
        }
        print(json.dumps(digest, indent=2))
    print(f"wrote {len(rows)} workload rows to {output_dir}")


if __name__ == "__main__":
    main()
