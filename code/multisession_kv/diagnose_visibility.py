#!/usr/bin/env python3
"""Probe why the ready-set term contributes nothing, and whether LCP's near
zero marginal value is a formula artefact or a property of the information.

Three questions:
1. Is the ready set ever informative? Count, at every eviction decision, how
   often residents differ in readiness.
2. Does the ready bonus matter at any weight?
3. Is LCP's marginal value suppressed by collinearity with the cached-share
   term, or does it stay near zero when that term is removed?
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import analyze_visibility_decomposition as vis
from analyze_oracle import simulate_policy, token_optimal_reuse
from analyze_workload_sensitivity import (
    load_trace_sessions,
    phone_group_layout,
    sorted_interleave,
)


def readiness_stats(
    order: list[str], capacity: int
) -> Counter:
    """At each eviction, how do residents split on readiness?"""
    next_at: list[dict[str, int]] = [{} for _ in order]
    future: dict[str, int] = {}
    for index in range(len(order) - 1, -1, -1):
        next_at[index] = dict(future)
        future[order[index]] = index

    final = {s: i for i, s in enumerate(order)}
    cache: list[str] = []
    counts: Counter = Counter()
    for index, session in enumerate(order):
        if session in cache:
            cache.remove(session)
            cache.append(session)
        else:
            if len(cache) >= capacity:
                ready = [r for r in cache if r in next_at[index]]
                counts["evictions"] += 1
                if len(ready) == len(cache):
                    counts["all_ready"] += 1
                elif len(ready) == 0:
                    counts["none_ready"] += 1
                else:
                    counts["mixed"] += 1
            cache.append(session)
        if final[session] == index and session in cache:
            cache.remove(session)
    return counts


def sweep_weights(
    order: list[str],
    reuse: list[int],
    inputs: list[int],
    capacity: int,
) -> dict[str, Any]:
    """Vary the ready weight and the LCP/cached coupling."""
    lru = simulate_policy(order, reuse, capacity, "lru")
    oracle = token_optimal_reuse(order, reuse, capacity)
    total_input = sum(inputs)

    def reduce(level: str, **kwargs: Any) -> float:
        result = vis.simulate_visibility(
            order, reuse, inputs, capacity, level, **kwargs
        )
        reused = result["reused_tokens"]
        denom = total_input - lru
        return (reused - lru) / denom * 100 if denom > 0 else 0.0

    out: dict[str, Any] = {"lru": lru, "oracle": oracle, "input": total_input}
    for weight in (0.0, 0.5, 2.0, 10.0, 100.0):
        out[f"set_w{weight:g}"] = reduce("set", w_ready=weight)
    for level in ("none", "set", "rank", "lcp", "rank+lcp"):
        out[f"level_{level}"] = reduce(level)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/hammerbench_en_50/fixed_tools.jsonl"),
    )
    parser.add_argument(
        "--points", default="4:2,4:3,8:2,8:4,8:6,8:7"
    )
    args = parser.parse_args()

    points: set[tuple[int, int]] = set()
    for item in args.points.split(","):
        n, c = item.strip().lower().replace("n", "").replace("c", "").split(":")
        points.add((int(n), int(c)))

    order, sessions = load_trace_sessions(args.trace)
    groups: dict[int, list[list[list[tuple[int, int]]]]] = {}
    for n in sorted({x for x, _ in points}):
        groups[n] = phone_group_layout(order, sessions, n)

    print("=== readiness at eviction decisions (phone groupings) ===")
    for (n, c) in sorted(points):
        total: Counter = Counter()
        for rows in groups[n]:
            seq, _, _ = sorted_interleave(rows)
            total.update(readiness_stats(seq, c))
        ev = total["evictions"]
        if ev == 0:
            print(f"N={n} C={c}: no evictions")
            continue
        print(
            f"N={n} C={c}: {ev:4d} evictions | "
            f"all-ready {total['all_ready']/ev:6.1%} | "
            f"mixed {total['mixed']/ev:6.1%} | "
            f"none-ready {total['none_ready']/ev:6.1%}"
        )

    print()
    print("=== ready-weight sweep: does the set term ever move? ===")
    for (n, c) in sorted(points):
        agg: dict[str, list[float]] = {}
        for rows in groups[n]:
            seq, reuse, inputs = sorted_interleave(rows)
            res = sweep_weights(seq, reuse, inputs, c)
            for key, value in res.items():
                if isinstance(value, float):
                    agg.setdefault(key, []).append(value)
        if not agg:
            continue
        mean = {k: sum(v) / len(v) for k, v in agg.items()}
        weights = [k for k in mean if k.startswith("set_w")]
        spread = max(mean[k] for k in weights) - min(mean[k] for k in weights)
        print(
            f"N={n} C={c}: "
            + " ".join(f"{k.split('_w')[1]}={mean[k]:.1f}" for k in weights)
            + f"  | spread {spread:.2f} pts"
        )

    print()
    print("=== full ladder and LCP-vs-cached coupling ===")
    for (n, c) in sorted(points):
        agg: dict[str, list[float]] = {}
        for rows in groups[n]:
            seq, reuse, inputs = sorted_interleave(rows)
            res = sweep_weights(seq, reuse, inputs, c)
            for key, value in res.items():
                if key.startswith("level_") and isinstance(value, float):
                    agg.setdefault(key, []).append(value)
        mean = {k: sum(v) / len(v) for k, v in agg.items()}
        print(
            f"N={n} C={c}: "
            + " ".join(f"{k[6:]}={mean[k]:5.1f}" for k in sorted(mean))
        )


if __name__ == "__main__":
    main()
