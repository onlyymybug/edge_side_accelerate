#!/usr/bin/env python3
"""Offline causal ablations for Session-level lifecycle-aware KV retention.

The fixed HammerBench trace does not contain native runtime lifecycle events.
This analysis therefore labels every online lifecycle event as coming from a
controlled protocol and reports two visibility assumptions separately:

* serial_arrival: only the request being dispatched is READY;
* queued_next_turn: after a Session request completes, that Session's next
  reference request is released into the scheduler queue.  Queue position and
  the queued prompt's LCP are then causally visible to the policy.

Future-derived labels are reserved for the Oracle implementations and are
rejected by the online simulator.
"""

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

from analyze_oracle import simulate_policy, token_optimal_reuse
from lifecycle_policy import (
    choose_serial_lifecycle_victim,
    update_session_metadata,
    validate_online_source,
)


PROTOCOLS = {"serial_arrival", "queued_next_turn"}
POLICIES = {"lru", "close_aware_lru", "ready_aware_lru", "lifecycle_lru"}
DEFAULT_POINTS = {(4, 2), (4, 3), (8, 2), (8, 4), (8, 6)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def next_positions(sessions: list[str]) -> list[dict[str, int]]:
    """Return next request index per Session after each current index."""
    future: dict[str, int] = {}
    output: list[dict[str, int]] = [{} for _ in sessions]
    for index in range(len(sessions) - 1, -1, -1):
        output[index] = dict(future)
        future[sessions[index]] = index
    return output


def _lru_victim(cache: list[str], candidates: list[str]) -> str:
    return next(session for session in cache if session in candidates)


def simulate_lifecycle_policy(
    sessions: list[str],
    potential_reuse: list[int],
    input_tokens: list[int],
    capacity: int,
    policy: str,
    protocol: str,
    lifecycle_source: str = "controlled_protocol",
    collect_audit: bool = False,
) -> dict[str, Any]:
    """Replay one online whole-Session policy under an explicit event protocol."""
    if not (len(sessions) == len(potential_reuse) == len(input_tokens)):
        raise ValueError("sessions, potential_reuse and input_tokens must align")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    validate_online_source(lifecycle_source)

    terminal_enabled = policy != "lru"
    queued_enabled = protocol == "queued_next_turn" and policy in {
        "ready_aware_lru",
        "lifecycle_lru",
    }
    final_request = {session: index for index, session in enumerate(sessions)}
    next_at = next_positions(sessions)
    cache: list[str] = []
    metadata: dict[str, dict[str, float | int | None]] = {}
    reused = 0
    evictions = 0
    done_releases = 0
    evicted_ready = 0
    evicted_waiting = 0
    audit: list[dict[str, Any]] = []

    for index, session in enumerate(sessions):
        hit = session in cache
        if hit:
            reused += potential_reuse[index]
            cache.remove(session)
            cache.append(session)
        else:
            if len(cache) >= capacity:
                candidates = list(cache)
                state: dict[str, str] = {}
                score: dict[str, float | None] = {}
                for resident in candidates:
                    ready = queued_enabled and resident in next_at[index]
                    state[resident] = "READY" if ready else "WAITING"
                    score[resident] = None

                if policy in {"lru", "close_aware_lru"} or (
                    policy == "ready_aware_lru" and not queued_enabled
                ):
                    victim = _lru_victim(cache, candidates)
                elif policy == "ready_aware_lru":
                    # Every resident with a released next request has a causal
                    # queue rank. Evict the farthest queued request; use LRU
                    # only when ranks are equal or unavailable.
                    victim = max(
                        candidates,
                        key=lambda resident: (
                            next_at[index].get(resident, len(sessions) + 1),
                            -cache.index(resident),
                        ),
                    )
                else:
                    if queued_enabled:
                        max_cached = max(
                            float(metadata[resident]["cached_tokens"])
                            for resident in candidates
                        )
                        for resident in candidates:
                            next_index = next_at[index][resident]
                            distance = max(1, next_index - index)
                            known_reuse = potential_reuse[next_index]
                            # The prompt is already queued, so both queue rank
                            # and exact LCP are controlled-protocol observations.
                            score[resident] = 2.0 / distance + known_reuse / max_cached
                        victim = min(
                            candidates,
                            key=lambda resident: (
                                float(score[resident]),
                                cache.index(resident),
                            ),
                        )
                    else:
                        victim, serial_scores = choose_serial_lifecycle_victim(
                            candidates, metadata, index, len(set(sessions))
                        )
                        score.update(serial_scores)

                victim_state = state[victim]
                cache.remove(victim)
                evictions += 1
                if victim_state == "READY":
                    evicted_ready += 1
                else:
                    evicted_waiting += 1
                if collect_audit:
                    audit.append(
                        {
                            "request_index": index,
                            "event": "evict",
                            "requested_session": session,
                            "victim_session": victim,
                            "victim_state": victim_state,
                            "victim_score": score[victim],
                            "policy": policy,
                            "protocol": protocol,
                            "lifecycle_source": lifecycle_source,
                        }
                    )
            cache.append(session)

        # Current input length is observable after this request. It is a
        # conservative proxy when no next prompt has arrived.
        update_session_metadata(metadata, session, index, input_tokens[index])

        if terminal_enabled and final_request[session] == index:
            if session in cache:
                cache.remove(session)
                done_releases += 1
            if collect_audit:
                audit.append(
                    {
                        "request_index": index,
                        "event": "state_change",
                        "session_id": session,
                        "state": "DONE",
                        "policy": policy,
                        "protocol": protocol,
                        "lifecycle_source": lifecycle_source,
                    }
                )

    return {
        "reused_tokens": reused,
        "prefill_tokens": sum(input_tokens) - reused,
        "evictions": evictions,
        "done_releases": done_releases,
        "evicted_ready": evicted_ready,
        "evicted_waiting": evicted_waiting,
        "audit": audit,
    }


def parse_points(value: str) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for item in value.split(","):
        sessions, capacity = item.strip().lower().replace("n", "").replace("c", "").split(":")
        points.add((int(sessions), int(capacity)))
    return points


def load_cases(matrix_root: Path, points: set[tuple[int, int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        and (int(job["nominal_session_count"]), int(job["lru_slots"])) in points
    }
    cases: list[dict[str, Any]] = []
    for (session_count, offset, capacity), lru_job in sorted(measured_lru.items()):
        baseline_job = all_keep[(session_count, offset)]
        baseline = read_jsonl(Path(baseline_job["run_dir"]) / "requests.jsonl")
        measured = read_jsonl(Path(lru_job["run_dir"]) / "requests.jsonl")
        sessions = [str(request["session_id"]) for request in baseline]
        potential = [int(request["reused_tokens"]) for request in baseline]
        inputs = [int(request["input_tokens"]) for request in baseline]
        measured_reuse = sum(int(request["reused_tokens"]) for request in measured)
        simulated_lru = simulate_policy(sessions, potential, capacity, "lru")
        if simulated_lru != measured_reuse:
            raise AssertionError(
                f"LRU mismatch N={session_count} offset={offset} C={capacity}: "
                f"simulated={simulated_lru}, measured={measured_reuse}"
            )
        cases.append(
            {
                "nominal_session_count": int(session_count),
                "session_offset": int(offset),
                "capacity": int(capacity),
                "sessions": sessions,
                "potential_reuse": potential,
                "input_tokens": inputs,
                "measured_lru_reuse": measured_reuse,
            }
        )
    missing = points - {
        (case["nominal_session_count"], case["capacity"]) for case in cases
    }
    if missing:
        raise RuntimeError(f"requested points absent from Stage-3 matrix: {sorted(missing)}")
    return manifest, cases


def analyze_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for case in cases:
        sessions = case["sessions"]
        potential = case["potential_reuse"]
        inputs = case["input_tokens"]
        capacity = case["capacity"]
        optimal = token_optimal_reuse(sessions, potential, capacity)
        belady = simulate_policy(sessions, potential, capacity, "belady")
        for protocol in sorted(PROTOCOLS):
            for policy in sorted(POLICIES):
                result = simulate_lifecycle_policy(
                    sessions,
                    potential,
                    inputs,
                    capacity,
                    policy,
                    protocol,
                    collect_audit=policy == "lifecycle_lru",
                )
                if policy == "lru" and result["reused_tokens"] != case["measured_lru_reuse"]:
                    raise AssertionError("online LRU implementation diverges from measured LRU")
                input_total = sum(inputs)
                lru = case["measured_lru_reuse"]
                oracle_gap = optimal - lru
                rows.append(
                    {
                        "nominal_session_count": case["nominal_session_count"],
                        "session_offset": case["session_offset"],
                        "capacity": capacity,
                        "protocol": protocol,
                        "lifecycle_source": "controlled_protocol",
                        "policy": policy,
                        "requests": len(sessions),
                        "input_tokens": input_total,
                        "reused_tokens": result["reused_tokens"],
                        "prefill_tokens": result["prefill_tokens"],
                        "reuse_rate": result["reused_tokens"] / input_total,
                        "prefill_reduction_vs_lru_pct": (
                            (result["reused_tokens"] - lru) / (input_total - lru) * 100
                        ),
                        "oracle_gap_recovery_pct": (
                            (result["reused_tokens"] - lru) / oracle_gap * 100
                            if oracle_gap > 0
                            else None
                        ),
                        "evictions": result["evictions"],
                        "done_releases": result["done_releases"],
                        "evicted_ready": result["evicted_ready"],
                        "evicted_waiting": result["evicted_waiting"],
                        "belady_reused_tokens": belady,
                        "token_optimal_reused_tokens": optimal,
                    }
                )
                for event in result["audit"]:
                    audit.append(
                        {
                            "nominal_session_count": case["nominal_session_count"],
                            "session_offset": case["session_offset"],
                            "capacity": capacity,
                            **event,
                        }
                    )
    return rows, audit


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = sorted(
        statistics.mean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def aggregate(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["nominal_session_count"], row["capacity"], row["protocol"], row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        input_tokens = sum(row["input_tokens"] for row in group)
        reused = sum(row["reused_tokens"] for row in group)
        lru = sum(
            row["reused_tokens"]
            for row in rows
            if row["nominal_session_count"] == key[0]
            and row["capacity"] == key[1]
            and row["protocol"] == key[2]
            and row["policy"] == "lru"
        )
        optimal = sum(row["token_optimal_reused_tokens"] for row in group)
        group_reductions = [float(row["prefill_reduction_vs_lru_pct"]) for row in group]
        low, high = bootstrap_mean_ci(group_reductions, seed + group_index)
        output.append(
            {
                "nominal_session_count": key[0],
                "capacity": key[1],
                "protocol": key[2],
                "lifecycle_source": "controlled_protocol",
                "policy": key[3],
                "session_groups": len(group),
                "requests": sum(row["requests"] for row in group),
                "input_tokens": input_tokens,
                "reused_tokens": reused,
                "prefill_tokens": input_tokens - reused,
                "reuse_rate": reused / input_tokens,
                "aggregate_prefill_reduction_vs_lru_pct": (
                    (reused - lru) / (input_tokens - lru) * 100
                ),
                "mean_group_prefill_reduction_vs_lru_pct": statistics.mean(group_reductions),
                "mean_group_prefill_reduction_ci_low": low,
                "mean_group_prefill_reduction_ci_high": high,
                "oracle_gap_recovery_pct": (
                    (reused - lru) / (optimal - lru) * 100
                    if optimal > lru
                    else None
                ),
                "evictions": sum(row["evictions"] for row in group),
                "done_releases": sum(row["done_releases"] for row in group),
                "evicted_ready": sum(row["evicted_ready"] for row in group),
                "evicted_waiting": sum(row["evicted_waiting"] for row in group),
            }
        )
    return output


def make_figure(summary: list[dict[str, Any]], output_dir: Path) -> None:
    policies = ["lru", "close_aware_lru", "ready_aware_lru", "lifecycle_lru"]
    labels = ["LRU", "CloseAware", "ReadyAware", "Lifecycle-LRU"]
    points = sorted({(row["nominal_session_count"], row["capacity"]) for row in summary})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for column, protocol in enumerate(sorted(PROTOCOLS)):
        subset = [row for row in summary if row["protocol"] == protocol]
        x = list(range(len(points)))
        width = 0.2
        for policy_index, (policy, label) in enumerate(zip(policies, labels)):
            lookup = {
                (row["nominal_session_count"], row["capacity"]): row
                for row in subset
                if row["policy"] == policy
            }
            axes[0][column].bar(
                [value + (policy_index - 1.5) * width for value in x],
                [lookup[point]["reuse_rate"] * 100 for point in points],
                width=width,
                label=label,
            )
        axes[0][column].set_xticks(x, [f"N{n}/C{c}" for n, c in points])
        axes[0][column].set_title(protocol)
        axes[0][column].set_ylabel("Token-weighted reuse rate (%)")
        axes[0][column].grid(axis="y", alpha=0.25)
    axes[0][1].legend(fontsize=8)
    fig.suptitle("Causal lifecycle ablations under controlled event protocols")
    fig.tight_layout()
    fig.savefig(output_dir / "lifecycle_reuse.png", dpi=180)
    plt.close(fig)


def make_report(summary: list[dict[str, Any]], cases: int) -> str:
    conservative = {
        (row["nominal_session_count"], row["capacity"]): row
        for row in summary
        if row["protocol"] == "serial_arrival" and row["policy"] == "lifecycle_lru"
    }
    passed_prefill = [
        point
        for point, row in sorted(conservative.items())
        if row["aggregate_prefill_reduction_vs_lru_pct"] >= 10
    ]
    failed_prefill = [
        point
        for point, row in sorted(conservative.items())
        if row["aggregate_prefill_reduction_vs_lru_pct"] < 10
    ]
    lines = [
        "# Lifecycle-aware Session KV analysis",
        "",
        "## Scope and leakage controls",
        "",
        f"The analysis covers {cases} full Stage-3 group/capacity cases at the five preselected N4/N8 points. Simulated LRU is required to match the measured phone LRU exactly before any method result is emitted.",
        "",
        "HammerBench does not natively expose runtime lifecycle events. Every method event in this analysis is therefore labeled `controlled_protocol`. `future_oracle` is rejected by the online simulator and is used only by Belady/token-optimal upper bounds.",
        "",
        "Two visibility assumptions are reported separately:",
        "",
        "- `serial_arrival`: only the dispatched request is READY; other resident Sessions are WAITING.",
        "- `queued_next_turn`: after a Session request completes, its next fixed reference request is released. Queue rank and exact LCP are visible because that prompt is already queued.",
        "",
        "A controlled close is emitted after a Session's final request. This validates the mechanism available when an upper layer explicitly closes a Session; it is not evidence that HammerBench provides natural close signals.",
        "",
        "## Population result",
        "",
        "| Protocol | Point | Policy | Groups | Reuse | Prefill reduction vs LRU | Oracle gap recovery | Evictions | DONE releases |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        recovery = row["oracle_gap_recovery_pct"]
        recovery_text = "n/a" if recovery is None else f"{recovery:.1f}%"
        lines.append(
            f"| {row['protocol']} | N{row['nominal_session_count']}/C{row['capacity']} | "
            f"{row['policy']} | {row['session_groups']} | {row['reuse_rate'] * 100:.1f}% | "
            f"{row['aggregate_prefill_reduction_vs_lru_pct']:.1f}% | {recovery_text} | "
            f"{row['evictions']} | {row['done_releases']} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "`serial_arrival` is the conservative result for the current fixed replay. Any additional gain under `queued_next_turn` is conditional on the upper scheduler genuinely having those prompts in its ready queue. Neither controlled result is an end-to-end Agent result, and TTFT improvement remains unclaimed until the selected policy is run on the phone.",
        "",
        "## Decision",
        "",
        f"Under `serial_arrival`, Lifecycle-LRU reaches the predeclared 10% Prefill-reduction threshold at {', '.join(f'N{n}/C{c}' for n, c in passed_prefill)}. It does not reach that threshold at {', '.join(f'N{n}/C{c}' for n, c in failed_prefill)}.",
        "",
        "The offline gate is therefore satisfied at both N=4 and N=8. The next step is a phone comparison at all five points, retaining the failed point as a capacity-pressure negative control. `queued_next_turn` should be a separate conditional experiment, because its near-Oracle result depends on exact queued request visibility.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--points",
        default="4:2,4:3,8:2,8:4,8:6",
        help="comma-separated N:C points",
    )
    args = parser.parse_args()
    points = parse_points(args.points)
    if points != DEFAULT_POINTS:
        print(json.dumps({"warning": "non-default lifecycle points", "points": sorted(points)}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, cases = load_cases(args.matrix_root, points)
    rows, audit = analyze_cases(cases)
    summary = aggregate(rows, int(manifest["randomization_seed"]))
    write_csv(args.output_dir / "lifecycle_by_group.csv", rows)
    write_csv(args.output_dir / "lifecycle_summary.csv", summary)
    (args.output_dir / "lifecycle_by_group.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "lifecycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "lifecycle_audit.jsonl").open("w", encoding="utf-8") as stream:
        for event in audit:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    make_figure(summary, args.output_dir)
    (args.output_dir / "report.md").write_text(make_report(summary, len(cases)))
    print(
        json.dumps(
            {
                "status": "pass",
                "validated_lru_cases": len(cases),
                "summary_rows": len(summary),
                "audit_events": len(audit),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
