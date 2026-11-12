#!/usr/bin/env python3
"""Scoping probe: is the real method opportunity in the *granularity*?

The current design makes a binary whole-Session keep/evict decision inside a
fixed number of equal slots. This probe measures three alternatives under an
identical token budget, so the comparison is about representation and not
about getting more memory:

  whole   - keep a Session's full KV or nothing (today's design);
  prefix  - keep an arbitrary-length prefix of a Session's KV;
  ideal   - an upper bound that keeps whatever the future needs, for free.

If `prefix` is far above `whole`, the finding is that the paper's unit of
decision is misplaced: the interesting question is not *which* Session to
evict but *how much of each Session* to retain.

Run: python probe_granularity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACE = Path("data/hammerbench_en_50/fixed_tools.jsonl")


def load_sessions(trace: Path) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(row["session_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["turn_id"])
    return list(grouped.values())


def round_robin(sessions: list[list[dict[str, Any]]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    depth = max(len(rows) for rows in sessions)
    for turn in range(depth):
        for index, rows in enumerate(sessions):
            if turn < len(rows):
                order.append((index, turn))
    return order


def simulate(
    sessions: list[list[dict[str, Any]]],
    order: list[tuple[int, int]],
    budget: int,
    granularity: str,
    prefix_quantum: int = 128,
) -> dict[str, float]:
    """Replay one retention granularity under a hard token budget.

    With partial retention the resident KV is modelled as a prefix of the
    Session's last prompt, so a partially-retained Session still serves the
    overlapping part of its next prompt. Retained accounting is in tokens.
    """
    held: dict[int, int] = {}  # session -> retained tokens
    reused = 0
    total = 0
    evictions = 0

    for index, turn in order:
        row = sessions[index][turn]
        need = int(row["input_tokens"])
        overlap = int(row["previous_turn_lcp_tokens"])
        total += need

        # What we already hold for this Session can serve the overlap.
        have = held.pop(index, 0)
        if overlap > 0:
            reused += min(have, overlap)

        # After serving, keep as much as this granularity and the budget allow.
        if granularity == "whole":
            want = need
        elif granularity == "prefix":
            want = need
        else:  # ideal
            want = need
        want = min(want, budget)

        if granularity == "whole":
            # All-or-nothing: either the full Session fits or nothing is kept.
            while held and sum(held.values()) + want > budget:
                del held[next(iter(held))]
                evictions += 1
            if sum(held.values()) + want <= budget:
                held[index] = want
        else:
            # Partial: shrink the least-recently-used residents just enough.
            while held and sum(held.values()) + want > budget:
                victim = next(iter(held))
                excess = sum(held.values()) + want - budget
                if granularity == "prefix":
                    give = min(held[victim], excess)
                    give = max(give, min(excess, prefix_quantum))
                else:
                    give = held[victim]
                held[victim] -= give
                if held[victim] < prefix_quantum:
                    del held[victim]
                    evictions += 1
            held[index] = want

    return {
        "reused": reused,
        "total": total,
        "reuse_rate": reused / total,
        "evictions": evictions,
        "peak_resident": len(held),
    }


def main() -> None:
    sessions = load_sessions(TRACE)
    print(f"sessions={len(sessions)}  turns={sorted(len(s) for s in sessions)}")

    for count in (4, 8):
        picked = sessions[:count]
        order = round_robin(picked)
        total = sum(
            int(picked[i][t]["input_tokens"]) for i, t in order
        )
        print(f"\n=== N={count} round-robin, {len(order)} requests, "
              f"{total} total input tokens ===")
        print("budget(tok) | whole | prefix | ideal |  whole/ideal")
        print("         --- |  ---: |   ---: |  ---: |  ---:")
        # Budget expressed as "slots x 1024", matching the phone's fixed slots.
        for slots in (2, 3, 4, 6, 7):
            budget = slots * 1024
            w = simulate(picked, order, budget, "whole")
            p = simulate(picked, order, budget, "prefix")
            i = simulate(picked, order, budget, "ideal")
            print(
                f"{budget:11d} | {w['reuse_rate']:5.1%} | {p['reuse_rate']:6.1%} | "
                f"{i['reuse_rate']:5.1%} | {w['reuse_rate']/max(i['reuse_rate'],1e-9):6.1%}"
            )


if __name__ == "__main__":
    main()
