#!/usr/bin/env python3
"""Validate prepared traces and completed replay runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def lcp(lhs: list[int], rhs: list[int]) -> int:
    result = 0
    while result < min(len(lhs), len(rhs)) and lhs[result] == rhs[result]:
        result += 1
    return result


def validate_trace(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    reports = {}
    selected = manifest["selected_sessions"]
    trace_files = manifest.get(
        "trace_files", {"native_tools": "native_tools.jsonl", "fixed_tools": "fixed_tools.jsonl"}
    )
    for variant, filename in trace_files.items():
        rows = read_jsonl(directory / filename)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        snapshot_ids: set[str] = set()
        for row in rows:
            assert row["variant"] == variant
            assert row["source_snapshot_id"] not in snapshot_ids
            snapshot_ids.add(row["source_snapshot_id"])
            assert row["input_tokens"] == len(row["input_token_ids"])
            assert row["input_tokens"] + manifest["max_new_tokens"] <= manifest["context_tokens"]
            grouped[row["session_id"]].append(row)
        assert list(grouped) == selected
        for session_id, session_rows in grouped.items():
            session_rows.sort(key=lambda row: row["turn_id"])
            assert [row["turn_id"] for row in session_rows] == list(
                range(1, len(session_rows) + 1)
            )
            previous: list[int] = []
            first_tools = session_rows[0]["tools"]
            for row in session_rows:
                expected = lcp(previous, row["input_token_ids"]) if previous else 0
                assert row["previous_turn_lcp_tokens"] == expected
                assert row["required_new_tokens"] == row["input_tokens"] - expected
                if variant == "fixed_tools" or variant.startswith("bfcl_v4_full_tools_"):
                    assert row["tools"] == first_tools
                previous = row["input_token_ids"]
        reports[variant] = {
            "sessions": len(grouped),
            "requests": len(rows),
            "input_tokens_min": min(row["input_tokens"] for row in rows),
            "input_tokens_max": max(row["input_tokens"] for row in rows),
            "lcp_ratio": sum(row["previous_turn_lcp_tokens"] for row in rows)
            / sum(row["input_tokens"] for row in rows),
        }
    return {"status": "pass", "manifest": str(directory / "manifest.json"), "variants": reports}


def validate_run(directory: Path) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    capabilities = json.loads((directory / "capabilities.json").read_text(encoding="utf-8"))
    requests = read_jsonl(directory / "requests.jsonl")
    events = read_jsonl(directory / "events.jsonl")
    assert requests
    assert all(row["status"] == "ok" for row in requests)
    assert all(row["model_load_count"] == 1 for row in requests)
    assert {row["server_pid"] for row in requests} == {capabilities["server_pid"]}
    assert all(row["request_end_ns"] >= row["request_received_ns"] for row in requests)
    assert all(row["first_token_ns"] >= row["request_received_ns"] for row in requests)
    assert all(row["prefill_tokens"] + row["reused_tokens"] == row["input_tokens"] for row in requests)
    assert all(row["queue_ms"] == 0.0 for row in requests)
    assert all(row["kv_budget_bytes"] >= 0 for row in requests)
    lifecycle_report = None
    if config["policy"] == "lifecycle_lru":
        assert config["lifecycle_protocol"] == "serial_arrival"
        assert config["lifecycle_source"] == "controlled_protocol"
        lifecycle_events = [row for row in events if "lifecycle_source" in row]
        assert lifecycle_events
        assert {row["lifecycle_source"] for row in lifecycle_events} <= {
            "runtime_explicit",
            "controlled_protocol",
        }
        assert all(float(row["policy_decision_ms"]) >= 0 for row in requests)
        states = {
            state: sum(row.get("state") == state for row in lifecycle_events)
            for state in ("READY", "RUNNING", "WAITING", "DONE")
        }
        session_count = len(config["selected_sessions"])
        assert states["READY"] == len(requests)
        assert states["RUNNING"] == len(requests)
        assert states["DONE"] == session_count
        assert states["WAITING"] == len(requests) - session_count
        done_sessions = {
            row["session_id"] for row in lifecycle_events if row.get("state") == "DONE"
        }
        assert done_sessions == set(config["selected_sessions"])
        assert all(
            row.get("response", {}).get("status") == "ok"
            for row in lifecycle_events
            if row.get("state") == "DONE"
        )
        lifecycle_evictions = [
            row for row in events if row.get("reason") == "lifecycle_capacity"
        ]
        assert all(row.get("victim_state") == "WAITING" for row in lifecycle_evictions)
        assert all(
            row.get("response", {}).get("status") == "ok"
            for row in lifecycle_evictions
        )
        lifecycle_report = {
            "protocol": config["lifecycle_protocol"],
            "source": config["lifecycle_source"],
            "states": states,
            "capacity_evictions": len(lifecycle_evictions),
            "policy_decision_ms_total": sum(
                float(row["policy_decision_ms"]) for row in requests
            ),
        }
    result = {
        "status": "pass",
        "run_id": config["run_id"],
        "requests": len(requests),
        "total_prefill_tokens": sum(row["prefill_tokens"] for row in requests),
        "total_reused_tokens": sum(row["reused_tokens"] for row in requests),
    }
    if lifecycle_report is not None:
        result["lifecycle"] = lifecycle_report
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("--trace-dir", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate_trace(args.trace_dir) if args.kind == "trace" else validate_run(args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
