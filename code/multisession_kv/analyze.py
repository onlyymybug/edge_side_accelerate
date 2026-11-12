#!/usr/bin/env python3
"""Aggregate replay directories into machine-readable CSV and JSON summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def summarize(directory: Path) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    requests = read_jsonl(directory / "requests.jsonl")
    events = read_jsonl(directory / "events.jsonl")
    telemetry = read_jsonl(directory / "telemetry.jsonl")
    ttft = [float(row["ttft_ms"]) for row in requests]
    latency = [float(row["request_ms"]) for row in requests]
    input_tokens = sum(int(row["input_tokens"]) for row in requests)
    reused_tokens = sum(int(row["reused_tokens"]) for row in requests)
    temperatures = [
        float(row["skin_temp_c"])
        for row in telemetry
        if row.get("skin_temp_c") is not None
    ]
    rss = [int(row["rss_kb"]) for row in telemetry if row.get("rss_kb") is not None]
    return {
        "run_dir": str(directory),
        "run_id": config["run_id"],
        "variant": config["trace_variant"],
        "session_count": config["session_count"],
        "session_offset": config.get("session_offset", 0),
        "policy": config["policy"],
        "schedule": config["schedule"],
        "lru_slots": config["lru_slots"],
        "max_new_tokens": config["max_new_tokens"],
        "request_count": len(requests),
        "input_tokens": input_tokens,
        "prefill_tokens": sum(int(row["prefill_tokens"]) for row in requests),
        "reused_tokens": reused_tokens,
        "token_reuse_rate": reused_tokens / input_tokens,
        "ttft_ms_p50": statistics.median(ttft),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "request_ms_p50": statistics.median(latency),
        "request_ms_p95": percentile(latency, 0.95),
        "trace_makespan_ms": sum(latency),
        "save_restore_ms": sum(
            float(row["save_ms"]) + float(row["restore_ms"]) for row in requests
        ),
        "evictions": sum(row.get("event") == "evict" for row in events),
        "resets": sum(row.get("event") == "reset_all" for row in events),
        "rss_kb_peak": max(rss) if rss else None,
        "skin_temp_c_start": temperatures[0] if temperatures else None,
        "skin_temp_c_end": temperatures[-1] if temperatures else None,
        "skin_temp_c_peak": max(temperatures) if temperatures else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(directory) for directory in args.run_dirs]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
