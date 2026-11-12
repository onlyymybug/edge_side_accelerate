#!/usr/bin/env python3
"""Measure an idle-phone thermal baseline before choosing a recovery gate."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from collect_android import AndroidSampler


def battery_state(serial: str) -> str:
    result = subprocess.run(
        [
            "adb",
            "-s",
            serial,
            "shell",
            "dumpsys battery | grep -E 'powered:|status:|level:|temperature:'",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--duration-seconds", type=float, default=60)
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--sensor", choices=("skin", "nsp0"), default="skin")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("duration must be positive")

    # PID zero intentionally yields no process memory while retaining thermal data.
    sampler = AndroidSampler(args.serial, 0, args.interval_seconds)
    sampler.start()
    time.sleep(args.duration_seconds)
    sampler.stop()
    key = f"{args.sensor}_temp_c"
    values = [float(row[key]) for row in sampler.samples if row.get(key) is not None]
    if not values:
        raise RuntimeError(f"no {args.sensor} readings were collected")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial": args.serial,
        "sensor": args.sensor,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "battery_state": battery_state(args.serial),
        "samples": len(values),
        "minimum_c": min(values),
        "median_c": statistics.median(values),
        "maximum_c": max(values),
        "range_c": max(values) - min(values),
        "suggested_max_start_c": round(statistics.median(values) + 1.0, 3),
        "suggestion_note": "local convenience threshold only; inspect stability and choose explicitly for the formal config",
        "raw_samples": sampler.samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_samples"}, indent=2))


if __name__ == "__main__":
    main()

