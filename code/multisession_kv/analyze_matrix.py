#!/usr/bin/env python3
"""Aggregate all passed jobs recorded by a matrix manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analyze import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest_path = args.matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        status_counts[job["status"]] = status_counts.get(job["status"], 0) + 1
    incomplete = len(manifest["jobs"]) - status_counts.get("pass", 0)
    if args.require_complete and incomplete:
        raise RuntimeError(f"matrix is incomplete: {status_counts}")

    rows: list[dict[str, Any]] = []
    for sequence, job in enumerate(manifest["jobs"]):
        if job["status"] != "pass":
            continue
        run_dir = Path(job["run_dir"])
        row = summarize(run_dir)
        row.update(
            {
                "matrix_sequence": sequence,
                "job_id": job["job_id"],
                "repeat": job["repeat"],
                "experiment": job.get("experiment"),
                "nominal_session_count": job["nominal_session_count"],
                "nominal_budget_ratios": json.dumps(job.get("nominal_budget_ratios")),
                "actual_slot_ratio": job.get("actual_slot_ratio"),
                "attempts": job["attempts"],
            }
        )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"matrix has no passed jobs: {status_counts}")

    csv_path = args.csv or args.matrix_root / "summary.csv"
    json_path = args.json or args.matrix_root / "summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "status_counts": status_counts,
                "summarized_runs": len(rows),
                "csv": str(csv_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
