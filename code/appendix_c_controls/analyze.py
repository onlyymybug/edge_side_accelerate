#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from appendix_common import read_csv, truth


def number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        try:
            if row.get(key, "") != "":
                return float(row[key])
        except ValueError:
            pass
    return None


def summarize(rows: list[dict[str, str]], keys: list[str], metrics: list[tuple[str, ...]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for group, items in sorted(groups.items()):
        record: dict[str, Any] = dict(zip(keys, group))
        record["n"] = len(items)
        for aliases in metrics:
            values = [value for item in items if (value := number(item, *aliases)) is not None]
            label = aliases[0]
            record[label + "_mean"] = statistics.mean(values) if values else ""
            record[label + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one Appendix C phone run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = read_csv(args.run_dir / "summary.csv")
    experiment = rows[0]["experiment"] if rows else ""
    if experiment == "graph_boundary":
        result = summarize(rows, ["runtime", "mode", "prefix_tokens", "expected_prefill_calls"],
                           [("ttft_ms", "prefill_latency_ms"), ("energy_j",)])
        name = "graph_boundary_summary.csv"
    elif experiment == "tmac_threads":
        result = summarize(rows, ["batch_threads", "decode_threads"],
                           [("ttft_ms", "prefill_latency_ms"), ("decode_latency_ms",), ("e2e_latency_ms",), ("energy_j",)])
        name = "tmac_threads_summary.csv"
    elif experiment == "budget_termination":
        result = summarize(rows, ["task", "historical_category", "budget_name", "output_budget"],
                           [("generated_tokens",), ("e2e_latency_ms",), ("energy_j",)])
        by_group: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_group[(row.get("task", ""), row.get("historical_category", ""), row.get("budget_name", ""))].append(row)
        rates = {key: sum(truth(r.get("score_correct")) for r in group) / len(group) for key, group in by_group.items() if group}
        for item in result:
            key = (item["task"], item["historical_category"], item["budget_name"])
            item["accuracy"] = rates.get(key, "")
        name = "budget_termination_summary.csv"
    else:
        raise SystemExit("summary.csv is empty or has an unknown experiment")
    output = args.run_dir / name
    write_csv(output, result)
    errors = sum(row.get("status") != "ok" for row in rows)
    missing_calls = sum(experiment == "graph_boundary" and not row.get("actual_prefill_calls") for row in rows if row.get("status") == "ok")
    report = [f"# Validation: {experiment}", "", f"- Recorded requests: {len(rows)}", f"- Errors: {errors}"]
    if experiment == "graph_boundary":
        report.append(f"- Runs without native actual-prefill-call count: {missing_calls}")
    report.extend(["", f"Aggregate table: `{name}`", ""])
    (args.run_dir / "VALIDATION.md").write_text("\n".join(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
