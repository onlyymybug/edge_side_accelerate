#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from appendix_common import (
    DEFAULT_CONFIG,
    PLAN_DIR,
    file_sha256,
    is_length_stop,
    read_config,
    read_csv,
    stable_id,
    truth,
    write_jsonl,
)


def graph_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    section = cfg["graph_boundary"]
    rng = random.Random(section["seed"])
    rows = []
    order = 0
    for runtime in section["runtimes"]:
        for repeat in range(1, section["repeats"] + 1):
            lengths = list(section["prefix_lengths"])
            rng.shuffle(lengths)
            for length in lengths:
                order += 1
                rows.append({
                    "request_id": "graph_" + stable_id(runtime, repeat, length),
                    "experiment": "graph_boundary", "runtime": runtime,
                    "mode": section["modes"][runtime], "prefix_tokens": length,
                    "expected_prefill_calls": (length + section["block_widths"][runtime] - 1) // section["block_widths"][runtime],
                    "output_budget": section["output_budget"], "repeat": repeat,
                    "order": order,
                })
    return rows


def _spread(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    valid = [r for r in rows if r.get("returncode", "0") == "0"]
    valid.sort(key=lambda r: (int(float(r.get("prompt_tokens") or 0)) + int(float(r.get("generated_tokens") or 0)), r["sample_id"]))
    if len(valid) < count:
        raise ValueError(f"need {count} historical rows, found {len(valid)}")
    if count == 1:
        return [valid[len(valid) // 2]]
    indices = [round(i * (len(valid) - 1) / (count - 1)) for i in range(count)]
    return [valid[i] for i in indices]


def threads_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    section = cfg["tmac_threads"]
    chosen = _spread(read_csv(Path(section["historical_summary"])), section["sample_count"])
    rng = random.Random(section["seed"])
    rows = []
    order = 0
    for repeat in range(1, section["repeats"] + 1):
        sample_order = list(chosen)
        rng.shuffle(sample_order)
        for sample in sample_order:
            settings = list(section["settings"])
            rng.shuffle(settings)
            for batch_threads, decode_threads in settings:
                order += 1
                rows.append({
                    "request_id": "threads_" + stable_id(sample["sample_id"], repeat, batch_threads, decode_threads),
                    "experiment": "tmac_threads", "task": "GSM8K",
                    "sample_id": sample["sample_id"], "repeat": repeat,
                    "batch_threads": batch_threads, "decode_threads": decode_threads,
                    "output_budget": section["output_budget"], "order": order,
                    "historical_prompt_tokens": sample.get("prompt_tokens", ""),
                    "historical_generated_tokens": sample.get("generated_tokens", ""),
                })
    return rows


def _paired_categories(short_path: Path, long_path: Path) -> dict[str, list[dict[str, str]]]:
    short = {r["sample_id"]: r for r in read_csv(short_path)}
    long = {r["sample_id"]: r for r in read_csv(long_path)}
    groups = {"eos_unchanged": [], "truncated_improved": [], "truncated_still_wrong": []}
    for sample_id in sorted(short.keys() & long.keys()):
        a, b = short[sample_id], long[sample_id]
        if a.get("returncode", "0") != "0" or b.get("returncode", "0") != "0":
            continue
        same_text = file_sha256(a.get("completion_path", "")) == file_sha256(b.get("completion_path", "")) != ""
        if a.get("finish_reason") == "eos" and b.get("finish_reason") == "eos" and same_text:
            groups["eos_unchanged"].append(a)
        elif is_length_stop(a) and not truth(a.get("score_correct")) and truth(b.get("score_correct")):
            groups["truncated_improved"].append(a)
        elif is_length_stop(a) and not truth(a.get("score_correct")) and not truth(b.get("score_correct")):
            groups["truncated_still_wrong"].append(a)
    return groups


def budget_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    section = cfg["budget_termination"]
    rng = random.Random(section["seed"])
    rows = []
    order = 0
    for task, task_cfg in section["tasks"].items():
        groups = _paired_categories(Path(task_cfg["short_summary"]), Path(task_cfg["long_summary"]))
        selected = []
        for category, candidates in groups.items():
            rng.shuffle(candidates)
            if len(candidates) < section["per_category"]:
                raise ValueError(f"{task}/{category}: need {section['per_category']}, found {len(candidates)}")
            selected.extend((category, row) for row in candidates[:section["per_category"]])
        for repeat in range(1, section["repeats"] + 1):
            pairs = [(category, sample, budget_name, task_cfg[f"{budget_name}_budget"]) for category, sample in selected for budget_name in ("short", "long")]
            rng.shuffle(pairs)
            for category, sample, budget_name, budget in pairs:
                order += 1
                rows.append({
                    "request_id": "budget_" + stable_id(task, sample["sample_id"], repeat, budget_name),
                    "experiment": "budget_termination", "task": task,
                    "sample_id": sample["sample_id"], "historical_category": category,
                    "budget_name": budget_name, "output_budget": budget,
                    "repeat": repeat, "batch_threads": 6, "decode_threads": 6,
                    "order": order,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Appendix C schedules")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-dir", type=Path, default=PLAN_DIR)
    args = parser.parse_args()
    cfg = read_config(args.config)
    plans = {"graph_boundary": graph_plan(cfg), "tmac_threads": threads_plan(cfg), "budget_termination": budget_plan(cfg)}
    for name, rows in plans.items():
        path = args.plan_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        print(f"{name}: {len(rows)} requests -> {path}")


if __name__ == "__main__":
    main()
