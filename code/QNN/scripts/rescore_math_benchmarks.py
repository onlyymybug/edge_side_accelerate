#!/usr/bin/env python3
"""Rescore existing GSM8K/MATH-500 runs without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from math_benchmarks import (  # noqa: E402
    SCORER_VERSION,
    canonical_benchmark_name,
    score_benchmark_answer,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def benchmark_for_row(row: dict[str, str]) -> str:
    value = row.get("dataset") or row.get("task") or ""
    return canonical_benchmark_name(value)


def completion_for_row(result_dir: Path, row: dict[str, str]) -> str:
    sample_id = row["sample_id"]
    candidates: list[Path] = []
    output_path = row.get("output_path")
    if output_path:
        output = Path(output_path)
        candidates.extend(
            [
                output.parent / "assistant.txt",
                output.parent / "completion.txt",
                output,
            ]
        )
    candidates.extend(result_dir.glob(f"**/{sample_id}/assistant.txt"))
    candidates.extend(result_dir.glob(f"**/{sample_id}/completion.txt"))
    candidates.extend(result_dir.glob(f"**/{sample_id}/generated_text.txt"))
    candidates.extend(result_dir.glob(f"**/{sample_id}/output.txt"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name in ("generated_text.txt", "output.txt"):
            for marker in (
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|im_start|>assistant\n",
            ):
                if marker in text:
                    text = text.rsplit(marker, 1)[-1]
        for token in ("<|eot_id|>", "<|endoftext|>", "<|im_end|>"):
            text = text.replace(token, "")
        return text.strip()
    raise FileNotFoundError(f"no saved completion found for {sample_id}")


def number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def is_true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def is_success(row: dict[str, Any]) -> bool:
    key = "runner_returncode" if "runner_returncode" in row else "returncode"
    return str(row.get(key, "")) == "0"


def write_deadline_rescore(
    result_dir: Path, rows: list[dict[str, Any]], output_path: Path
) -> None:
    source = result_dir / "deadline_summary.csv"
    if not source.is_file():
        return
    deadline_rows = read_csv(source)
    output_rows: list[dict[str, Any]] = []
    for aggregate in deadline_rows:
        task = aggregate.get("task", "")
        selected = [
            row
            for row in rows
            if not task or (row.get("task") or row.get("dataset")) == task
        ]
        completed = [row for row in selected if is_success(row)]
        measured = [
            row
            for row in completed
            if row.get("energy_measurement_status")
            == "measured_signed_battery_energy"
            and number(row, "energy_j") is not None
        ]
        deadline_ms = int(aggregate["deadline_ms"])
        correct = [
            row
            for row in completed
            if is_true(row.get("answer_correct", row.get("score_correct")))
            and number(row, "inference_ms") is not None
            and float(row["inference_ms"]) <= deadline_ms
        ]
        measured_ids = {row.get("sample_id") for row in measured}
        measured_correct = [row for row in correct if row.get("sample_id") in measured_ids]
        total_energy = sum(number(row, "energy_j") or 0.0 for row in measured)
        num_tasks = int(aggregate.get("num_dataset_tasks") or len(selected))
        full_coverage = len(completed) == num_tasks and len(measured) == num_tasks
        provisional = total_energy / len(measured_correct) if measured_correct else ""
        updated = dict(aggregate)
        updated.update(
            {
                "num_completed_tasks": len(completed),
                "num_energy_measured_tasks": len(measured),
                "num_correct_under_deadline": len(correct),
                "num_measured_correct_under_deadline": len(measured_correct),
                "total_measured_energy_j": total_energy,
                "provisional_energy_per_correct_task_under_deadline_j": provisional,
                "energy_per_correct_task_under_deadline_j": provisional if full_coverage else "",
                "valid_for_full_dataset_reporting": full_coverage,
            }
        )
        output_rows.append(updated)
    write_csv(output_path, output_rows)


def rescore(result_dir: Path, in_place: bool) -> dict[str, Any]:
    summary_path = result_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary.csv: {result_dir}")
    rows = read_csv(summary_path)
    errors: list[str] = []
    rescored: list[dict[str, Any]] = []
    for row in rows:
        updated: dict[str, Any] = dict(row)
        try:
            benchmark = benchmark_for_row(row)
            try:
                completion = completion_for_row(result_dir, row)
            except FileNotFoundError:
                if is_success(row):
                    raise
                completion = ""
            scored = score_benchmark_answer(benchmark, completion, row["gold_answer"])
            updated.update(scored)
            if "answer_correct" in updated:
                updated["answer_correct"] = scored["score_correct"]
        except Exception as exc:
            errors.append(f"{row.get('sample_id', '?')}: {type(exc).__name__}: {exc}")
            updated.update(
                {
                    "scorer_version": SCORER_VERSION,
                    "answer_extractable": False,
                    "format_compliant": False,
                    "score_correct": False,
                    "answer_correct": (
                        False
                        if "answer_correct" in updated
                        else updated.get("answer_correct", "")
                    ),
                    "verifier_method": "rescore_error",
                    "verifier_error": errors[-1],
                }
            )
        rescored.append(updated)

    if in_place:
        backup = result_dir / "summary.pre_rescore_v1.csv"
        if not backup.exists():
            shutil.copy2(summary_path, backup)
        output_summary = summary_path
        deadline_output = result_dir / "deadline_summary.csv"
        deadline_source = deadline_output
        if deadline_source.exists():
            deadline_backup = result_dir / "deadline_summary.pre_rescore_v1.csv"
            if not deadline_backup.exists():
                shutil.copy2(deadline_source, deadline_backup)
    else:
        output_summary = result_dir / "summary.rescored_v2.csv"
        deadline_output = result_dir / "deadline_summary.rescored_v2.csv"
    write_csv(output_summary, rescored)
    write_deadline_rescore(result_dir, rescored, deadline_output)

    correct_key = (
        "answer_correct" if "answer_correct" in rescored[0] else "score_correct"
    )
    report = {
        "scorer_version": SCORER_VERSION,
        "source_summary": str(summary_path),
        "output_summary": str(output_summary),
        "samples": len(rescored),
        "runtime_successes": sum(is_success(row) for row in rescored),
        "answer_extractable": sum(
            is_true(row.get("answer_extractable")) for row in rescored
        ),
        "format_compliant": sum(
            is_true(row.get("format_compliant")) for row in rescored
        ),
        "correct": sum(is_true(row.get(correct_key)) for row in rescored),
        "accuracy": (
            sum(is_true(row.get(correct_key)) for row in rescored) / len(rescored)
            if rescored
            else None
        ),
        "parse_methods": Counter(row.get("answer_parse_method", "") for row in rescored),
        "verifier_methods": Counter(row.get("verifier_method", "") for row in rescored),
        "rescore_errors": errors,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    report_path = result_dir / (
        "evaluation_summary.json" if in_place else "evaluation_summary.rescored_v2.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace summary/deadline files after creating pre_rescore_v1 backups",
    )
    args = parser.parse_args()
    failed = False
    for result_dir in args.result_dirs:
        try:
            report = rescore(result_dir.resolve(), args.in_place)
            print(json.dumps(report, ensure_ascii=False, default=dict))
        except Exception as exc:
            failed = True
            print(f"{result_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
