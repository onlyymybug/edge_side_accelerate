#!/usr/bin/env python3
"""Run GSM8K and MATH-500 through the Android Qwen2.5-3B T-MAC bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(CODE_ROOT))

from math_benchmarks import (  # noqa: E402
    BENCHMARKS,
    SCORER_VERSION,
    build_benchmark_prompt,
    load_benchmark_samples,
    score_benchmark_answer,
)
from runtime import (  # noqa: E402
    TmacConfig,
    TmacPhoneRunner,
    is_transport_error,
    qwen_chatml,
)


PACE_ROOT = CODE_ROOT.parent
DEFAULT_DATASETS = {
    "GSM8K": PACE_ROOT / "datasets" / "gsm8k_test.jsonl",
    "MATH500": PACE_ROOT / "datasets" / "math500_test.jsonl",
}
DEFAULT_DEADLINES = (5000, 10000, 20000, 30000)


def parse_budget(value: str | int | None) -> int | None:
    if value is None or str(value).strip().lower() in {"none", "no_budget", "no-budget"}:
        return None
    budget = int(value)
    if budget <= 0:
        raise ValueError("output budget must be positive")
    return budget


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def backfill_cold_start_metrics(row: dict[str, Any]) -> None:
    """Add cold-start latency fields to rows produced before these fields existed."""
    for target, second in (
        ("cold_start_ttft_ms", "ttft_ms"),
        ("cold_start_e2e_latency_ms", "inference_ms"),
    ):
        if row.get(target) not in (None, ""):
            continue
        if row.get("model_load_ms") in (None, "") or row.get(second) in (None, ""):
            row[target] = ""
            continue
        row[target] = float(row["model_load_ms"]) + float(row[second])


def truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def write_deadline_summary(
    path: Path,
    rows: list[dict[str, Any]],
    tasks: list[str],
    deadlines: list[int],
    dataset_sizes: dict[str, int],
) -> None:
    output: list[dict[str, Any]] = []
    for task in tasks:
        selected = [row for row in rows if row.get("task") == task]
        for deadline in deadlines:
            completed = [
                row
                for row in selected
                if str(row.get("runner_returncode", "")) == "0"
                and row.get("inference_ms") not in (None, "")
            ]
            correct = [
                row
                for row in completed
                if truth(row.get("score_correct"))
                and float(row["inference_ms"]) <= deadline
            ]
            measured_correct = [row for row in correct if row.get("energy_j") not in (None, "")]
            measured = [row for row in completed if row.get("energy_j") not in (None, "")]
            total_energy = sum(float(row["energy_j"]) for row in measured)
            valid = len(selected) == dataset_sizes[task] and len(measured) == len(selected)
            energy_per_correct = total_energy / len(measured_correct) if measured_correct else ""
            output.append(
                {
                    "task": task,
                    "deadline_ms": deadline,
                    "num_dataset_tasks": dataset_sizes[task],
                    "num_completed_tasks": len(completed),
                    "num_energy_measured_tasks": len(measured),
                    "num_correct_under_deadline": len(correct),
                    "num_measured_correct_under_deadline": len(measured_correct),
                    "total_measured_energy_j": total_energy if measured else "",
                    "provisional_energy_per_correct_task_under_deadline_j": energy_per_correct,
                    "energy_per_correct_task_under_deadline_j": energy_per_correct if valid else "",
                    "valid_for_full_dataset_reporting": valid,
                }
            )
    write_csv(path, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--gsm8k-dataset", type=Path, default=DEFAULT_DATASETS["GSM8K"])
    parser.add_argument("--math500-dataset", type=Path, default=DEFAULT_DATASETS["MATH500"])
    parser.add_argument("--gsm8k-budget", default="512")
    parser.add_argument("--math500-budget", default="512")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deadlines-ms", default=",".join(map(str, DEFAULT_DEADLINES)))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        budgets = {
            "GSM8K": parse_budget(args.gsm8k_budget),
            "MATH500": parse_budget(args.math500_budget),
        }
        deadlines = sorted({int(value) for value in args.deadlines_ms.split(",")})
    except ValueError as exc:
        parser.error(str(exc))
    if not deadlines or deadlines[0] <= 0:
        parser.error("deadlines must be positive")

    datasets = {"GSM8K": args.gsm8k_dataset.resolve(), "MATH500": args.math500_dataset.resolve()}
    loaded: dict[str, list[dict[str, Any]]] = {}
    dataset_sizes: dict[str, int] = {}
    for task in args.tasks:
        if not datasets[task].is_file():
            parser.error(f"dataset missing: {datasets[task]}")
        all_samples = load_benchmark_samples(datasets[task], task)
        dataset_sizes[task] = len(all_samples)
        loaded[task] = all_samples[: args.limit] if args.limit is not None else all_samples
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": {task: len(loaded[task]) for task in args.tasks},
                    "budgets": budgets,
                    "serial": os.environ.get("ANDROID_SERIAL", ""),
                    "phone_dir": os.environ.get(
                        "TMAC_PHONE_DIR", "/data/local/tmp/qwen25_3b_tmac_android_bundle"
                    ),
                },
                indent=2,
            )
        )
        return 0

    config = TmacConfig.from_env()
    runner = TmacPhoneRunner(config)
    device_label = os.environ.get("DEVICE_LABEL", "redmi")
    result_model = os.environ.get("RESULT_MODEL_DIR_NAME", "qwen2.5-3B-tmac-w4a16")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PACE_ROOT / "results" / "T-MAC" / device_label / result_model
    )
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root = output_root / run_id
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"cannot resume missing run: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "backend": "T-MAC CPU",
        "device_label": device_label,
        "adb_transport_serial": config.serial,
        "device_identity": runner.device_identity(),
        "phone_dir": config.phone_dir,
        "model_result_name": "qwen2.5-3b",
        "quantization_result_name": "GPTQ-INT4-W4A16-G128-INT_N",
        "result_model_dir_name": result_model,
        "max_seq_len": config.max_seq_len,
        "threads": config.threads,
        "tokenizer": str(config.local_tokenizer),
        "tokenizer_sha256": sha256(config.local_tokenizer),
        "prompt_template_version": "qwen_chatml_raw_v1",
        "scorer_version": SCORER_VERSION,
        "timing_policy": {
            "compatibility_target": "PACE QNN PyTorchObserver",
            "ttft": "llama prompt_eval_time; QNN convention first_token==prompt_eval_end",
            "absolute_window": "backproject llama total_time from device process_end_ms",
            "model_load": "process_start_ms to backprojected inference_start_ms",
            "cold_start_ttft_ms": "model_load_ms + ttft_ms",
            "cold_start_e2e_latency_ms": "model_load_ms + inference_ms",
        },
        "energy_per_token_formula": "energy_j / generated_tokens",
        "edp_formula": "energy_j * inference_ms / 1000",
        "runs": {
            task: {
                "dataset": str(datasets[task]),
                "dataset_sha256": sha256(datasets[task]),
                "dataset_tasks": dataset_sizes[task],
                "requested_budget": budgets[task],
                "deadlines_ms": deadlines,
            }
            for task in args.tasks
        },
    }
    if args.resume:
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("backend", "phone_dir", "max_seq_len", "threads", "runs"):
            if original.get(key) != manifest.get(key):
                parser.error(f"resume configuration differs: {key}")
        original["timing_policy"] = manifest["timing_policy"]
        atomic_json(manifest_path, original)
        manifest = original
    else:
        atomic_json(manifest_path, manifest)

    summary_path = root / "summary.csv"
    rows = load_csv(summary_path) if args.resume else []
    for row in rows:
        backfill_cold_start_metrics(row)
    if args.resume and rows:
        write_csv(summary_path, rows)
    rows_by_id = {(str(row.get("task")), str(row.get("sample_id"))): row for row in rows}
    remote_namespace = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{device_label}_{run_id}")
    for task in args.tasks:
        for index, sample in enumerate(loaded[task], 1):
            key = (task, sample["sample_id"])
            if args.resume and key in rows_by_id and str(rows_by_id[key].get("runner_returncode")) == "0":
                print(f"[{index}/{len(loaded[task])}] {task} {sample['sample_id']}: already completed")
                continue
            raw_prompt = build_benchmark_prompt(task, sample["question"])
            formatted_prompt = qwen_chatml(raw_prompt)
            local_dir = root / task.lower() / sample["sample_id"]
            try:
                runtime = runner.run(
                    task=task,
                    sample_id=sample["sample_id"],
                    formatted_prompt=formatted_prompt,
                    output_budget=budgets[task],
                    local_dir=local_dir,
                    remote_namespace=remote_namespace,
                )
                completion = str(runtime.pop("completion"))
                row = {
                    "task": task,
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    "subject": sample["subject"],
                    "difficulty": sample["difficulty"],
                    **runtime,
                    **score_benchmark_answer(task, completion, sample["gold_answer"]),
                }
            except Exception as exc:
                row = {
                    "task": task,
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    "subject": sample["subject"],
                    "difficulty": sample["difficulty"],
                    "requested_budget_tokens": "none" if budgets[task] is None else budgets[task],
                    "returncode": -1,
                    "runner_returncode": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "score_correct": False,
                }
                if is_transport_error(exc):
                    rows_by_id[key] = row
                    rows = [
                        rows_by_id[(selected_task, selected["sample_id"])]
                        for selected_task in args.tasks
                        for selected in loaded[selected_task]
                        if (selected_task, selected["sample_id"]) in rows_by_id
                    ]
                    write_csv(summary_path, rows)
                    write_deadline_summary(
                        root / "deadline_summary.csv",
                        rows,
                        args.tasks,
                        deadlines,
                        dataset_sizes,
                    )
                    print(
                        f"[{index}/{len(loaded[task])}] {task} {sample['sample_id']} "
                        "ADB transport disconnected; stopping run",
                        flush=True,
                    )
                    raise RuntimeError(
                        f"ADB transport disconnected while running {sample['sample_id']}; "
                        "result checkpoint saved, reconnect and use --resume"
                    ) from exc
                if not args.continue_on_error:
                    rows_by_id[key] = row
                    write_csv(summary_path, list(rows_by_id.values()))
                    raise
            rows_by_id[key] = row
            rows = [
                rows_by_id[(selected_task, selected["sample_id"])]
                for selected_task in args.tasks
                for selected in loaded[selected_task]
                if (selected_task, selected["sample_id"]) in rows_by_id
            ]
            write_csv(summary_path, rows)
            write_deadline_summary(root / "deadline_summary.csv", rows, args.tasks, deadlines, dataset_sizes)
            print(
                f"[{index}/{len(loaded[task])}] {task} {sample['sample_id']} "
                f"rc={row.get('runner_returncode')} gen={row.get('generated_tokens')} "
                f"ttft={row.get('ttft_ms')}ms decode={row.get('decode_token_per_sec')} "
                f"correct={row.get('score_correct')}"
            )
            if row.get("device_connection_status") == "offline" or is_transport_error(
                row.get("error")
            ):
                raise RuntimeError(
                    f"ADB transport disconnected while running {sample['sample_id']}; "
                    "result checkpoint saved, reconnect and use --resume"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
