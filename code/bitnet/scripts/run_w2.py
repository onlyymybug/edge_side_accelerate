#!/usr/bin/env python3
"""Run W2, GSM8K, or MATH-500 with the deployed BitNet Hybrid-1024 stack."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from math_benchmarks import (  # noqa: E402
    BENCHMARKS,
    build_benchmark_prompt,
    load_benchmark_samples,
    score_benchmark_answer,
)
from bitnet.support.runtime_telemetry import (  # noqa: E402
    RuntimeTelemetrySampler,
    summarize_inference_window,
    write_telemetry,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_w1 import (
    adb,
    adb_base,
    formatted_prompt_for_counting,
    parse_observer,
    required_env,
    sha256,
    shell_quote,
    stream_process,
)


PROMPT_TEMPLATE = (
    "Solve the math problem. Show your reasoning briefly. End with exactly: "
    "The answer is <scalar>. In that final sentence, <scalar> must be a plain "
    "integer, decimal, or a/b fraction. Do not use LaTeX, units, currency "
    "symbols, commas, or percent signs in the final answer.\n\n"
    "Problem: {question}"
)


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [
                key for key in ("sample_id", "question", "gold_answer") if key not in row
            ]
            if missing:
                raise ValueError(f"Missing {missing} at {path}:{line_number}")
            rows.append(row)
    return rows


def assistant_text(full_output: str) -> str:
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    text = full_output.rsplit(marker, 1)[-1]
    for token in ("<|eot_id|>", "<|endoftext|>", "<|im_end|>"):
        text = text.replace(token, "")
    return text.strip()


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "dataset",
        "source_id",
        "subject",
        "difficulty",
        "question",
        "gold_solution",
        "gold_answer",
        "predicted_answer",
        "normalized_predicted_answer",
        "normalized_gold_answer",
        "answer_parse_method",
        "score_method",
        "answer_correct",
        "estimated_prompt_tokens",
        "runner_prompt_tokens",
        "seq_len_prompt_tokens",
        "prompt_token_source",
        "requested_output_tokens",
        "target_output_tokens",
        "seq_len",
        "generated_tokens",
        "decode_tokens_per_second",
        "ttft_ms",
        "prefill_latency_ms",
        "decode_latency_ms",
        "inference_ms",
        "model_load_ms",
        "energy_j",
        "average_power_w",
        "energy_per_token_j",
        "edp_j_s",
        "start_voltage_v",
        "end_voltage_v",
        "start_current_a",
        "end_current_a",
        "start_skin_temp_c",
        "end_skin_temp_c",
        "max_skin_temp_c",
        "peak_rss_mb",
        "peak_vmhwm_mb",
        "memory_use_mb",
        "telemetry_sample_count",
        "power_sample_count",
        "energy_measurement_status",
        "telemetry_path",
        "returncode",
        "output_path",
        "log_path",
        "error",
    ]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_deadline_summary(
    path: Path,
    rows: list[dict[str, Any]],
    deadlines_ms: list[int],
    total_dataset_tasks: int,
) -> None:
    columns = [
        "deadline_ms",
        "num_dataset_tasks",
        "num_completed_tasks",
        "num_energy_measured_tasks",
        "num_correct_under_deadline",
        "num_measured_correct_under_deadline",
        "total_measured_energy_j",
        "provisional_energy_per_correct_task_under_deadline_j",
        "energy_per_correct_task_under_deadline_j",
        "valid_for_full_dataset_reporting",
    ]

    def number(row: dict[str, Any], key: str) -> float | None:
        try:
            value = row.get(key, "")
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    completed = [row for row in rows if str(row.get("returncode", "")) == "0"]
    measured = [
        row
        for row in completed
        if row.get("energy_measurement_status")
        == "measured_signed_battery_energy"
        and number(row, "energy_j") is not None
    ]
    total_energy_j = sum(number(row, "energy_j") or 0.0 for row in measured)
    full_coverage = (
        len(completed) == total_dataset_tasks
        and len(measured) == total_dataset_tasks
    )
    output_rows: list[dict[str, Any]] = []
    for deadline_ms in deadlines_ms:
        correct = [
            row
            for row in completed
            if str(row.get("answer_correct", "")).lower() == "true"
            and (number(row, "inference_ms") or float("inf")) <= deadline_ms
        ]
        measured_correct = [row for row in correct if row in measured]
        provisional = (
            total_energy_j / len(measured_correct) if measured_correct else ""
        )
        output_rows.append(
            {
                "deadline_ms": deadline_ms,
                "num_dataset_tasks": total_dataset_tasks,
                "num_completed_tasks": len(completed),
                "num_energy_measured_tasks": len(measured),
                "num_correct_under_deadline": len(correct),
                "num_measured_correct_under_deadline": len(measured_correct),
                "total_measured_energy_j": total_energy_j,
                "provisional_energy_per_correct_task_under_deadline_j": provisional,
                "energy_per_correct_task_under_deadline_j": (
                    provisional if full_coverage else ""
                ),
                "valid_for_full_dataset_reporting": full_coverage,
            }
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-kind",
        choices=("w2", "gsm8k", "math500"),
        default="w2",
        help="input schema and scoring policy (default: w2)",
    )
    parser.add_argument("--limit", type=int, help="run only the first N samples")
    parser.add_argument("--run-id", help="result directory name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run and skip samples completed successfully",
    )
    parser.add_argument(
        "--model-max-seq-len",
        action="store_true",
        help="use the PTE maximum total sequence length instead of an output budget",
    )
    parser.add_argument(
        "--prompt-token-calibration-csv",
        type=Path,
        help="reuse runner_prompt_tokens from a previous run for exact output budgets",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue after a failed sample",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=None,
        help="seconds between thermalservice samples (default: config.env)",
    )
    parser.add_argument(
        "--deadlines-ms",
        help="comma-separated deadlines used for aggregate metrics",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    telemetry_interval = (
        args.telemetry_interval
        if args.telemetry_interval is not None
        else float(required_env("TELEMETRY_INTERVAL_SECONDS"))
    )
    if telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")

    root = Path(required_env("BUNDLE_ROOT")).resolve()
    benchmark = args.dataset_kind.upper() if args.dataset_kind != "w2" else None
    if args.dataset_kind == "w2":
        dataset_path = Path(required_env("W2_DATASET_PATH")).resolve()
        target_output_tokens = int(required_env("W2_TARGET_OUTPUT_TOKENS"))
        task_name = "W2_reasoning"
        output_label = "w2"
        prompt_template = PROMPT_TEMPLATE
    elif args.dataset_kind == "gsm8k":
        dataset_path = Path(required_env("GSM8K_DATASET_PATH")).resolve()
        target_output_tokens = int(required_env("GSM8K_TARGET_OUTPUT_TOKENS"))
        task_name = "GSM8K"
        output_label = "gsm8k"
        prompt_template = BENCHMARKS["GSM8K"].prompt_template
    else:
        dataset_path = Path(required_env("MATH500_DATASET_PATH")).resolve()
        target_output_tokens = int(required_env("MATH500_TARGET_OUTPUT_TOKENS"))
        task_name = "MATH500"
        output_label = "math500"
        prompt_template = BENCHMARKS["MATH500"].prompt_template
    deadline_text = args.deadlines_ms or required_env("DEADLINES_MS")
    try:
        deadlines_ms = sorted({int(value) for value in deadline_text.split(",")})
    except ValueError:
        parser.error("--deadlines-ms must contain comma-separated integers")
    if not deadlines_ms or deadlines_ms[0] <= 0:
        parser.error("deadlines must be positive")
    tokenizer_path = Path(required_env("TOKENIZER_PATH")).resolve()
    device_dir = required_env("DEVICE_DIR").rstrip("/")
    max_seq_len = int(required_env("MAX_SEQ_LEN"))
    seq_len_compensation = int(required_env("W2_SEQ_LEN_COMPENSATION"))
    run_id = args.run_id or (
        f"{output_label}_{required_env('MODEL_RESULT_NAME')}_"
        f"{required_env('QUANTIZATION_RESULT_NAME')}_"
        f"{required_env('PERFORMANCE_MODE_RESULT_NAME')}"
    )
    result_dir = Path(required_env("RESULTS_ROOT")).resolve() / run_id
    summary_path = result_dir / "summary.csv"
    if args.resume:
        if not result_dir.is_dir():
            parser.error(f"cannot resume missing result directory: {result_dir}")
    else:
        result_dir.mkdir(parents=True, exist_ok=False)

    bitnet_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(bitnet_root))
    from support.w2_answer_parser import (  # pylint: disable=import-outside-toplevel
        answers_equal,
        parse_final_answer,
        parse_gold_answer,
    )

    samples = (
        load_benchmark_samples(dataset_path, benchmark)
        if benchmark is not None
        else load_samples(dataset_path)
    )
    total_dataset_tasks = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    calibrated_prompt_tokens: dict[str, int] = {}
    calibration_path: Path | None = None
    if args.prompt_token_calibration_csv is not None:
        calibration_path = args.prompt_token_calibration_csv.resolve()
        with calibration_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                sample_id = row.get("sample_id", "")
                runner_count = row.get("runner_prompt_tokens", "")
                if sample_id and runner_count:
                    calibrated_prompt_tokens[sample_id] = int(runner_count)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device_output_dir = f"{device_dir}/outputs/{output_label}_hybrid_1024/{run_id}"
    adb("shell", f"mkdir -p {shell_quote(device_output_dir)}")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": task_name,
        "device_serial": required_env("ANDROID_SERIAL"),
        "remote_model": f"{device_dir}/{required_env('REMOTE_MODEL_NAME')}",
        "local_model_sha256": sha256(root / "hybrid_llama_qnn.pte"),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "qnn_version": "2.28.0.241029232508_102474",
        "model_mode": "hybrid",
        "max_seq_len": max_seq_len,
        "requested_output_budget_tokens": (
            "none" if args.model_max_seq_len else target_output_tokens
        ),
        "sequence_length_policy": (
            "model_max" if args.model_max_seq_len else "prompt_plus_output_budget"
        ),
        "seq_len_compensation": seq_len_compensation,
        "prompt_token_calibration_csv": (
            str(calibration_path) if calibration_path is not None else None
        ),
        "prompt_token_calibration_sha256": (
            sha256(calibration_path) if calibration_path is not None else None
        ),
        "budget_execution_mode": "physical",
        "htp_performance_mode": "runner_default_not_controllable",
        "token_timestamps": "unsupported_by_runner",
        "score_method": (
            "w2_numeric_exact"
            if benchmark is None
            else (
                "gsm8k_numeric_exact"
                if benchmark == "GSM8K"
                else "math500_normalized_exact_not_symbolic_equivalence"
            )
        ),
        "prompt_template": prompt_template,
        "telemetry": {
            "source": "dumpsys thermalservice HAL vbat/ibat/skin",
            "power_formula": "signed_power_w = vbat_v * ibat_a",
            "energy_formula": "trapezoidal integral over observer inference window",
            "sampling_interval_seconds": telemetry_interval,
            "memory_source": "/proc/PID/status VmRSS and VmHWM",
            "boundary_policy": "linear interpolation to inference_start_ms/inference_end_ms",
        },
        "deadlines_ms": deadlines_ms,
        "energy_per_correct_task_under_deadline_formula": (
            "sum energy of all tasks / count(correct tasks with inference_ms <= deadline)"
        ),
        "deadline_latency_field": "inference_ms",
        "edp_formula": "energy_j * inference_ms / 1000",
        "energy_per_token_formula": "energy_j / generated_tokens",
        "memory_use_field": "process VmHWM converted from KiB to MiB",
        "measurement_scope_note": (
            "battery-level signed energy includes the whole phone and telemetry overhead; "
            "it is not isolated NPU energy"
        ),
    }
    manifest_path = result_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"cannot resume without manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "task",
            "device_serial",
            "remote_model",
            "local_model_sha256",
            "dataset_sha256",
            "max_seq_len",
            "requested_output_budget_tokens",
            "sequence_length_policy",
        )
        mismatches = [
            key
            for key in compatibility_keys
            if existing_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            parser.error(
                "resume configuration differs from the original run: "
                + ", ".join(mismatches)
            )
        existing_manifest.update(
            {
                "telemetry": manifest["telemetry"],
                "deadlines_ms": deadlines_ms,
                "energy_per_correct_task_under_deadline_formula": manifest[
                    "energy_per_correct_task_under_deadline_formula"
                ],
                "deadline_latency_field": manifest["deadline_latency_field"],
                "edp_formula": manifest["edp_formula"],
                "energy_per_token_formula": manifest["energy_per_token_formula"],
                "memory_use_field": manifest["memory_use_field"],
                "measurement_scope_note": manifest["measurement_scope_note"],
            }
        )
        manifest_path.write_text(
            json.dumps(existing_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    existing_rows = load_summary(summary_path) if args.resume else []
    completed_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in existing_rows:
        sample_id = row.get("sample_id", "")
        sample_dir = result_dir / sample_id
        if (
            sample_id
            and row.get("returncode") == "0"
            and (sample_dir / "output.txt").is_file()
            and (sample_dir / "run.log").is_file()
        ):
            row["output_path"] = str(sample_dir / "output.txt")
            row["log_path"] = str(sample_dir / "run.log")
            completed_ids.add(sample_id)
            rows.append(row)
    if args.resume:
        print(
            f"Resuming {run_id}: {len(completed_ids)} completed samples will be skipped",
            flush=True,
        )
        write_summary(summary_path, rows)
        write_deadline_summary(
            result_dir / "deadline_summary.csv",
            rows,
            deadlines_ms,
            total_dataset_tasks,
        )

    any_failure = False
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        if sample_id in completed_ids:
            print(f"[{index}/{len(samples)}] {sample_id}: already completed, skipping")
            continue
        prompt = (
            build_benchmark_prompt(benchmark, sample["question"])
            if benchmark is not None
            else PROMPT_TEMPLATE.format(**sample)
        )
        estimated_prompt_tokens = len(
            tokenizer.encode(formatted_prompt_for_counting(prompt)).ids
        )
        seq_len_prompt_tokens = estimated_prompt_tokens
        prompt_token_source = "host_tokenizer_estimate"
        if calibration_path is not None:
            if sample_id not in calibrated_prompt_tokens:
                raise RuntimeError(
                    f"{sample_id}: no runner_prompt_tokens in {calibration_path}"
                )
            seq_len_prompt_tokens = calibrated_prompt_tokens[sample_id]
            prompt_token_source = "previous_runner_observation"

        sample_output_capacity = target_output_tokens
        if args.model_max_seq_len:
            seq_len = max_seq_len
            sample_output_capacity = (
                max_seq_len - estimated_prompt_tokens - seq_len_compensation
            )
        elif calibration_path is not None:
            available = max_seq_len - seq_len_prompt_tokens - 1
            if available < 1:
                raise RuntimeError(
                    f"{sample_id}: calibrated prompt leaves no output capacity"
                )
            sample_output_capacity = min(target_output_tokens, available)
            seq_len = seq_len_prompt_tokens + sample_output_capacity + 1
        else:
            available = max_seq_len - estimated_prompt_tokens - seq_len_compensation
            if available < 1:
                raise RuntimeError(
                    f"{sample_id}: prompt({estimated_prompt_tokens}) leaves no "
                    f"output capacity in max_seq_len={max_seq_len}"
                )
            sample_output_capacity = min(target_output_tokens, available)
            seq_len = (
                estimated_prompt_tokens
                + sample_output_capacity
                + seq_len_compensation
            )

        sample_dir = result_dir / sample_id
        sample_dir.mkdir(exist_ok=args.resume)
        for stale_name in ("output.txt", "inference_speed.txt", "assistant.txt"):
            (sample_dir / stale_name).unlink(missing_ok=True)
        remote_output = f"{device_output_dir}/{sample_id}_output.txt"
        remote_speed = f"{device_output_dir}/{sample_id}_inference_speed.txt"
        runner_args = [
            "./qnn_llama_runner",
            "--model_path",
            required_env("REMOTE_MODEL_NAME"),
            "--tokenizer_path",
            "tokenizer.json",
            "--prompt",
            prompt,
            "--system_prompt",
            "",
            "--seq_len",
            str(seq_len),
            "--eval_mode",
            required_env("EVAL_MODE"),
            "--kv_updater",
            required_env("KV_UPDATER"),
            "--temperature",
            required_env("TEMPERATURE"),
            "--logits_scale",
            required_env("LOGITS_SCALE"),
            "--logits_offset",
            required_env("LOGITS_OFFSET"),
            "--num_iters",
            "1",
            "--output_path",
            remote_output,
            "--performance_output_path",
            remote_speed,
        ]
        remote_command = "\n".join(
            [
                f"cd {shell_quote(device_dir)} || exit 1",
                'export LD_LIBRARY_PATH="$PWD"',
                'export LD_PRELOAD="$PWD/libc++_shared.so"',
                'export ADSP_LIBRARY_PATH="$PWD${ADSP_LIBRARY_PATH:+;$ADSP_LIBRARY_PATH}"',
                "unset QNN_OP_PACKAGE_PATHS",
                shlex.join(runner_args),
            ]
        )
        command = [*adb_base(), "shell", remote_command]
        (sample_dir / "command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )

        print(
            f"\n[{index}/{len(samples)}] {sample_id}: "
            f"prompt_tokens={estimated_prompt_tokens} seq_len={seq_len}",
            flush=True,
        )
        log_path = sample_dir / "run.log"
        telemetry = RuntimeTelemetrySampler(adb, telemetry_interval)
        returncode = stream_process(command, log_path, telemetry=telemetry)
        telemetry_path = sample_dir / "telemetry.csv"
        write_telemetry(telemetry_path, telemetry.samples)
        error = ""
        output_path = sample_dir / "output.txt"
        speed_path = sample_dir / "inference_speed.txt"
        if returncode == 0:
            for remote_path, local_path in (
                (remote_output, output_path),
                (remote_speed, speed_path),
            ):
                pulled = adb(
                    "pull",
                    "-a",
                    remote_path,
                    str(local_path),
                    check=False,
                    capture_output=True,
                )
                if pulled.returncode != 0:
                    returncode = pulled.returncode
                    error = pulled.stderr.strip() or pulled.stdout.strip()
                    break

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        observer = parse_observer(log_text)
        telemetry_summary: dict[str, Any] = {
            "energy_j": "",
            "average_power_w": "",
            "start_voltage_v": "",
            "end_voltage_v": "",
            "start_current_a": "",
            "end_current_a": "",
            "start_skin_temp_c": "",
            "end_skin_temp_c": "",
            "max_skin_temp_c": "",
            "peak_rss_mb": "",
            "peak_vmhwm_mb": "",
            "memory_use_mb": "",
            "telemetry_sample_count": len(telemetry.samples),
            "power_sample_count": sum(
                sample.valid_power for sample in telemetry.samples
            ),
            "energy_measurement_status": "missing_observer_inference_window",
        }
        if "inference_start_ms" in observer and "inference_end_ms" in observer:
            telemetry_summary = summarize_inference_window(
                telemetry.samples,
                float(observer["inference_start_ms"]),
                float(observer["inference_end_ms"]),
            )
        predicted_answer = "NA"
        normalized_predicted_answer = "NA"
        normalized_gold_answer = "NA"
        parse_method = "no_output"
        score_method = (
            "w2_numeric_exact"
            if benchmark is None
            else (
                "gsm8k_numeric_exact"
                if benchmark == "GSM8K"
                else "math500_normalized_exact_not_symbolic_equivalence"
            )
        )
        answer_correct = False
        if output_path.exists():
            generated = assistant_text(
                output_path.read_text(encoding="utf-8", errors="replace")
            )
            if benchmark is not None:
                scored = score_benchmark_answer(
                    benchmark, generated, sample["gold_answer"]
                )
                predicted_answer = scored["predicted_answer"]
                normalized_predicted_answer = scored[
                    "normalized_predicted_answer"
                ]
                normalized_gold_answer = scored["normalized_gold_answer"]
                parse_method = scored["answer_parse_method"]
                score_method = scored["score_method"]
                answer_correct = bool(scored["score_correct"])
            else:
                parsed = parse_final_answer(generated)
                gold = parse_gold_answer(sample["gold_answer"])
                predicted_answer = parsed.answer
                normalized_predicted_answer = parsed.answer
                normalized_gold_answer = gold.answer
                parse_method = parsed.method
                answer_correct = answers_equal(parsed.answer, gold.answer)
            (sample_dir / "assistant.txt").write_text(
                generated + "\n", encoding="utf-8"
            )

        def elapsed(start_key: str, end_key: str) -> int | str:
            if start_key not in observer or end_key not in observer:
                return ""
            return int(observer[end_key]) - int(observer[start_key])

        generated_tokens = observer.get("generated_tokens", "")
        inference_ms = elapsed("inference_start_ms", "inference_end_ms")
        energy_j = telemetry_summary["energy_j"]
        energy_per_token_j: float | str = ""
        edp_j_s: float | str = ""
        if energy_j != "" and generated_tokens not in ("", 0):
            energy_per_token_j = float(energy_j) / int(generated_tokens)
        if energy_j != "" and inference_ms != "":
            edp_j_s = float(energy_j) * (int(inference_ms) / 1000)

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": sample.get("dataset", ""),
                "source_id": sample.get("source_id", ""),
                "subject": sample.get("subject", ""),
                "difficulty": sample.get("difficulty", ""),
                "question": sample["question"],
                "gold_solution": sample.get("gold_solution", ""),
                "gold_answer": sample["gold_answer"],
                "predicted_answer": predicted_answer,
                "normalized_predicted_answer": normalized_predicted_answer,
                "normalized_gold_answer": normalized_gold_answer,
                "answer_parse_method": parse_method,
                "score_method": score_method,
                "answer_correct": answer_correct,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "runner_prompt_tokens": observer.get("prompt_tokens", ""),
                "seq_len_prompt_tokens": seq_len_prompt_tokens,
                "prompt_token_source": prompt_token_source,
                "requested_output_tokens": target_output_tokens,
                "target_output_tokens": sample_output_capacity,
                "seq_len": seq_len,
                "generated_tokens": generated_tokens,
                "decode_tokens_per_second": observer.get(
                    "decode_token_per_sec", observer.get("tokens_per_second", "")
                ),
                "ttft_ms": elapsed("inference_start_ms", "first_token_ms"),
                "prefill_latency_ms": elapsed(
                    "inference_start_ms", "prompt_eval_end_ms"
                ),
                "decode_latency_ms": elapsed("first_token_ms", "inference_end_ms"),
                "inference_ms": inference_ms,
                "model_load_ms": elapsed("model_load_start_ms", "model_load_end_ms"),
                "energy_j": energy_j,
                "average_power_w": telemetry_summary["average_power_w"],
                "energy_per_token_j": energy_per_token_j,
                "edp_j_s": edp_j_s,
                "start_voltage_v": telemetry_summary["start_voltage_v"],
                "end_voltage_v": telemetry_summary["end_voltage_v"],
                "start_current_a": telemetry_summary["start_current_a"],
                "end_current_a": telemetry_summary["end_current_a"],
                "start_skin_temp_c": telemetry_summary["start_skin_temp_c"],
                "end_skin_temp_c": telemetry_summary["end_skin_temp_c"],
                "max_skin_temp_c": telemetry_summary["max_skin_temp_c"],
                "peak_rss_mb": telemetry_summary["peak_rss_mb"],
                "peak_vmhwm_mb": telemetry_summary["peak_vmhwm_mb"],
                "memory_use_mb": telemetry_summary["peak_vmhwm_mb"],
                "telemetry_sample_count": telemetry_summary[
                    "telemetry_sample_count"
                ],
                "power_sample_count": telemetry_summary["power_sample_count"],
                "energy_measurement_status": telemetry_summary[
                    "energy_measurement_status"
                ],
                "telemetry_path": str(telemetry_path),
                "returncode": returncode,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "error": error or ("" if returncode == 0 else "runner_failed"),
            }
        )
        write_summary(summary_path, rows)
        write_deadline_summary(
            result_dir / "deadline_summary.csv",
            rows,
            deadlines_ms,
            total_dataset_tasks,
        )
        if returncode != 0:
            any_failure = True
            if not args.continue_on_error:
                break

    print(f"\nResults: {result_dir}")
    print(f"Summary: {summary_path}")
    print(f"Deadline summary: {result_dir / 'deadline_summary.csv'}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
