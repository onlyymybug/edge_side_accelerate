#!/usr/bin/env python3
"""Run PACE W3 action prompts with the deployed BitNet Hybrid-1024 stack."""

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
        "difficulty",
        "request",
        "gold_action",
        "gold_arguments",
        "action_name",
        "action_arguments",
        "action_valid",
        "action_correct",
        "action_error_type",
        "action_error_message",
        "estimated_prompt_tokens",
        "seq_len_prompt_tokens",
        "prompt_token_source",
        "runner_prompt_tokens",
        "target_output_tokens",
        "seq_len",
        "generated_tokens",
        "decode_tokens_per_second",
        "ttft_ms",
        "inference_ms",
        "model_load_ms",
        "returncode",
        "output_path",
        "log_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="run only the first N samples")
    parser.add_argument("--run-id", help="result directory name")
    parser.add_argument(
        "--model-max-seq-len",
        action="store_true",
        help="use the PTE maximum total sequence length instead of the action budget",
    )
    parser.add_argument(
        "--prompt-token-calibration-csv",
        type=Path,
        help="reuse runner_prompt_tokens from a previous run for exact output budgets",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    root = Path(required_env("BUNDLE_ROOT")).resolve()
    dataset_path = Path(required_env("W3_DATASET_PATH")).resolve()
    config_path = Path(required_env("W3_CONFIG_PATH")).resolve()
    tokenizer_path = Path(required_env("TOKENIZER_PATH")).resolve()
    device_dir = required_env("DEVICE_DIR").rstrip("/")
    max_seq_len = int(required_env("MAX_SEQ_LEN"))
    target_output_tokens = int(required_env("W3_TARGET_OUTPUT_TOKENS"))
    run_id = args.run_id or f"w3_{datetime.now():%Y%m%d_%H%M%S}"
    result_dir = Path(required_env("RESULTS_ROOT")).resolve() / run_id
    result_dir.mkdir(parents=True, exist_ok=False)

    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    from support.w3_action_data import build_action_prompt, read_action_samples
    from support.w3_action_validator import (
        ACTION_VALIDATOR_VERSION,
        classify_action,
        parse_action_text,
        validate_action_schema,
        validate_action_setup,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    action_schema = experiment["action_schema"]
    prompt_template = experiment["prompt_template"]
    samples = read_action_samples(dataset_path)
    validate_action_setup(action_schema, samples)
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
    device_output_dir = f"{device_dir}/outputs/w3_hybrid_1024/{run_id}"
    adb("shell", f"mkdir -p {shell_quote(device_output_dir)}")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "W3_action",
        "device_serial": required_env("ANDROID_SERIAL"),
        "remote_model": f"{device_dir}/{required_env('REMOTE_MODEL_NAME')}",
        "local_model_sha256": sha256(root / "hybrid_llama_qnn.pte"),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "pace_config": str(config_path),
        "qnn_version": "2.28.0.241029232508_102474",
        "model_mode": "hybrid",
        "max_seq_len": max_seq_len,
        "action_budget_tokens": (
            "none" if args.model_max_seq_len else target_output_tokens
        ),
        "sequence_length_policy": (
            "model_max" if args.model_max_seq_len else "prompt_plus_action_budget"
        ),
        "prompt_token_calibration_csv": (
            str(calibration_path) if calibration_path is not None else None
        ),
        "prompt_token_calibration_sha256": (
            sha256(calibration_path) if calibration_path is not None else None
        ),
        "validator_version": ACTION_VALIDATOR_VERSION,
        "htp_performance_mode": "runner_default_not_controllable",
        "token_timestamps": "unsupported_by_runner",
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    any_failure = False
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        prompt = build_action_prompt(sample, prompt_template, action_schema)
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

        if args.model_max_seq_len:
            seq_len = max_seq_len
        elif calibration_path is not None:
            seq_len = seq_len_prompt_tokens + target_output_tokens + 1
        else:
            seq_len = estimated_prompt_tokens + target_output_tokens + 3
        if seq_len > max_seq_len:
            raise RuntimeError(f"{sample_id}: required seq_len {seq_len} > {max_seq_len}")

        sample_dir = result_dir / sample_id
        sample_dir.mkdir()
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
        returncode = stream_process(command, log_path)
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

        observer = parse_observer(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
        action_name: Any = "NA"
        action_arguments: Any = "NA"
        action_valid = False
        action_correct = False
        action_error_type = "parse_error"
        action_error_message = "no output"
        if output_path.exists():
            generated = assistant_text(
                output_path.read_text(encoding="utf-8", errors="replace")
            )
            (sample_dir / "assistant.txt").write_text(
                generated + "\n", encoding="utf-8"
            )
            parsed = parse_action_text(generated)
            validation = validate_action_schema(parsed, action_schema)
            (
                action_valid,
                action_correct,
                action_error_type,
                action_error_message,
            ) = classify_action(
                parsed,
                validation,
                str(sample["gold_action"]),
                sample["gold_arguments"],
            )
            action_name = parsed.action
            if isinstance(parsed.arguments, dict):
                action_arguments = json.dumps(
                    parsed.arguments, ensure_ascii=False, sort_keys=True
                )

        def elapsed(start_key: str, end_key: str) -> int | str:
            if start_key not in observer or end_key not in observer:
                return ""
            return int(observer[end_key]) - int(observer[start_key])

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": sample.get("dataset", ""),
                "difficulty": sample.get("difficulty", ""),
                "request": sample["request"],
                "gold_action": sample["gold_action"],
                "gold_arguments": json.dumps(
                    sample["gold_arguments"], ensure_ascii=False, sort_keys=True
                ),
                "action_name": action_name,
                "action_arguments": action_arguments,
                "action_valid": action_valid,
                "action_correct": action_correct,
                "action_error_type": action_error_type,
                "action_error_message": action_error_message,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "seq_len_prompt_tokens": seq_len_prompt_tokens,
                "prompt_token_source": prompt_token_source,
                "runner_prompt_tokens": observer.get("prompt_tokens", ""),
                "target_output_tokens": (
                    max_seq_len - estimated_prompt_tokens - 3
                    if args.model_max_seq_len
                    else target_output_tokens
                ),
                "seq_len": seq_len,
                "generated_tokens": observer.get("generated_tokens", ""),
                "decode_tokens_per_second": observer.get(
                    "decode_token_per_sec", observer.get("tokens_per_second", "")
                ),
                "ttft_ms": elapsed("inference_start_ms", "first_token_ms"),
                "inference_ms": elapsed("inference_start_ms", "inference_end_ms"),
                "model_load_ms": elapsed("model_load_start_ms", "model_load_end_ms"),
                "returncode": returncode,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "error": error or ("" if returncode == 0 else "runner_failed"),
            }
        )
        write_summary(result_dir / "summary.csv", rows)
        if returncode != 0:
            any_failure = True
            if not args.continue_on_error:
                break

    print(f"\nResults: {result_dir}")
    print(f"Summary: {result_dir / 'summary.csv'}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
