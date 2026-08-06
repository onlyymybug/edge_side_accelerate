#!/usr/bin/env python3
"""Run PACE W1 prompts with the already-deployed BitNet Hybrid-1024 QNN stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


OBSERVER_RE = re.compile(r"PyTorchObserver\s+(\{.*\})")


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adb_base() -> list[str]:
    command = [required_env("ADB_BIN")]
    serial = required_env("ANDROID_SERIAL")
    if serial:
        command.extend(["-s", serial])
    return command


def adb(
    *args: str,
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*adb_base(), *args],
        check=check,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def formatted_prompt_for_counting(prompt: str) -> str:
    # The BitNet runner applies the Llama-3 chat wrapper. The tokenizer itself
    # adds BOS, so do not add <|begin_of_text|> here.
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + prompt
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row or "prompt" not in row:
                raise ValueError(f"Missing sample_id/prompt at {path}:{line_number}")
            samples.append(row)
    return samples


def parse_observer(log_text: str) -> dict[str, Any]:
    matches = OBSERVER_RE.findall(log_text)
    if not matches:
        return {}
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}


def stream_process(command: list[str], log_path: Path, telemetry: Any = None) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if telemetry is not None:
            telemetry.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return int(process.wait())
        finally:
            if telemetry is not None:
                telemetry.stop()


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "dataset",
        "difficulty",
        "prompt",
        "prompt_tokens",
        "target_output_tokens",
        "seq_len",
        "returncode",
        "generated_tokens",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "ttft_ms",
        "inference_ms",
        "model_load_ms",
        "e2e_output_tokens_per_second",
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
        "--continue-on-error",
        action="store_true",
        help="continue after a failed sample",
    )
    args = parser.parse_args()

    root = Path(required_env("BUNDLE_ROOT")).resolve()
    dataset_path = Path(required_env("DATASET_PATH")).resolve()
    tokenizer_path = Path(required_env("TOKENIZER_PATH")).resolve()
    device_dir = required_env("DEVICE_DIR").rstrip("/")
    remote_model = f"{device_dir}/{required_env('REMOTE_MODEL_NAME')}"
    max_seq_len = int(required_env("MAX_SEQ_LEN"))
    target_output_tokens = int(required_env("TARGET_OUTPUT_TOKENS"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(required_env("RESULTS_ROOT")).resolve() / run_id
    result_dir.mkdir(parents=True, exist_ok=False)

    samples = load_samples(dataset_path)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        samples = samples[: args.limit]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device_output_dir = f"{device_dir}/outputs/w1_hybrid_1024/{run_id}"
    adb("shell", f"mkdir -p {shell_quote(device_output_dir)}")

    device_props = adb(
        "shell",
        "printf 'model='; getprop ro.product.model; "
        "printf ' soc='; getprop ro.soc.model; "
        "printf ' android='; getprop ro.build.version.release; "
        "printf ' build='; getprop ro.build.display.id",
        capture_output=True,
    ).stdout.strip()

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "device_serial": required_env("ANDROID_SERIAL"),
        "device_properties": device_props,
        "device_dir": device_dir,
        "remote_model": remote_model,
        "local_model_sha256": sha256(root / "hybrid_llama_qnn.pte"),
        "tokenizer_sha256": sha256(tokenizer_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "qnn_version": "2.28.0.241029232508_102474",
        "htp_architecture": "v79",
        "model_mode": "hybrid",
        "prefill_ar_len": int(required_env("PREFILL_AR_LEN")),
        "max_seq_len": max_seq_len,
        "target_output_tokens": target_output_tokens,
        "htp_performance_mode": "runner_default_not_controllable",
        "token_timestamps": "unsupported_by_runner",
        "logits_scale": float(required_env("LOGITS_SCALE")),
        "logits_offset": int(required_env("LOGITS_OFFSET")),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    any_failure = False

    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        prompt = str(sample["prompt"])
        prompt_tokens = len(tokenizer.encode(formatted_prompt_for_counting(prompt)).ids)
        seq_len = prompt_tokens + target_output_tokens
        if seq_len > max_seq_len:
            raise RuntimeError(
                f"{sample_id}: prompt_tokens({prompt_tokens}) + "
                f"target_output_tokens({target_output_tokens}) exceeds {max_seq_len}"
            )

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
            f"prompt_tokens={prompt_tokens} seq_len={seq_len}",
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

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        observer = parse_observer(log_text)
        model_load_ms = ""
        inference_ms = ""
        ttft_ms = ""
        if observer:
            model_load_ms = observer["model_load_end_ms"] - observer["model_load_start_ms"]
            inference_ms = observer["inference_end_ms"] - observer["inference_start_ms"]
            ttft_ms = observer["first_token_ms"] - observer["inference_start_ms"]

        e2e_tps = ""
        if speed_path.exists():
            e2e_tps = speed_path.read_text(encoding="utf-8").strip()

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": sample.get("dataset", ""),
                "difficulty": sample.get("difficulty", ""),
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "target_output_tokens": target_output_tokens,
                "seq_len": seq_len,
                "returncode": returncode,
                "generated_tokens": observer.get("generated_tokens", ""),
                "prefill_tokens_per_second": observer.get("prefill_token_per_sec", ""),
                "decode_tokens_per_second": observer.get("decode_token_per_sec", observer.get("tokens_per_second", "")),
                "ttft_ms": ttft_ms,
                "inference_ms": inference_ms,
                "model_load_ms": model_load_ms,
                "e2e_output_tokens_per_second": e2e_tps,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "error": error or ("" if returncode == 0 else "runner_failed"),
            }
        )
        write_summary(result_dir / "summary.csv", rows)

        if returncode != 0:
            any_failure = True
            print(f"{sample_id} failed with return code {returncode}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(f"\nResults: {result_dir}")
    print(f"Summary: {result_dir / 'summary.csv'}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
