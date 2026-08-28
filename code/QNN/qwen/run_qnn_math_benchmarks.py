#!/usr/bin/env python3
"""Run GSM8K and MATH-500 with a deployed Qwen/Llama QNN bundle."""

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
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
QNN_CODE = SCRIPT_DIR.parent
BUNDLE = Path(
    os.environ.get("QWEN_BUNDLE_ROOT", "/home/lyyyy/qwen25_3b_phone_bundle")
).resolve()
MODEL_ROOT = Path(
    os.environ.get(
        "QWEN_MODEL_ROOT",
        "/home/lyyyy/qwen25_3b_qnn_artifacts_hybrid_1024_balanced",
    )
).resolve()
PACE_CODE = Path(
    os.environ.get("PACE_CODE_ROOT", str(SCRIPT_DIR.parents[1]))
).resolve()
PACE_ROOT = PACE_CODE.parent
sys.path.insert(0, str(PACE_CODE))
sys.path.insert(0, str(QNN_CODE))

from math_benchmarks import (  # noqa: E402
    BENCHMARKS,
    SCORER_VERSION,
    build_benchmark_prompt,
    load_benchmark_samples,
    score_benchmark_answer,
)
from common.runtime_telemetry import (  # noqa: E402
    RuntimeTelemetrySampler,
    summarize_inference_window,
    write_telemetry,
)
ADB = Path(
    os.environ.get("ADB_BIN", "adb")
)
ADB_SERVER_PORT = os.environ.get("ADB_SERVER_PORT", "5037")
SERIAL = os.environ.get("ANDROID_SERIAL", "3B15B800A0R00000")
PHONE_DIR = os.environ.get(
    "PHONE_DIR", "/data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced"
)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", str(PACE_ROOT / "results" / "QNN"))
).resolve()
DEVICE_LABEL = os.environ.get("DEVICE_LABEL", "oneplus")
MODEL_RESULT_NAME = os.environ.get("MODEL_RESULT_NAME", "qwen2.5-3b")
QUANTIZATION_RESULT_NAME = os.environ.get(
    "QUANTIZATION_RESULT_NAME", "w4a16-block16"
)
PERFORMANCE_MODE_RESULT_NAME = os.environ.get(
    "PERFORMANCE_MODE_RESULT_NAME", "balanced"
)
RESULT_MODEL_DIR_NAME = os.environ.get(
    "RESULT_MODEL_DIR_NAME", "qwen2.5-3B-w4a16"
)
TELEMETRY_INTERVAL_SECONDS = float(
    os.environ.get("TELEMETRY_INTERVAL_SECONDS", "0.5")
)
DEFAULT_DEADLINES_MS = os.environ.get(
    "DEADLINES_MS", "5000,10000,20000,30000"
)
HTP_PERFORMANCE_MODE_NAME = os.environ.get(
    "HTP_PERFORMANCE_MODE_NAME", "balanced_runtime_override"
)
QNN_RUNNER_MAX_SECONDS = float(os.environ.get("QNN_RUNNER_MAX_SECONDS", "1800"))
RUNNER_MODEL_PROFILE = os.environ.get("RUNNER_MODEL_PROFILE", "qwen").strip().lower()
BITNET_RAW_PROMPT = os.environ.get("BITNET_RAW_PROMPT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
DECODER_MODEL_VERSION = os.environ.get(
    "DECODER_MODEL_VERSION",
    "llama3" if RUNNER_MODEL_PROFILE == "llama" else "qwen2_5",
).strip()
QNN_RUNNER_NAME = os.environ.get(
    "QNN_RUNNER_NAME", "qnn_llama_runner_htp_mode"
).strip()
QNN_RUNNER_LOCAL_PATH = Path(
    os.environ.get(
        "QNN_RUNNER_LOCAL_PATH",
        "/home/lyyyy/qwen25_runner_htp_mode/qnn_llama_runner",
    )
).resolve()
HTP_PERFORMANCE_MODE = int(os.environ.get("HTP_PERFORMANCE_MODE", "8"))
REMOTE_MODEL_NAME = os.environ.get("REMOTE_MODEL_NAME", "hybrid_llama_qnn.pte")
QNN_RUNNER_REMOTE_PATH = os.environ.get(
    "QNN_RUNNER_REMOTE_PATH", f"{PHONE_DIR}/{QNN_RUNNER_NAME}"
).strip()
BITNET_KV_UPDATER = os.environ.get("KV_UPDATER", "ShiftPointer")
BITNET_LOGITS_SCALE = os.environ.get("LOGITS_SCALE", "0.0012398984981700778")
BITNET_LOGITS_OFFSET = os.environ.get("LOGITS_OFFSET", "32310")
OBSERVER_RE = re.compile(r"PyTorchObserver\s+(\{.*?\})")
QWEN_BENCHMARK_CHAT_TEMPLATE_VERSION = "qwen2.5_default_system_user_v1"
QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
COMPLETION_TERMINATION_POLICY = (
    "truncate_at_first_llama_eos_v1"
    if RUNNER_MODEL_PROFILE == "llama"
    else "truncate_at_first_qwen_eos_v1"
)
ADB_TRANSPORT_FAILURE_MARKERS = (
    "device offline",
    "device not found",
    "no devices/emulators found",
    "device unauthorized",
    "cannot connect to daemon",
    "failed to connect to",
    "connection refused",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def windows_path(path: Path) -> str:
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())], text=True
    ).strip()


def adb_local_path(path: Path) -> str:
    return windows_path(path) if str(ADB).lower().endswith(".exe") else str(path.resolve())


def adb(
    *args: str,
    timeout: float | None = 240,
    check: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(ADB), "-P", ADB_SERVER_PORT, "-s", SERIAL, *args],
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        timeout=timeout,
    )
    if check:
        result.check_returncode()
    return result


def adb_transport_failure(*messages: str | None) -> str:
    """Return the first ADB transport failure message, if one is present."""
    for message in messages:
        if not message:
            continue
        normalized = " ".join(str(message).split())
        lowered = normalized.casefold()
        device_not_found = re.search(
            r"(?:error:\s*)?device(?:\s+\S+)?\s+not found", lowered
        )
        if device_not_found or any(
            marker in lowered for marker in ADB_TRANSPORT_FAILURE_MARKERS
        ):
            return normalized
    return ""


def battery_power_state() -> dict[str, Any]:
    result = adb("shell", "dumpsys battery", timeout=15)
    powered: dict[str, bool] = {}
    for source, value in re.findall(
        r"^\s*(AC|USB|Wireless|Dock) powered:\s*(true|false)\s*$",
        result.stdout or "",
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        powered[source.casefold()] = value.casefold() == "true"
    return {
        "external_powered": any(powered.values()),
        "power_sources": "+".join(
            source for source, active in powered.items() if active
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_identity() -> dict[str, str]:
    command = (
        "printf 'hardware_serial='; getprop ro.serialno; "
        "printf 'model='; getprop ro.product.model; "
        "printf 'soc='; getprop ro.soc.model; "
        "printf 'abi='; getprop ro.product.cpu.abi"
    )
    result = adb("shell", command, timeout=15, check=True)
    identity: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            identity[key.strip()] = value.strip()
    return identity


def observer_metrics(log_text: str) -> dict[str, Any]:
    match = OBSERVER_RE.search(log_text)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    start = data.get("inference_start_ms")
    end = data.get("inference_end_ms")
    first = data.get("first_token_ms")
    load_start = data.get("model_load_start_ms")
    load_end = data.get("model_load_end_ms")
    data["model_load_ms"] = (
        load_end - load_start if load_start is not None and load_end is not None else None
    )
    data["ttft_ms"] = first - start if first is not None and start is not None else None
    data["e2e_latency_ms"] = end - start if end is not None and start is not None else None
    prompt_end = data.get("prompt_eval_end_ms")
    data["prefill_latency_ms"] = (
        prompt_end - start if prompt_end is not None and start is not None else None
    )
    data["decode_latency_ms"] = (
        end - first if end is not None and first is not None else None
    )
    data["cold_start_ttft_ms"] = (
        data["model_load_ms"] + data["ttft_ms"]
        if data["model_load_ms"] is not None and data["ttft_ms"] is not None
        else None
    )
    data["cold_start_e2e_latency_ms"] = (
        data["model_load_ms"] + data["e2e_latency_ms"]
        if data["model_load_ms"] is not None
        and data["e2e_latency_ms"] is not None
        else None
    )
    if data.get("decode_token_per_sec") is None:
        data["decode_token_per_sec"] = data.get("tokens_per_second")
    return data


def backfill_cold_start_metrics(row: dict[str, Any]) -> None:
    """Add cold-start latency fields to rows produced before these fields existed."""
    for target, second in (
        ("cold_start_ttft_ms", "ttft_ms"),
        ("cold_start_e2e_latency_ms", "e2e_latency_ms"),
    ):
        if row.get(target) not in (None, ""):
            continue
        if row.get("model_load_ms") in (None, "") or row.get(second) in (None, ""):
            row[target] = ""
            continue
        row[target] = float(row["model_load_ms"]) + float(row[second])


def completion_from_output(
    raw: str, prompt: str, runner_model_profile: str | None = None
) -> str:
    profile = runner_model_profile or RUNNER_MODEL_PROFILE
    prompt_removed = False
    if raw.startswith(prompt):
        raw = raw[len(prompt) :].lstrip()
        prompt_removed = True
    if not prompt_removed:
        markers = (
            ("<|start_header_id|>assistant<|end_header_id|>",)
            if profile in ("bitnet", "llama")
            else (
                "<|im_start|>assistant\n",
                "<|start_header_id|>assistant<|end_header_id|>",
            )
        )
        for marker in markers:
            expected_position = prompt.rfind(marker)
            positions = [match.start() for match in re.finditer(re.escape(marker), raw)]
            if positions:
                # The legacy Llama runner always appends its own assistant
                # header after --prompt. Nested diagnostic templates may also
                # contain assistant markers, so the outermost/last one is the
                # actual generation boundary.
                if profile in ("bitnet", "llama"):
                    boundary = positions[-1]
                else:
                    boundary = (
                        min(
                            positions,
                            key=lambda position: abs(position - expected_position),
                        )
                        if expected_position >= 0
                        else positions[0]
                    )
                raw = raw[boundary + len(marker) :].lstrip()
                break
    terminal_positions = [
        position
        for token in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>")
        if (position := raw.find(token)) >= 0
    ]
    if terminal_positions:
        raw = raw[: min(terminal_positions)]
    return raw.rstrip()


def qwen_benchmark_chat_prompt(user_prompt: str) -> str:
    """Apply the Qwen2.5 chat protocol used for GSM8K and MATH500."""
    return (
        f"<|im_start|>system\n{QWEN_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def runner_prompt_for_counting(prompt: str, system_prompt: str = "") -> str:
    """Render the chat wrapper added internally by the legacy Llama runner."""
    if RUNNER_MODEL_PROFILE not in ("bitnet", "llama"):
        return prompt
    if RUNNER_MODEL_PROFILE == "bitnet" and BITNET_RAW_PROMPT:
        return prompt
    chunks: list[str] = []
    if system_prompt:
        chunks.append(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + system_prompt
            + "<|eot_id|>"
        )
    chunks.append(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + prompt
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    # tokenizer.json inserts BOS, so it is intentionally omitted here.
    return "".join(chunks)


def prompt_for(task: str, sample: dict[str, Any], cfg: dict[str, Any]) -> str:
    del cfg
    return build_benchmark_prompt(task, sample["question"])


def score(task: str, sample: dict[str, Any], completion: str, cfg: dict[str, Any]) -> dict[str, Any]:
    del cfg
    return score_benchmark_answer(task, completion, sample["gold_answer"])


def run_sample(
    *,
    task: str,
    sample: dict[str, Any],
    cfg: dict[str, Any],
    budget: int | None,
    tokenizer: Tokenizer,
    remote_root: str,
    local_root: Path,
    telemetry_interval: float,
    prompt_builder: Callable[[str, dict[str, Any], dict[str, Any]], str] | None = None,
    system_prompt_builder: Callable[
        [str, dict[str, Any], dict[str, Any]], str
    ]
    | None = None,
    score_builder: Callable[
        [str, dict[str, Any], str, dict[str, Any]], dict[str, Any]
    ]
    | None = None,
    tokenizer_path_arg: str = "tokenizer.json",
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    raw_prompt = (prompt_builder or prompt_for)(task, sample, cfg)
    system_prompt = (
        system_prompt_builder(task, sample, cfg) if system_prompt_builder else ""
    )
    prompt = raw_prompt
    if RUNNER_MODEL_PROFILE == "qwen" and task in BENCHMARKS:
        prompt = qwen_benchmark_chat_prompt(raw_prompt)
    prompt_for_counting = runner_prompt_for_counting(prompt, system_prompt)
    estimated_prompt_tokens = len(tokenizer.encode(prompt_for_counting).ids)
    available_output_tokens = MAX_SEQ_LEN - estimated_prompt_tokens
    if available_output_tokens < 1:
        raise ValueError(
            f"{sample_id}: prompt_tokens={estimated_prompt_tokens} leaves no output "
            f"capacity in max_seq_len={MAX_SEQ_LEN}"
        )
    # A missing task-level budget means "generate until EOS or the PTE context
    # limit".  The PTE still has a hard total sequence limit, so this is not an
    # unbounded allocation: all remaining context slots are made available.
    effective_budget = (
        available_output_tokens
        if budget is None
        else min(budget, available_output_tokens)
    )
    seq_len = estimated_prompt_tokens + effective_budget

    remote_dir = f"{remote_root}/{task.lower()}/{sample_id}"
    local_dir = local_root / task.lower() / sample_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "generated_text.txt",
        "speed.txt",
        "runner.log",
        "completion.txt",
        "raw_prompt.txt",
        "system_prompt.txt",
        "telemetry.csv",
        "telemetry.raw",
    ):
        (local_dir / stale_name).unlink(missing_ok=True)
    (local_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if system_prompt:
        (local_dir / "system_prompt.txt").write_text(
            system_prompt, encoding="utf-8"
        )
    if prompt != raw_prompt:
        (local_dir / "raw_prompt.txt").write_text(raw_prompt, encoding="utf-8")

    if RUNNER_MODEL_PROFILE == "bitnet":
        runner_args = [
            f"./{QNN_RUNNER_NAME}",
            "--model_path", REMOTE_MODEL_NAME,
            "--tokenizer_path", tokenizer_path_arg,
            "--prompt", prompt,
            "--system_prompt", system_prompt,
            "--seq_len", str(seq_len),
            "--eval_mode", "1",
            "--kv_updater", BITNET_KV_UPDATER,
            "--temperature", "0",
            "--logits_scale", BITNET_LOGITS_SCALE,
            "--logits_offset", BITNET_LOGITS_OFFSET,
            "--htp_performance_mode", str(HTP_PERFORMANCE_MODE),
            "--num_iters", "1",
            "--output_path", f"{remote_dir}/generated_text.txt",
            "--performance_output_path", f"{remote_dir}/speed.txt",
        ]
        if BITNET_RAW_PROMPT:
            runner_args.append("--raw_prompt")
        runtime_environment = [
            "export LD_LIBRARY_PATH=$PWD;",
            "export LD_PRELOAD=$PWD/libc++_shared.so;",
            "export ADSP_LIBRARY_PATH=$PWD;",
            "unset QNN_OP_PACKAGE_PATHS;",
        ]
    else:
        runner_args = [
            QNN_RUNNER_REMOTE_PATH,
            "--decoder_model_version", DECODER_MODEL_VERSION,
            "--model_path", REMOTE_MODEL_NAME,
            # EOS ids are loaded from tokenizer_config.json, so this must be
            # the directory containing both tokenizer files.
            "--tokenizer_path", ".",
            "--prompt", prompt,
            "--system_prompt", system_prompt,
            "--seq_len", str(seq_len),
            "--eval_mode", "1",
            "--temperature", "0",
            "--num_iters", "1",
            "--shared_buffer",
            "--htp_performance_mode", str(HTP_PERFORMANCE_MODE),
            "--output_path", f"{remote_dir}/generated_text.txt",
            "--performance_output_path", f"{remote_dir}/speed.txt",
        ]
        runtime_environment = [
            "export LD_LIBRARY_PATH=$PWD;",
            "export ADSP_LIBRARY_PATH=$PWD;",
            "export QNN_OP_PACKAGE_PATHS='';",
        ]
    runner_log_remote = f"{remote_dir}/runner.log"
    # The runner is invoked plainly and its output redirected to runner.log
    # on-device; the exit code is appended as a marker line for the rc regex.
    # Telemetry is NOT embedded in this command: an on-device dumpsys/proc
    # sampling loop competes with the runner for SoC cycles and corrupts the
    # measured decode speed and energy.  Instead RuntimeTelemetrySampler polls
    # thermalservice host-side over a separate adb connection (see below).
    command = " ".join(
        [
            f"cd {shlex.quote(PHONE_DIR)} || exit 1;",
            f"mkdir -p {shlex.quote(remote_dir)};",
            f"rm -f {shlex.quote(remote_dir + '/generated_text.txt')} "
            f"{shlex.quote(remote_dir + '/speed.txt')} "
            f"{shlex.quote(remote_dir + '/telemetry.raw')} "
            f"{shlex.quote(runner_log_remote)};",
            *runtime_environment,
            f"{shlex.join(runner_args)} > {shlex.quote(runner_log_remote)} 2>&1;",
            f"printf 'runner_returncode=%s\\n' \"$?\" "
            f">> {shlex.quote(runner_log_remote)}",
        ]
    )
    (local_dir / "command.txt").write_text(
        shlex.join([str(ADB), "-P", ADB_SERVER_PORT, "-s", SERIAL, "shell", command])
        + "\n",
        encoding="utf-8",
    )
    battery_state_start = battery_power_state()
    # Derive the process name from the executable actually placed in this
    # sample's command. This stays correct for both Qwen and BitNet profiles
    # even when QNN_RUNNER_NAME is overridden by the environment.
    runner_process_name = Path(runner_args[0]).name
    sampler = RuntimeTelemetrySampler(
        adb,
        telemetry_interval,
        runner_process_name=runner_process_name,
    )
    sampler.start()
    try:
        shell_result = adb("shell", command, timeout=QNN_RUNNER_MAX_SECONDS + 30)
    finally:
        sampler.stop()
    battery_state_end = battery_power_state()
    pull_result = adb("pull", "-a", remote_dir + "/.", adb_local_path(local_dir))
    telemetry = sampler.samples
    telemetry_path = local_dir / "telemetry.csv"
    write_telemetry(telemetry_path, telemetry)

    log_path = local_dir / "runner.log"
    output_path = local_dir / "generated_text.txt"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    raw_output = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    completion = completion_from_output(raw_output, prompt)
    (local_dir / "completion.txt").write_text(completion, encoding="utf-8")

    metrics = observer_metrics(log_text)
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
        "telemetry_sample_count": len(telemetry),
        "power_sample_count": sum(sample.valid_power for sample in telemetry),
        "energy_measurement_status": "missing_observer_inference_window",
    }
    if metrics.get("inference_start_ms") is not None and metrics.get(
        "inference_end_ms"
    ) is not None:
        telemetry_summary = summarize_inference_window(
            telemetry,
            float(metrics["inference_start_ms"]),
            float(metrics["inference_end_ms"]),
        )
    rc_match = re.search(r"runner_returncode=(\d+)", log_text)
    reported_runner_rc = (
        int(rc_match.group(1)) if rc_match else shell_result.returncode
    )
    transport_error = adb_transport_failure(
        shell_result.stdout,
        pull_result.stdout,
        *(sample.error for sample in telemetry),
    )
    # A successful/missing remote return code cannot make a sample valid when
    # the ADB transport disappeared before its artifacts were collected.
    runner_rc = -1 if transport_error else reported_runner_rc
    generated_tokens = metrics.get("generated_tokens")
    inference_ms = metrics.get("e2e_latency_ms")
    energy_j = telemetry_summary["energy_j"]
    external_powered = bool(
        battery_state_start["external_powered"]
        or battery_state_end["external_powered"]
    )
    if external_powered and energy_j != "":
        telemetry_summary["signed_net_battery_energy_j"] = energy_j
        telemetry_summary["signed_average_battery_power_w"] = telemetry_summary[
            "average_power_w"
        ]
        telemetry_summary["energy_j"] = ""
        telemetry_summary["average_power_w"] = ""
        telemetry_summary[
            "energy_measurement_status"
        ] = "external_power_connected_not_valid_for_consumption"
        energy_j = ""
    energy_per_token_j: float | str = ""
    edp_j_s: float | str = ""
    if energy_j != "" and generated_tokens not in (None, "", 0):
        energy_per_token_j = float(energy_j) / int(generated_tokens)
    if energy_j != "" and inference_ms not in (None, ""):
        edp_j_s = float(energy_j) * (float(inference_ms) / 1000)
    start_power_w: float | str = ""
    end_power_w: float | str = ""
    if (
        telemetry_summary["start_voltage_v"] != ""
        and telemetry_summary["start_current_a"] != ""
    ):
        start_power_w = float(telemetry_summary["start_voltage_v"]) * float(
            telemetry_summary["start_current_a"]
        )
    if (
        telemetry_summary["end_voltage_v"] != ""
        and telemetry_summary["end_current_a"] != ""
    ):
        end_power_w = float(telemetry_summary["end_voltage_v"]) * float(
            telemetry_summary["end_current_a"]
        )
    row: dict[str, Any] = {
        "task": task,
        "sample_id": sample_id,
        "dataset": sample.get("dataset", task),
        "source_id": sample.get("source_id", ""),
        "subject": sample.get("subject", ""),
        "difficulty": sample.get("difficulty", ""),
        "requested_budget_tokens": "none" if budget is None else budget,
        "budget_tokens": effective_budget,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "seq_len": seq_len,
        "returncode": runner_rc,
        "runner_returncode": runner_rc,
        "runner_reported_returncode": reported_runner_rc,
        "adb_pull_returncode": pull_result.returncode,
        "device_connection_status": "offline" if transport_error else "",
        "error": transport_error,
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": generated_tokens,
        "model_load_ms": metrics.get("model_load_ms"),
        "ttft_ms": metrics.get("ttft_ms"),
        "cold_start_ttft_ms": metrics.get("cold_start_ttft_ms"),
        "prefill_latency_ms": metrics.get("prefill_latency_ms"),
        "decode_latency_ms": metrics.get("decode_latency_ms"),
        "inference_ms": inference_ms,
        "e2e_latency_ms": metrics.get("e2e_latency_ms"),
        "cold_start_e2e_latency_ms": metrics.get(
            "cold_start_e2e_latency_ms"
        ),
        "prefill_token_per_sec": metrics.get("prefill_token_per_sec"),
        "decode_token_per_sec": metrics.get("decode_token_per_sec"),
        "seq_len_stop": "Generation stopped at seq_len limit" in log_text,
        "qnn_shards_restored": log_text.count(
            "Use cached delegate handle for current method: kv_forward"
        ),
        "output_chars": len(completion),
        "replacement_chars": completion.count("\ufffd"),
        "energy_j": energy_j,
        "average_power_w": telemetry_summary["average_power_w"],
        "energy_per_token_j": energy_per_token_j,
        "edp_j_s": edp_j_s,
        "start_voltage_v": telemetry_summary["start_voltage_v"],
        "end_voltage_v": telemetry_summary["end_voltage_v"],
        "start_current_a": telemetry_summary["start_current_a"],
        "end_current_a": telemetry_summary["end_current_a"],
        "start_power_w": start_power_w,
        "end_power_w": end_power_w,
        "start_skin_temp_c": telemetry_summary["start_skin_temp_c"],
        "end_skin_temp_c": telemetry_summary["end_skin_temp_c"],
        "max_skin_temp_c": telemetry_summary["max_skin_temp_c"],
        "peak_rss_mb": telemetry_summary["peak_rss_mb"],
        "peak_vmhwm_mb": telemetry_summary["peak_vmhwm_mb"],
        "memory_use_mb": telemetry_summary["peak_vmhwm_mb"],
        "telemetry_sample_count": telemetry_summary["telemetry_sample_count"],
        "power_sample_count": telemetry_summary["power_sample_count"],
        "energy_measurement_status": telemetry_summary[
            "energy_measurement_status"
        ],
        "signed_net_battery_energy_j": telemetry_summary.get(
            "signed_net_battery_energy_j", ""
        ),
        "signed_average_battery_power_w": telemetry_summary.get(
            "signed_average_battery_power_w", ""
        ),
        "external_powered_start": battery_state_start["external_powered"],
        "external_powered_end": battery_state_end["external_powered"],
        "power_sources_start": battery_state_start["power_sources"],
        "power_sources_end": battery_state_end["power_sources"],
        "telemetry_path": str(telemetry_path),
        "output_path": str(output_path),
        "log_path": str(log_path),
    }
    row.update((score_builder or score)(task, sample, completion, cfg))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_deadline_summary(
    path: Path,
    rows: list[dict[str, Any]],
    tasks: list[str],
    deadlines_by_task: dict[str, list[int]],
    total_tasks_by_task: dict[str, int],
) -> None:
    columns = [
        "task",
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

    output_rows: list[dict[str, Any]] = []
    for task in tasks:
        selected = [row for row in rows if row.get("task") == task]
        completed = [
            row for row in selected if str(row.get("runner_returncode", "")) == "0"
        ]
        measured = [
            row
            for row in completed
            if row.get("energy_measurement_status")
            == "measured_signed_battery_energy"
            and number(row, "energy_j") is not None
        ]
        total_energy_j = sum(number(row, "energy_j") or 0.0 for row in measured)
        full_coverage = (
            len(completed) == total_tasks_by_task[task]
            and len(measured) == total_tasks_by_task[task]
        )
        for deadline_ms in deadlines_by_task[task]:
            correct = [
                row
                for row in completed
                if str(row.get("score_correct", "")).lower() == "true"
                and number(row, "inference_ms") is not None
                and float(row["inference_ms"]) <= deadline_ms
            ]
            measured_ids = {
                (row.get("task"), row.get("sample_id")) for row in measured
            }
            measured_correct = [
                row
                for row in correct
                if (row.get("task"), row.get("sample_id")) in measured_ids
            ]
            provisional = (
                total_energy_j / len(measured_correct) if measured_correct else ""
            )
            output_rows.append(
                {
                    "task": task,
                    "deadline_ms": deadline_ms,
                    "num_dataset_tasks": total_tasks_by_task[task],
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
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_path.replace(path)


def summarize(rows: list[dict[str, Any]], tasks: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        correct_values = [row.get("score_correct") for row in selected]
        scored = [
            value if isinstance(value, bool) else str(value).lower() == "true"
            for value in correct_values
            if isinstance(value, bool) or str(value).lower() in ("true", "false")
        ]
        numeric = lambda key: [
            float(row[key]) for row in selected if row.get(key) not in (None, "")
        ]
        summary[task] = {
            "samples": len(selected),
            "runtime_successes": sum(
                str(row["runner_returncode"]) == "0" for row in selected
            ),
            "scored_samples": len(scored),
            "correct": sum(scored),
            "accuracy": sum(scored) / len(scored) if scored else None,
            "mean_ttft_ms": mean(numeric("ttft_ms")) if numeric("ttft_ms") else None,
            "mean_cold_start_ttft_ms": (
                mean(numeric("cold_start_ttft_ms"))
                if numeric("cold_start_ttft_ms")
                else None
            ),
            "mean_e2e_latency_ms": mean(numeric("e2e_latency_ms")) if numeric("e2e_latency_ms") else None,
            "mean_cold_start_e2e_latency_ms": (
                mean(numeric("cold_start_e2e_latency_ms"))
                if numeric("cold_start_e2e_latency_ms")
                else None
            ),
            "mean_decode_token_per_sec": (
                mean(numeric("decode_token_per_sec")) if numeric("decode_token_per_sec") else None
            ),
        }
    return summary


def optional_output_budget(value: str) -> int | None:
    """Parse a fixed output budget or select all remaining model context."""
    if value.lower() in ("none", "no_budget"):
        return None
    try:
        budget = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "budget must be a positive integer, none, or no_budget"
        ) from exc
    if budget < 1:
        raise argparse.ArgumentTypeError("budget must be positive")
    return budget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="fixed result directory name")
    parser.add_argument("--tag", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run and skip successfully measured samples",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue after a failed sample",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=lambda value: value.upper().replace("-", ""),
        choices=("GSM8K", "MATH500"),
        default=("GSM8K", "MATH500"),
        help="benchmarks to run; default: GSM8K MATH500",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N samples of each selected task",
    )
    parser.add_argument(
        "--gsm8k-budget",
        type=optional_output_budget,
        default=BENCHMARKS["GSM8K"].default_budget,
        help="output-token limit, or no_budget to use all remaining context",
    )
    parser.add_argument(
        "--math500-budget",
        type=optional_output_budget,
        default=BENCHMARKS["MATH500"].default_budget,
        help="output-token limit, or no_budget to use all remaining context",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="local parent directory for this run (default: pace/results)",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=TELEMETRY_INTERVAL_SECONDS,
        help="seconds between thermalservice samples",
    )
    parser.add_argument(
        "--deadlines-ms",
        help="comma-separated deadlines overriding task configuration",
    )
    args = parser.parse_args()
    if args.run_id and args.tag:
        parser.error("use --run-id, not both --run-id and legacy --tag")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")
    if RUNNER_MODEL_PROFILE not in ("qwen", "llama", "bitnet"):
        parser.error("RUNNER_MODEL_PROFILE must be qwen, llama, or bitnet")
    if not QNN_RUNNER_NAME or "/" in QNN_RUNNER_NAME:
        parser.error("QNN_RUNNER_NAME must be a plain remote filename")
    if not QNN_RUNNER_REMOTE_PATH:
        parser.error("QNN_RUNNER_REMOTE_PATH must not be empty")
    if not 0 <= HTP_PERFORMANCE_MODE <= 8:
        parser.error("HTP_PERFORMANCE_MODE must be an integer from 0 through 8")

    device = adb("get-state", timeout=15)
    if device.returncode != 0 or device.stdout.strip() != "device":
        raise RuntimeError(f"phone is not connected or authorized: {SERIAL}")
    identity = device_identity()
    required_remote_files = (
        "hybrid_llama_qnn.pte",
        "tokenizer.json",
        "tokenizer_config.json",
        "libqnn_executorch_backend.so",
        "libQnnHtp.so",
        "libQnnHtpV79Stub.so",
        "libQnnHtpV79Skel.so",
        "libQnnHtpPrepare.so",
        "libQnnSystem.so",
    )
    remote_check = " && ".join(
        f"test -f {shlex.quote(PHONE_DIR + '/' + filename)}"
        for filename in required_remote_files
    )
    remote_check += f" && test -f {shlex.quote(QNN_RUNNER_REMOTE_PATH)}"
    if adb("shell", remote_check, timeout=20).returncode != 0:
        raise RuntimeError(
            f"{RUNNER_MODEL_PROFILE} runtime/model is incomplete: {PHONE_DIR}"
        )

    model_path = MODEL_ROOT / "hybrid_llama_qnn.pte"
    tokenizer_path = MODEL_ROOT / "tokenizer.json"
    tokenizer_config_path = MODEL_ROOT / "tokenizer_config.json"
    runner_path = QNN_RUNNER_LOCAL_PATH
    backend_path = BUNDLE / "runtime/libqnn_executorch_backend.so"
    if not all(
        path.is_file()
        for path in (
            model_path,
            tokenizer_path,
            tokenizer_config_path,
            runner_path,
            backend_path,
        )
    ):
        raise FileNotFoundError(
            f"missing {RUNNER_MODEL_PROFILE} artifact under {MODEL_ROOT}"
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_definitions = {
        "GSM8K": (
            PACE_ROOT / "datasets/gsm8k_test.jsonl",
            None,
            args.gsm8k_budget,
        ),
        "MATH500": (
            PACE_ROOT / "datasets/math500_test.jsonl",
            None,
            args.math500_budget,
        ),
    }
    tasks = list(dict.fromkeys(args.tasks))
    definitions = [(task, *all_definitions[task]) for task in tasks]
    task_label = "_".join(task.lower() for task in tasks)
    single_benchmark = len(tasks) == 1
    if single_benchmark:
        benchmark_budget = (
            args.gsm8k_budget if tasks[0] == "GSM8K" else args.math500_budget
        )
        default_run_id = (
            "output_budget_no_budget"
            if benchmark_budget is None
            else f"output_budget_{benchmark_budget}"
        )
        default_output_parent = (
            RESULTS_ROOT
            / DEVICE_LABEL
            / tasks[0].lower()
            / RESULT_MODEL_DIR_NAME
            / PERFORMANCE_MODE_RESULT_NAME
        )
    else:
        default_run_id = (
            f"{task_label}_{MODEL_RESULT_NAME}_{QUANTIZATION_RESULT_NAME}_"
            f"{PERFORMANCE_MODE_RESULT_NAME}"
        )
        default_output_parent = RESULTS_ROOT / DEVICE_LABEL
    run_id = args.run_id or args.tag or default_run_id
    output_parent = (
        args.output_root.resolve() if args.output_root else default_output_parent
    )
    local_root = output_parent / run_id
    summary_path = local_root / "summary.csv"
    manifest_path = local_root / "manifest.json"
    if args.resume:
        if not local_root.is_dir():
            parser.error(f"cannot resume missing result directory: {local_root}")
    else:
        local_root.mkdir(parents=True, exist_ok=False)
    remote_root = f"{PHONE_DIR}/outputs/{run_id}"

    loaded_all: dict[str, list[dict[str, Any]]] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    total_tasks_by_task: dict[str, int] = {}
    deadlines_by_task: dict[str, list[int]] = {}
    override_deadlines: list[int] | None = None
    if args.deadlines_ms:
        try:
            override_deadlines = sorted(
                {int(value) for value in args.deadlines_ms.split(",")}
            )
        except ValueError:
            parser.error("--deadlines-ms must contain comma-separated integers")
        if not override_deadlines or override_deadlines[0] <= 0:
            parser.error("deadlines must be positive")
    default_deadlines = [int(value) for value in DEFAULT_DEADLINES_MS.split(",")]
    for task, dataset_path, cfg_path, _ in definitions:
        samples = load_benchmark_samples(dataset_path, task)
        loaded_all[task] = samples
        loaded[task] = samples[: args.limit] if args.limit is not None else samples
        total_tasks_by_task[task] = len(samples)
        cfg = load_json(cfg_path) if cfg_path is not None else {}
        configs[task] = cfg
        configured = cfg.get("experiment", {}).get("deadlines_ms", default_deadlines)
        deadlines_by_task[task] = override_deadlines or [int(value) for value in configured]

    manifest = {
        "run_id": run_id,
        "device_label": DEVICE_LABEL,
        "model_result_name": MODEL_RESULT_NAME,
        "quantization_result_name": QUANTIZATION_RESULT_NAME,
        "performance_mode_result_name": PERFORMANCE_MODE_RESULT_NAME,
        "timestamp": datetime.now().astimezone().isoformat(),
        "adb_transport_serial": SERIAL,
        "device_identity": identity,
        "phone_dir": PHONE_DIR,
        "local_model": str(model_path),
        "local_model_sha256": sha256(model_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "tokenizer_config_sha256": sha256(tokenizer_config_path),
        "runner_tokenizer_path": ".",
        "completion_termination_policy": COMPLETION_TERMINATION_POLICY,
        "local_runner": str(runner_path),
        "local_runner_sha256": sha256(runner_path),
        "remote_runner": QNN_RUNNER_REMOTE_PATH,
        "local_backend_sha256": sha256(backend_path),
        "qnn_version": "2.37",
        "model_mode": "hybrid",
        "decoder_model_version": DECODER_MODEL_VERSION,
        "runner_model_profile": RUNNER_MODEL_PROFILE,
        "eval_mode": 1,
        "temperature": 0,
        "shared_buffer": True,
        "budget_execution_mode": "physical",
        "scorer_version": SCORER_VERSION,
        "format_required_for_correctness": False,
        "token_timestamps": "unsupported_by_runner",
        "qnn_op_package_paths": "",
        "max_seq_len": MAX_SEQ_LEN,
        "htp_performance_mode": HTP_PERFORMANCE_MODE,
        "htp_performance_mode_name": HTP_PERFORMANCE_MODE_NAME,
        "latency_fields": {
            "ttft_ms": "first_token_ms - inference_start_ms; excludes model load",
            "e2e_latency_ms": "inference_end_ms - inference_start_ms; excludes model load",
            "cold_start_ttft_ms": "model_load_ms + ttft_ms",
            "cold_start_e2e_latency_ms": "model_load_ms + e2e_latency_ms",
        },
        "telemetry": {
            "source": "dumpsys thermalservice HAL vbat/ibat/skin",
            "collection_mode": "host-side ADB polling",
            "power_formula": "signed_power_w = vbat_v * ibat_a",
            "energy_formula": "trapezoidal integral over observer inference window",
            "sampling_interval_seconds": args.telemetry_interval,
            "memory_source": "/proc/PID/status VmRSS and VmHWM",
            "boundary_policy": (
                "linear interpolation to inference_start_ms/inference_end_ms"
            ),
            "watchdog_seconds": QNN_RUNNER_MAX_SECONDS,
            "external_power_policy": (
                "retain signed net battery flow but invalidate consumption metrics"
            ),
        },
        "energy_per_token_formula": "energy_j / generated_tokens",
        "edp_formula": "energy_j * inference_ms / 1000",
        "memory_use_field": "process VmHWM converted from KiB to MiB",
        "measurement_scope_note": (
            "battery-level signed energy includes the whole phone and telemetry overhead; "
            "it is not isolated NPU energy"
        ),
        "runs": {
            task: {
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
                "dataset_tasks": total_tasks_by_task[task],
                "requested_budget": budget,
                "prompt_template_version": (
                    QWEN_BENCHMARK_CHAT_TEMPLATE_VERSION
                    if RUNNER_MODEL_PROFILE == "qwen" and task in BENCHMARKS
                    else (
                        "llama3_runner_chat_template_v1"
                        if RUNNER_MODEL_PROFILE == "llama" and task in BENCHMARKS
                        else "runner_or_task_specific"
                    )
                ),
                "deadlines_ms": deadlines_by_task[task],
            }
            for task, dataset, _, budget in definitions
        },
    }
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"cannot resume without manifest: {manifest_path}")
        existing_manifest = load_json(manifest_path)
        compatibility_checks = {
            "phone_dir": existing_manifest.get("phone_dir") == PHONE_DIR,
            "local_model_sha256": existing_manifest.get("local_model_sha256")
            == manifest["local_model_sha256"],
            "tokenizer_sha256": existing_manifest.get("tokenizer_sha256")
            == manifest["tokenizer_sha256"],
            "tokenizer_config_sha256": existing_manifest.get(
                "tokenizer_config_sha256"
            )
            == manifest["tokenizer_config_sha256"],
            "runner_tokenizer_path": existing_manifest.get(
                "runner_tokenizer_path"
            )
            == manifest["runner_tokenizer_path"],
            "completion_termination_policy": existing_manifest.get(
                "completion_termination_policy"
            )
            == manifest["completion_termination_policy"],
            "local_runner_sha256": existing_manifest.get(
                "local_runner_sha256", manifest["local_runner_sha256"]
            )
            == manifest["local_runner_sha256"],
            "local_backend_sha256": existing_manifest.get(
                "local_backend_sha256", manifest["local_backend_sha256"]
            )
            == manifest["local_backend_sha256"],
            "device_hardware_serial": existing_manifest.get("device_identity", {}).get(
                "hardware_serial"
            )
            == identity.get("hardware_serial"),
            "runs": existing_manifest.get("runs") == manifest["runs"],
        }
        mismatches = [key for key, matches in compatibility_checks.items() if not matches]
        if mismatches:
            parser.error(
                "resume configuration differs from the original run: "
                + ", ".join(mismatches)
            )
        existing_manifest.update(
            {
                "adb_transport_serial": SERIAL,
                "local_runner": manifest["local_runner"],
                "local_runner_sha256": manifest["local_runner_sha256"],
                "local_backend_sha256": manifest["local_backend_sha256"],
                "qnn_version": manifest["qnn_version"],
                "model_mode": manifest["model_mode"],
                "budget_execution_mode": manifest["budget_execution_mode"],
                "scorer_version": manifest["scorer_version"],
                "format_required_for_correctness": False,
                "token_timestamps": manifest["token_timestamps"],
                "telemetry": manifest["telemetry"],
                "latency_fields": manifest["latency_fields"],
            }
        )
        manifest_path.write_text(
            json.dumps(existing_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    existing_rows = load_csv(summary_path) if args.resume else []
    rows: list[dict[str, Any]] = []
    completed_ids: set[tuple[str, str]] = set()
    for row in existing_rows:
        backfill_cold_start_metrics(row)
        task = str(row.get("task", ""))
        sample_id = str(row.get("sample_id", ""))
        sample_dir = local_root / task.lower() / sample_id
        if (
            task in tasks
            and sample_id
            and str(row.get("runner_returncode", "")) == "0"
            and row.get("energy_measurement_status")
            == "measured_signed_battery_energy"
            and (sample_dir / "generated_text.txt").is_file()
            and (sample_dir / "runner.log").is_file()
            and (sample_dir / "telemetry.csv").is_file()
        ):
            row["output_path"] = str(sample_dir / "generated_text.txt")
            row["log_path"] = str(sample_dir / "runner.log")
            row["telemetry_path"] = str(sample_dir / "telemetry.csv")
            rows.append(row)
            completed_ids.add((task, sample_id))
    if args.resume:
        print(
            f"Resuming {run_id}: {len(completed_ids)} completed samples will be skipped",
            flush=True,
        )
        write_csv(summary_path, rows)
        write_deadline_summary(
            local_root / "deadline_summary.csv",
            rows,
            tasks,
            deadlines_by_task,
            total_tasks_by_task,
        )

    total = sum(len(samples) for samples in loaded.values())
    completed = 0
    any_failure = False
    fatal_transport_failure = False
    for task, _, cfg_path, budget in definitions:
        samples = loaded[task]
        cfg = configs[task]
        for sample in samples:
            completed += 1
            sample_id = str(sample["sample_id"])
            if (task, sample_id) in completed_ids:
                print(f"[{completed}/{total}] {task} {sample_id}: already completed, skipping")
                continue
            try:
                row = run_sample(
                    task=task,
                    sample=sample,
                    cfg=cfg,
                    budget=budget,
                    tokenizer=tokenizer,
                    remote_root=remote_root,
                    local_root=local_root,
                    telemetry_interval=args.telemetry_interval,
                )
            except Exception as exception:
                any_failure = True
                transport_error = adb_transport_failure(str(exception))
                row = {
                    "task": task,
                    "sample_id": sample_id,
                    "dataset": sample.get("dataset", task),
                    "runner_returncode": -1,
                    "returncode": -1,
                    "energy_measurement_status": "runner_or_telemetry_exception",
                    "error": f"{type(exception).__name__}: {exception}",
                    "device_connection_status": (
                        "offline" if transport_error else ""
                    ),
                }
            rows.append(row)
            write_csv(summary_path, rows)
            write_deadline_summary(
                local_root / "deadline_summary.csv",
                rows,
                tasks,
                deadlines_by_task,
                total_tasks_by_task,
            )
            correctness = row.get("score_correct", "unscored")
            print(
                f"[{completed}/{total}] {task} {row['sample_id']} rc={row['runner_returncode']} "
                f"prompt={row.get('prompt_tokens')} gen={row.get('generated_tokens')} "
                f"ttft={row.get('ttft_ms')}ms decode={row.get('decode_token_per_sec')} "
                f"correct={correctness}",
                flush=True,
            )
            if row.get("device_connection_status") == "offline":
                any_failure = True
                fatal_transport_failure = True
                print(
                    "Fatal ADB transport failure detected; stopping the batch "
                    "immediately even though --continue-on-error is enabled. "
                    f"Details: {row.get('error', 'device offline')}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if int(row.get("runner_returncode", -1)) != 0:
                any_failure = True
                if not args.continue_on_error:
                    break
        if fatal_transport_failure or (any_failure and not args.continue_on_error):
            break

    summary = summarize(rows, tasks)
    aggregate_path = local_root / "aggregate.json"
    aggregate_temporary = aggregate_path.with_suffix(".json.tmp")
    aggregate_temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    aggregate_temporary.replace(aggregate_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results={local_root}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
