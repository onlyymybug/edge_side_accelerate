#!/usr/bin/env python3
"""Shared Android T-MAC execution and QNN-compatible metric collection."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

CODE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(CODE_ROOT))

from common.runtime_telemetry import (  # noqa: E402
    RuntimeTelemetrySampler,
    summarize_inference_window,
    write_telemetry,
)


PERF_RE = {
    "load": re.compile(r"load time\s*=\s*([0-9.]+) ms"),
    "prompt": re.compile(
        r"prompt eval time\s*=\s*([0-9.]+) ms\s*/\s*(\d+) tokens"
        r".*?([0-9.]+) tokens per second"
    ),
    "decode": re.compile(
        r"(?<!prompt )eval time\s*=\s*([0-9.]+) ms\s*/\s*(\d+) runs"
        r".*?([0-9.]+) tokens per second"
    ),
    "total": re.compile(r"total time\s*=\s*([0-9.]+) ms\s*/\s*(\d+) tokens"),
}
TMAC_INIT_RE = re.compile(r"^\[[0-9:]+\].*?ggml_tmac_init\s*$", re.MULTILINE)
TERMINAL_MARKERS = (
    "[end of text]",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
)
TRANSPORT_MARKERS = (
    "device offline",
    "device not found",
    "error: device '",
    "no devices/emulators found",
    "device unauthorized",
    "cannot connect to daemon",
    "failed to connect to",
    "connection refused",
)


def qwen_chatml(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Render the exact Qwen ChatML boundary consumed as a raw llama.cpp prompt."""
    chunks: list[str] = []
    if system_prompt:
        chunks.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    chunks.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def clean_completion(raw: str) -> tuple[str, bool]:
    """Remove only runner diagnostics and terminal markers from raw stdout."""
    text = TMAC_INIT_RE.sub("", raw).lstrip("\r\n")
    terminal_positions = [
        position
        for marker in TERMINAL_MARKERS
        if (position := text.find(marker)) >= 0
    ]
    ended_with_eos = bool(terminal_positions)
    if terminal_positions:
        text = text[: min(terminal_positions)]
    return text.rstrip(), ended_with_eos


def _number(match: re.Match[str] | None, group: int = 1) -> float | None:
    return float(match.group(group)) if match else None


def _integer(match: re.Match[str] | None, group: int) -> int | None:
    return int(match.group(group)) if match else None


def parse_perf(stderr: str) -> dict[str, Any]:
    load = PERF_RE["load"].search(stderr)
    prompt = PERF_RE["prompt"].search(stderr)
    decode = PERF_RE["decode"].search(stderr)
    total = PERF_RE["total"].search(stderr)
    return {
        "llama_reported_load_ms": _number(load),
        "prompt_eval_ms": _number(prompt),
        "prompt_tokens": _integer(prompt, 2),
        "llama_prefill_token_per_sec": _number(prompt, 3),
        "llama_decode_eval_ms": _number(decode),
        "decode_eval_runs": _integer(decode, 2),
        "llama_decode_token_per_sec": _number(decode, 3),
        "llama_total_ms": _number(total),
        "llama_total_tokens": _integer(total, 2),
    }


def _meta(text: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}=(\d+)$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _compact(text: str) -> str:
    return " ".join(text.split())


def _transport_error(*texts: str) -> str:
    compact = _compact(" ".join(texts))
    lowered = compact.casefold()
    return compact if any(marker in lowered for marker in TRANSPORT_MARKERS) else ""


def is_transport_error(*texts: object) -> bool:
    """Return whether an exception/result message indicates a lost ADB transport."""
    return bool(_transport_error(*(str(text) for text in texts if text is not None)))


@dataclass(frozen=True)
class TmacConfig:
    serial: str
    adb_bin: str = "adb"
    adb_server_port: str = "5037"
    phone_dir: str = "/data/local/tmp/qwen25_3b_tmac_android_bundle"
    local_tokenizer: Path = Path("/home/lyyyy/qwen25_3b_phone_bundle/model/tokenizer.json")
    max_seq_len: int = 1024
    threads: int = 4
    batch_threads: int | None = None
    telemetry_interval: float = 0.5
    timeout_seconds: float = 1800

    @classmethod
    def from_env(cls) -> "TmacConfig":
        serial = os.environ.get("ANDROID_SERIAL", "").strip()
        if not serial:
            raise ValueError("ANDROID_SERIAL must identify exactly one target device")
        return cls(
            serial=serial,
            adb_bin=os.environ.get("ADB_BIN", "adb"),
            adb_server_port=os.environ.get("ADB_SERVER_PORT", "5037"),
            phone_dir=os.environ.get(
                "TMAC_PHONE_DIR", "/data/local/tmp/qwen25_3b_tmac_android_bundle"
            ),
            local_tokenizer=Path(
                os.environ.get(
                    "TMAC_TOKENIZER_PATH",
                    "/home/lyyyy/qwen25_3b_phone_bundle/model/tokenizer.json",
                )
            ).resolve(),
            max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "1024")),
            threads=int(os.environ.get("TMAC_THREADS", "4")),
            batch_threads=int(
                os.environ.get(
                    "TMAC_BATCH_THREADS", os.environ.get("TMAC_THREADS", "4")
                )
            ),
            telemetry_interval=float(
                os.environ.get("TELEMETRY_INTERVAL_SECONDS", "0.5")
            ),
            timeout_seconds=float(os.environ.get("TMAC_RUNNER_MAX_SECONDS", "1800")),
        )


class TmacPhoneRunner:
    """Run one prompt and emit the same core runtime fields as the QNN harness."""

    def __init__(self, config: TmacConfig | None = None) -> None:
        self.config = config or TmacConfig.from_env()
        if self.config.max_seq_len <= 1:
            raise ValueError("MAX_SEQ_LEN must be greater than one")
        if self.config.threads <= 0:
            raise ValueError("TMAC_THREADS must be positive")
        if self.batch_threads <= 0:
            raise ValueError("TMAC_BATCH_THREADS must be positive")
        if self.config.telemetry_interval <= 0:
            raise ValueError("TELEMETRY_INTERVAL_SECONDS must be positive")
        if not self.config.local_tokenizer.is_file():
            raise FileNotFoundError(
                f"Qwen tokenizer is missing: {self.config.local_tokenizer}"
            )
        self.tokenizer = Tokenizer.from_file(str(self.config.local_tokenizer))
        self._check_phone()

    @property
    def batch_threads(self) -> int:
        """Prefill/batch threads; defaults to decode threads for compatibility."""
        return self.config.batch_threads or self.config.threads

    def adb_command(self, *args: str) -> list[str]:
        return [
            self.config.adb_bin,
            "-P",
            self.config.adb_server_port,
            "-s",
            self.config.serial,
            *args,
        ]

    def adb(
        self,
        *args: str,
        timeout: float | None = 240,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("ADB_SERVER_SOCKET", None)
        return subprocess.run(
            self.adb_command(*args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=environment,
        )

    def _adb_sampler(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.adb(*args, timeout=kwargs.get("timeout", 240))

    def _check_phone(self) -> None:
        state = self.adb("get-state", timeout=15)
        if state.returncode != 0 or state.stdout.strip() != "device":
            raise RuntimeError(
                f"Android device unavailable: {self.config.serial}: {_compact(state.stdout)}"
            )
        required = (
            "bin/llama-cli",
            "lib/libggml.so",
            "lib/libllama.so",
            "config/kcfg.ini",
            "model/Qwen2.5-3B-Instruct-GPTQ-Int4.INT_N.gguf",
        )
        command = " && ".join(
            f"test -f {shlex.quote(self.config.phone_dir + '/' + name)}"
            for name in required
        )
        checked = self.adb("shell", command, timeout=30)
        if checked.returncode != 0:
            raise RuntimeError(
                f"T-MAC bundle incomplete under {self.config.phone_dir}: "
                f"{_compact(checked.stdout)}"
            )

    def device_identity(self) -> dict[str, str]:
        result = self.adb(
            "shell",
            "printf 'hardware_serial='; getprop ro.serialno; "
            "printf 'model='; getprop ro.product.model; "
            "printf 'soc='; getprop ro.soc.model; "
            "printf 'android='; getprop ro.build.version.release; "
            "printf 'abi='; getprop ro.product.cpu.abi",
            timeout=15,
        )
        identity: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                identity[key] = value.strip()
        return identity

    def battery_state(self) -> dict[str, Any]:
        now = time.monotonic()
        cached = getattr(self, "_battery_state_cache", None)
        cached_at = getattr(self, "_battery_state_cache_time", 0.0)
        if cached is not None and now - cached_at < 300.0:
            return dict(cached)
        result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(3):
            try:
                result = self.adb("shell", "dumpsys battery", timeout=15)
                if result.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                result = None
            if attempt < 2:
                time.sleep(1)
        if result is None or result.returncode != 0:
            return {
                "external_powered": True,
                "power_sources": "unknown_battery_state",
                "battery_status": "",
            }
        powered = {
            source.casefold(): value.casefold() == "true"
            for source, value in re.findall(
                r"^\s*(AC|USB|Wireless|Dock) powered:\s*(true|false)\s*$",
                result.stdout,
                flags=re.MULTILINE | re.IGNORECASE,
            )
        }
        status = re.search(r"^\s*status:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
        state = {
            "external_powered": any(powered.values()),
            "power_sources": "+".join(k for k, active in powered.items() if active),
            "battery_status": int(status.group(1)) if status else "",
        }
        self._battery_state_cache = state
        self._battery_state_cache_time = now
        return dict(state)

    def run(
        self,
        *,
        task: str,
        sample_id: str,
        formatted_prompt: str,
        output_budget: int | None,
        local_dir: Path,
        remote_namespace: str,
    ) -> dict[str, Any]:
        local_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = local_dir / "formatted_prompt.txt"
        prompt_path.write_text(formatted_prompt, encoding="utf-8")
        estimated_prompt_tokens = len(
            self.tokenizer.encode(formatted_prompt, add_special_tokens=False).ids
        )
        capacity = self.config.max_seq_len - estimated_prompt_tokens
        if capacity < 1:
            raise ValueError(
                f"{sample_id}: prompt_tokens={estimated_prompt_tokens} does not fit "
                f"MAX_SEQ_LEN={self.config.max_seq_len}"
            )
        effective_budget = capacity if output_budget is None else min(output_budget, capacity)
        if effective_budget <= 0:
            raise ValueError("output budget must be positive")

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample_id)
        remote_dir = (
            f"{self.config.phone_dir}/outputs/pace/{remote_namespace}/{task.lower()}/"
            f"{safe_id}"
        )
        mkdir = self.adb("shell", f"mkdir -p {shlex.quote(remote_dir)}", timeout=30)
        if mkdir.returncode != 0:
            raise RuntimeError(f"cannot create {remote_dir}: {_compact(mkdir.stdout)}")
        push = self.adb(
            "push",
            str(prompt_path),
            remote_dir + "/prompt.txt",
            timeout=120,
        )
        if push.returncode != 0:
            raise RuntimeError(f"cannot push prompt: {_compact(push.stdout)}")

        cli_args = [
            "./bin/llama-cli",
            "-m",
            "model/Qwen2.5-3B-Instruct-GPTQ-Int4.INT_N.gguf",
            "--no-mmap",
            "-t",
            str(self.config.threads),
            "-tb",
            str(self.batch_threads),
            "-c",
            str(self.config.max_seq_len),
            "-n",
            str(effective_budget),
            "-ngl",
            "0",
            "--temp",
            "0",
            "--no-display-prompt",
            "--simple-io",
            "-f",
            remote_dir + "/prompt.txt",
        ]
        raw_remote = remote_dir + "/raw_output.txt"
        stderr_remote = remote_dir + "/runner_stderr.log"
        meta_remote = remote_dir + "/runner_meta.txt"
        command = " ".join(
            [
                f"cd {shlex.quote(self.config.phone_dir)} || exit 1;",
                f"rm -f {shlex.quote(raw_remote)} {shlex.quote(stderr_remote)} "
                f"{shlex.quote(meta_remote)};",
                "export LD_LIBRARY_PATH=$PWD/lib;",
                "export TMAC_KCFG_FILE=$PWD/config/kcfg.ini;",
                f"printf '__PACE_PROCESS_START_MS__=%s\\n' \"$(date +%s%3N)\" "
                f"> {shlex.quote(meta_remote)};",
                f"{shlex.join(cli_args)} > {shlex.quote(raw_remote)} "
                f"2> {shlex.quote(stderr_remote)};",
                "pace_rc=$?;",
                f"printf '__PACE_PROCESS_END_MS__=%s\\nrunner_returncode=%s\\n' "
                f"\"$(date +%s%3N)\" \"$pace_rc\" >> {shlex.quote(meta_remote)};",
                "exit \"$pace_rc\"",
            ]
        )
        (local_dir / "command.txt").write_text(
            shlex.join(self.adb_command("shell", command)) + "\n", encoding="utf-8"
        )

        battery_start = self.battery_state()
        sampler = RuntimeTelemetrySampler(
            self._adb_sampler,
            self.config.telemetry_interval,
            runner_process_name="llama-cli",
        )
        sampler.start()
        try:
            shell = self.adb(
                "shell", command, timeout=self.config.timeout_seconds + 30
            )
        finally:
            sampler.stop()
        battery_end = self.battery_state()
        pull = self.adb("pull", "-a", remote_dir + "/.", str(local_dir), timeout=180)
        telemetry_path = local_dir / "telemetry.csv"
        write_telemetry(telemetry_path, sampler.samples)
        transport_error = _transport_error(shell.stdout, pull.stdout)

        raw_path = local_dir / "raw_output.txt"
        stderr_path = local_dir / "runner_stderr.log"
        meta_path = local_dir / "runner_meta.txt"
        raw = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.is_file() else ""
        stderr = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path.is_file()
            else ""
        )
        meta = meta_path.read_text(encoding="utf-8", errors="replace") if meta_path.is_file() else ""
        completion, ended_with_eos = clean_completion(raw)
        (local_dir / "completion.txt").write_text(completion, encoding="utf-8")

        perf = parse_perf(stderr)
        process_start = _meta(meta, "__PACE_PROCESS_START_MS__")
        process_end = _meta(meta, "__PACE_PROCESS_END_MS__")
        rc_value = _meta(meta, "runner_returncode")
        runner_rc = -1 if transport_error else (rc_value if rc_value is not None else shell.returncode)
        prompt_tokens = perf["prompt_tokens"] or estimated_prompt_tokens
        visible_output_tokens = len(
            self.tokenizer.encode(completion, add_special_tokens=False).ids
        )
        eval_runs = perf["decode_eval_runs"]
        generated_tokens = (
            int(eval_runs) + 1
            if eval_runs is not None and (completion or ended_with_eos)
            else visible_output_tokens + int(ended_with_eos)
        )
        finish_reason = (
            "eos"
            if ended_with_eos
            else "length"
            if generated_tokens >= effective_budget
            else "error"
            if runner_rc != 0
            else "unknown"
        )

        inference_ms = perf["llama_total_ms"]
        prompt_eval_ms = perf["prompt_eval_ms"]
        inference_end = float(process_end) if process_end is not None and inference_ms is not None else None
        inference_start = inference_end - float(inference_ms) if inference_end is not None else None
        prompt_eval_end = (
            inference_start + float(prompt_eval_ms)
            if inference_start is not None and prompt_eval_ms is not None
            else None
        )
        # PACE QNN records first_token_ms == prompt_eval_end_ms. Preserve that
        # convention so TTFT and prefill latency remain directly comparable.
        first_token = prompt_eval_end
        model_load_end = inference_start
        model_load_ms = (
            model_load_end - float(process_start)
            if model_load_end is not None and process_start is not None
            else None
        )
        cold_start_ttft_ms = (
            model_load_ms + float(prompt_eval_ms)
            if model_load_ms is not None and prompt_eval_ms is not None
            else None
        )
        cold_start_e2e_latency_ms = (
            model_load_ms + float(inference_ms)
            if model_load_ms is not None and inference_ms is not None
            else None
        )
        decode_latency_ms = (
            inference_end - first_token
            if inference_end is not None and first_token is not None
            else None
        )
        aligned_decode_tps = (
            generated_tokens * 1000.0 / decode_latency_ms
            if generated_tokens and decode_latency_ms and decode_latency_ms > 0
            else None
        )
        aligned_prefill_tps = (
            prompt_tokens * 1000.0 / float(prompt_eval_ms)
            if prompt_tokens and prompt_eval_ms and prompt_eval_ms > 0
            else None
        )

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
            "telemetry_sample_count": len(sampler.samples),
            "power_sample_count": sum(sample.valid_power for sample in sampler.samples),
            "energy_measurement_status": "missing_inference_window",
        }
        if inference_start is not None and inference_end is not None:
            telemetry_summary = summarize_inference_window(
                sampler.samples, inference_start, inference_end
            )
        signed_energy = telemetry_summary["energy_j"]
        external_powered = bool(
            battery_start["external_powered"] or battery_end["external_powered"]
        )
        energy_j: float | str = signed_energy
        average_power_w: float | str = telemetry_summary["average_power_w"]
        energy_status = telemetry_summary["energy_measurement_status"]
        if external_powered and signed_energy != "":
            energy_j = ""
            average_power_w = ""
            energy_status = "external_power_connected_not_valid_for_consumption"
        energy_per_token: float | str = ""
        edp: float | str = ""
        if energy_j != "" and generated_tokens:
            energy_per_token = float(energy_j) / generated_tokens
        if energy_j != "" and inference_ms is not None:
            edp = float(energy_j) * float(inference_ms) / 1000.0

        start_voltage = telemetry_summary["start_voltage_v"]
        end_voltage = telemetry_summary["end_voltage_v"]
        start_current = telemetry_summary["start_current_a"]
        end_current = telemetry_summary["end_current_a"]
        observer = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "visible_output_tokens": visible_output_tokens,
            "model_load_start_ms": process_start,
            "model_load_end_ms": model_load_end,
            "inference_start_ms": inference_start,
            "prompt_eval_end_ms": prompt_eval_end,
            "first_token_ms": first_token,
            "inference_end_ms": inference_end,
            "cold_start_ttft_ms": cold_start_ttft_ms,
            "cold_start_e2e_latency_ms": cold_start_e2e_latency_ms,
            "prefill_token_per_sec": aligned_prefill_tps,
            "decode_token_per_sec": aligned_decode_tps,
            "finish_reason": finish_reason,
            "timing_source": "llama_perf_backprojected_from_device_process_end",
        }
        runner_log = stderr.rstrip() + "\nTmacObserver " + json.dumps(observer, ensure_ascii=False) + "\n" + meta
        (local_dir / "runner.log").write_text(runner_log, encoding="utf-8")
        (local_dir / "metrics.json").write_text(
            json.dumps(observer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        return {
            "returncode": runner_rc,
            "runner_returncode": runner_rc,
            "runner_reported_returncode": rc_value,
            "adb_pull_returncode": pull.returncode,
            "device_connection_status": "offline" if transport_error else "",
            "error": transport_error,
            "requested_budget_tokens": "none" if output_budget is None else output_budget,
            "budget_tokens": effective_budget,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_token_delta": prompt_tokens - estimated_prompt_tokens,
            "prompt_token_source": "llama_prompt_eval",
            "seq_len": self.config.max_seq_len,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generated_token_source": "llama_decode_eval_runs_plus_first_token",
            "visible_output_tokens": visible_output_tokens,
            "model_load_ms": model_load_ms,
            "ttft_ms": prompt_eval_ms,
            "cold_start_ttft_ms": cold_start_ttft_ms,
            "prefill_latency_ms": prompt_eval_ms,
            "decode_latency_ms": decode_latency_ms,
            "inference_ms": inference_ms,
            "e2e_latency_ms": inference_ms,
            "cold_start_e2e_latency_ms": cold_start_e2e_latency_ms,
            "timing_source": "llama_perf_backprojected_from_device_process_end",
            "prefill_token_per_sec": aligned_prefill_tps,
            "decode_token_per_sec": aligned_decode_tps,
            "runner_decode_token_per_sec": perf["llama_decode_token_per_sec"],
            "finish_reason": finish_reason,
            "seq_len_stop": finish_reason == "length",
            "qnn_shards_restored": "",
            "output_chars": len(completion),
            "replacement_chars": completion.count("\ufffd"),
            "energy_j": energy_j,
            "average_power_w": average_power_w,
            "energy_per_token_j": energy_per_token,
            "edp_j_s": edp,
            "signed_net_battery_energy_j": signed_energy,
            "signed_average_battery_power_w": telemetry_summary["average_power_w"],
            "external_powered_start": battery_start["external_powered"],
            "external_powered_end": battery_end["external_powered"],
            "power_sources_start": battery_start["power_sources"],
            "power_sources_end": battery_end["power_sources"],
            "start_voltage_v": start_voltage,
            "end_voltage_v": end_voltage,
            "start_current_a": start_current,
            "end_current_a": end_current,
            "start_power_w": (
                float(start_voltage) * float(start_current)
                if start_voltage != "" and start_current != ""
                else ""
            ),
            "end_power_w": (
                float(end_voltage) * float(end_current)
                if end_voltage != "" and end_current != ""
                else ""
            ),
            "start_skin_temp_c": telemetry_summary["start_skin_temp_c"],
            "end_skin_temp_c": telemetry_summary["end_skin_temp_c"],
            "max_skin_temp_c": telemetry_summary["max_skin_temp_c"],
            "peak_rss_mb": telemetry_summary["peak_rss_mb"],
            "peak_vmhwm_mb": telemetry_summary["peak_vmhwm_mb"],
            "memory_use_mb": telemetry_summary["peak_vmhwm_mb"],
            "telemetry_sample_count": telemetry_summary["telemetry_sample_count"],
            "power_sample_count": telemetry_summary["power_sample_count"],
            "energy_measurement_status": energy_status,
            "telemetry_path": str(telemetry_path),
            "output_path": str(raw_path),
            "completion_path": str(local_dir / "completion.txt"),
            "log_path": str(local_dir / "runner.log"),
            "completion": completion,
        }
