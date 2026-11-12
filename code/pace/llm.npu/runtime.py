#!/usr/bin/env python3
"""Android mllm QNN AOT execution and PACE-compatible metric collection."""

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


ANSI_RE = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
PERF_RE = {
    "prefill_time": re.compile(r"Prefill time\s*:\s*([0-9.]+)\s*[μu]s"),
    "prefill_tps": re.compile(
        r"Prefill time\s*:.*?\(\s*([0-9.]+)\s+tokens/s\)"
    ),
    "decode_time": re.compile(r"Decode time\s*:\s*([0-9.]+)\s*[μu]s"),
    "decode_tps": re.compile(
        r"Decode time\s*:.*?\(\s*([0-9.]+)\s+tokens/s\)"
    ),
    "prefill_tokens": re.compile(r"Prefill tokens\s*:\s*(\d+)"),
    "decode_tokens": re.compile(r"Decode tokens\s*:\s*(\d+)"),
}
PROMPT_READY_RE = re.compile(r"PromptProcessor\.cpp:\d+\s+num_tokens:\s*\d+\s*\r?\n")
PERFORMANCE_MARKER = "============== Performance Summary ==============="
HTP_MODE_RE = re.compile(r"QNN HTP performance mode:\s*(\w+)")
QNN_CONTEXT_RE = re.compile(r"QNN context loaded successfully")
QNN_GRAPH_RE = re.compile(r"Successfully created QNNModel for graph:\s*(\S+)")
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


QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


def qwen_chatml(
    user_prompt: str, system_prompt: str = QWEN_DEFAULT_SYSTEM_PROMPT
) -> str:
    """Render the exact Qwen2.5 template used by the existing QNN runner."""
    chunks: list[str] = []
    if system_prompt:
        chunks.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    chunks.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def clean_completion(raw: str) -> tuple[str, bool]:
    """Extract generated text from mllm's combined stdout/stderr stream."""
    text = ANSI_RE.sub("", raw)
    prompt_markers = list(PROMPT_READY_RE.finditer(text))
    if prompt_markers:
        text = text[prompt_markers[-1].end() :]
    performance = text.find(PERFORMANCE_MARKER)
    if performance >= 0:
        text = text[:performance]
    text = text.lstrip("\r\n")
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


def parse_perf(log_text: str) -> dict[str, Any]:
    """Parse the mllm performance summary; runner times are microseconds."""
    text = ANSI_RE.sub("", log_text)
    prefill_us = _number(PERF_RE["prefill_time"].search(text))
    decode_us = _number(PERF_RE["decode_time"].search(text))
    prefill_ms = prefill_us / 1000.0 if prefill_us is not None else None
    decode_ms = decode_us / 1000.0 if decode_us is not None else None
    return {
        "prompt_eval_ms": prefill_ms,
        "decode_eval_ms": decode_ms,
        "inference_ms": (
            prefill_ms + decode_ms
            if prefill_ms is not None and decode_ms is not None
            else None
        ),
        "prompt_tokens": _integer(PERF_RE["prefill_tokens"].search(text), 1),
        "generated_tokens": _integer(PERF_RE["decode_tokens"].search(text), 1),
        "runner_prefill_token_per_sec": _number(PERF_RE["prefill_tps"].search(text)),
        "runner_decode_token_per_sec": _number(PERF_RE["decode_tps"].search(text)),
    }


def parse_htp(log_text: str, requested_mode: str) -> dict[str, Any]:
    """Verify the requested DCVS mode and both AOT QNN graphs were loaded."""
    text = ANSI_RE.sub("", log_text)
    mode_match = HTP_MODE_RE.search(text)
    actual_mode = mode_match.group(1) if mode_match else ""
    graphs = sorted(set(QNN_GRAPH_RE.findall(text)))
    context_loaded = QNN_CONTEXT_RE.search(text) is not None
    verified = bool(
        context_loaded
        and actual_mode == requested_mode
        and "model.0.s32" in graphs
        and "model.0.s1" in graphs
    )
    return {
        "htp_performance_requested": requested_mode,
        "htp_performance_actual": actual_mode,
        "qnn_context_loaded": context_loaded,
        "qnn_graphs_loaded": ",".join(graphs),
        "htp_execution_verified": verified,
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
class LlmNpuConfig:
    serial: str
    adb_bin: str = "adb"
    adb_server_port: str = "5037"
    phone_dir: str = "/data/local/tmp/mllm-qwen25-3b-sm8750"
    context_file: str = "qwen2.5-3B-lpbq-sha-sm8750.bin"
    local_tokenizer: Path = Path(
        "/home/lyyyy/qwen25_3b_mllm_sm8750_bundle/payload/tokenizer.json"
    )
    max_seq_len: int = 1024
    ar_len: int = 32
    htp_performance: str = "burst"
    telemetry_interval: float = 0.5
    timeout_seconds: float = 1800
    require_htp: bool = True
    allow_active_runner: bool = False

    @classmethod
    def from_env(cls) -> "LlmNpuConfig":
        serial = os.environ.get("ANDROID_SERIAL", "").strip()
        if not serial:
            raise ValueError("ANDROID_SERIAL must identify exactly one target device")
        return cls(
            serial=serial,
            adb_bin=os.environ.get("ADB_BIN", "adb"),
            adb_server_port=os.environ.get("ADB_SERVER_PORT", "5037"),
            phone_dir=os.environ.get(
                "LLM_NPU_PHONE_DIR",
                "/data/local/tmp/mllm-qwen25-3b-sm8750",
            ),
            context_file=os.environ.get(
                "LLM_NPU_CONTEXT_FILE",
                "qwen2.5-3B-lpbq-sha-sm8750.bin",
            ),
            local_tokenizer=Path(
                os.environ.get(
                    "LLM_NPU_TOKENIZER_PATH",
                    "/home/lyyyy/qwen25_3b_mllm_sm8750_bundle/payload/tokenizer.json",
                )
            ).resolve(),
            max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "1024")),
            ar_len=int(os.environ.get("LLM_NPU_AR_LEN", "32")),
            htp_performance=os.environ.get("HTP_PERFORMANCE", "burst").strip(),
            telemetry_interval=float(
                os.environ.get("TELEMETRY_INTERVAL_SECONDS", "0.5")
            ),
            timeout_seconds=float(os.environ.get("LLM_NPU_RUNNER_MAX_SECONDS", "1800")),
            require_htp=os.environ.get("LLM_NPU_REQUIRE_HTP", "1").strip().lower()
            not in {"0", "false", "no"},
            allow_active_runner=os.environ.get(
                "LLM_NPU_ALLOW_ACTIVE_RUNNER", "0"
            ).strip().lower()
            in {"1", "true", "yes"},
        )


class LlmNpuPhoneRunner:
    """Run one prompt and emit the same core runtime fields as the QNN harness."""

    def __init__(self, config: LlmNpuConfig | None = None) -> None:
        self.config = config or LlmNpuConfig.from_env()
        if self.config.max_seq_len <= 1:
            raise ValueError("MAX_SEQ_LEN must be greater than one")
        if self.config.max_seq_len != 1024:
            raise ValueError("this AOT context requires MAX_SEQ_LEN=1024")
        if self.config.ar_len != 32:
            raise ValueError("this AOT context requires LLM_NPU_AR_LEN=32")
        if self.config.htp_performance not in {
            "burst",
            "balanced",
            "low_power_saver",
        }:
            raise ValueError(
                "HTP_PERFORMANCE must be burst, balanced, or low_power_saver"
            )
        if self.config.telemetry_interval <= 0:
            raise ValueError("TELEMETRY_INTERVAL_SECONDS must be positive")
        if not self.config.local_tokenizer.is_file():
            raise FileNotFoundError(
                f"Qwen tokenizer is missing: {self.config.local_tokenizer}"
            )
        self.tokenizer = Tokenizer.from_file(str(self.config.local_tokenizer))
        self._check_phone()

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
        active = self.adb(
            "shell",
            "for p in $(pidof mllm-qwen2-aot-runner 2>/dev/null); do "
            "printf 'pid=%s command=' \"$p\"; "
            "tr '\\000' ' ' < /proc/$p/cmdline 2>/dev/null; echo; done; exit 0",
            timeout=15,
        )
        if active.stdout.strip() and not self.config.allow_active_runner:
            raise RuntimeError(
                "another mllm runner is already active on the target; refusing "
                "to contaminate either experiment. Wait for it to finish, or set "
                "LLM_NPU_ALLOW_ACTIVE_RUNNER=1 only for an intentional diagnostic: "
                f"{_compact(active.stdout)}"
            )
        required = (
            "mllm-qwen2-aot-runner",
            "run_on_phone.sh",
            "config_3B_1024.json",
            "tokenizer.json",
            "libMllmQNNBackend.so",
            "libMllmRT.so",
            "libQnnHtp.so",
            "libQnnHtpV79Skel.so",
            "libQnnLLaMAPackage_HTP.so",
            self.config.context_file,
        )
        command = " && ".join(
            f"test -f {shlex.quote(self.config.phone_dir + '/' + name)}"
            for name in required
        )
        checked = self.adb("shell", command, timeout=30)
        if checked.returncode != 0:
            raise RuntimeError(
                f"llm.npu bundle incomplete under {self.config.phone_dir}: "
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
            # Conservatively invalidate consumption energy instead of losing an
            # otherwise completed inference after a transient wireless-ADB stall.
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

        cli_args = ["./run_on_phone.sh", "--no_template"]
        raw_remote = remote_dir + "/raw_output.txt"
        stderr_remote = remote_dir + "/runner_stderr.log"
        meta_remote = remote_dir + "/runner_meta.txt"
        command = " ".join(
            [
                f"cd {shlex.quote(self.config.phone_dir)} || exit 1;",
                f"rm -f {shlex.quote(raw_remote)} {shlex.quote(stderr_remote)} "
                f"{shlex.quote(meta_remote)};",
                f"printf '__PACE_PROCESS_START_MS__=%s\\n' \"$(date +%s%3N)\" "
                f"> {shlex.quote(meta_remote)};",
                f"HTP_PERFORMANCE={shlex.quote(self.config.htp_performance)} "
                f"GEN_LEN={effective_budget} {shlex.join(cli_args)} "
                f"< {shlex.quote(remote_dir + '/prompt.txt')} "
                f"> {shlex.quote(raw_remote)} 2> {shlex.quote(stderr_remote)};",
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
            runner_process_name="mllm-qwen2-aot-runner",
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
        combined_log = raw + ("\n" + stderr if stderr else "")
        completion, ended_with_eos = clean_completion(combined_log)
        (local_dir / "completion.txt").write_text(completion, encoding="utf-8")

        perf = parse_perf(combined_log)
        htp = parse_htp(combined_log, self.config.htp_performance)
        process_start = _meta(meta, "__PACE_PROCESS_START_MS__")
        process_end = _meta(meta, "__PACE_PROCESS_END_MS__")
        rc_value = _meta(meta, "runner_returncode")
        runner_rc = -1 if transport_error else (rc_value if rc_value is not None else shell.returncode)
        prompt_eval_error = ""
        if runner_rc == 0 and perf["prompt_tokens"] is None:
            runner_rc = -3
            prompt_eval_error = "mllm performance summary did not report prefill tokens"
        htp_error = ""
        if runner_rc == 0 and self.config.require_htp and not htp["htp_execution_verified"]:
            runner_rc = -2
            htp_error = (
                "HTP execution verification failed: requested mode or QNN s32/s1 "
                "graph evidence is missing"
            )
        prompt_tokens = perf["prompt_tokens"] or estimated_prompt_tokens
        visible_output_tokens = len(
            self.tokenizer.encode(completion, add_special_tokens=False).ids
        )
        generated_tokens = (
            int(perf["generated_tokens"])
            if perf["generated_tokens"] is not None
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

        inference_ms = perf["inference_ms"]
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
        decode_latency_ms = perf["decode_eval_ms"]
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
            "timing_source": "mllm_perf_backprojected_from_device_process_end",
            **htp,
        }
        runner_log = (
            combined_log.rstrip()
            + "\nLlmNpuObserver "
            + json.dumps(observer, ensure_ascii=False)
            + "\n"
            + meta
        )
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
            "error": transport_error or prompt_eval_error or htp_error,
            "requested_budget_tokens": "none" if output_budget is None else output_budget,
            "budget_tokens": effective_budget,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_token_delta": prompt_tokens - estimated_prompt_tokens,
            "prompt_token_source": "mllm_prefill_summary",
            "seq_len": self.config.max_seq_len,
            "ar_len": self.config.ar_len,
            "htp_performance": self.config.htp_performance,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generated_token_source": "mllm_decode_summary",
            "visible_output_tokens": visible_output_tokens,
            "model_load_ms": model_load_ms,
            "ttft_ms": prompt_eval_ms,
            "cold_start_ttft_ms": cold_start_ttft_ms,
            "prefill_latency_ms": prompt_eval_ms,
            "decode_latency_ms": decode_latency_ms,
            "inference_ms": inference_ms,
            "e2e_latency_ms": inference_ms,
            "cold_start_e2e_latency_ms": cold_start_e2e_latency_ms,
            "timing_source": "mllm_perf_backprojected_from_device_process_end",
            "prefill_token_per_sec": aligned_prefill_tps,
            "decode_token_per_sec": aligned_decode_tps,
            "runner_prefill_token_per_sec": perf["runner_prefill_token_per_sec"],
            "runner_decode_token_per_sec": perf["runner_decode_token_per_sec"],
            **htp,
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
