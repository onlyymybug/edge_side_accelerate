"""Sample battery, skin temperature, and runner RSS through Android ADB."""

from __future__ import annotations

import csv
import math
import re
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_TEMPERATURE_RE = re.compile(
    r"Temperature\{mValue=([-+0-9.eE]+),\s*mType=\d+,\s*"
    r"mName=([^,}]+),\s*mStatus=(-?\d+)\}"
)
_INTEGER_FIELD_RE = re.compile(r"^([A-Za-z_]+)=(\d+)$", re.MULTILINE)


@dataclass
class TelemetrySample:
    timestamp_ms: float
    host_timestamp_ms: float
    adb_latency_ms: float
    voltage_v: float | None
    current_a: float | None
    power_w: float | None
    skin_temp_c: float | None
    thermal_status: int | None
    runner_pid: int | None
    rss_kb: int | None
    vmhwm_kb: int | None
    valid_power: bool
    error: str

    def as_row(self) -> dict[str, Any]:
        return {
            "timestamp_ms": f"{self.timestamp_ms:.3f}",
            "host_timestamp_ms": f"{self.host_timestamp_ms:.3f}",
            "adb_latency_ms": f"{self.adb_latency_ms:.3f}",
            "voltage_v": "" if self.voltage_v is None else self.voltage_v,
            "current_a": "" if self.current_a is None else self.current_a,
            "power_w": "" if self.power_w is None else self.power_w,
            "skin_temp_c": "" if self.skin_temp_c is None else self.skin_temp_c,
            "thermal_status": (
                "" if self.thermal_status is None else self.thermal_status
            ),
            "runner_pid": "" if self.runner_pid is None else self.runner_pid,
            "rss_kb": "" if self.rss_kb is None else self.rss_kb,
            "vmhwm_kb": "" if self.vmhwm_kb is None else self.vmhwm_kb,
            "valid_power": self.valid_power,
            "error": self.error,
        }


class RuntimeTelemetrySampler:
    """Poll OEM thermalservice while a runner process is alive."""

    REMOTE_COMMAND = r"""
printf '__PACE_TS_BEGIN__='; date +%s%3N
dumpsys thermalservice
printf '__PACE_MEMORY__\n'
pid="$(pidof qnn_llama_runner 2>/dev/null | awk '{print $1}')"
if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
  printf 'pid=%s\n' "$pid"
  awk '/^VmRSS:/{print "rss_kb=" $2} /^VmHWM:/{print "vmhwm_kb=" $2}' "/proc/$pid/status"
fi
printf '__PACE_TS_END__='; date +%s%3N
""".strip()

    def __init__(
        self,
        adb: Callable[..., Any],
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self._adb = adb
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[TelemetrySample] = []

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler has already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        # Always take a final battery/temperature point after runner completion.
        self.samples.append(self._sample_once())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.samples.append(self._sample_once())
            remaining = self._interval_seconds - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _sample_once(self) -> TelemetrySample:
        host_start_ms = time.time_ns() / 1_000_000
        monotonic_start = time.monotonic()
        try:
            result = self._adb(
                "shell",
                self.REMOTE_COMMAND,
                check=False,
                capture_output=True,
                timeout=10,
            )
        except Exception as exception:  # Preserve the experiment on sampling failure.
            latency_ms = (time.monotonic() - monotonic_start) * 1000
            host_end_ms = time.time_ns() / 1_000_000
            return TelemetrySample(
                timestamp_ms=(host_start_ms + host_end_ms) / 2,
                host_timestamp_ms=(host_start_ms + host_end_ms) / 2,
                adb_latency_ms=latency_ms,
                voltage_v=None,
                current_a=None,
                power_w=None,
                skin_temp_c=None,
                thermal_status=None,
                runner_pid=None,
                rss_kb=None,
                vmhwm_kb=None,
                valid_power=False,
                error=f"{type(exception).__name__}: {exception}".replace("\n", " "),
            )
        latency_ms = (time.monotonic() - monotonic_start) * 1000
        host_end_ms = time.time_ns() / 1_000_000
        text = result.stdout or ""
        error = ""
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "adb_failed").strip()

        begin_match = re.search(r"__PACE_TS_BEGIN__=(\d+)", text)
        end_match = re.search(r"__PACE_TS_END__=(\d+)", text)
        if begin_match and end_match:
            timestamp_ms = (float(begin_match.group(1)) + float(end_match.group(1))) / 2
        else:
            timestamp_ms = (host_start_ms + host_end_ms) / 2
            error = error or "missing_phone_timestamp"

        readings: dict[str, tuple[float, int]] = {}
        for value, name, status in _TEMPERATURE_RE.findall(text):
            readings[name.strip().lower()] = (float(value), int(status))
        voltage = readings.get("vbat", (None, 0))[0]
        current = readings.get("ibat", (None, 0))[0]
        skin = readings.get("skin", (None, 0))[0]
        skin_status = readings.get("skin", (0, None))[1]
        valid_power = voltage is not None and current is not None
        power = voltage * current if valid_power else None
        if not valid_power:
            error = error or "missing_vbat_or_ibat"

        integer_fields = {
            key: int(value) for key, value in _INTEGER_FIELD_RE.findall(text)
        }
        return TelemetrySample(
            timestamp_ms=timestamp_ms,
            host_timestamp_ms=(host_start_ms + host_end_ms) / 2,
            adb_latency_ms=latency_ms,
            voltage_v=voltage,
            current_a=current,
            power_w=power,
            skin_temp_c=skin,
            thermal_status=skin_status,
            runner_pid=integer_fields.get("pid"),
            rss_kb=integer_fields.get("rss_kb"),
            vmhwm_kb=integer_fields.get("vmhwm_kb"),
            valid_power=valid_power,
            error=error.replace("\n", " "),
        )


def write_telemetry(path: Path, samples: list[TelemetrySample]) -> None:
    columns = list(TelemetrySample(0, 0, 0, None, None, None, None, None, None, None, None, False, "").as_row())
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sample.as_row() for sample in samples)
    temporary_path.replace(path)


def parse_phone_telemetry(path: Path) -> list[TelemetrySample]:
    """Parse samples written by the on-device telemetry loop."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    samples: list[TelemetrySample] = []
    for block in text.split("__PACE_SAMPLE_BEGIN__")[1:]:
        payload, separator, _ = block.partition("__PACE_SAMPLE_END__")
        if not separator:
            continue
        begin_match = re.search(r"__PACE_TS_BEGIN__=(\d+)", payload)
        end_match = re.search(r"__PACE_TS_END__=(\d+)", payload)
        if not begin_match or not end_match:
            continue
        begin_ms = float(begin_match.group(1))
        end_ms = float(end_match.group(1))
        readings: dict[str, tuple[float, int]] = {}
        for value, name, status in _TEMPERATURE_RE.findall(payload):
            readings[name.strip().lower()] = (float(value), int(status))
        voltage = readings.get("vbat", (None, 0))[0]
        current = readings.get("ibat", (None, 0))[0]
        skin = readings.get("skin", (None, 0))[0]
        skin_status = readings.get("skin", (0, None))[1]
        valid_power = voltage is not None and current is not None
        fields = {key: int(value) for key, value in _INTEGER_FIELD_RE.findall(payload)}
        samples.append(
            TelemetrySample(
                timestamp_ms=(begin_ms + end_ms) / 2,
                host_timestamp_ms=(begin_ms + end_ms) / 2,
                adb_latency_ms=end_ms - begin_ms,
                voltage_v=voltage,
                current_a=current,
                power_w=voltage * current if valid_power else None,
                skin_temp_c=skin,
                thermal_status=skin_status,
                runner_pid=fields.get("pid"),
                rss_kb=fields.get("rss_kb"),
                vmhwm_kb=fields.get("vmhwm_kb"),
                valid_power=valid_power,
                error="" if valid_power else "missing_vbat_or_ibat",
            )
        )
    return samples


def phone_telemetry_runner_command(
    *,
    runner_command: str,
    runner_log_path: str,
    telemetry_raw_path: str,
    telemetry_stop_path: str,
    runner_pid_path: str,
    sampler_pid_path: str,
    interval_seconds: float,
    maximum_seconds: float,
) -> str:
    """Wrap a runner and a bounded telemetry loop in one Android shell job.

    The sampler exits when the runner exits, a stop marker appears, or the
    maximum duration is reached.  Its timeout also terminates a stranded
    runner, so an ADB disconnect cannot leave either process alive forever.
    """
    if interval_seconds <= 0 or maximum_seconds <= 0:
        raise ValueError("telemetry interval and maximum duration must be positive")
    interval = format(interval_seconds, ".6g")
    maximum = int(math.ceil(maximum_seconds))
    log_q = shlex.quote(runner_log_path)
    raw_q = shlex.quote(telemetry_raw_path)
    stop_q = shlex.quote(telemetry_stop_path)
    runner_pid_q = shlex.quote(runner_pid_path)
    sampler_pid_q = shlex.quote(sampler_pid_path)
    return "\n".join(
        [
            f"rm -f {raw_q} {stop_q} {runner_pid_q} {sampler_pid_q};",
            "pace_sample() {",
            f"  printf '__PACE_SAMPLE_BEGIN__\\n' >> {raw_q};",
            f"  printf '__PACE_TS_BEGIN__=' >> {raw_q}; date +%s%3N >> {raw_q};",
            f"  dumpsys thermalservice >> {raw_q} 2>&1;",
            f"  printf '__PACE_MEMORY__\\n' >> {raw_q};",
            "  if [ -r \"/proc/$runner_pid/status\" ]; then",
            f"    printf 'pid=%s\\n' \"$runner_pid\" >> {raw_q};",
            "    awk '/^VmRSS:/{print \"rss_kb=\" $2} "
            "         /^VmHWM:/{print \"vmhwm_kb=\" $2}' "
            f"\"/proc/$runner_pid/status\" >> {raw_q};",
            "  fi;",
            f"  printf '__PACE_TS_END__=' >> {raw_q}; date +%s%3N >> {raw_q};",
            f"  printf '__PACE_SAMPLE_END__\\n' >> {raw_q};",
            "};",
            f"{runner_command} > {log_q} 2>&1 &",
            "runner_pid=$!;",
            f"printf '%s\\n' \"$runner_pid\" > {runner_pid_q};",
            "(",
            "  sampler_started=$(date +%s);",
            "  pace_sample;",
            "  while kill -0 \"$runner_pid\" 2>/dev/null; do",
            f"    [ -f {stop_q} ] && break;",
            "    sampler_now=$(date +%s);",
            f"    if [ $((sampler_now - sampler_started)) -ge {maximum} ]; then",
            f"      printf '__PACE_WATCHDOG_TIMEOUT__={maximum}\\n' >> {raw_q};",
            "      kill -TERM \"$runner_pid\" 2>/dev/null || true;",
            "      sleep 1;",
            "      kill -KILL \"$runner_pid\" 2>/dev/null || true;",
            "      break;",
            "    fi;",
            f"    sleep {interval};",
            "    pace_sample;",
            "  done;",
            "  pace_sample;",
            ") &",
            "sampler_pid=$!;",
            f"printf '%s\\n' \"$sampler_pid\" > {sampler_pid_q};",
            "pace_cleanup() {",
            f"  touch {stop_q};",
            "  kill -TERM \"$runner_pid\" 2>/dev/null || true;",
            "  wait \"$sampler_pid\" 2>/dev/null || true;",
            "};",
            "trap 'pace_cleanup; exit 130' HUP INT TERM;",
            "wait \"$runner_pid\"; runner_rc=$?;",
            f"printf '\\nrunner_returncode=%s\\n' \"$runner_rc\" >> {log_q};",
            f"touch {stop_q};",
            "wait \"$sampler_pid\" 2>/dev/null || true;",
            "trap - HUP INT TERM;",
            f"rm -f {stop_q} {runner_pid_q} {sampler_pid_q};",
            "exit \"$runner_rc\"",
        ]
    )


def phone_telemetry_cleanup_command(
    runner_pid_path: str, sampler_pid_path: str, telemetry_stop_path: str
) -> str:
    """Stop processes left by an interrupted sample before it is resumed."""
    runner_pid_q = shlex.quote(runner_pid_path)
    sampler_pid_q = shlex.quote(sampler_pid_path)
    stop_q = shlex.quote(telemetry_stop_path)
    return " ".join(
        [
            f"touch {stop_q};",
            f"if [ -r {runner_pid_q} ]; then",
            f"pace_pid=$(cat {runner_pid_q});",
            "pace_cmd=$(tr '\\000' ' ' < \"/proc/$pace_pid/cmdline\" 2>/dev/null || true);",
            "case \"$pace_cmd\" in",
            "*qnn_llama_runner*) kill -TERM \"$pace_pid\" 2>/dev/null || true;;",
            "esac; fi;",
            # The stop marker lets the sampler exit itself after its current
            # dumpsys call.  Do not kill an unverified, potentially reused PID.
            f"rm -f {runner_pid_q} {sampler_pid_q};",
        ]
    )


def summarize_inference_window(
    samples: list[TelemetrySample], start_ms: float, end_ms: float
) -> dict[str, Any]:
    """Integrate signed vbat*ibat power over observer inference boundaries."""
    valid = sorted(
        (sample for sample in samples if sample.valid_power),
        key=lambda sample: sample.timestamp_ms,
    )
    result: dict[str, Any] = {
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
        "telemetry_sample_count": len(samples),
        "power_sample_count": len(valid),
        "energy_measurement_status": "insufficient_samples",
    }
    if end_ms <= start_ms or len(valid) < 2:
        return result
    if valid[0].timestamp_ms > start_ms or valid[-1].timestamp_ms < end_ms:
        result["energy_measurement_status"] = "inference_window_not_bracketed"
        return result

    def interpolate(left: TelemetrySample, right: TelemetrySample, timestamp: float) -> dict[str, float | None]:
        span = right.timestamp_ms - left.timestamp_ms
        ratio = 0.0 if span == 0 else (timestamp - left.timestamp_ms) / span

        def value(name: str) -> float | None:
            first = getattr(left, name)
            second = getattr(right, name)
            if first is None or second is None:
                return None
            return float(first) + (float(second) - float(first)) * ratio

        return {
            "timestamp_ms": timestamp,
            "voltage_v": value("voltage_v"),
            "current_a": value("current_a"),
            "power_w": value("power_w"),
            "skin_temp_c": value("skin_temp_c"),
        }

    def boundary(timestamp: float) -> dict[str, float | None]:
        for left, right in zip(valid, valid[1:]):
            if left.timestamp_ms <= timestamp <= right.timestamp_ms:
                return interpolate(left, right, timestamp)
        raise RuntimeError("bracket check and boundary interpolation disagree")

    start_point = boundary(start_ms)
    end_point = boundary(end_ms)
    points: list[dict[str, float | None]] = [start_point]
    points.extend(
        {
            "timestamp_ms": sample.timestamp_ms,
            "voltage_v": sample.voltage_v,
            "current_a": sample.current_a,
            "power_w": sample.power_w,
            "skin_temp_c": sample.skin_temp_c,
        }
        for sample in valid
        if start_ms < sample.timestamp_ms < end_ms
    )
    points.append(end_point)
    energy_j = 0.0
    for left, right in zip(points, points[1:]):
        duration_s = (float(right["timestamp_ms"]) - float(left["timestamp_ms"])) / 1000
        energy_j += (float(left["power_w"]) + float(right["power_w"])) * 0.5 * duration_s

    memory_samples = [
        sample for sample in samples if start_ms <= sample.timestamp_ms <= end_ms
    ]
    rss_values = [sample.rss_kb for sample in memory_samples if sample.rss_kb is not None]
    vmhwm_values = [sample.vmhwm_kb for sample in memory_samples if sample.vmhwm_kb is not None]
    skin_values = [
        float(point["skin_temp_c"])
        for point in points
        if point["skin_temp_c"] is not None
    ]
    duration_s = (end_ms - start_ms) / 1000
    result.update(
        {
            "energy_j": energy_j,
            "average_power_w": energy_j / duration_s,
            "start_voltage_v": start_point["voltage_v"],
            "end_voltage_v": end_point["voltage_v"],
            "start_current_a": start_point["current_a"],
            "end_current_a": end_point["current_a"],
            "start_skin_temp_c": start_point["skin_temp_c"],
            "end_skin_temp_c": end_point["skin_temp_c"],
            "max_skin_temp_c": max(skin_values) if skin_values else "",
            "peak_rss_mb": max(rss_values) / 1024 if rss_values else "",
            "peak_vmhwm_mb": max(vmhwm_values) / 1024 if vmhwm_values else "",
            "energy_measurement_status": "measured_signed_battery_energy",
        }
    )
    return result
