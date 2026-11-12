#!/usr/bin/env python3
"""Low-frequency Android process and thermal sampler."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class AndroidSampler:
    def __init__(self, serial: str, pid: int, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.serial = serial
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self.samples.append(self._sample())
            remaining = self.interval_seconds - (time.monotonic() - started)
            self._stop.wait(max(0, remaining))

    def _sample(self) -> dict[str, Any]:
        host_start_ns = time.time_ns()
        monotonic_start = time.monotonic()
        command = (
            f"date +%s%3N; awk '/^VmRSS:|^VmHWM:/{{print}}' /proc/{self.pid}/status; "
            "dumpsys thermalservice | grep 'Temperature{'"
        )
        try:
            result = subprocess.run(
                ["adb", "-s", self.serial, "shell", command],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            text = result.stdout
            error = result.stderr.strip() if result.returncode else ""
        except Exception as exception:
            text = ""
            error = f"{type(exception).__name__}: {exception}"

        def integer(pattern: str) -> int | None:
            match = re.search(pattern, text, re.MULTILINE)
            return int(match.group(1)) if match else None

        def last_temperature(name: str) -> float | None:
            values = re.findall(
                rf"Temperature\{{mValue=([-+0-9.]+),\s*mType=\d+,\s*mName={re.escape(name)},",
                text,
            )
            return float(values[-1]) if values else None

        return {
            "host_timestamp_ns": (host_start_ns + time.time_ns()) // 2,
            "phone_timestamp_ms": integer(r"^(\d{13})$"),
            "probe_ms": (time.monotonic() - monotonic_start) * 1000,
            "pid": self.pid,
            "rss_kb": integer(r"^VmRSS:\s+(\d+)\s+kB"),
            "vmhwm_kb": integer(r"^VmHWM:\s+(\d+)\s+kB"),
            "skin_temp_c": last_temperature("skin"),
            "nsp0_temp_c": last_temperature("nsp0"),
            "vbat_v": last_temperature("vbat"),
            "ibat_a": last_temperature("ibat"),
            "error": error,
        }

    def write(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for sample in self.samples:
                stream.write(json.dumps(sample, separators=(",", ":")) + "\n")

