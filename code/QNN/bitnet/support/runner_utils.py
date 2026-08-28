"""Shared host-side helpers for BitNet QNN experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


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
