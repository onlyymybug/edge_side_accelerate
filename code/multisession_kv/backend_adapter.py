#!/usr/bin/env python3
"""Adapter for the QNN multi-session phone runner."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


class QnnPhoneBackend:
    def __init__(
        self,
        *,
        bundle: Path,
        serial: str,
        max_sessions: int,
        htp_mode: int,
        device_dir: str,
        raw_log: Path,
        request_timeout: float,
    ) -> None:
        host_dir = (bundle / "host").resolve()
        if not (host_dir / "multisession_client.py").is_file():
            raise FileNotFoundError(f"bundle host client missing: {host_dir}")
        sys.path.insert(0, str(host_dir))
        from multisession_client import JsonlRunner, adb_command

        args = Namespace(
            adb="adb",
            serial=serial,
            device_dir=device_dir,
            binary="qnn_llama_runner_multisession",
            model="hybrid_llama_qnn.pte",
            tokenizer=".",
            decoder_model_version="qwen2_5",
            htp_performance_mode=htp_mode,
            max_resident_sessions=max_sessions,
            request_timeout=request_timeout,
        )
        self._log = raw_log.open("w", encoding="utf-8")
        self._runner = JsonlRunner(
            adb_command(args), self._log, request_timeout=request_timeout
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runner.request(payload)

    def close(self) -> None:
        try:
            self._runner.close()
        finally:
            self._log.close()

    def __enter__(self) -> "QnnPhoneBackend":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

