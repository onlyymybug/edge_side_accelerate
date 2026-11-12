#!/usr/bin/env python3
"""Short, reversible DDR-frequency probe for the QNN phone runner."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import threading
import time
from pathlib import Path

from backend_adapter import QnnPhoneBackend


DDR_DIR = "/sys/devices/system/cpu/bus_dcvs/DDR"


def adb_root(serial: str, command: str) -> str:
    result = subprocess.run(
        ["adb", "-s", serial, "shell", f"su -c {shlex.quote(command)}"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return result.stdout.strip()


def read_number(serial: str, name: str) -> int:
    return int(adb_root(serial, f"cat {DDR_DIR}/{name}"))


def set_boost(serial: str, value: int) -> None:
    adb_root(serial, f"echo {value} > {DDR_DIR}/boost_freq")
    actual = read_number(serial, "boost_freq")
    if actual != value:
        raise RuntimeError(f"DDR boost readback {actual}, requested {value}")


def skin_c(serial: str) -> float:
    result = subprocess.run(
        ["adb", "-s", serial, "shell", "dumpsys thermalservice"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    values = re.findall(r"Temperature\{mValue=([-+0-9.]+),\s*mType=\d+,\s*mName=skin,", result.stdout)
    if not values:
        raise RuntimeError("skin temperature unavailable")
    return float(values[-1])


def sampled_request(
    backend: QnnPhoneBackend,
    serial: str,
    payload: dict,
    *,
    switch_after_s: float | None = None,
    switch_to_khz: int | None = None,
) -> dict:
    samples: list[int] = []
    stop = threading.Event()
    start = time.monotonic()
    switch_info: dict = {}

    def sample() -> None:
        while not stop.is_set():
            if switch_after_s is not None and not switch_info and time.monotonic() - start >= switch_after_s:
                try:
                    set_boost(serial, int(switch_to_khz))
                    switch_info.update({"at_s": time.monotonic() - start,
                                        "sample_index": len(samples), "boost_khz": switch_to_khz})
                except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                    switch_info.update({"error": str(error)})
            try:
                samples.append(read_number(serial, "cur_freq"))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.25)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        response = backend.request(payload)
    finally:
        stop.set()
        thread.join(timeout=2)
    if response.get("status") != "ok":
        raise RuntimeError(f"runner request failed: {response}")
    if "error" in switch_info:
        raise RuntimeError(f"DDR switch failed: {switch_info['error']}")
    return {
        "prefill_ms": response.get("prefill_ms"),
        "decode_ms": response.get("decode_ms"),
        "ttft_ms": response.get("ttft_ms"),
        "request_ms": response.get("request_ms"),
        "output_tokens": response.get("output_tokens"),
        "prefill_tokens": response.get("prefill_tokens"),
        "reused_tokens": response.get("reused_tokens"),
        "ddr_samples_khz": samples,
        "ddr_median_khz": statistics.median(samples) if samples else None,
        "mid_request_switch": switch_info or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession_ctx6144")
    parser.add_argument("--boost-khz", type=int, default=3187000)
    parser.add_argument("--mid-run", action="store_true", help="switch DDR boost 5 s into long decode")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    available = [int(x) for x in adb_root(args.serial, f"cat {DDR_DIR}/available_frequencies").split()]
    if args.boost_khz not in available:
        raise ValueError(f"boost-khz must be one of {available}")
    original = read_number(args.serial, "boost_freq")
    result = {
        "serial": args.serial,
        "available_khz": available,
        "original_boost_khz": original,
        "requested_boost_khz": args.boost_khz,
        "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with QnnPhoneBackend(
            bundle=args.bundle,
            serial=args.serial,
            max_sessions=1,
            htp_mode=8,
            device_dir=args.device_dir,
            raw_log=args.output.with_suffix(".runner.log"),
            request_timeout=600,
        ) as backend:
            caps = backend.request({"command": "get_capabilities"})
            if caps.get("status") != "ok" or caps.get("model_load_count") != 1:
                raise RuntimeError(f"capability check failed: {caps}")
            result["model_load_count"] = 1
            # Warm the model once; evict after each call so every trial starts cold.
            backend.request({"command": "create_session", "session_id": "ddr_probe"})
            sampled_request(
                backend,
                args.serial,
                {"command": "run_request", "request_id": "warmup", "session_id": "ddr_probe",
                 "input_token_ids": [1000 + i % 257 for i in range(256)],
                 "max_new_tokens": 4, "temperature": 0, "stop_on_eos": False},
            )
            backend.request({"command": "evict_session", "session_id": "ddr_probe"})

            labels = ("normal", "switch", "normal", "switch") if args.mid_run else ("normal", "boost", "normal", "boost")
            for index, label in enumerate(labels):
                start_skin = skin_c(args.serial)
                if start_skin > 35.0:
                    raise RuntimeError(f"phone too warm for DDR probe: skin={start_skin:.2f} C")
                set_boost(args.serial, args.boost_khz if label == "boost" else original)
                time.sleep(0.5)
                request_id = f"ddr_{index}"
                metrics = sampled_request(
                    backend,
                    args.serial,
                    {"command": "run_request", "request_id": request_id, "session_id": "ddr_probe",
                     "input_token_ids": [1000 + i % 257 for i in range(512)],
                     "max_new_tokens": 96 if args.mid_run else 32,
                     "temperature": 0, "stop_on_eos": False},
                    switch_after_s=5.0 if label == "switch" else None,
                    switch_to_khz=args.boost_khz if label == "switch" else None,
                )
                backend.request({"command": "evict_session", "session_id": "ddr_probe"})
                result["trials"].append({"condition": label, "skin_start_c": start_skin,
                                         "skin_end_c": skin_c(args.serial),
                                         "boost_khz": read_number(args.serial, "boost_freq"), **metrics})
                print(json.dumps(result["trials"][-1], ensure_ascii=False), flush=True)
    finally:
        set_boost(args.serial, original)
        result["restored_boost_khz"] = read_number(args.serial, "boost_freq")
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "restored_boost_khz": result["restored_boost_khz"]}))


if __name__ == "__main__":
    main()
