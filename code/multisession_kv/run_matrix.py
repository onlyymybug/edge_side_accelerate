#!/usr/bin/env python3
"""Randomized, resumable HammerBench replay matrix launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def trace_sessions(path: Path) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        session_id = str(json.loads(line)["session_id"])
        if session_id not in seen:
            seen.add(session_id)
            output.append(session_id)
    return output


def expand_policy(policy: dict[str, Any], session_count: int) -> list[dict[str, Any]]:
    name = str(policy["name"])
    if name not in {"lru", "lifecycle_lru"}:
        return [{"policy": name, "lru_slots": None, "nominal_budget_ratios": None}]
    by_slots: dict[int, list[float]] = {}
    for ratio in policy.get("budget_ratios", []):
        ratio = float(ratio)
        if not 0 < ratio <= 1:
            raise ValueError(f"invalid {name} budget ratio: {ratio}")
        slots = max(1, min(session_count, math.ceil(session_count * ratio)))
        by_slots.setdefault(slots, []).append(ratio)
    output = []
    for slots, ratios in by_slots.items():
        output.append(
            {
                "policy": name,
                "lru_slots": slots,
                "nominal_budget_ratios": ratios,
                "actual_slot_ratio": slots / session_count,
            }
        )
    return output


def battery_state(serial: str) -> dict[str, Any]:
    result = subprocess.run(
        ["adb", "-s", serial, "shell", "dumpsys battery"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "dumpsys battery failed")
    powered = {
        name.lower(): value == "true"
        for name, value in re.findall(
            r"^\s*(AC|USB|Wireless|Dock) powered:\s*(true|false)\s*$",
            result.stdout,
            re.MULTILINE | re.IGNORECASE,
        )
    }
    if not powered:
        raise RuntimeError("could not parse battery power state")
    status_match = re.search(r"^\s*status:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    level_match = re.search(r"^\s*level:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    return {
        "powered": powered,
        "status": int(status_match.group(1)) if status_match else None,
        "level": int(level_match.group(1)) if level_match else None,
    }


def ensure_not_charging(serial: str) -> dict[str, Any]:
    state = battery_state(serial)
    active = [name for name, enabled in state["powered"].items() if enabled]
    if active:
        raise RuntimeError(f"phone is externally powered: {', '.join(active)}")
    return state


def build_jobs(spec: dict[str, Any], sessions: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    configured_blocks = spec.get("experiments")
    blocks = configured_blocks or [spec]
    for block in blocks:
        session_counts = block.get("session_counts")
        if session_counts is None:
            session_counts = spec["session_counts"]
        schedules = block.get("schedules")
        if schedules is None:
            schedules = spec["schedules"]
        policies = block.get("policies")
        if policies is None:
            policies = spec["policies"]
        repeats = int(block.get("repeats", spec["repeats"]))
        maximum = block.get(
            "max_groups_per_session_count",
            spec.get("max_groups_per_session_count"),
        )
        experiment_name = str(block["name"]) if configured_blocks else None
        for nominal_count in session_counts:
            nominal_count = int(nominal_count)
            groups = [
                (offset, sessions[offset : offset + nominal_count])
                for offset in range(0, len(sessions), nominal_count)
            ]
            if maximum is not None:
                groups = groups[: int(maximum)]
            for offset, group in groups:
                actual_count = len(group)
                expanded_policies = [
                    item
                    for policy in policies
                    for item in expand_policy(policy, actual_count)
                ]
                for repeat in range(repeats):
                    for schedule in schedules:
                        for policy in expanded_policies:
                            identity = {
                                "nominal_session_count": nominal_count,
                                "session_offset": offset,
                                "session_count": actual_count,
                                "sessions": group,
                                "repeat": repeat,
                                "schedule": schedule,
                                **policy,
                            }
                            if experiment_name is not None:
                                identity["experiment"] = experiment_name
                            digest = hashlib.sha256(
                                json.dumps(identity, sort_keys=True).encode("utf-8")
                            ).hexdigest()[:10]
                            jobs.append(
                                {
                                    "job_id": digest,
                                    **identity,
                                    "status": "pending",
                                    "attempts": 0,
                                }
                            )
    random.Random(int(spec["seed"])).shuffle(jobs)
    return jobs


def thermal_reading(serial: str, sensor: str) -> float:
    result = subprocess.run(
        ["adb", "-s", serial, "shell", "dumpsys thermalservice | grep 'Temperature{'"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "thermalservice failed")
    values = re.findall(
        rf"Temperature\{{mValue=([-+0-9.]+),\s*mType=\d+,\s*mName={re.escape(sensor)},",
        result.stdout,
    )
    if not values:
        raise RuntimeError(f"no current {sensor} temperature")
    return float(values[-1])


def wait_for_temperature(serial: str, gate: dict[str, Any]) -> list[dict[str, Any]]:
    if not gate.get("enabled", False):
        return []
    threshold = gate.get("max_start_c")
    if threshold is None:
        raise ValueError("temperature gate is enabled but max_start_c is null")
    sensor = str(gate.get("sensor", "skin"))
    consecutive_required = int(gate.get("consecutive_samples", 3))
    poll_seconds = float(gate.get("poll_seconds", 10))
    deadline = time.monotonic() + float(gate.get("timeout_seconds", 1800))
    samples: list[dict[str, Any]] = []
    consecutive = 0
    while True:
        value = thermal_reading(serial, sensor)
        sample = {
            "host_timestamp_ns": time.time_ns(),
            "sensor": sensor,
            "temperature_c": value,
            "threshold_c": float(threshold),
        }
        samples.append(sample)
        consecutive = consecutive + 1 if value <= float(threshold) else 0
        if consecutive >= consecutive_required:
            return samples
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"temperature did not recover below {threshold} C for "
                f"{consecutive_required} consecutive samples"
            )
        time.sleep(poll_seconds)


def preserve_attempt(job_dir: Path, attempts_dir: Path, job_id: str, attempt: int) -> None:
    if not job_dir.exists():
        return
    attempts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = attempts_dir / f"{job_id}_attempt{attempt}_{timestamp}"
    job_dir.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--min-battery-level", type=int, default=None,
        help="stop cleanly before starting another job when battery level is below this percentage",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    if args.min_battery_level is not None and not 0 <= args.min_battery_level <= 100:
        parser.error("--min-battery-level must be between 0 and 100")

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sessions = trace_sessions(args.trace)
    if not sessions:
        raise ValueError("trace contains no Sessions")
    spec_identity = {
        "config_sha256": sha256(args.config),
        "trace_sha256": sha256(args.trace),
        "bundle_runner_sha256": sha256(
            args.bundle / "device" / "qnn_llama_runner_multisession"
        ),
        "session_ids": sessions,
    }
    bundle_pte = args.bundle / "device" / "hybrid_llama_qnn.pte"
    bundle_tokenizer = args.bundle / "device" / "tokenizer.json"
    if bundle_pte.is_file():
        spec_identity["bundle_pte_sha256"] = sha256(bundle_pte)
    if bundle_tokenizer.is_file():
        spec_identity["bundle_tokenizer_sha256"] = sha256(bundle_tokenizer)
    if args.device_dir != "/data/local/tmp/qnn_multisession":
        spec_identity["device_dir"] = args.device_dir
    spec_hash = hashlib.sha256(
        json.dumps(spec_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = args.results_root / "matrix_manifest.json"
    args.results_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["spec_hash"] != spec_hash:
            raise RuntimeError(
                "existing matrix manifest belongs to a different config/trace/runner"
            )
        for job in manifest["jobs"]:
            if job["status"] == "running":
                job["status"] = "pending"
                job["last_error"] = "previous launcher stopped while job was running"
    else:
        jobs = build_jobs(spec, sessions)
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_hash": spec_hash,
            "spec_identity": spec_identity,
            "config": spec,
            "trace": str(args.trace.resolve()),
            "bundle": str(args.bundle.resolve()),
            "randomization_seed": int(spec["seed"]),
            "jobs": jobs,
        }
        atomic_json(manifest_path, manifest)

    counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(json.dumps({"manifest": str(manifest_path), "jobs": len(manifest["jobs"]), "status_counts": counts}, indent=2), flush=True)
    if args.dry_run:
        return

    if spec.get("require_not_charging", True):
        initial_battery = ensure_not_charging(args.serial)
        print(json.dumps({"battery_preflight": initial_battery}), flush=True)

    completed_this_invocation = 0
    replay_script = Path(__file__).with_name("replay.py")
    validate_script = Path(__file__).with_name("validate_results.py")
    for job in manifest["jobs"]:
        if job["status"] == "pass":
            continue
        if job["status"] == "failed" and not args.retry_failed:
            continue
        if args.max_jobs is not None and completed_this_invocation >= args.max_jobs:
            break
        if args.min_battery_level is not None:
            level = battery_state(args.serial)["level"]
            if level is None:
                raise RuntimeError("battery level unavailable; cannot enforce --min-battery-level")
            if level < args.min_battery_level:
                print(
                    json.dumps({"battery_stop": {"level": level, "minimum": args.min_battery_level},
                                "completed_this_invocation": completed_this_invocation}),
                    flush=True,
                )
                break
        job_dir = args.results_root / "runs" / job["job_id"]
        preserve_attempt(
            job_dir,
            args.results_root / "attempts",
            job["job_id"],
            int(job["attempts"]),
        )
        try:
            job_battery = None
            if spec.get("require_not_charging", True):
                job_battery = ensure_not_charging(args.serial)
            recovery = wait_for_temperature(
                args.serial, dict(spec.get("temperature_gate", {}))
            )
        except Exception as error:
            job["status"] = "failed"
            job["last_error"] = f"temperature_gate: {type(error).__name__}: {error}"
            atomic_json(manifest_path, manifest)
            if args.stop_on_failure:
                raise
            continue
        job["status"] = "running"
        job["attempts"] = int(job["attempts"]) + 1
        job["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        job_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(replay_script),
            "--trace",
            str(args.trace),
            "--output-dir",
            str(job_dir),
            "--bundle",
            str(args.bundle),
            "--serial",
            args.serial,
            "--device-dir",
            args.device_dir,
            "--session-offset",
            str(job["session_offset"]),
            "--session-limit",
            str(job["session_count"]),
            "--schedule",
            str(job["schedule"]),
            "--policy",
            str(job["policy"]),
            "--max-new-tokens",
            str(spec["max_new_tokens"]),
            "--htp-mode",
            str(spec["htp_mode"]),
            "--request-timeout",
            str(spec["request_timeout_seconds"]),
        ]
        if spec.get("warmup", True):
            command.append("--warmup")
        interval = float(spec.get("telemetry_interval_seconds", 0))
        if interval > 0:
            command += ["--telemetry-interval", str(interval)]
        if job["policy"] in {"lru", "lifecycle_lru"}:
            command += ["--lru-slots", str(job["lru_slots"])]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        job["stdout_tail"] = result.stdout[-4000:]
        job["stderr_tail"] = result.stderr[-4000:]
        if result.returncode == 0:
            validation = subprocess.run(
                [sys.executable, str(validate_script), "run", "--run-dir", str(job_dir)],
                text=True,
                capture_output=True,
                check=False,
            )
        else:
            validation = None
        if result.returncode == 0 and validation and validation.returncode == 0:
            job["status"] = "pass"
            job["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            job["run_dir"] = str(job_dir)
            job["launch_battery"] = job_battery
            job["recovery_samples"] = recovery
            job["validation"] = json.loads(validation.stdout)
        else:
            job["status"] = "failed"
            job["last_error"] = (
                f"replay_rc={result.returncode}; validation_rc="
                f"{validation.returncode if validation else 'not_run'}"
            )
            if validation:
                job["validation_stdout_tail"] = validation.stdout[-4000:]
                job["validation_stderr_tail"] = validation.stderr[-4000:]
        atomic_json(manifest_path, manifest)
        completed_this_invocation += 1
        print(json.dumps({"job_id": job["job_id"], "status": job["status"], "completed_this_invocation": completed_this_invocation}), flush=True)
        if job["status"] == "failed" and args.stop_on_failure:
            raise RuntimeError(job["last_error"])


if __name__ == "__main__":
    main()
