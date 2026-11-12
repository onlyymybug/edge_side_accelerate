#!/usr/bin/env python3
"""Replay fixed token traces against the resident QNN phone runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import uuid
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_adapter import QnnPhoneBackend
from collect_android import AndroidSampler
from lifecycle_policy import choose_serial_lifecycle_victim, update_session_metadata


DEFAULT_MODEL_SHA256 = "bd94b61a35a1b236a6d27da6bdce8148eda5742db9bf6b3bb8efbc79a053872a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def schedule_rows(
    grouped: dict[str, list[dict[str, Any]]], schedule: str
) -> list[dict[str, Any]]:
    if schedule == "grouped":
        return [row for session_id in grouped for row in grouped[session_id]]
    output: list[dict[str, Any]] = []
    maximum = max(len(rows) for rows in grouped.values())
    for turn_index in range(maximum):
        for rows in grouped.values():
            if turn_index < len(rows):
                output.append(rows[turn_index])
    return output


def adb_value(serial: str, command: str) -> str:
    result = subprocess.run(
        ["adb", "-s", serial, "shell", command],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--session-limit", type=int, required=True)
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--schedule", choices=("grouped", "round_robin"), required=True)
    parser.add_argument(
        "--policy",
        choices=("no_reuse", "all_keep", "single_slot", "lru", "lifecycle_lru"),
        required=True,
    )
    parser.add_argument("--lru-slots", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--htp-mode", type=int, default=8)
    parser.add_argument("--model-sha256")
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--telemetry-interval", type=float, default=0)
    args = parser.parse_args()
    local_pte = args.bundle / "device" / "hybrid_llama_qnn.pte"
    model_sha256 = args.model_sha256 or (
        sha256(local_pte) if local_pte.is_file() else DEFAULT_MODEL_SHA256
    )
    if args.session_limit < 1:
        parser.error("--session-limit must be positive")
    if args.policy in {"lru", "lifecycle_lru"} and not 1 <= args.lru_slots <= args.session_limit:
        parser.error("--lru-slots must be between 1 and --session-limit")

    rows = read_jsonl(args.trace)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    selected_ids = list(dict.fromkeys(row["session_id"] for row in rows))[
        args.session_offset : args.session_offset + args.session_limit
    ]
    if len(selected_ids) != args.session_limit:
        parser.error("session offset/limit exceeds available trace Sessions")
    grouped = {session_id: grouped[session_id] for session_id in selected_ids}
    for session_rows in grouped.values():
        session_rows.sort(key=lambda row: row["turn_id"])
    scheduled = schedule_rows(grouped, args.schedule)
    final_turn = {
        session_id: max(int(row["turn_id"]) for row in session_rows)
        for session_id, session_rows in grouped.items()
    }

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requests: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    resident_lru: OrderedDict[str, None] = OrderedDict()
    lifecycle_metadata: dict[str, dict[str, float | int | None]] = {}
    active_session: str | None = None
    trace_manifest_path = args.trace.parent / "manifest.json"
    trace_manifest = (
        json.loads(trace_manifest_path.read_text(encoding="utf-8"))
        if trace_manifest_path.is_file()
        else None
    )

    config = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace.resolve()),
        "trace_variant": rows[0]["variant"],
        "selected_sessions": selected_ids,
        "session_count": len(selected_ids),
        "session_offset": args.session_offset,
        "schedule": args.schedule,
        "policy": args.policy,
        "lru_slots": args.lru_slots if args.policy in {"lru", "lifecycle_lru"} else None,
        "lifecycle_protocol": "serial_arrival" if args.policy == "lifecycle_lru" else None,
        "lifecycle_source": "controlled_protocol" if args.policy == "lifecycle_lru" else None,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "stop_on_eos": not args.ignore_eos,
        "htp_performance_mode": args.htp_mode,
        "warmup": args.warmup,
        "telemetry_interval_seconds": args.telemetry_interval or None,
        "serial": args.serial,
        "host": platform.node(),
        "device_model": adb_value(args.serial, "getprop ro.product.model"),
        "device_abi": adb_value(args.serial, "getprop ro.product.cpu.abi"),
        "battery_state": adb_value(
            args.serial,
            "dumpsys battery | grep -E 'powered:|status:|level:|temperature:'",
        ),
        "trace_sha256": sha256(args.trace),
        "trace_manifest": str(trace_manifest_path.resolve()) if trace_manifest else None,
        "trace_manifest_sha256": sha256(trace_manifest_path) if trace_manifest else None,
        "dataset_sha256": trace_manifest.get("dataset_sha256") if trace_manifest else None,
        "tokenizer_sha256": trace_manifest.get("tokenizer_sha256") if trace_manifest else None,
        "model_sha256": model_sha256,
        "device_dir": args.device_dir,
        "runner_sha256": sha256(args.bundle / "device" / "qnn_llama_runner_multisession"),
        "replay_sha256": sha256(Path(__file__)),
        "lifecycle_policy_sha256": sha256(Path(__file__).with_name("lifecycle_policy.py")),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with QnnPhoneBackend(
        bundle=args.bundle,
        serial=args.serial,
        max_sessions=len(selected_ids),
        htp_mode=args.htp_mode,
        device_dir=args.device_dir,
        raw_log=args.output_dir / "runner.log",
        request_timeout=args.request_timeout,
    ) as backend:
        sampler: AndroidSampler | None = None
        capabilities = backend.request({"command": "get_capabilities"})
        if capabilities.get("status") != "ok" or capabilities.get("model_load_count") != 1:
            raise RuntimeError(capabilities)
        (args.output_dir / "capabilities.json").write_text(
            json.dumps(capabilities, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for session_id in selected_ids:
            response = backend.request({"command": "create_session", "session_id": session_id})
            if response.get("status") != "ok":
                raise RuntimeError(response)
            events.append(
                {"run_id": run_id, "host_monotonic_ns": time.monotonic_ns(), "event": "create_session", "session_id": session_id, "response": response}
            )

        if args.warmup:
            first = scheduled[0]
            warmup_response = backend.request(
                {
                    "command": "run_request",
                    "request_id": f"{run_id}_warmup",
                    "session_id": first["session_id"],
                    "input_token_ids": first["input_token_ids"],
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0,
                    "stop_on_eos": not args.ignore_eos,
                }
            )
            if warmup_response.get("status") != "ok":
                raise RuntimeError(warmup_response)
            reset_response = backend.request({"command": "reset_all"})
            (args.output_dir / "warmup.json").write_text(
                json.dumps(
                    {"request": warmup_response, "reset": reset_response},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.telemetry_interval:
            sampler = AndroidSampler(
                args.serial,
                int(capabilities["server_pid"]),
                args.telemetry_interval,
            )
            sampler.start()

        started_ns = time.monotonic_ns()
        try:
            for index, trace_row in enumerate(scheduled):
                session_id = trace_row["session_id"]
                request_eviction_reason = None
                policy_decision_ns = 0
                if args.policy == "lifecycle_lru":
                    events.append(
                        {
                            "run_id": run_id,
                            "host_monotonic_ns": time.monotonic_ns(),
                            "event": "state_change",
                            "request_index": index,
                            "session_id": session_id,
                            "state": "READY",
                            "lifecycle_source": "controlled_protocol",
                        }
                    )
                if args.policy == "no_reuse":
                    response = backend.request({"command": "reset_all"})
                    resident_lru.clear()
                    active_session = None
                    events.append(
                        {"run_id": run_id, "host_monotonic_ns": time.monotonic_ns(), "event": "reset_all", "request_index": index, "response": response}
                    )
                    request_eviction_reason = "no_reuse_reset_all"
                elif args.policy == "single_slot" and active_session not in (None, session_id):
                    response = backend.request(
                        {"command": "evict_session", "session_id": active_session}
                    )
                    events.append(
                        {"run_id": run_id, "host_monotonic_ns": time.monotonic_ns(), "event": "evict", "reason": "single_slot_switch", "request_index": index, "session_id": active_session, "response": response}
                    )
                    request_eviction_reason = "single_slot_switch"
                    resident_lru.pop(active_session, None)
                    active_session = None
                elif args.policy == "lru" and session_id not in resident_lru:
                    while len(resident_lru) >= args.lru_slots:
                        victim, _ = resident_lru.popitem(last=False)
                        response = backend.request(
                            {"command": "evict_session", "session_id": victim}
                        )
                        events.append(
                            {"run_id": run_id, "host_monotonic_ns": time.monotonic_ns(), "event": "evict", "reason": "lru_capacity", "request_index": index, "session_id": victim, "response": response}
                        )
                        request_eviction_reason = "lru_capacity"
                        if active_session == victim:
                            active_session = None
                elif args.policy == "lifecycle_lru" and session_id not in resident_lru:
                    while len(resident_lru) >= args.lru_slots:
                        decision_start_ns = time.monotonic_ns()
                        victim, scores = choose_serial_lifecycle_victim(
                            list(resident_lru),
                            lifecycle_metadata,
                            index,
                            len(selected_ids),
                        )
                        policy_decision_ns += time.monotonic_ns() - decision_start_ns
                        resident_lru.pop(victim)
                        response = backend.request(
                            {"command": "evict_session", "session_id": victim}
                        )
                        events.append(
                            {
                                "run_id": run_id,
                                "host_monotonic_ns": time.monotonic_ns(),
                                "event": "evict",
                                "reason": "lifecycle_capacity",
                                "request_index": index,
                                "session_id": victim,
                                "victim_state": "WAITING",
                                "victim_score": scores[victim],
                                "candidate_scores": scores,
                                "lifecycle_source": "controlled_protocol",
                                "response": response,
                            }
                        )
                        request_eviction_reason = "lifecycle_capacity"
                        if active_session == victim:
                            active_session = None

                if args.policy == "lifecycle_lru":
                    events.append(
                        {
                            "run_id": run_id,
                            "host_monotonic_ns": time.monotonic_ns(),
                            "event": "state_change",
                            "request_index": index,
                            "session_id": session_id,
                            "state": "RUNNING",
                            "lifecycle_source": "controlled_protocol",
                        }
                    )

                payload = {
                    "command": "run_request",
                    "request_id": f"{run_id}_{index:05d}",
                    "session_id": session_id,
                    "input_token_ids": trace_row["input_token_ids"],
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0,
                    "stop_on_eos": not args.ignore_eos,
                }
                host_start_ns = time.monotonic_ns()
                response = backend.request(payload)
                host_end_ns = time.monotonic_ns()
                result = {
                    "run_id": run_id,
                    "request_index": index,
                    "policy": args.policy,
                    "schedule": args.schedule,
                    "trace_variant": trace_row["variant"],
                    "turn_id": trace_row["turn_id"],
                    "source_snapshot_id": trace_row["source_snapshot_id"],
                    "reference_lcp_tokens": trace_row["previous_turn_lcp_tokens"],
                    "reference_required_new_tokens": trace_row["required_new_tokens"],
                    "required_new_tokens": trace_row["required_new_tokens"],
                    "kv_budget_bytes": (
                        0
                        if args.policy == "no_reuse"
                        else int(capabilities["session_backing_bytes"])
                        * (
                            args.lru_slots
                            if args.policy in {"lru", "lifecycle_lru"}
                            else (1 if args.policy == "single_slot" else len(selected_ids))
                        )
                    ),
                    "eviction_count": sum(event["event"] == "evict" for event in events),
                    "eviction_reason": request_eviction_reason,
                    "queue_ms": 0.0,
                    "policy_decision_ms": policy_decision_ns / 1_000_000,
                    "host_start_ns": host_start_ns,
                    "host_end_ns": host_end_ns,
                    "host_roundtrip_ms": (host_end_ns - host_start_ns) / 1_000_000,
                    **response,
                }
                requests.append(result)
                if response.get("status") != "ok":
                    write_jsonl(args.output_dir / "requests.jsonl", requests)
                    write_jsonl(args.output_dir / "events.jsonl", events)
                    raise RuntimeError(response)
                if response.get("server_pid") != capabilities.get("server_pid"):
                    raise AssertionError("runner PID changed")
                if response.get("model_load_count") != 1:
                    raise AssertionError("model was loaded more than once")
                resident_lru.pop(session_id, None)
                resident_lru[session_id] = None
                active_session = session_id
                if args.policy == "lifecycle_lru":
                    update_session_metadata(
                        lifecycle_metadata,
                        session_id,
                        index,
                        int(trace_row["input_tokens"]),
                    )
                    if int(trace_row["turn_id"]) == final_turn[session_id]:
                        response = backend.request(
                            {"command": "evict_session", "session_id": session_id}
                        )
                        resident_lru.pop(session_id, None)
                        active_session = None
                        events.append(
                            {
                                "run_id": run_id,
                                "host_monotonic_ns": time.monotonic_ns(),
                                "event": "state_change",
                                "request_index": index,
                                "session_id": session_id,
                                "state": "DONE",
                                "reason": "controlled_protocol_close",
                                "lifecycle_source": "controlled_protocol",
                                "response": response,
                            }
                        )
                    else:
                        events.append(
                            {
                                "run_id": run_id,
                                "host_monotonic_ns": time.monotonic_ns(),
                                "event": "state_change",
                                "request_index": index,
                                "session_id": session_id,
                                "state": "WAITING",
                                "lifecycle_source": "controlled_protocol",
                            }
                        )
            finished_ns = time.monotonic_ns()
        finally:
            if sampler:
                sampler.stop()
                sampler.write(args.output_dir / "telemetry.jsonl")

        duration_ms = (finished_ns - started_ns) / 1_000_000

    write_jsonl(args.output_dir / "requests.jsonl", requests)
    write_jsonl(args.output_dir / "events.jsonl", events)
    summary = {
        "run_id": run_id,
        "status": "pass",
        "requests": len(requests),
        "duration_ms": duration_ms,
        "total_input_tokens": sum(row["input_tokens"] for row in requests),
        "total_prefill_tokens": sum(row["prefill_tokens"] for row in requests),
        "total_reused_tokens": sum(row["reused_tokens"] for row in requests),
        "evictions": sum(row["event"] == "evict" for row in events),
        "resets": sum(row["event"] == "reset_all" for row in events),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
