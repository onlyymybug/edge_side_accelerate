#!/usr/bin/env python3
"""Conditional KV-policy replay of a recorded AppWorld phone run.

The event order, prompt token IDs and generated token IDs are frozen. This
isolates KV decisions but does not simulate latency-induced changes in Agent
arrival order, task success, or actual AppWorld execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from appworld_multisession_smoke import choose_lru_victim


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def freeze_run(source: Path) -> dict[str, Any]:
    paths = {name: source / name for name in
             ("config.json", "events.jsonl", "requests.jsonl", "runner.log")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("source must contain config, events, requests, runner.log")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    events = read_jsonl(paths["events.jsonl"])
    responses = {row["request_id"]: row for row in read_jsonl(paths["requests.jsonl"])}
    requests: dict[str, dict[str, Any]] = {}
    for line in paths["runner.log"].read_text(encoding="utf-8").splitlines():
        if not line.startswith("REQUEST "):
            continue
        payload = json.loads(line.removeprefix("REQUEST "))
        if payload.get("command") != "run_request":
            continue
        request_id = payload["request_id"]
        if request_id not in responses or request_id in requests:
            raise ValueError(f"missing or duplicate response for {request_id}")
        response = responses[request_id]
        if payload["session_id"] != response["task_id"]:
            raise ValueError(f"session mismatch for {request_id}")
        requests[request_id] = {
            "request_id": request_id,
            "task_id": response["task_id"],
            "input_token_ids": payload["input_token_ids"],
            "generated_token_ids": response["generated_token_ids"],
            "final_generated_token_cached": response["final_generated_token_cached"],
            "observed_reused_tokens": response["reused_tokens"],
            "observed_prefill_tokens": response["prefill_tokens"],
        }
    if set(requests) != set(responses):
        raise ValueError("runner request log does not match requests.jsonl")
    selected_events = []
    for row in events:
        if row["event"] in {"llm_ready", "inference_start", "task_end"}:
            selected_events.append({key: row[key] for key in
                                    ("event", "task_id", "step", "request_id")
                                    if key in row})
    return {
        "source": str(source.resolve()),
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "source_policy": config.get("policy", "lru"),
        "source_release_on_done": config.get("release_on_done", True),
        "source_kv_slots": config.get("kv_slots", config.get("capacity")),
        "task_ids": config["task_ids"],
        "events": selected_events,
        "requests": requests,
        "limitations": "fixed source event order and model output; KV token replay only",
    }


def simulate(trace: dict[str, Any], *, policy: str, kv_slots: int,
             release_on_done: bool) -> dict[str, Any]:
    if policy not in {"lru", "ready_lru"} or kv_slots < 1:
        raise ValueError("unsupported policy or invalid KV slots")
    resident: OrderedDict[str, None] = OrderedDict()
    cached: dict[str, list[int]] = {}
    queued: set[str] = set()
    results: list[dict[str, Any]] = []
    evictions: list[dict[str, Any]] = []
    for item in trace["events"]:
        task_id = item["task_id"]
        if item["event"] == "llm_ready":
            if task_id in queued:
                raise ValueError(f"duplicate ready event for {task_id}")
            queued.add(task_id)
        elif item["event"] == "task_end":
            if release_on_done:
                resident.pop(task_id, None)
                cached.pop(task_id, None)
        elif item["event"] == "inference_start":
            if task_id not in queued:
                raise ValueError(f"request dispatched before ready: {item}")
            queued.remove(task_id)
            request = trace["requests"][item["request_id"]]
            if request["task_id"] != task_id:
                raise ValueError(f"request/session mismatch: {item}")
            if task_id not in resident:
                while len(resident) >= kv_slots:
                    victim = choose_lru_victim(resident, queued, policy)
                    resident.pop(victim)
                    cached.pop(victim, None)
                    evictions.append({"before_request_id": item["request_id"],
                                      "victim": victim, "queued_task_ids": sorted(queued)})
            prompt = request["input_token_ids"]
            reused = common_prefix(cached.get(task_id, []), prompt)
            generated = request["generated_token_ids"]
            if not request["final_generated_token_cached"]:
                generated = generated[:-1]
            cached[task_id] = prompt + generated
            resident[task_id] = None
            resident.move_to_end(task_id)
            results.append({"request_id": item["request_id"], "task_id": task_id,
                            "input_tokens": len(prompt), "reused_tokens": reused,
                            "prefill_tokens": len(prompt) - reused,
                            "observed_reused_tokens": request.get("observed_reused_tokens")})
    if queued or len(results) != len(trace["requests"]):
        raise ValueError("incomplete event/request sequence")
    return {"policy": policy, "release_on_done": release_on_done,
            "kv_slots": kv_slots, "requests": results, "evictions": evictions,
            "total_reused_tokens": sum(row["reused_tokens"] for row in results),
            "total_prefill_tokens": sum(row["prefill_tokens"] for row in results),
            "capacity_evictions": len(evictions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-slots", type=int)
    args = parser.parse_args()
    trace = freeze_run(args.source_results)
    kv_slots = args.kv_slots or trace["source_kv_slots"]
    if kv_slots is None:
        parser.error("--kv-slots required when source config lacks kv_slots")
    if trace["source_kv_slots"] is None:
        parser.error("source config must record its original KV slots for validation")
    source = simulate(trace, policy=trace["source_policy"],
                      kv_slots=trace["source_kv_slots"],
                      release_on_done=trace["source_release_on_done"])
    mismatches = [row for row in source["requests"]
                  if row["reused_tokens"] != row["observed_reused_tokens"]]
    if mismatches:
        raise RuntimeError(f"source replay failed to match phone: {mismatches}")
    variants = {
        "lru_no_close": simulate(trace, policy="lru", kv_slots=kv_slots,
                                 release_on_done=False),
        "lru_close": simulate(trace, policy="lru", kv_slots=kv_slots,
                              release_on_done=True),
        "ready_lru_close": simulate(trace, policy="ready_lru", kv_slots=kv_slots,
                                    release_on_done=True),
    }
    report = {"trace": trace, "source_validation": "exact per-request KV reuse match",
              "variants": variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({name: {key: result[key] for key in
                             ("capacity_evictions", "total_reused_tokens",
                              "total_prefill_tokens")}
                      for name, result in variants.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
