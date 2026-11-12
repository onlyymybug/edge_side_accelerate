#!/usr/bin/env python3
"""Real AppWorld tasks, one serialized QNN phone service; plumbing smoke only.

Each AppWorld environment lives in its own process. The parent alone owns the
phone runner and dispatches one request at a time. Tool delay is synthetic and
must not be reported as measured AppWorld API latency.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import heapq
import json
import multiprocessing as mp
import queue
import re
import time
import traceback
from pathlib import Path
from typing import Any

from appworld import AppWorld, update_root
from tokenizers import Tokenizer

from agent_closed_loop import DEFAULT_BUNDLE, clean_generation, write_jsonl
from appworld_phone_pilot import (
    PROFILES, call_tool, load_tool_docs, normalize_action_answer, parse_call,
    validate_profile_instruction,
    prepare_parameters, real_monotonic_ns, render,
)
from backend_adapter import QnnPhoneBackend


class ScriptedBackend:
    """Scripted follower-count calls for process/queue integration, never accuracy."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.turns: dict[str, int] = {}

    def __enter__(self) -> "ScriptedBackend":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload["command"]
        if command == "get_capabilities":
            return {"status": "ok", "server_pid": 0, "model_load_count": 1}
        if command in {"create_session", "evict_session"}:
            return {"status": "ok"}
        if command != "run_request":
            raise ValueError(command)
        task_id = payload["session_id"]
        turn = self.turns.get(task_id, 0)
        self.turns[task_id] = turn + 1
        prompt = self.tokenizer.decode(payload["input_token_ids"], skip_special_tokens=False)
        observations = re.findall(r"<\|im_start\|>tool\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
        if turn == 0:
            name, parameters = "spotify__show_current_song", {}
        elif turn == 1:
            artist_id = json.loads(observations[-1])["artists"][0]["id"]
            name, parameters = "spotify__show_artist", {"artist_id": artist_id}
        elif turn == 2:
            count = json.loads(observations[-1])["follower_count"]
            name, parameters = "supervisor__complete_task", {"answer": str(count)}
        else:
            raise RuntimeError("scripted backend exceeded three calls")
        time.sleep(0.04)
        output = json.dumps({"name": name, "parameters": parameters},
                            ensure_ascii=False, separators=(",", ":"))
        return {"status": "ok", "server_pid": 0, "model_load_count": 1,
                "generated_text": output, "output_tokens": len(self.tokenizer.encode(output).ids)}


def event(event_type: str, task_id: str, **fields: Any) -> dict[str, Any]:
    return {"event": event_type, "task_id": task_id,
            "host_monotonic_ns": real_monotonic_ns(), **fields}


def choose_lru_victim(resident_lru: OrderedDict[str, None],
                      queued_task_ids: set[str], policy: str) -> str:
    """Use only currently queued requests; never inspect a future AppWorld turn."""
    if policy == "ready_lru":
        for task_id in resident_lru:
            if task_id not in queued_task_ids:
                return task_id
    return next(iter(resident_lru))


def task_worker(settings: dict[str, Any], inbox: Any, outbox: Any,
                start_barrier: Any) -> None:
    task_id = settings["task_id"]
    started_ns = real_monotonic_ns()
    steps = 0
    repair_count = 0
    filtered_count = 0
    error: str | None = None
    success = False
    try:
        update_root(settings["appworld_root"])
        bundle = Path(settings["bundle"])
        tokenizer = Tokenizer.from_file(str(bundle / "device" / "tokenizer.json"))
        all_docs = load_tool_docs(Path(settings["appworld_root"]))
        execution_docs = {row["name"]: row for row in all_docs}
        docs = [json.loads(json.dumps(row)) for row in all_docs
                if row["name"] in PROFILES[settings["profile"]]]
        if settings["profile"] in {"most_played_song", "queue_popular_songs"}:
            visible_fields = {"spotify__search_artists": {"query"}}
            if settings["profile"] == "most_played_song":
                visible_fields["spotify__search_songs"] = {"artist_id", "sort_by", "page_limit"}
            else:
                visible_fields["spotify__search_songs"] = {"artist_id", "min_play_count", "page_index"}
                visible_fields["spotify__add_to_queue"] = {"song_id"}
            docs.sort(key=lambda row: {"spotify__search_artists": 0,
                                       "spotify__search_songs": 1,
                                       "spotify__add_to_queue": 2,
                                       "supervisor__complete_task": 3}[row["name"]])
            for row in docs:
                fields = visible_fields.get(row["name"])
                if fields is not None:
                    parameters = row["parameters"]
                    parameters["properties"] = {key: value for key, value in
                                                parameters["properties"].items()
                                                if key in fields}
                    if "required" in parameters:
                        parameters["required"] = [key for key in parameters["required"]
                                                  if key in fields]
        with AppWorld(task_id=task_id, experiment_name=settings["experiment_name"],
                      raise_on_unsafe_syntax=True) as world:
            validate_profile_instruction(settings["profile"], world.task.instruction)
            profile = json.loads(call_tool(world, "supervisor__show_profile", {}, execution_docs))
            accounts = json.loads(call_tool(world, "supervisor__show_account_passwords", {}, execution_docs))
            spotify = next(row for row in accounts if row["account_name"] == "spotify")
            login = json.loads(call_tool(world, "spotify__login", {
                "username": profile["email"], "password": spotify["password"],
            }, execution_docs))
            bootstrap_token = login["access_token"]
            for row in docs:
                if row["name"].startswith("spotify__"):
                    row["parameters"]["properties"].pop("access_token", None)
            visible_docs = {row["name"]: row for row in docs}
            zero_arg_names = {name for name, doc in visible_docs.items()
                              if not doc["parameters"].get("properties", {})}
            messages = [{"role": "user", "content": world.task.instruction}]
            start_barrier.wait(timeout=settings["worker_timeout"])
            if settings["arrival_delay_ms"]:
                time.sleep(settings["arrival_delay_ms"] / 1000)
            outbox.put(event("task_start", task_id))
            for step in range(1, settings["max_steps"] + 1):
                prepare_start_ns = real_monotonic_ns()
                prompt = render(messages, docs, bootstrap_token,
                                settings["profile"], "direct_json")
                token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                prepare_end_ns = real_monotonic_ns()
                if len(token_ids) + settings["max_new_tokens"] > settings["context_tokens"]:
                    error = "context overflow"
                    break
                ready_ns = real_monotonic_ns()
                outbox.put({"type": "ready", "task_id": task_id, "step": step,
                            "ready_ns": ready_ns, "token_ids": token_ids,
                            "prompt_prepare_ms": (prepare_end_ns - prepare_start_ns) / 1_000_000})
                response = inbox.get(timeout=settings["worker_timeout"])
                steps += 1
                if response.get("status") != "ok":
                    error = f"runner status: {response.get('status')}"
                    break
                output = clean_generation(str(response.get("generated_text", "")))
                try:
                    name, parameters, repaired, ignored_extra = parse_call(
                        output, zero_arg_names, "direct_json",
                        flat_parameter_names={name: set(doc["parameters"].get("properties", {}))
                                              for name, doc in visible_docs.items()})
                    parameters, dropped = prepare_parameters(
                        name, parameters, visible_docs, execution_docs, bootstrap_token)
                    parameters, normalized_answer = normalize_action_answer(
                        settings["profile"], name, parameters)
                    tool_start_ns = real_monotonic_ns()
                    result = call_tool(world, name, parameters, execution_docs)
                    enriched = False
                    if (settings["enrich_navigation"] and
                            name in {"spotify__previous_song", "spotify__next_song"}):
                        current = call_tool(world, "spotify__show_current_song",
                                            {"access_token": bootstrap_token}, execution_docs)
                        result = json.dumps({"move": json.loads(result),
                                             "current_song": json.loads(current)},
                                            ensure_ascii=False, separators=(",", ":"))
                        enriched = True
                    if settings["tool_delay_ms"]:
                        time.sleep(settings["tool_delay_ms"] / 1000)
                    tool_end_ns = real_monotonic_ns()
                except ValueError as exc:
                    error = str(exc)
                    break
                repair_count += int(repaired)
                filtered_count += bool(dropped)
                outbox.put(event("tool_end", task_id, name=name,
                                 tool_start_ns=tool_start_ns,
                                 tool_end_ns=tool_end_ns,
                                 tool_ms=(tool_end_ns - tool_start_ns) / 1_000_000,
                                 synthetic_tool_delay_ms=settings["tool_delay_ms"],
                                 parse_repaired=repaired, dropped_parameters=dropped,
                                 ignored_extra_calls=ignored_extra,
                                 enriched_observation=enriched,
                                 normalized_action_answer=normalized_answer))
                visible_parameters = {key: value for key, value in parameters.items()
                                      if key in visible_docs[name]["parameters"].get("properties", {})}
                messages.append({"role": "assistant", "content": json.dumps(
                    {"name": name, "parameters": visible_parameters},
                    ensure_ascii=False, separators=(",", ":"))})
                messages.append({"role": "tool", "content": result[:2500]})
                if name == "supervisor__complete_task":
                    break
            else:
                error = "max_steps reached without complete_task"
            evaluation = world.evaluate().to_dict()
            success = bool(evaluation["success"])
    except Exception as exc:  # worker failure must be reported to dispatcher
        error = f"{type(exc).__name__}: {exc}"
        outbox.put(event("worker_exception", task_id, traceback=traceback.format_exc()))
    finally:
        finished_ns = real_monotonic_ns()
        outbox.put({"type": "done", "task_id": task_id,
                    "summary": {"task_id": task_id, "success": success,
                                "profile": settings["profile"],
                                "steps": steps, "error": error,
                                "completion_ms": (finished_ns - started_ns) / 1_000_000,
                                "repaired_tool_calls": repair_count,
                                "filtered_tool_calls": filtered_count}})


def run_dispatcher(backend: Any, task_ids: list[str], workers: dict[str, Any],
                   inboxes: dict[str, Any], outbox: Any, *, max_new_tokens: int,
                   worker_timeout: float, kv_slots: int, policy: str = "lru",
                   release_on_done: bool = True) -> tuple[list[dict[str, Any]],
                                                    list[dict[str, Any]],
                                                    list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    ready: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0
    created: set[str] = set()
    resident_lru: OrderedDict[str, None] = OrderedDict()
    caps = backend.request({"command": "get_capabilities"})
    if caps.get("status") != "ok" or caps.get("model_load_count") != 1:
        raise RuntimeError(f"runner not resident: {caps}")
    server_pid = caps.get("server_pid")

    def receive(message: dict[str, Any]) -> None:
        nonlocal sequence
        if message.get("type") == "ready":
            sequence += 1
            heapq.heappush(ready, (message["ready_ns"], sequence, message))
            events.append(event("llm_ready", message["task_id"],
                                ready_ns=message["ready_ns"], step=message["step"]))
        elif message.get("type") == "done":
            task_id = message["task_id"]
            summaries[task_id] = message["summary"]
            events.append(event("task_end", task_id,
                                success=message["summary"]["success"],
                                error=message["summary"]["error"]))
            if release_on_done and task_id in created:
                response = backend.request({"command": "evict_session", "session_id": task_id})
                if response.get("status") != "ok":
                    raise RuntimeError(f"evict_session failed: {response}")
                events.append(event("session_release", task_id, source="runtime_explicit"))
                resident_lru.pop(task_id, None)
                created.remove(task_id)
        else:
            events.append(message)

    while len(summaries) < len(task_ids):
        try:
            if not ready:
                receive(outbox.get(timeout=1.0))
            while True:
                receive(outbox.get_nowait())
        except queue.Empty:
            pass
        if not ready:
            if any(not worker.is_alive() and task_id not in summaries
                   for task_id, worker in workers.items()):
                raise RuntimeError("AppWorld worker exited without a done message")
            continue
        _, _, item = heapq.heappop(ready)
        task_id = item["task_id"]
        if task_id in summaries:
            raise RuntimeError(f"completed task was queued: {task_id}")
        if task_id not in created:
            response = backend.request({"command": "create_session", "session_id": task_id})
            if response.get("status") != "ok":
                raise RuntimeError(f"create_session failed: {response}")
            created.add(task_id)
            events.append(event("session_create", task_id))
        policy_hit = task_id in resident_lru
        if not policy_hit:
            while len(resident_lru) >= kv_slots:
                queued_task_ids = {row[2]["task_id"] for row in ready}
                victim = choose_lru_victim(resident_lru, queued_task_ids, policy)
                resident_lru.pop(victim)
                response = backend.request({"command": "evict_session", "session_id": victim})
                if response.get("status") != "ok":
                    raise RuntimeError(f"capacity eviction failed: {response}")
                events.append(event("session_evict", victim,
                                    reason=f"{policy}_capacity",
                                    admitted_task_id=task_id,
                                    queued_task_ids=sorted(queued_task_ids),
                                    response=response))
        dequeue_ns = real_monotonic_ns()
        request_id = f"{task_id}_{item['step']:02d}"
        queue_ms = (dequeue_ns - item["ready_ns"]) / 1_000_000
        events.append(event("inference_start", task_id, request_id=request_id,
                            queue_ms=queue_ms, input_tokens=len(item["token_ids"])))
        response = backend.request({
            "command": "run_request", "request_id": request_id,
            "session_id": task_id, "input_token_ids": item["token_ids"],
            "max_new_tokens": max_new_tokens, "temperature": 0,
            "stop_on_eos": True,
        })
        finish_ns = real_monotonic_ns()
        if response.get("status") == "ok":
            resident_lru[task_id] = None
            resident_lru.move_to_end(task_id)
        if (response.get("status") == "ok" and
                (response.get("server_pid") != server_pid or
                 response.get("model_load_count") != 1)):
            raise AssertionError("runner PID changed or model reloaded")
        requests.append({"task_id": task_id, "request_id": request_id,
                         "step": item["step"], "queue_ms": queue_ms,
                         "policy_kv_hit": policy_hit,
                         "prompt_prepare_ms": item["prompt_prepare_ms"],
                         "host_start_ns": dequeue_ns, "host_end_ns": finish_ns,
                         "host_roundtrip_ms": (finish_ns - dequeue_ns) / 1_000_000,
                         "input_tokens_host": len(item["token_ids"]), **response})
        events.append(event("inference_end", task_id, request_id=request_id,
                            status=response.get("status")))
        inboxes[task_id].put(response)
    return events, requests, [summaries[task_id] for task_id in task_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("phone", "mock"), default="phone")
    parser.add_argument("--serial")
    parser.add_argument("--task-ids", nargs="+", default=["6bdbc26_2", "6bdbc26_3"])
    parser.add_argument("--profile", choices=tuple(PROFILES), default="follower_count")
    parser.add_argument("--task-profile", action="append", default=[], metavar="TASK=PROFILE",
                        help="override the task profile")
    parser.add_argument("--enrich-navigation-task", action="append", default=[], metavar="TASK",
                        help="add current-song observation after navigation move")
    parser.add_argument("--kv-slots", "--capacity", dest="kv_slots", type=int, default=2,
                        help="host policy KV budget; --capacity is a legacy alias")
    parser.add_argument("--policy", choices=("lru", "ready_lru"), default="lru")
    parser.add_argument("--release-on-done", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--max-sessions", type=int,
                        help="runner session ID capacity; defaults to number of tasks")
    parser.add_argument("--tool-delay-ms", type=float, default=1000.0)
    parser.add_argument("--task-tool-delay-ms", action="append", default=[], metavar="TASK=MS",
                        help="override synthetic tool delay for one task")
    parser.add_argument("--task-arrival-delay-ms", action="append", default=[], metavar="TASK=MS",
                        help="delay first task arrival after all workers initialize")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=6144)
    parser.add_argument("--worker-timeout", type=float, default=240.0)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--appworld-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "appworld_data")
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession_ctx6144")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "phone" and not args.serial:
        parser.error("--serial is required for --backend phone")
    if len(set(args.task_ids)) != len(args.task_ids) or args.kv_slots < 1:
        parser.error("task IDs must be unique and kv-slots positive")
    if args.max_sessions is None:
        args.max_sessions = len(args.task_ids)
    if args.max_sessions < len(args.task_ids):
        parser.error("runner max-sessions must be at least the number of distinct task IDs")
    if args.tool_delay_ms < 0 or args.max_steps < 1:
        parser.error("delay must be nonnegative and max-steps positive")
    def parse_overrides(specs: list[str], label: str) -> dict[str, float]:
        overrides: dict[str, float] = {}
        for spec in specs:
            try:
                task_id, value = spec.rsplit("=", 1)
                delay_ms = float(value)
            except ValueError:
                parser.error(f"invalid {label}: {spec!r}")
            if task_id not in args.task_ids or task_id in overrides or delay_ms < 0:
                parser.error(f"unknown, duplicate, or negative {label}: {spec!r}")
            overrides[task_id] = delay_ms
        return overrides

    task_tool_delays = parse_overrides(args.task_tool_delay_ms, "--task-tool-delay-ms")
    task_arrival_delays = parse_overrides(args.task_arrival_delay_ms,
                                          "--task-arrival-delay-ms")
    task_profiles: dict[str, str] = {task_id: args.profile for task_id in args.task_ids}
    overridden_profiles: set[str] = set()
    for spec in args.task_profile:
        try:
            task_id, profile = spec.rsplit("=", 1)
        except ValueError:
            parser.error(f"invalid --task-profile: {spec!r}")
        if task_id not in task_profiles or task_id in overridden_profiles or profile not in PROFILES:
            parser.error(f"unknown, duplicate, or unsupported --task-profile: {spec!r}")
        task_profiles[task_id] = profile
        overridden_profiles.add(task_id)
    enriched_tasks = set(args.enrich_navigation_task)
    if (len(enriched_tasks) != len(args.enrich_navigation_task) or
            any(task_id not in task_profiles or task_profiles[task_id] != "song_navigation"
                for task_id in enriched_tasks)):
        parser.error("--enrich-navigation-task requires unique song_navigation task IDs")
    if args.backend == "mock" and any(profile != "follower_count"
                                      for profile in task_profiles.values()):
        parser.error("the scripted mock supports only follower_count")
    # Check task semantics before allocating a phone runner or creating output files.
    for task_id, profile in task_profiles.items():
        specs_path = args.appworld_root / "data" / "tasks" / task_id / "specs.json"
        try:
            instruction = json.loads(specs_path.read_text(encoding="utf-8"))["instruction"]
            validate_profile_instruction(profile, instruction)
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"invalid task/profile for {task_id}: {exc}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "backend": args.backend,
        "task_ids": args.task_ids, "profile": args.profile,
        "task_profiles": task_profiles,
        "enrich_navigation_tasks": sorted(enriched_tasks),
        "max_sessions": args.max_sessions, "kv_slots": args.kv_slots,
        "policy": args.policy, "release_on_done": args.release_on_done,
        "tool_delay_ms": args.tool_delay_ms, "max_steps": args.max_steps,
        "task_tool_delay_ms": task_tool_delays,
        "task_arrival_delay_ms": task_arrival_delays,
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens, "serial": args.serial,
        "experiment_name": args.experiment_name,
        "bootstrap_auth": True, "prompt_style": "direct_json",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "small multi-session plumbing smoke; synthetic tool delay, not benchmark",
    }, indent=2) + "\n", encoding="utf-8")
    context = mp.get_context("spawn")
    outbox = context.Queue()
    inboxes = {task_id: context.Queue() for task_id in args.task_ids}
    start_barrier = context.Barrier(len(args.task_ids))
    settings = {"appworld_root": str(args.appworld_root.resolve()),
                "bundle": str(args.bundle.resolve()), "profile": args.profile,
                "experiment_name": args.experiment_name,
                "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
                "context_tokens": args.context_tokens,
                "tool_delay_ms": args.tool_delay_ms,
                "worker_timeout": args.worker_timeout}
    workers: dict[str, Any] = {}
    try:
        if args.backend == "phone":
            backend_context: Any = QnnPhoneBackend(
                bundle=args.bundle, serial=args.serial, max_sessions=args.max_sessions,
                htp_mode=8, device_dir=args.device_dir,
                raw_log=args.output_dir / "runner.log", request_timeout=180,
            )
        else:
            backend_context = ScriptedBackend(Tokenizer.from_file(
                str(args.bundle / "device" / "tokenizer.json")))
        with backend_context as backend:
            for task_id in args.task_ids:
                worker = context.Process(
                    target=task_worker, args=({**settings, "task_id": task_id,
                                               "profile": task_profiles[task_id],
                                               "enrich_navigation": task_id in enriched_tasks,
                                               "tool_delay_ms": task_tool_delays.get(
                                                   task_id, args.tool_delay_ms),
                                               "arrival_delay_ms": task_arrival_delays.get(
                                                   task_id, 0)},
                                              inboxes[task_id], outbox,
                                              start_barrier),
                    name=f"appworld-{task_id}")
                worker.start()
                workers[task_id] = worker
            events, requests, summaries = run_dispatcher(
                backend, args.task_ids, workers, inboxes, outbox,
                max_new_tokens=args.max_new_tokens,
                worker_timeout=args.worker_timeout, kv_slots=args.kv_slots,
                policy=args.policy, release_on_done=args.release_on_done)
    except Exception:
        (args.output_dir / "failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for worker in workers.values():
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    write_jsonl(args.output_dir / "events.jsonl", events)
    write_jsonl(args.output_dir / "requests.jsonl", requests)
    write_jsonl(args.output_dir / "tasks.jsonl", summaries)
    report = {"tasks": len(summaries),
              "successes": sum(row["success"] for row in summaries),
              "requests": len(requests),
              "max_sessions": args.max_sessions,
              "kv_slots": args.kv_slots,
              "policy": args.policy,
              "release_on_done": args.release_on_done,
              "capacity_evictions": sum(row["event"] == "session_evict" for row in events),
              "queue_ms_total": sum(row["queue_ms"] for row in requests),
              "queue_ms_max": max((row["queue_ms"] for row in requests), default=0),
              "note": ("scripted mock with real AppWorld tools; no model accuracy"
                       if args.backend == "mock" else
                       "phone model with real AppWorld tools; controlled extra wait; not a benchmark")}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
