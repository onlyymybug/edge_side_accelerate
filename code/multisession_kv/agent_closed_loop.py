#!/usr/bin/env python3
"""Small closed-loop Agent/phone-service smoke test, not a benchmark result.

The model chooses a tool; the tool executes; its result becomes the model's
next prompt. Independent tasks share one resident model, so requests can queue
while other tasks' tools run. No future turn or terminal label is read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import heapq
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from backend_adapter import QnnPhoneBackend
from collect_android import AndroidSampler


DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "qnn_multisession_phone_bundle_ctx6144"
DEFAULT_TASKS = Path(__file__).resolve().parent / "data" / "agent_closed_loop_smoke.json"
SYSTEM = (
    "You are a tool-using assistant. To look up a requested item, respond with "
    'exactly {"name":"lookup_item","parameters":{"item":"ITEM"}} and no other text. '
    "After a tool result is returned, answer the user's question concisely. "
    "Never invent a lookup result."
)


def clean_generation(value: str) -> str:
    return re.split(r"<\|(?:im_end|endoftext|fim_suffix)\|>", value, maxsplit=1)[0].strip()


def parse_tool_call(value: str) -> str:
    value = clean_generation(value)
    try:
        call = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a single JSON tool call: {value[:120]!r}") from exc
    if not isinstance(call, dict) or call.get("name") != "lookup_item":
        raise ValueError(f"unexpected tool call: {call!r}")
    parameters = call.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("item"), str):
        raise ValueError(f"tool call lacks item: {call!r}")
    return parameters["item"]


def render_prompt(messages: list[dict[str, str]]) -> str:
    chunks = [f"<|im_start|>system\n{SYSTEM}<|im_end|>"]
    chunks.extend(
        f"<|im_start|>{row['role']}\n{row['content']}<|im_end|>" for row in messages
    )
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def execute_tool(catalog: dict[str, str], item: str, delay_ms: float) -> dict[str, Any]:
    start_ns = time.monotonic_ns()
    if delay_ms:
        time.sleep(delay_ms / 1000)
    value = catalog.get(item)
    return {
        "item": item,
        "value": value,
        "error": None if value is not None else "not_found",
        "tool_start_ns": start_ns,
        "tool_end_ns": time.monotonic_ns(),
    }


class MockBackend:
    """Protocol-compatible scripted backend for plumbing tests only."""

    def __init__(self, tasks: list[dict[str, str]]) -> None:
        self.tasks = {task["task_id"]: task for task in tasks}
        self.calls: dict[str, int] = {}

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload["command"]
        if command == "get_capabilities":
            return {"status": "ok", "model_load_count": 1, "server_pid": 0}
        if command in {"create_session", "evict_session"}:
            return {"status": "ok"}
        if command != "run_request":
            raise ValueError(command)
        task_id = payload["session_id"]
        turn = self.calls.get(task_id, 0)
        self.calls[task_id] = turn + 1
        time.sleep(0.01)
        if turn == 0:
            content = json.dumps(
                {"name": "lookup_item", "parameters": {"item": self.tasks[task_id]["item"]}},
                separators=(",", ":"),
            )
        else:
            content = f"The value is {self.tasks[task_id]['expected']}."
        return {
            "status": "ok", "generated_text": content, "ttft_ms": 1.0,
            "prefill_ms": 1.0, "decode_ms": 1.0, "model_load_count": 1,
            "server_pid": 0,
        }


@dataclass
class TaskState:
    task_id: str
    item: str
    expected: str
    question: str
    messages: list[dict[str, str]] = field(default_factory=list)
    phase: str = "ready"
    started_ns: int = 0
    completed_ns: int = 0
    success: bool = False
    error: str | None = None
    final_answer: str | None = None


def run_closed_loop(
    backend: Any,
    tokenizer: Tokenizer,
    tasks_data: list[dict[str, str]],
    catalog: dict[str, str],
    *,
    tool_delay_ms: float,
    max_new_tokens: int,
    context_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    tasks = {
        row["task_id"]: TaskState(**row, messages=[{"role": "user", "content": row["question"]}])
        for row in tasks_data
    }
    ready: list[tuple[int, int, str]] = []
    order = 0

    def event(name: str, task_id: str, **fields: Any) -> None:
        events.append({"host_monotonic_ns": time.monotonic_ns(), "event": name,
                       "task_id": task_id, **fields})

    def enqueue(task_id: str, ready_ns: int) -> None:
        nonlocal order
        order += 1
        heapq.heappush(ready, (ready_ns, order, task_id))
        tasks[task_id].phase = "ready"
        event("llm_ready", task_id, ready_ns=ready_ns)

    def finish(task: TaskState, error: str | None = None) -> None:
        task.phase = "done"
        task.error = error
        task.completed_ns = time.monotonic_ns()
        event("task_end", task.task_id, success=task.success, error=error)
        response = backend.request({"command": "evict_session", "session_id": task.task_id})
        if response.get("status") != "ok":
            raise RuntimeError(f"evict_session failed: {response}")
        event("session_release", task.task_id, source="runtime_explicit")

    capabilities = backend.request({"command": "get_capabilities"})
    if capabilities.get("status") != "ok" or capabilities.get("model_load_count") != 1:
        raise RuntimeError(f"runner not resident: {capabilities}")
    server_pid = capabilities.get("server_pid")
    for task in tasks.values():
        response = backend.request({"command": "create_session", "session_id": task.task_id})
        if response.get("status") != "ok":
            raise RuntimeError(f"create_session failed: {response}")
        task.started_ns = time.monotonic_ns()
        event("task_start", task.task_id)
        enqueue(task.task_id, task.started_ns)

    pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        while any(task.phase != "done" for task in tasks.values()):
            for future, task_id in list(pending.items()):
                if not future.done():
                    continue
                del pending[future]
                task = tasks[task_id]
                result = future.result()
                event("tool_end", task_id, item=result["item"], value=result["value"],
                      tool_start_ns=result["tool_start_ns"], tool_end_ns=result["tool_end_ns"])
                task.messages.append({"role": "tool", "content": json.dumps(
                    {"item": result["item"], "value": result["value"], "error": result["error"]},
                    separators=(",", ":"))})
                enqueue(task_id, result["tool_end_ns"])

            if not ready:
                if not pending:
                    raise RuntimeError("unfinished tasks have no runnable request or tool")
                concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                continue

            ready_ns, _, task_id = heapq.heappop(ready)
            dequeue_ns = time.monotonic_ns()
            queue_ms = (dequeue_ns - ready_ns) / 1_000_000
            task = tasks[task_id]
            prepare_start_ns = time.monotonic_ns()
            prompt = render_prompt(task.messages)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            prompt_prepare_ms = (time.monotonic_ns() - prepare_start_ns) / 1_000_000
            if len(token_ids) + max_new_tokens > context_tokens:
                finish(task, f"context overflow: {len(token_ids)} + {max_new_tokens}")
                continue
            request_id = f"{task_id}_{len([r for r in requests if r['task_id'] == task_id]) + 1}"
            start_ns = time.monotonic_ns()
            task.phase = "running"
            event("inference_start", task_id, request_id=request_id, queue_ms=queue_ms,
                  prompt_prepare_ms=prompt_prepare_ms, input_tokens=len(token_ids))
            response = backend.request({
                "command": "run_request", "request_id": request_id,
                "session_id": task_id, "input_token_ids": token_ids,
                "max_new_tokens": max_new_tokens, "temperature": 0, "stop_on_eos": True,
            })
            end_ns = time.monotonic_ns()
            record = {"task_id": task_id, "request_id": request_id,
                      "host_start_ns": start_ns, "host_end_ns": end_ns,
                      "queue_ms": queue_ms, "prompt_prepare_ms": prompt_prepare_ms,
                      "host_roundtrip_ms": (end_ns - start_ns) / 1_000_000,
                      "input_tokens_host": len(token_ids), **response}
            requests.append(record)
            event("inference_end", task_id, request_id=request_id,
                  status=response.get("status"))
            if response.get("status") != "ok":
                finish(task, f"runner response: {response}")
                continue
            if response.get("model_load_count") != 1 or response.get("server_pid") != server_pid:
                raise AssertionError("runner PID changed or model was reloaded")
            output = clean_generation(str(response.get("generated_text", "")))
            if len(task.messages) == 1:
                try:
                    item = parse_tool_call(output)
                    if item not in catalog:
                        raise ValueError(f"unknown item: {item}")
                except ValueError as exc:
                    finish(task, str(exc))
                    continue
                task.messages.append({"role": "assistant", "content": output})
                event("tool_start", task_id, item=item)
                task.phase = "tool_wait"
                pending[pool.submit(execute_tool, catalog, item, tool_delay_ms)] = task_id
            else:
                task.final_answer = output
                task.success = task.expected.casefold() in output.casefold()
                finish(task, None if task.success else "expected value absent from final answer")

    results = [
        {"task_id": task.task_id, "success": task.success, "error": task.error,
         "answer": task.final_answer, "completion_ms":
         (task.completed_ns - task.started_ns) / 1_000_000}
        for task in tasks.values()
    ]
    return events, requests, results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "phone"), default="mock")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--serial")
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession_ctx6144")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tool-delay-ms", type=float, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-tokens", type=int, default=6144)
    parser.add_argument("--htp-mode", type=int, default=8)
    parser.add_argument("--telemetry-interval", type=float, default=1)
    args = parser.parse_args()
    if args.backend == "phone" and not args.serial:
        parser.error("--serial is required for --backend phone")
    if args.tool_delay_ms < 0 or args.max_new_tokens < 1 or args.context_tokens < 1:
        parser.error("delay and token limits must be non-negative/positive")
    fixture = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks_data = fixture["tasks"]
    catalog = fixture["catalog"]
    tokenizer = Tokenizer.from_file(str(args.bundle / "device" / "tokenizer.json"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "backend": args.backend, "tasks": str(args.tasks), "task_count": len(tasks_data),
        "bundle": str(args.bundle), "serial": args.serial,
        "tool_delay_ms": args.tool_delay_ms, "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens, "htp_mode": args.htp_mode,
        "note": "synthetic closed-loop plumbing smoke; not a benchmark result",
    }, indent=2) + "\n", encoding="utf-8")

    if args.backend == "mock":
        backend: Any = MockBackend(tasks_data)
    else:
        backend = QnnPhoneBackend(
            bundle=args.bundle, serial=args.serial,
            max_sessions=len(tasks_data), htp_mode=args.htp_mode,
            device_dir=args.device_dir, raw_log=args.output_dir / "runner.log",
            request_timeout=180,
        )
    sampler: AndroidSampler | None = None
    try:
        if args.backend == "phone" and args.telemetry_interval > 0:
            caps = backend.request({"command": "get_capabilities"})
            if caps.get("status") != "ok":
                raise RuntimeError(caps)
            sampler = AndroidSampler(args.serial, int(caps["server_pid"]), args.telemetry_interval)
            sampler.start()
        events, requests, results = run_closed_loop(
            backend, tokenizer, tasks_data, catalog,
            tool_delay_ms=args.tool_delay_ms, max_new_tokens=args.max_new_tokens,
            context_tokens=args.context_tokens,
        )
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.write(args.output_dir / "telemetry.jsonl")
        if args.backend == "phone":
            backend.close()
    write_jsonl(args.output_dir / "events.jsonl", events)
    write_jsonl(args.output_dir / "requests.jsonl", requests)
    write_jsonl(args.output_dir / "tasks.jsonl", results)
    summary = {"tasks": len(results), "successes": sum(row["success"] for row in results),
               "requests": len(requests),
               "queue_ms_total": sum(row["queue_ms"] for row in requests),
               "note": "smoke test only; no paper-level claim"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
