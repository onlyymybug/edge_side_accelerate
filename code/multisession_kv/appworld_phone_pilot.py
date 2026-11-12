#!/usr/bin/env python3
"""A restricted, single-task AppWorld agent with LLM inference on the phone.

Only a small allowlist of AppWorld functional APIs is exposed. The model never
supplies Python code: each JSON call is converted to a fixed, validated API
invocation. This is a feasibility pilot, not an AppWorld benchmark submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from appworld import AppWorld, update_root
from jinja2 import Environment
from tokenizers import Tokenizer

from agent_closed_loop import DEFAULT_BUNDLE, clean_generation, write_jsonl
from backend_adapter import QnnPhoneBackend


ALLOWLIST = (
    "supervisor__show_profile",
    "supervisor__show_account_passwords",
    "spotify__login",
    "spotify__show_current_song",
    "spotify__show_artist",
    "spotify__search_artists",
    "spotify__search_songs",
    "spotify__add_to_queue",
    "spotify__previous_song",
    "spotify__next_song",
    "supervisor__complete_task",
)
PROFILES = {
    "follower_count": {
        "spotify__show_current_song", "spotify__show_artist",
        "supervisor__complete_task",
    },
    "song_navigation": {
        "spotify__show_current_song", "spotify__previous_song",
        "spotify__next_song", "supervisor__complete_task",
    },
    "most_played_song": {
        "spotify__search_artists", "spotify__search_songs",
        "supervisor__complete_task",
    },
    "queue_popular_songs": {
        "spotify__search_artists", "spotify__search_songs",
        "spotify__add_to_queue", "supervisor__complete_task",
    },
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def validate_profile_instruction(profile: str, instruction: str) -> None:
    """Reject a task/profile mismatch before a profile-specific prompt is sent."""
    if profile == "most_played_song" and "most played song" not in instruction.casefold():
        raise ValueError(
            f"profile {profile!r} requires a 'most played song' instruction; "
            f"got {instruction!r}")
    if profile == "queue_popular_songs" and not (
            "add all the songs" in instruction.casefold()
            and "player queue" in instruction.casefold()
            and "played over" in instruction.casefold()):
        raise ValueError(f"profile {profile!r} does not match {instruction!r}")


def real_monotonic_ns() -> int:
    """Freezegun in AppWorld patches monotonic_ns, but not clock_gettime_ns."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def load_tool_docs(root: Path) -> list[dict[str, Any]]:
    directory = root / "data" / "api_docs" / "function_calling"
    selected = []
    for app in ("supervisor", "spotify"):
        for entry in json.loads((directory / f"{app}.json").read_text(encoding="utf-8")):
            function = entry["function"]
            if function["name"] in ALLOWLIST:
                selected.append(function)
    if {row["name"] for row in selected} != set(ALLOWLIST):
        raise RuntimeError("AppWorld API docs missing an allowlisted function")
    return selected


def parse_call(output: str, zero_arg_names: set[str] | None = None,
               prompt_style: str = "direct_json",
               flat_parameter_names: dict[str, set[str]] | None = None
               ) -> tuple[str, dict[str, Any], bool, int]:
    value = clean_generation(output)
    repaired = False
    extra_calls = 0
    if prompt_style in {"qwen_hermes", "qwen_bundle_template"}:
        matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>",
                                   value, re.DOTALL))
        if not matches:
            raise ValueError(f"not a Hermes tool call: {value[:120]!r}")
        extra_calls = max(0, value.count("<tool_call>") - 1)
        value = matches[0].group(1).strip()
        if value.startswith('"name"'):
            value = "{" + value + "}"
            repaired = True
    try:
        call = json.loads(value)
    except json.JSONDecodeError as exc:
        # A narrowly scoped, logged repair for a malformed key and/or one
        # trailing quote. Do not infer a missing tool name or parameter value.
        candidate = value.replace('"parameters:', '"parameters":', 1)
        candidate = candidate.replace('"parameters {}"', '"parameters":{}', 1)
        if candidate.endswith('}"'):
            candidate = candidate[:-1]
        try:
            call = json.loads(candidate)
        except json.JSONDecodeError:
            raise ValueError(f"not a single JSON call: {value[:120]!r}") from exc
        repaired = candidate != value
    if prompt_style in {"qwen_hermes", "qwen_bundle_template"} and isinstance(call, dict):
        if set(call) == {"name", "arguments"}:
            call["parameters"] = call.pop("arguments")
    if (isinstance(call, dict) and set(call) == {"name"}
            and call["name"] in (zero_arg_names or set())):
        call["parameters"] = {}
        repaired = True
    if (isinstance(call, dict) and isinstance(call.get("name"), str)
            and "parameters" not in call):
        allowed = (flat_parameter_names or {}).get(call["name"], set())
        flat_keys = set(call) - {"name"}
        if flat_keys and flat_keys <= allowed:
            call = {"name": call["name"],
                    "parameters": {key: call[key] for key in flat_keys}}
            repaired = True
    if not isinstance(call, dict) or set(call) != {"name", "parameters"}:
        raise ValueError("call must contain only name and parameters")
    name, parameters = call["name"], call["parameters"]
    if name not in ALLOWLIST or not isinstance(parameters, dict):
        raise ValueError(f"API not allowed or parameters invalid: {name!r}")
    return name, parameters, repaired, extra_calls


def call_tool(world: AppWorld, name: str, parameters: dict[str, Any],
              docs: dict[str, dict[str, Any]]) -> str:
    """Invoke one allowlisted API; model text cannot become executable code."""
    if name not in ALLOWLIST or name not in docs:
        raise ValueError(f"API not allowlisted: {name}")
    allowed = set(docs[name]["parameters"].get("properties", {}))
    if not set(parameters) <= allowed:
        raise ValueError(f"unexpected parameters for {name}: {set(parameters) - allowed}")
    if not all(IDENTIFIER.fullmatch(key) for key in parameters):
        raise ValueError("invalid parameter identifier")
    app, api = name.split("__", 1)
    if not IDENTIFIER.fullmatch(app) or not IDENTIFIER.fullmatch(api):
        raise ValueError("invalid API identifier")
    for value in parameters.values():
        if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
            raise ValueError("unsupported parameter type")
    kwargs = ", ".join(f"{key}={value!r}" for key, value in parameters.items())
    return world.execute(f"print(apis.{app}.{api}({kwargs}))")


def prepare_parameters(name: str, parameters: dict[str, Any],
                       visible_docs: dict[str, dict[str, Any]],
                       execution_docs: dict[str, dict[str, Any]],
                       bootstrap_token: str | None) -> tuple[dict[str, Any], list[str]]:
    if name not in visible_docs:
        raise ValueError(f"API not exposed at this stage: {name}")
    allowed_visible = set(visible_docs[name]["parameters"].get("properties", {}))
    dropped = sorted(set(parameters) - allowed_visible)
    if bootstrap_token is not None:
        # Drop extra fields, but never invent a task-specific argument/ID.
        parameters = {key: value for key, value in parameters.items()
                      if key in allowed_visible}
    if (bootstrap_token is not None and name.startswith("spotify__")
            and "access_token" in execution_docs[name]["parameters"].get("properties", {})):
        parameters["access_token"] = bootstrap_token
    return parameters, dropped


def normalize_action_answer(profile: str, name: str,
                            parameters: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if profile not in {"song_navigation", "queue_popular_songs"} or name != "supervisor__complete_task":
        return parameters, False
    normalized = parameters.get("answer") is not None
    return {**parameters, "answer": None}, normalized


def render(messages: list[dict[str, str]], docs: list[dict[str, Any]],
           bootstrap_token: str | None = None,
           profile: str = "follower_count",
           prompt_style: str = "direct_json",
           chat_template: str | None = None) -> str:
    if prompt_style == "qwen_bundle_template":
        if chat_template is None:
            raise ValueError("bundle chat_template.jinja is required")
        system = (
            "Solve the user task in the simulated app. Call exactly one tool per "
            "turn and wait for its result before deciding the next action. "
            "Use only the provided functions."
        )
        if bootstrap_token is not None:
            system += (
                " The Agent Service already logged into Spotify. Do not include "
                "access_token; the service attaches it securely."
            )
        if profile == "follower_count":
            system += (
                " Inspect the current song, then its artist, then submit the "
                "follower count with supervisor__complete_task."
            )
        elif profile == "song_navigation":
            system += (
                " Move one song in the requested direction, inspect the current "
                "song after each move, and stop only when the target artist is "
                "reached. This is an action task, so complete_task needs no answer."
            )
        elif profile == "most_played_song":
            system += (
                " Search the named artist, then search that artist's songs "
                "sorted by -play_count with page_limit=1, then submit the "
                "returned title with supervisor__complete_task."
            )
        elif profile == "queue_popular_songs":
            system += (
                " Search the named artist, search the artist's songs above "
                "the requested play count, add every returned song to the "
                "queue, then complete the task without an answer."
            )
        tool_specs = [{"type": "function", "function": row} for row in docs]
        return Environment(autoescape=False).from_string(chat_template).render(
            messages=[{"role": "system", "content": system}, *messages],
            tools=tool_specs, add_generation_prompt=True)
    tool_lines = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                           for row in docs)
    if prompt_style == "qwen_hermes":
        tool_lines = "\n".join(json.dumps({"type": "function", "function": row},
                                         ensure_ascii=False, separators=(",", ":"))
                               for row in docs)
        system = (
            "You are solving a task in a simulated app environment.\n\n# Tools\n\n"
            "You may call one function per turn. Function signatures are in "
            "<tools></tools> XML tags:\n<tools>\n" + tool_lines + "\n</tools>\n\n"
            "For each function call, return one JSON object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n"
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\nDo not write text outside the tool_call tags. "
            "Use only the provided functions."
        )
    else:
        if profile in {"most_played_song", "queue_popular_songs"}:
            output_rule = (
                "On every turn output exactly one JSON object with only the "
                "keys name and parameters. The name must be one of the provided "
                "function names, never a placeholder. Your first function must "
                "be spotify__search_artists, with parameters.query equal to the "
                "artist's name in the user request. No Markdown or explanation. "
            )
        else:
            output_rule = (
                'On every turn output exactly one JSON object: {"name":"app__api",'
                '"parameters":{"parameter":"value"}}. No Markdown or explanation. '
            )
        system = (
            "You are solving a task in a simulated app environment. "
            "You may call only these functions:\n<tools>\n" + tool_lines + "\n</tools>\n"
            + output_rule +
            "First inspect the supervisor profile and account passwords if login is needed. "
            "Read tool results carefully; use returned IDs and credentials exactly. "
            "When you know the answer, call supervisor__complete_task with the answer. "
            "Never call an API outside the provided list."
        )
    if bootstrap_token is not None:
        system += (
            " The Agent Service has already authenticated to Spotify using "
            "online account data. Do not include access_token in any call; "
            "the service attaches it securely. Never add top-level keys "
            "other than name and parameters. "
        )
        if profile == "follower_count":
            if prompt_style == "qwen_hermes":
                system += (
                    " First call spotify__show_current_song. Then call "
                    "spotify__show_artist with its returned artist ID and "
                    "supervisor__complete_task with the follower count."
                )
            else:
                system += (
                    'Your first reply must be exactly '
                    '{"name":"spotify__show_current_song","parameters":{}}. '
                    "After seeing that result, call spotify__show_artist with the "
                    "returned artist ID, then supervisor__complete_task with the "
                    "artist's follower count."
                )
        elif profile == "song_navigation":
            system += (
                "Inspect the current song and its artist. Move one song at a "
                "time in the direction requested by the user, inspecting the "
                "result after each move. When the target artist is reached, "
                "call supervisor__complete_task with status success and no answer."
            )
        elif profile == "most_played_song":
            system += (
                "First call spotify__search_artists with the artist name from "
                "the user task. Then call spotify__search_songs with that "
                "artist_id, sort_by='-play_count', page_limit=1. Finally call "
                "supervisor__complete_task with the first song's title."
            )
        elif profile == "queue_popular_songs":
            system += (
                "Use the exact matching artist_id from search_artists, not an "
                "unrelated artist's ID. Search songs using artist_id and "
                "min_play_count from the user request. For each returned song, "
                "call spotify__add_to_queue with its song_id. Then call "
                "supervisor__complete_task with status success and no answer."
            )
    chunks = [f"<|im_start|>system\n{system}<|im_end|>"]
    chunks.extend(f"<|im_start|>{row['role']}\n{row['content']}<|im_end|>"
                  for row in messages)
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def run_task(backend: Any, tokenizer: Tokenizer, world: AppWorld,
             tool_docs: list[dict[str, Any]], *, max_steps: int,
             max_new_tokens: int, context_tokens: int,
             bootstrap_token: str | None = None,
             profile: str = "follower_count",
             prompt_style: str = "direct_json",
             chat_template: str | None = None,
             enrich_navigation: bool = False,
             execution_docs: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]],
                                                                 list[dict[str, Any]], dict[str, Any]]:
    validate_profile_instruction(profile, world.task.instruction)
    visible_docs = {row["name"]: row for row in tool_docs}
    zero_arg_names = {name for name, doc in visible_docs.items()
                      if not doc["parameters"].get("properties", {})}
    docs_by_name = execution_docs or visible_docs
    events: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    messages = [{"role": "user", "content": world.task.instruction}]
    session_id = world.task.id
    started_ns = real_monotonic_ns()
    caps = backend.request({"command": "get_capabilities"})
    if caps.get("status") != "ok" or caps.get("model_load_count") != 1:
        raise RuntimeError(f"runner not resident: {caps}")
    response = backend.request({"command": "create_session", "session_id": session_id})
    if response.get("status") != "ok":
        raise RuntimeError(response)
    events.append({"event": "task_start", "task_id": session_id,
                   "host_monotonic_ns": started_ns})
    error: str | None = None
    try:
        for step in range(1, max_steps + 1):
            prompt = render(messages, tool_docs, bootstrap_token, profile,
                            prompt_style, chat_template)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if len(token_ids) + max_new_tokens > context_tokens:
                error = f"context overflow: {len(token_ids)} + {max_new_tokens}"
                break
            request_id = f"{session_id}_{step:02d}"
            start_ns = real_monotonic_ns()
            events.append({"event": "inference_start", "task_id": session_id,
                           "request_id": request_id, "host_monotonic_ns": start_ns,
                           "input_tokens": len(token_ids)})
            response = backend.request({
                "command": "run_request", "request_id": request_id,
                "session_id": session_id, "input_token_ids": token_ids,
                "max_new_tokens": max_new_tokens, "temperature": 0,
                "stop_on_eos": True,
            })
            end_ns = real_monotonic_ns()
            requests.append({"request_id": request_id, "step": step,
                             "host_start_ns": start_ns, "host_end_ns": end_ns,
                             "host_roundtrip_ms": (end_ns - start_ns) / 1_000_000,
                             "input_tokens_host": len(token_ids), **response})
            events.append({"event": "inference_end", "task_id": session_id,
                           "request_id": request_id, "host_monotonic_ns": end_ns,
                           "status": response.get("status")})
            if response.get("status") != "ok":
                error = f"runner returned {response.get('status')}"
                break
            if response.get("server_pid") != caps.get("server_pid") or response.get("model_load_count") != 1:
                raise AssertionError("runner PID changed or model reloaded")
            output = clean_generation(str(response.get("generated_text", "")))
            try:
                name, parameters, repaired, extra_calls = parse_call(
                    output, zero_arg_names, prompt_style)
                parameters, dropped_parameters = prepare_parameters(
                    name, parameters, visible_docs, docs_by_name, bootstrap_token)
                # This profile is an action, not a question. AppWorld's
                # completion schema requires answer=None for actions.
                parameters, normalized_action_answer = normalize_action_answer(
                    profile, name, parameters)
                tool_start_ns = real_monotonic_ns()
                result = call_tool(world, name, parameters, docs_by_name)
                enriched = False
                if enrich_navigation and name in {"spotify__previous_song", "spotify__next_song"}:
                    current_result = call_tool(world, "spotify__show_current_song",
                                               {"access_token": bootstrap_token}, docs_by_name)
                    result = json.dumps({"move": json.loads(result),
                                         "current_song": json.loads(current_result)},
                                        ensure_ascii=False, separators=(",", ":"))
                    enriched = True
                tool_end_ns = real_monotonic_ns()
            except ValueError as exc:
                error = str(exc)
                break
            events.append({"event": "tool_call", "task_id": session_id,
                           "host_monotonic_ns": tool_start_ns,
                           "tool_end_ns": tool_end_ns,
                           "name": name, "parse_repaired": repaired,
                           "ignored_extra_calls": extra_calls,
                           "dropped_parameters": dropped_parameters,
                           "normalized_action_answer": normalized_action_answer,
                           "enriched_observation": enriched,
                           "tool_ms": (tool_end_ns - tool_start_ns) / 1_000_000})
            visible_parameters = {key: value for key, value in parameters.items()
                                  if key in visible_docs[name]["parameters"].get("properties", {})}
            if prompt_style in {"qwen_hermes", "qwen_bundle_template"}:
                assistant_content = "<tool_call>\n" + json.dumps(
                    {"name": name, "arguments": visible_parameters},
                    ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"
            else:
                assistant_content = json.dumps(
                    {"name": name, "parameters": visible_parameters},
                    ensure_ascii=False, separators=(",", ":"))
            messages.append({"role": "assistant", "content": assistant_content})
            # AppWorld tool outputs can be long. Keep a bounded observation so
            # context overflow is detected rather than silently truncated.
            messages.append({"role": "tool", "content": result[:2500]})
            if name == "supervisor__complete_task":
                break
        else:
            error = "max_steps reached without complete_task"
        evaluation = world.evaluate().to_dict()
    finally:
        backend.request({"command": "evict_session", "session_id": session_id})
        events.append({"event": "session_release", "task_id": session_id,
                       "host_monotonic_ns": real_monotonic_ns(),
                       "source": "runtime_explicit"})
    finished_ns = real_monotonic_ns()
    summary = {"task_id": session_id, "steps": len(requests),
               "success": evaluation["success"], "error": error,
               "completion_ms": (finished_ns - started_ns) / 1_000_000,
               "evaluation": {"difficulty": evaluation.get("difficulty"),
                              "num_tests": evaluation.get("num_tests"),
                              "passed_tests": len(evaluation.get("passes", []))},
               "repaired_tool_calls": sum(row.get("parse_repaired", False)
                                          for row in events if row["event"] == "tool_call"),
               "ignored_extra_calls": sum(row.get("ignored_extra_calls", 0)
                                          for row in events if row["event"] == "tool_call"),
               "filtered_tool_calls": sum(bool(row.get("dropped_parameters"))
                                          for row in events if row["event"] == "tool_call"),
               "enriched_observations": sum(bool(row.get("enriched_observation"))
                                            for row in events if row["event"] == "tool_call"),
               "normalized_action_answers": sum(bool(row.get("normalized_action_answer"))
                                                for row in events if row["event"] == "tool_call"),
               "note": "single-task feasibility pilot; not a benchmark aggregate"}
    events.append({"event": "task_end", "task_id": session_id,
                   "host_monotonic_ns": finished_ns, "success": evaluation["success"]})
    return events, requests, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--task-id", default="6bdbc26_1")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="follower_count")
    parser.add_argument("--prompt-style", choices=("direct_json", "qwen_hermes",
                                                   "qwen_bundle_template"),
                        default="direct_json")
    parser.add_argument("--appworld-root", type=Path, default=Path(__file__).resolve().parents[2] / "appworld_data")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--device-dir", default="/data/local/tmp/qnn_multisession_ctx6144")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=6144)
    parser.add_argument("--bootstrap-auth", action="store_true",
                        help="Agent Service performs online Spotify login; not official benchmark protocol")
    parser.add_argument("--enrich-navigation", action="store_true",
                        help="After a song move, service also fetches current-song details")
    args = parser.parse_args()
    if args.enrich_navigation and (args.profile != "song_navigation" or not args.bootstrap_auth):
        parser.error("--enrich-navigation requires --profile song_navigation --bootstrap-auth")
    update_root(str(args.appworld_root.resolve()))
    docs = load_tool_docs(args.appworld_root)
    template_path = args.bundle / "device" / "chat_template.jinja"
    chat_template = (template_path.read_text(encoding="utf-8")
                     if args.prompt_style == "qwen_bundle_template" else None)
    tokenizer = Tokenizer.from_file(str(args.bundle / "device" / "tokenizer.json"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "task_id": args.task_id, "experiment_name": args.experiment_name,
        "profile": args.profile,
        "prompt_style": args.prompt_style,
        "bundle": str(args.bundle), "serial": args.serial,
        "allowlist": list(ALLOWLIST), "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "htp_mode": 8,
        "bootstrap_auth": args.bootstrap_auth,
        "enrich_navigation": args.enrich_navigation,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tool_docs_sha256": hashlib.sha256(json.dumps(
            docs, sort_keys=True).encode("utf-8")).hexdigest(),
        "chat_template_sha256": (hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
                                 if chat_template is not None else None),
        "note": "AppWorld single-task feasibility pilot",
    }, indent=2) + "\n", encoding="utf-8")
    with AppWorld(task_id=args.task_id, experiment_name=args.experiment_name,
                  raise_on_unsafe_syntax=True) as world:
        bootstrap_token = None
        execution_docs = {row["name"]: row for row in docs}
        if args.bootstrap_auth:
            profile = json.loads(call_tool(world, "supervisor__show_profile", {}, execution_docs))
            accounts = json.loads(call_tool(world, "supervisor__show_account_passwords", {}, execution_docs))
            spotify = next(row for row in accounts if row["account_name"] == "spotify")
            login = json.loads(call_tool(world, "spotify__login", {
                "username": profile["email"], "password": spotify["password"],
            }, execution_docs))
            bootstrap_token = login["access_token"]
            docs = [json.loads(json.dumps(row)) for row in docs
                    if row["name"] in PROFILES[args.profile]]
            for row in docs:
                if row["name"].startswith("spotify__"):
                    row["parameters"]["properties"].pop("access_token", None)
        with QnnPhoneBackend(
            bundle=args.bundle, serial=args.serial, max_sessions=1, htp_mode=8,
            device_dir=args.device_dir, raw_log=args.output_dir / "runner.log",
            request_timeout=180,
        ) as backend:
            events, requests, summary = run_task(
                backend, tokenizer, world, docs, max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens, context_tokens=args.context_tokens,
                bootstrap_token=bootstrap_token, profile=args.profile,
                prompt_style=args.prompt_style,
                chat_template=chat_template,
                enrich_navigation=args.enrich_navigation,
                execution_docs=execution_docs,
            )
    write_jsonl(args.output_dir / "events.jsonl", events)
    write_jsonl(args.output_dir / "requests.jsonl", requests)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("task_id", "steps", "success", "error", "completion_ms")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
