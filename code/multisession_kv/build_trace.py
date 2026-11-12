#!/usr/bin/env python3
"""Build reproducible HammerBench native/fixed-tools token traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer


PROMPT_TEMPLATE_VERSION = "hammerbench_official_appendix_d_qwen_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_snapshot_id(source_id: str) -> tuple[str, str, int]:
    data_type, conversation_id, turn_text = source_id.rsplit("_", 2)
    turn_id = int(turn_text)
    if turn_id < 1:
        raise ValueError("turn ID must start at one")
    return data_type, conversation_id, turn_id


def last_function_call(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    calls = [
        (index, message.get("content"))
        for index, message in enumerate(row.get("messages", []))
        if message.get("role") == "function call"
        and isinstance(message.get("content"), dict)
    ]
    if not calls:
        raise ValueError("snapshot has no function-call label")
    return calls[-1]


def visible_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    label_index, _ = last_function_call(row)
    return [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in row.get("messages", [])[:label_index]
        if message.get("role") in ("user", "assistant")
    ]


def stable_fixed_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in rows:
        for tool in row.get("tools", row.get("multiple_tools", [])):
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise ValueError("tool has no string name")
            encoded = canonical(tool)
            previous = by_name.get(tool["name"])
            if previous and previous[0] != encoded:
                raise ValueError(f"conflicting definitions for tool {tool['name']}")
            by_name[tool["name"]] = (encoded, tool)
    return [by_name[name][1] for name in sorted(by_name)]


def system_prompt(tools: list[dict[str, Any]]) -> str:
    tool_lines = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools
    )
    return (
        "You have access to the following functions.\n"
        f"<tools>\n{tool_lines}\n</tools>\n"
        "To call a function, respond with exactly one JSON object in this form:\n"
        '{"name":"fully.qualified.tool.name","parameters":{"parameter":"value"}}\n'
        "Do not hallucinate parameters or parameter values. Include every parameter "
        "defined by the selected function; if its value is not stated in the user "
        "request or conversation history, use an empty string. Preserve values exactly "
        "as the user expressed them. In particular, do not normalize or rewrite time "
        "expressions. Never ask the user for missing parameters; output the current "
        "function-call snapshot using empty strings. Do not include Markdown, "
        "explanations, or any text outside the JSON object. If none of the available "
        "tools is suitable, output exactly: Sorry, no tool is suitable for your request."
    )


def render_prompt(messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
    chunks = [f"<|im_start|>system\n{system_prompt(tools)}<|im_end|>"]
    chunks.extend(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
        for message in messages
    )
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def lcp(lhs: list[int], rhs: list[int]) -> int:
    result = 0
    while result < min(len(lhs), len(rhs)) and lhs[result] == rhs[result]:
        result += 1
    return result


def load_sessions(dataset: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("HammerBench multi-turn dataset must be a JSON array")
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for source_index, row in enumerate(raw):
        try:
            source_id = str(row["id"])
            data_type, conversation_id, turn_id = parse_snapshot_id(source_id)
            messages = visible_messages(row)
            _, label = last_function_call(row)
            tools = row.get("tools", row.get("multiple_tools", []))
            if not messages or not isinstance(tools, list) or not tools:
                raise ValueError("missing visible messages or tools")
            sessions[(data_type, conversation_id)].append(
                {
                    "source_index": source_index,
                    "source_id": source_id,
                    "data_type": data_type,
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "visible_messages": messages,
                    "tools": tools,
                    "label": label,
                }
            )
        except Exception as error:
            exclusions.append(
                {"source_index": source_index, "source_id": row.get("id"), "reason": str(error)}
            )
    valid: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in sessions.items():
        rows.sort(key=lambda item: item["turn_id"])
        turns = [item["turn_id"] for item in rows]
        expected = list(range(1, max(turns) + 1))
        if turns != expected:
            exclusions.append(
                {"session_key": list(key), "source_ids": [item["source_id"] for item in rows], "reason": "duplicate or non-contiguous turns"}
            )
        else:
            valid[key] = rows
    return valid, exclusions


def render_sessions(
    sessions: dict[tuple[str, str], list[dict[str, Any]]],
    tokenizer: Tokenizer,
    context_tokens: int,
    max_new_tokens: int,
) -> tuple[dict[tuple[str, str], dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    output: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    exclusions: list[dict[str, Any]] = []
    for key, rows in sessions.items():
        try:
            fixed_tools = stable_fixed_tools(rows)
            variants: dict[str, list[dict[str, Any]]] = {}
            for variant in ("native_tools", "fixed_tools"):
                rendered: list[dict[str, Any]] = []
                previous: list[int] = []
                for row in rows:
                    tools = row["tools"] if variant == "native_tools" else fixed_tools
                    prompt = render_prompt(row["visible_messages"], tools)
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                    prefix = lcp(previous, token_ids) if previous else 0
                    rendered.append(
                        {
                            "dataset": "hammerbench-en-multi-turn",
                            "variant": variant,
                            "session_id": f"{key[0]}::{key[1]}",
                            "data_type": key[0],
                            "conversation_id": key[1],
                            "turn_id": row["turn_id"],
                            "source_index": row["source_index"],
                            "source_snapshot_id": row["source_id"],
                            "tools": tools,
                            "visible_messages": row["visible_messages"],
                            "reference_function_call": row["label"],
                            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                            "prompt": prompt,
                            "input_token_ids": token_ids,
                            "input_tokens": len(token_ids),
                            "previous_turn_lcp_tokens": prefix,
                            "required_new_tokens": len(token_ids) - prefix,
                        }
                    )
                    previous = token_ids
                variants[variant] = rendered
            longest = max(
                item["input_tokens"]
                for variant_rows in variants.values()
                for item in variant_rows
            )
            if longest + max_new_tokens > context_tokens:
                raise ValueError(
                    f"context overflow: longest input {longest} + output {max_new_tokens} > {context_tokens}"
                )
            output[key] = variants
        except Exception as error:
            exclusions.append(
                {"session_key": list(key), "source_ids": [item["source_id"] for item in rows], "reason": str(error)}
            )
    return output, exclusions


def select_stratified(
    sessions: dict[tuple[str, str], dict[str, list[dict[str, Any]]]], count: int, seed: int
) -> list[tuple[str, str]]:
    if len(sessions) < count:
        raise ValueError(f"only {len(sessions)} eligible Sessions; requested {count}")
    lengths = sorted(
        max(item["input_tokens"] for item in variants["fixed_tools"])
        for variants in sessions.values()
    )
    cuts = [lengths[len(lengths) * index // 4] for index in (1, 2, 3)]
    strata: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for key, variants in sessions.items():
        turns = len(variants["fixed_tools"])
        turn_bin = min(turns, 5)
        final_length = variants["fixed_tools"][-1]["input_tokens"]
        length_bin = sum(final_length > cut for cut in cuts)
        strata[(turn_bin, length_bin)].append(key)
    rng = random.Random(seed)
    for values in strata.values():
        values.sort()
        rng.shuffle(values)
    selected: list[tuple[str, str]] = []
    ordered_strata = sorted(strata)
    while len(selected) < count:
        progressed = False
        for stratum in ordered_strata:
            if strata[stratum] and len(selected) < count:
                selected.append(strata[stratum].pop())
                progressed = True
        if not progressed:
            break
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-count", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = Path(cfg["hammerbench"]["dataset"]).resolve()
    repository = Path(cfg["hammerbench"]["repository"]).resolve()
    tokenizer_path = Path(cfg["hammerbench"]["tokenizer"]).resolve()
    count = args.session_count or int(cfg["hammerbench"]["session_count"])
    seed = int(cfg["experiment"]["seed"])
    context_tokens = int(cfg["experiment"]["context_tokens"])
    max_new_tokens = int(cfg["experiment"]["max_new_tokens"])
    minimum_turns = int(cfg["hammerbench"].get("minimum_turns", 2))

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    parsed, parse_exclusions = load_sessions(dataset)
    for key in list(parsed):
        if len(parsed[key]) < minimum_turns:
            rows = parsed.pop(key)
            parse_exclusions.append(
                {
                    "session_key": list(key),
                    "source_ids": [item["source_id"] for item in rows],
                    "reason": f"fewer than {minimum_turns} turns",
                }
            )
    rendered, render_exclusions = render_sessions(
        parsed, tokenizer, context_tokens, max_new_tokens
    )
    selected = select_stratified(rendered, count, seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("native_tools", "fixed_tools"):
        rows = [item for key in selected for item in rendered[key][variant]]
        write_jsonl(args.output_dir / f"{variant}.jsonl", rows)
    write_jsonl(args.output_dir / "exclusions.jsonl", parse_exclusions + render_exclusions)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_local_mtime_utc": datetime.fromtimestamp(
            dataset.stat().st_mtime, timezone.utc
        ).isoformat(),
        "download_time": None,
        "download_time_note": "not recoverable from the existing local checkout",
        "repository": str(repository),
        "repository_commit": git_commit(repository),
        "license_file": str(repository / "LICENSE"),
        "license_sha256": sha256(repository / "LICENSE"),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "seed": seed,
        "context_tokens": context_tokens,
        "max_new_tokens": max_new_tokens,
        "minimum_turns": minimum_turns,
        "raw_snapshots": sum(len(rows) for rows in parsed.values()),
        "parsed_complete_sessions": len(parsed),
        "eligible_sessions_both_variants": len(rendered),
        "selected_session_count": len(selected),
        "selected_sessions": [f"{key[0]}::{key[1]}" for key in selected],
        "selected_turn_histogram": dict(
            sorted(Counter(len(rendered[key]["fixed_tools"]) for key in selected).items())
        ),
        "exclusion_count": len(parse_exclusions) + len(render_exclusions),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
