#!/usr/bin/env python3
"""Build BFCL V4 fixed-reference-history traces for systems replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

from build_trace import lcp, render_prompt, sha256, write_jsonl


VARIANT = "bfcl_v4_full_tools_reference_history"
PROMPT_TEMPLATE_VERSION = "bfcl_v4_full_tools_hammer_chatml_system_replay_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    return sha256(path)


def select_stratified(
    sessions: dict[str, list[dict[str, Any]]], count: int, seed: int
) -> list[str]:
    """Round-robin sample from turn-count and final-prompt-length strata."""
    if len(sessions) < count:
        raise ValueError(f"only {len(sessions)} eligible Sessions; requested {count}")
    lengths = sorted(rows[-1]["input_tokens"] for rows in sessions.values())
    cuts = [lengths[len(lengths) * i // 4] for i in (1, 2, 3)]
    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for session_id, rows in sessions.items():
        final_length = rows[-1]["input_tokens"]
        stratum = (min(len(rows), 5), sum(final_length > cut for cut in cuts))
        strata[stratum].append(session_id)
    rng = random.Random(seed)
    for ids in strata.values():
        ids.sort()
        rng.shuffle(ids)
    selected: list[str] = []
    while len(selected) < count:
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def load_tools(
    row: dict[str, Any], docs_dir: Path, mapping: dict[str, str]
) -> tuple[list[dict[str, Any]], list[Path]]:
    tools: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for class_name in row["involved_classes"]:
        source = docs_dir / mapping[class_name]
        source_paths.append(source)
        tools.extend(read_jsonl(source))
    excluded = set(row.get("excluded_function", []))
    return [tool for tool in tools if tool["name"] not in excluded], source_paths


def render_session(
    row: dict[str, Any],
    ground_truth: list[list[str]],
    index: int,
    split: str,
    tokenizer: Tokenizer,
    tools: list[dict[str, Any]],
    context_tokens: int,
    max_new_tokens: int,
    execute_calls: Any,
) -> list[dict[str, Any]]:
    if len(row["question"]) != len(ground_truth) or len(ground_truth) < 2:
        raise ValueError("ground truth and question turns do not align or are not multi-turn")
    history: list[dict[str, str]] = []
    rendered: list[dict[str, Any]] = []
    previous_ids: list[int] = []
    for turn_id, (query, calls) in enumerate(zip(row["question"], ground_truth), 1):
        if not isinstance(query, list) or not query:
            raise ValueError(f"turn {turn_id} has no messages")
        if not all(isinstance(message, dict) and message.get("role") in {"user", "system"} for message in query):
            raise ValueError(f"turn {turn_id} has unexpected message roles")
        visible = history + query
        prompt = render_prompt(visible, tools)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(token_ids) + max_new_tokens > context_tokens:
            raise ValueError(
                f"context overflow: turn {turn_id} input {len(token_ids)} + "
                f"output {max_new_tokens} > {context_tokens}"
            )
        prefix = lcp(previous_ids, token_ids) if previous_ids else 0
        rendered.append(
            {
                "dataset": f"bfcl-v4-multi-turn-{split}",
                "variant": VARIANT,
                "session_id": row["id"],
                "turn_id": turn_id,
                "source_index": index,
                "source_snapshot_id": f"{row['id']}::turn_{turn_id}",
                "tools": tools,
                "visible_messages": visible,
                "reference_function_call": calls,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "prompt": prompt,
                "input_token_ids": token_ids,
                "input_tokens": len(token_ids),
                "previous_turn_lcp_tokens": prefix,
                "required_new_tokens": len(token_ids) - prefix,
            }
        )
        results, _ = execute_calls(
            calls,
            row["initial_config"],
            row["involved_classes"],
            f"bfcl_trace_{split}_{index}",
            row["id"],
            long_context=split == "long_context",
            is_evaL_run=False,
        )
        if len(results) != len(calls):
            raise ValueError(f"turn {turn_id} call/result count mismatch")
        history.extend(
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in query
        )
        for call, result in zip(calls, results):
            history.append({"role": "assistant", "content": str(call)})
            history.append({"role": "tool", "content": str(result)})
        previous_ids = token_ids
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("base", "long_context"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-count", type=int, default=50)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository = Path(config["bfcl"]["repository"]).resolve()
    sys.path.insert(0, str(repository))
    from bfcl_eval.constants.executable_backend_config import (  # noqa: E402
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
        execute_multi_turn_func_call,
    )

    dataset = Path(config["bfcl"][args.split]).resolve()
    answer = dataset.parent / "possible_answer" / dataset.name
    docs_dir = dataset.parent / "multi_turn_func_doc"
    tokenizer_path = Path(config["hammerbench"]["tokenizer"]).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    context_tokens = int(config["experiment"]["context_tokens"])
    max_new_tokens = int(config["experiment"]["max_new_tokens"])
    source_rows = read_jsonl(dataset)
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(answer)}
    eligible: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    docs_used: set[Path] = set()
    for index, row in enumerate(source_rows):
        try:
            tools, paths = load_tools(row, docs_dir, MULTI_TURN_FUNC_DOC_FILE_MAPPING)
            docs_used.update(paths)
            if not tools:
                raise ValueError("no available tool definitions")
            eligible[row["id"]] = render_session(
                row,
                answers[row["id"]],
                index,
                args.split,
                tokenizer,
                tools,
                context_tokens,
                max_new_tokens,
                execute_multi_turn_func_call,
            )
        except Exception as error:
            exclusions.append({"source_index": index, "source_id": row.get("id"), "reason": str(error)})
    selected = select_stratified(eligible, args.session_count, int(config["experiment"]["seed"]))
    rows = [request for session_id in selected for request in eligible[session_id]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "reference_history.jsonl"
    write_jsonl(trace_path, rows)
    write_jsonl(args.output_dir / "exclusions.jsonl", exclusions)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "ground_truth": str(answer),
        "ground_truth_sha256": file_sha256(answer),
        "tool_doc_sha256": {str(path): file_sha256(path) for path in sorted(docs_used)},
        "repository": str(repository),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "variant": VARIANT,
        "trace_files": {VARIANT: trace_path.name},
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "seed": int(config["experiment"]["seed"]),
        "context_tokens": context_tokens,
        "max_new_tokens": max_new_tokens,
        "raw_sessions": len(source_rows),
        "eligible_sessions": len(eligible),
        "selected_sessions": selected,
        "selected_session_count": len(selected),
        "selected_turn_histogram": dict(sorted(Counter(len(eligible[session_id]) for session_id in selected).items())),
        "selected_requests": len(rows),
        "selected_input_tokens_min": min(row["input_tokens"] for row in rows),
        "selected_input_tokens_median": statistics.median(row["input_tokens"] for row in rows),
        "selected_input_tokens_max": max(row["input_tokens"] for row in rows),
        "selected_weighted_lcp_rate": sum(row["previous_turn_lcp_tokens"] for row in rows) / sum(row["input_tokens"] for row in rows),
        "exclusion_count": len(exclusions),
        "trace_sha256": file_sha256(trace_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
