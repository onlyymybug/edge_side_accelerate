#!/usr/bin/env python3
"""Run HammerBench snapshots with the deployed Qwen QNN phone bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from tokenizers import Tokenizer

import run_pace_datasets as qrun


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    os.environ.get(
        "HAMMERBENCH_DATASET_ROOT", "/home/lyyyy/HammerBench/full_dataset/data"
    )
).resolve()
MODEL_PROFILE = os.environ.get("RUNNER_MODEL_PROFILE", "qwen").strip().lower()
PROMPT_TEMPLATE_VERSION = (
    "hammerbench_official_appendix_d_llama3_v2"
    if MODEL_PROFILE == "bitnet"
    else "hammerbench_official_appendix_d_qwen_v2"
)
TASK_NAME = "HAMMERBENCH"
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_component(value: str) -> str:
    return SAFE_COMPONENT_RE.sub("_", value).strip("_") or "sample"


def hammer_type(row: dict[str, Any], split: str) -> str:
    source_id = str(row.get("id", "unknown"))
    return source_id.rsplit("_", 2)[0] if split == "multi-turn" else source_id


def last_function_call(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    calls = [
        (index, message["content"])
        for index, message in enumerate(row.get("messages", []))
        if message.get("role") == "function call"
        and isinstance(message.get("content"), dict)
    ]
    if not calls:
        raise ValueError("HammerBench row has no function-call label")
    return calls[-1]


def visible_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    label_index, _ = last_function_call(row)
    return [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in row.get("messages", [])[:label_index]
        if message.get("role") in ("user", "assistant")
    ]


def visible_input(row: dict[str, Any]) -> str:
    return "\n".join(
        f"{message['role']}:{message['content']}" for message in visible_messages(row)
    )


def build_prompt(_task: str, sample: dict[str, Any], _cfg: dict[str, Any]) -> str:
    # Align with HammerBench, Appendix D, "Prompts of Function Calling".  In
    # particular, snapshots must preserve user wording, represent unavailable
    # parameter values as empty strings, and reject an irrelevant tool set.
    tool_lines = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        for tool in sample["tools"]
    )
    system_content = (
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
    if MODEL_PROFILE == "bitnet":
        # The legacy BitNet runner itself wraps --prompt as a Llama-3 user
        # message. Keep this payload plain to avoid a nested chat template.
        chunks = [system_content]
        for message in sample["visible_messages"]:
            chunks.append(f"{message['role']}:\n{message['content']}")
        return "\n".join(chunks)

    chunks = [f"<|im_start|>system\n{system_content}<|im_end|>"]
    for message in sample["visible_messages"]:
        chunks.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
        )
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def json_candidates(text: str) -> list[str]:
    stripped = text.replace("<|im_end|>", "").strip()
    candidates = [stripped]
    candidates.extend(FENCE_RE.findall(stripped))
    candidates.extend(TOOL_CALL_RE.findall(stripped))
    return candidates


def first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for candidate in json_candidates(text):
        try:
            return json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        for match in re.finditer(r"[\[{]", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
                return value
            except (ValueError, json.JSONDecodeError):
                continue
    raise ValueError("no JSON tool call found")


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    value = first_json_value(text)
    if isinstance(value, list):
        if not value:
            raise ValueError("empty tool-call list")
        value = value[0]
    if isinstance(value, dict) and "function" in value and isinstance(
        value["function"], dict
    ):
        value = value["function"]
    if not isinstance(value, dict):
        raise ValueError("tool call is not a JSON object")
    name = value.get("name")
    parameters = value.get("parameters", value.get("arguments", {}))
    if not isinstance(name, str) or not name:
        raise ValueError("tool call has no name")
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    if not isinstance(parameters, dict):
        raise ValueError("tool-call parameters are not an object")
    return name, parameters


def rouge_tokens(value: Any, language: str) -> list[str]:
    text = str(value).strip(".?!。！ ").lower()
    if language == "zh":
        text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)
    return text.split()


def rouge_l_f1(reference: Any, hypothesis: Any, language: str) -> float:
    ref = rouge_tokens(reference, language)
    hyp = rouge_tokens(hypothesis, language)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for column, hyp_token in enumerate(hyp, start=1):
            if ref_token == hyp_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def official_arguments_correct(
    gold: dict[str, Any], predicted: dict[str, Any], language: str
) -> bool:
    gold = {key: str(value) for key, value in gold.items() if value != ""}
    predicted = {
        key: str(value) for key, value in predicted.items() if value != ""
    }
    if any(key not in gold for key in predicted):
        return False
    if gold == predicted:
        return True
    gold.pop("kwargs", None)
    predicted.pop("kwargs", None)
    if gold == predicted:
        return True
    scores = [
        rouge_l_f1(value, predicted[key], language) if key in predicted else 0.0
        for key, value in gold.items()
    ]
    return bool(scores) and min(scores) >= 0.7


def score_output(
    _task: str,
    sample: dict[str, Any],
    completion: str,
    _cfg: dict[str, Any],
) -> dict[str, Any]:
    gold_name = sample["label_name"]
    gold_parameters = sample["label_parameters"]
    predicted_name = ""
    predicted_parameters: dict[str, Any] = {}
    parse_error = ""
    try:
        predicted_name, predicted_parameters = parse_tool_call(completion)
        predicted_parameters = {
            key: value
            for key, value in predicted_parameters.items()
            if value not in ("", "\\")
        }
    except Exception as error:  # Keep malformed model output as an evaluated row.
        parse_error = str(error)
    function_correct = not parse_error and predicted_name == gold_name
    arguments_correct = (
        official_arguments_correct(
            gold_parameters, predicted_parameters, sample["language"]
        )
        if not parse_error
        else False
    )
    gold_keys = set(gold_parameters)
    predicted_keys = set(predicted_parameters)
    return {
        "hammer_type": sample["hammer_type"],
        "language": sample["language"],
        "split": sample["split"],
        "source_index": sample["source_index"],
        "original_id": sample["original_id"],
        "predicted_name": predicted_name,
        "gold_name": gold_name,
        "predicted_parameters_json": json.dumps(
            predicted_parameters, ensure_ascii=False, separators=(",", ":")
        ),
        "gold_parameters_json": json.dumps(
            gold_parameters, ensure_ascii=False, separators=(",", ":")
        ),
        "output_parsed": not bool(parse_error),
        "output_parse_error": parse_error,
        "rejected": "sorry" in completion.lower(),
        "function_correct": function_correct,
        "arguments_correct": arguments_correct,
        "parameter_hallucination_count": len(predicted_keys - gold_keys),
        "parameter_missing_count": len(gold_keys - predicted_keys),
        "score_correct": function_correct and arguments_correct,
    }


def load_samples(
    dataset_path: Path, language: str, split: str
) -> list[dict[str, Any]]:
    raw = qrun.load_json(dataset_path)
    if not isinstance(raw, list):
        raise ValueError(f"HammerBench dataset is not a JSON array: {dataset_path}")
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        label_index, label = last_function_call(row)
        del label_index
        kind = hammer_type(row, split)
        source_id = str(row.get("id", kind))
        sample_id = (
            safe_component(source_id)
            if split == "multi-turn"
            else f"{safe_component(kind)}_{index:05d}"
        )
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": f"hammerbench-{language}-{split}",
                "source_id": source_id,
                "source_index": index,
                "original_id": source_id,
                "hammer_type": kind,
                "language": language,
                "split": split,
                "tools": row.get("tools", row.get("multiple_tools", [])),
                "visible_messages": visible_messages(row),
                "visible_input": visible_input(row),
                "label_name": str(label.get("name", "")),
                "label_parameters": dict(
                    label.get("arguments", label.get("parameters", {}))
                ),
            }
        )
    return samples


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
            if math.isfinite(value):
                values.append(value)
        except (TypeError, ValueError):
            pass
    return values


def truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def write_evaluation_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    groups = [("overall", rows)] + [
        (kind, [row for row in rows if row.get("hammer_type") == kind])
        for kind in sorted({str(row.get("hammer_type", "")) for row in rows})
    ]
    output: list[dict[str, Any]] = []
    for group, selected in groups:
        if not selected:
            continue
        function_correct = [row for row in selected if truth(row.get("function_correct"))]
        output.append(
            {
                "group": group,
                "samples": len(selected),
                "runtime_successes": sum(
                    str(row.get("runner_returncode", "")) == "0" for row in selected
                ),
                "parsed_outputs": sum(truth(row.get("output_parsed")) for row in selected),
                "rejection_rate": mean([truth(row.get("rejected")) for row in selected]),
                "output_string_ratio": mean(
                    [not truth(row.get("output_parsed")) for row in selected]
                ),
                "func_acc": mean(
                    [truth(row.get("function_correct")) for row in selected]
                ),
                "args_acc": mean(
                    [truth(row.get("arguments_correct")) for row in selected]
                ),
                "end_to_end_acc": mean(
                    [truth(row.get("score_correct")) for row in selected]
                ),
                "average_parameter_hallucinations": mean(
                    numeric(selected, "parameter_hallucination_count")
                ),
                "average_parameter_missing": mean(
                    numeric(selected, "parameter_missing_count")
                ),
                "pn_fp": (
                    mean(
                        [
                            float(row.get("parameter_hallucination_count", 0)) > 0
                            for row in function_correct
                        ]
                    )
                    if function_correct
                    else ""
                ),
                "pn_fn": (
                    mean(
                        [
                            float(row.get("parameter_missing_count", 0)) > 0
                            for row in function_correct
                        ]
                    )
                    if function_correct
                    else ""
                ),
            }
        )
    qrun.write_csv(path, output)


def write_thermal_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    groups = [("overall", rows)] + [
        (kind, [row for row in rows if row.get("hammer_type") == kind])
        for kind in sorted({str(row.get("hammer_type", "")) for row in rows})
    ]
    output: list[dict[str, Any]] = []
    for group, selected in groups:
        if not selected:
            continue

        def average(key: str) -> float | str:
            values = numeric(selected, key)
            return mean(values) if values else ""

        max_temps = numeric(selected, "max_skin_temp_c")
        energies = numeric(selected, "energy_j")
        output.append(
            {
                "group": group,
                "samples": len(selected),
                "temperature_measured_samples": len(max_temps),
                "energy_measured_samples": len(energies),
                "mean_start_voltage_v": average("start_voltage_v"),
                "mean_end_voltage_v": average("end_voltage_v"),
                "mean_start_current_a": average("start_current_a"),
                "mean_end_current_a": average("end_current_a"),
                "mean_start_power_w": average("start_power_w"),
                "mean_end_power_w": average("end_power_w"),
                "mean_average_power_w": average("average_power_w"),
                "total_energy_j": sum(energies),
                "mean_energy_j": mean(energies) if energies else "",
                "mean_energy_per_token_j": average("energy_per_token_j"),
                "mean_edp_j_s": average("edp_j_s"),
                "mean_start_skin_temp_c": average("start_skin_temp_c"),
                "mean_end_skin_temp_c": average("end_skin_temp_c"),
                "mean_max_skin_temp_c": mean(max_temps) if max_temps else "",
                "observed_max_skin_temp_c": max(max_temps) if max_temps else "",
                "mean_skin_temp_delta_c": (
                    mean(
                        [
                            float(row["end_skin_temp_c"])
                            - float(row["start_skin_temp_c"])
                            for row in selected
                            if row.get("start_skin_temp_c") not in (None, "")
                            and row.get("end_skin_temp_c") not in (None, "")
                        ]
                    )
                    if any(
                        row.get("start_skin_temp_c") not in (None, "")
                        and row.get("end_skin_temp_c") not in (None, "")
                        for row in selected
                    )
                    else ""
                ),
                "mean_peak_rss_mb": average("peak_rss_mb"),
                "mean_peak_vmhwm_mb": average("peak_vmhwm_mb"),
            }
        )
    qrun.write_csv(path, output)


def write_inference_results(
    path: Path, rows: list[dict[str, Any]], samples_by_id: dict[str, dict[str, Any]]
) -> None:
    output = []
    for row in rows:
        sample = samples_by_id.get(str(row.get("sample_id", "")))
        if sample is None:
            continue
        completion_path = Path(str(row.get("output_path", ""))).with_name(
            "completion.txt"
        )
        prediction = (
            completion_path.read_text(encoding="utf-8", errors="replace")
            if completion_path.is_file()
            else ""
        )
        output.append(
            {
                "id": sample["sample_id"],
                "original_id": sample["original_id"],
                "input": sample["visible_input"],
                "predict": prediction,
                "label": {
                    "name": sample["label_name"],
                    "arguments": sample["label_parameters"],
                },
            }
        )
    atomic_json(path, output)


def append_journal(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_journal(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sample_id"):
                rows[str(row["sample_id"])] = row
    return rows


def ordered_rows(
    rows_by_id: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [rows_by_id[sample["sample_id"]] for sample in samples if sample["sample_id"] in rows_by_id]


def write_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    deadlines_ms: list[int],
    full_selected_count: int,
) -> None:
    qrun.write_csv(root / "summary_metrics.csv", rows)
    write_evaluation_summary(root / "evaluation_summary.csv", rows)
    write_thermal_summary(root / "thermal_summary_metrics.csv", rows)
    write_inference_results(root / "inference_results.json", rows, samples_by_id)
    qrun.write_deadline_summary(
        root / "deadline_summary.csv",
        rows,
        [TASK_NAME],
        {TASK_NAME: deadlines_ms},
        {TASK_NAME: full_selected_count},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--split", choices=("single-turn", "multi-turn"), default="single-turn"
    )
    parser.add_argument(
        "--types",
        nargs="*",
        help="optional HammerBench types, for example ST-Perfect ir-ST-Perfect",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-budget",
        type=int,
        default=None,
        help=(
            "optional generated-token cap; by default use all remaining space "
            "in the 1024-token PTE context"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--telemetry-interval", type=float, default=qrun.TELEMETRY_INTERVAL_SECONDS
    )
    parser.add_argument("--deadlines-ms", default=qrun.DEFAULT_DEADLINES_MS)
    args = parser.parse_args()
    if args.output_budget is not None and args.output_budget <= 0:
        parser.error("--output-budget must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    try:
        deadlines_ms = sorted({int(value) for value in args.deadlines_ms.split(",")})
    except ValueError:
        parser.error("--deadlines-ms must contain comma-separated integers")
    if not deadlines_ms or deadlines_ms[0] <= 0:
        parser.error("deadlines must be positive")

    dataset_root = args.dataset_root.resolve()
    language_layout_path = dataset_root / args.language / f"{args.split}.json"
    flat_layout_path = dataset_root / f"{args.split}.json"
    dataset_path = (
        language_layout_path
        if language_layout_path.is_file()
        else flat_layout_path
    )
    if not dataset_path.is_file():
        parser.error(
            "HammerBench dataset is missing; checked: "
            f"{language_layout_path}, {flat_layout_path}"
        )
    all_samples = load_samples(dataset_path, args.language, args.split)
    selected_types = sorted(set(args.types or []))
    if selected_types:
        available_types = {sample["hammer_type"] for sample in all_samples}
        unknown = sorted(set(selected_types) - available_types)
        if unknown:
            parser.error(f"unknown HammerBench types: {', '.join(unknown)}")
        all_samples = [
            sample for sample in all_samples if sample["hammer_type"] in selected_types
        ]
    full_selected_count = len(all_samples)
    samples = all_samples[: args.limit] if args.limit is not None else all_samples
    tokenizer_path = Path(
        os.environ.get(
            "HAMMERBENCH_TOKENIZER_PATH",
            str(qrun.MODEL_ROOT / "tokenizer.json"),
        )
    ).resolve()
    if not tokenizer_path.is_file():
        parser.error(f"Qwen tokenizer is missing: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    prompt_lengths = [
        len(tokenizer.encode(build_prompt(TASK_NAME, sample, {})).ids)
        for sample in samples
    ]
    context_overflows = sum(length >= qrun.MAX_SEQ_LEN for length in prompt_lengths)
    if args.dry_run:
        sorted_lengths = sorted(prompt_lengths)
        percentile = lambda fraction: sorted_lengths[
            min(len(sorted_lengths) - 1, math.ceil(fraction * len(sorted_lengths)) - 1)
        ]
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "full_selected_samples": full_selected_count,
                    "run_samples": len(samples),
                    "types": selected_types or "all",
                    "output_budget": (
                        args.output_budget
                        if args.output_budget is not None
                        else "none_model_context_limit_only"
                    ),
                    "prompt_tokens": {
                        "min": min(prompt_lengths),
                        "median": sorted_lengths[len(sorted_lengths) // 2],
                        "p95": percentile(0.95),
                        "max": max(prompt_lengths),
                    },
                    "prompts_with_no_output_capacity": context_overflows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    identity = qrun.device_identity()
    model_path = qrun.MODEL_ROOT / "hybrid_llama_qnn.pte"
    runner_path = (
        qrun.BUNDLE / "runtime" / "qnn_llama_runner"
        if MODEL_PROFILE == "bitnet"
        else qrun.QNN_RUNNER_LOCAL_PATH
    )
    backend_path = qrun.BUNDLE / "runtime" / "libqnn_executorch_backend.so"
    for required in (model_path, runner_path, backend_path):
        if not required.is_file():
            parser.error(f"required Qwen artifact is missing: {required}")
    required_remote = (
        (
            "qnn_llama_runner",
            qrun.REMOTE_MODEL_NAME,
            "tokenizer.json",
            "libc++_shared.so",
            "libqnn_executorch_backend.so",
            "libQnnHtp.so",
            "libQnnHtpV79Stub.so",
            "libQnnHtpV79Skel.so",
            "libQnnHtpPrepare.so",
            "libQnnSystem.so",
        )
        if MODEL_PROFILE == "bitnet"
        else (
            qrun.QNN_RUNNER_NAME,
            "hybrid_llama_qnn.pte",
            "tokenizer.json",
            "tokenizer_config.json",
            "libqnn_executorch_backend.so",
            "libQnnHtp.so",
            "libQnnHtpV79Stub.so",
            "libQnnHtpV79Skel.so",
            "libQnnHtpPrepare.so",
            "libQnnSystem.so",
        )
    )
    remote_check = " && ".join(
        f"test -f {qrun.shlex.quote(qrun.PHONE_DIR + '/' + name)}"
        for name in required_remote
    )
    check = qrun.adb("shell", remote_check, timeout=30)
    if check.returncode != 0:
        parser.error(
            f"{MODEL_PROFILE} phone bundle is incomplete under {qrun.PHONE_DIR}"
        )

    default_run_id = qrun.PERFORMANCE_MODE_RESULT_NAME
    run_id = args.run_id or default_run_id
    task_dir = (
        "hammerbench-single" if args.split == "single-turn" else "hammerbench-multi"
    )
    output_parent = (
        args.output_root.resolve()
        if args.output_root
        else (
            qrun.RESULTS_ROOT
            / qrun.DEVICE_LABEL
            / task_dir
            / qrun.RESULT_MODEL_DIR_NAME
        )
    )
    root = output_parent / run_id
    manifest_path = root / "manifest.json"
    if args.resume:
        if not root.is_dir() or not manifest_path.is_file():
            parser.error(f"cannot resume missing HammerBench run: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "device_label": qrun.DEVICE_LABEL,
        "adb_transport_serial": qrun.SERIAL,
        "device_identity": identity,
        "phone_dir": qrun.PHONE_DIR,
        "model_result_name": qrun.MODEL_RESULT_NAME,
        "quantization_result_name": qrun.QUANTIZATION_RESULT_NAME,
        "performance_mode_result_name": qrun.PERFORMANCE_MODE_RESULT_NAME,
        "htp_performance_mode": qrun.HTP_PERFORMANCE_MODE,
        "htp_performance_mode_name": qrun.HTP_PERFORMANCE_MODE_NAME,
        "qnn_version": "2.28" if MODEL_PROFILE == "bitnet" else "2.37",
        "local_model": str(model_path),
        "local_model_sha256": qrun.sha256(model_path),
        "local_runner_sha256": qrun.sha256(runner_path),
        "remote_runner": (
            "qnn_llama_runner"
            if MODEL_PROFILE == "bitnet"
            else qrun.QNN_RUNNER_NAME
        ),
        "local_backend_sha256": qrun.sha256(backend_path),
        "tokenizer_sha256": qrun.sha256(tokenizer_path),
        "dataset": str(dataset_path),
        "dataset_sha256": qrun.sha256(dataset_path),
        "language": args.language,
        "split": args.split,
        "types": selected_types,
        "dataset_selected_samples": full_selected_count,
        "requested_samples": len(samples),
        "limit": args.limit,
        "max_seq_len": qrun.MAX_SEQ_LEN,
        "requested_output_budget_tokens": (
            args.output_budget if args.output_budget is not None else "none"
        ),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "decoder_model_version": "llama3" if MODEL_PROFILE == "bitnet" else "qwen2_5",
        "eval_mode": 1,
        "temperature": 0,
        "qnn_op_package_paths": "",
        "telemetry": {
            "source": "dumpsys thermalservice HAL vbat/ibat/skin",
            "collection_mode": "host-side ADB polling",
            "sampling_interval_seconds": args.telemetry_interval,
            "power_formula": "signed_power_w = vbat_v * ibat_a",
            "energy_formula": "trapezoidal integral over inference window",
            "memory_source": "/proc/PID/status VmRSS and VmHWM",
            "external_power_policy": "invalidate consumption metrics while powered",
        },
        "scoring": {
            "function": "exact name match",
            "arguments": "HammerBench Rouge-L F1 >= 0.7 policy",
            "end_to_end": "function_correct and arguments_correct",
        },
        "deadlines_ms": deadlines_ms,
    }
    if args.resume:
        previous = qrun.load_json(manifest_path)
        checks = {
            "device_hardware_serial": previous.get("device_identity", {}).get(
                "hardware_serial"
            )
            == identity.get("hardware_serial"),
            "local_model_sha256": previous.get("local_model_sha256")
            == manifest["local_model_sha256"],
            "local_runner_sha256": previous.get("local_runner_sha256")
            == manifest["local_runner_sha256"],
            "dataset_sha256": previous.get("dataset_sha256")
            == manifest["dataset_sha256"],
            "language": previous.get("language") == args.language,
            "split": previous.get("split") == args.split,
            "types": previous.get("types", []) == selected_types,
            "limit": previous.get("limit") == args.limit,
            "requested_samples": previous.get("requested_samples") == len(samples),
            "requested_output_budget_tokens": previous.get(
                "requested_output_budget_tokens"
            )
            == (args.output_budget if args.output_budget is not None else "none"),
            "prompt_template_version": previous.get("prompt_template_version")
            == PROMPT_TEMPLATE_VERSION,
        }
        failed = [key for key, compatible in checks.items() if not compatible]
        if failed:
            parser.error(
                "resume configuration differs from original run: " + ", ".join(failed)
            )
        previous.update(
            {
                "resumed_at": datetime.now().astimezone().isoformat(),
                "adb_transport_serial": qrun.SERIAL,
                "telemetry": manifest["telemetry"],
            }
        )
        atomic_json(manifest_path, previous)
    else:
        atomic_json(manifest_path, manifest)

    journal_path = root / "rows.jsonl"
    rows_by_id = load_journal(journal_path)
    if not rows_by_id and (root / "summary_metrics.csv").is_file():
        rows_by_id = {
            str(row["sample_id"]): row
            for row in qrun.load_csv(root / "summary_metrics.csv")
        }
    # Result directories may be moved into the canonical hierarchy between
    # runs. Rebase path-valued columns in memory so the next checkpoint does
    # not carry obsolete absolute paths forward. Existing result files are not
    # rewritten merely by loading them.
    for sample_id, row in rows_by_id.items():
        sample_dir = root / TASK_NAME.lower() / sample_id
        row["output_path"] = str(sample_dir / "generated_text.txt")
        row["log_path"] = str(sample_dir / "runner.log")
        row["telemetry_path"] = str(sample_dir / "telemetry.csv")
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    completed = {
        sample_id
        for sample_id, row in rows_by_id.items()
        if str(row.get("runner_returncode", "")) == "0"
    }
    remote_root = f"{qrun.PHONE_DIR}/outputs/{run_id}"
    processed_since_checkpoint = 0
    exit_code = 0
    try:
        for position, sample in enumerate(samples, start=1):
            sample_id = sample["sample_id"]
            if args.resume and sample_id in completed:
                continue
            sample_dir = root / TASK_NAME.lower() / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "visible_input.txt").write_text(
                sample["visible_input"], encoding="utf-8"
            )
            atomic_json(
                sample_dir / "label.json",
                {
                    "name": sample["label_name"],
                    "arguments": sample["label_parameters"],
                },
            )
            try:
                row = qrun.run_sample(
                    task=TASK_NAME,
                    sample=sample,
                    cfg={},
                    budget=args.output_budget,
                    tokenizer=tokenizer,
                    remote_root=remote_root,
                    local_root=root,
                    telemetry_interval=args.telemetry_interval,
                    prompt_builder=build_prompt,
                    score_builder=score_output,
                    tokenizer_path_arg=("tokenizer.json" if MODEL_PROFILE == "bitnet" else "."),
                )
            except ValueError as error:
                if "leaves no output capacity" not in str(error):
                    raise
                score_fields = score_output(TASK_NAME, sample, "", {})
                row = {
                    "task": TASK_NAME,
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    "requested_budget_tokens": (
                        args.output_budget if args.output_budget is not None else "none"
                    ),
                    "runner_returncode": "",
                    "status": "context_overflow",
                    "error": str(error),
                    **score_fields,
                }
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                transport_error = qrun.adb_transport_failure(str(error))
                row = {
                    "task": TASK_NAME,
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    "requested_budget_tokens": (
                        args.output_budget if args.output_budget is not None else "none"
                    ),
                    "runner_returncode": -1,
                    "device_connection_status": "offline" if transport_error else "",
                    "status": "error",
                    "error": str(error),
                    **score_output(TASK_NAME, sample, "", {}),
                }
                if transport_error or not args.continue_on_error:
                    append_journal(journal_path, row)
                    rows_by_id[sample_id] = row
                    raise RuntimeError(str(error)) from error
            append_journal(journal_path, row)
            rows_by_id[sample_id] = row
            processed_since_checkpoint += 1
            print(
                f"[{position}/{len(samples)}] {sample['hammer_type']} {sample_id} "
                f"rc={row.get('runner_returncode')} prompt={row.get('prompt_tokens')} "
                f"gen={row.get('generated_tokens')} ttft={row.get('ttft_ms')}ms "
                f"decode={row.get('decode_token_per_sec')} "
                f"func={row.get('function_correct')} args={row.get('arguments_correct')} "
                f"correct={row.get('score_correct')}",
                flush=True,
            )
            if row.get("device_connection_status") == "offline":
                raise RuntimeError(
                    f"ADB transport failed during {sample_id}; stopping immediately"
                )
            if processed_since_checkpoint >= args.checkpoint_every:
                rows = ordered_rows(rows_by_id, samples)
                write_outputs(
                    root,
                    rows,
                    samples_by_id,
                    deadlines_ms,
                    full_selected_count,
                )
                processed_since_checkpoint = 0
    except KeyboardInterrupt:
        print("Interrupted; writing a resumable checkpoint.", file=sys.stderr, flush=True)
        exit_code = 130
    finally:
        rows = ordered_rows(rows_by_id, samples)
        if rows:
            write_outputs(
                root,
                rows,
                samples_by_id,
                deadlines_ms,
                full_selected_count,
            )
    print(f"Results: {root}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
