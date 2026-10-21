#!/usr/bin/env python3
"""Run HammerBench with the Android Qwen2.5 mllm backend."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
QNN_QWEN = CODE_ROOT / "QNN" / "qwen"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(QNN_QWEN))

import run_hammerbench as hammer  # noqa: E402
from runtime import MllmConfig, MllmPhoneRunner, is_transport_error  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--split", choices=("single-turn", "multi-turn"), default="single-turn"
    )
    parser.add_argument("--types", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get(
                "HAMMERBENCH_DATASET_ROOT", "/home/lyyyy/HammerBench/full_dataset/data"
            )
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--deadlines-ms", default="5000,10000,20000,30000")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    try:
        deadlines = sorted({int(value) for value in args.deadlines_ms.split(",")})
    except ValueError:
        parser.error("--deadlines-ms must contain comma-separated integers")

    dataset_root = args.dataset_root.resolve()
    nested = dataset_root / args.language / f"{args.split}.json"
    flat = dataset_root / f"{args.split}.json"
    dataset_path = nested if nested.is_file() else flat
    if not dataset_path.is_file():
        parser.error(f"HammerBench dataset missing; checked {nested} and {flat}")
    all_samples = hammer.load_samples(dataset_path, args.language, args.split)
    selected_types = sorted(set(args.types or []))
    if selected_types:
        available = {sample["hammer_type"] for sample in all_samples}
        unknown = sorted(set(selected_types) - available)
        if unknown:
            parser.error("unknown HammerBench types: " + ", ".join(unknown))
        all_samples = [
            sample for sample in all_samples if sample["hammer_type"] in selected_types
        ]
    full_selected_count = len(all_samples)
    samples = all_samples[: args.limit] if args.limit is not None else all_samples
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "split": args.split,
                    "types": selected_types or "all",
                    "samples": len(samples),
                    "output_budget": "none_model_context_limit_only",
                },
                indent=2,
            )
        )
        return 0

    config = MllmConfig.from_env()
    runner = MllmPhoneRunner(config)
    device_identity = runner.device_identity()
    device_label = os.environ.get("DEVICE_LABEL", "redmi")
    result_model = os.environ.get(
        "RESULT_MODEL_DIR_NAME", "qwen2.5-3B-mllm-v2-native-lpbq-w4a16-kv-w8a8"
    )
    task_dir = "hammerbench-single" if args.split == "single-turn" else "hammerbench-multi"
    output_parent = (
        args.output_root.resolve()
        if args.output_root
        else CODE_ROOT.parent
        / "results"
        / "mllm"
        / device_label
        / task_dir
        / result_model
    )
    run_id = args.run_id or "default"
    root = output_parent / run_id
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"cannot resume missing run: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            manifest_path,
            {
                "run_id": run_id,
                "timestamp": datetime.now().astimezone().isoformat(),
                "backend": "mllm QNN AOT HTP/NPU",
                "device_label": device_label,
                "adb_transport_serial": config.serial,
                "device_identity": device_identity,
                "phone_dir": config.phone_dir,
                "model_result_name": "qwen2.5-3b",
                "quantization_result_name": "LPBQ-W4A16-KV-W8A8-G32",
                "max_seq_len": config.max_seq_len,
                "ar_len": config.ar_len,
                "htp_performance": config.htp_performance,
                "platform_library_path": config.platform_library_path,
                "runner_file": config.runner_file,
                "context_file": config.context_file,
                "remote_tokenizer_file": config.remote_tokenizer_file,
                "model_config_file": config.model_config_file,
                "require_htp_verification": config.require_htp,
                "dataset": str(dataset_path),
                "language": args.language,
                "split": args.split,
                "types": selected_types,
                "dataset_selected_samples": full_selected_count,
                "requested_samples": len(samples),
                "requested_output_budget_tokens": "none",
                "prompt_template_version": hammer.PROMPT_TEMPLATE_VERSION,
                "prompt_transport": "caller_rendered_chatml_via_no_template_stdin",
                "scoring_version": hammer.SCORING_VERSION,
                "deadlines_ms": deadlines,
                "timing_policy": "PACE-QNN-compatible backprojected mllm_perf",
            },
        )

    journal = root / "rows.jsonl"
    rows_by_id = hammer.load_journal(journal) if args.resume else {}
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    remote_namespace = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{device_label}_{task_dir}_{run_id}")
    processed = 0
    exit_code = 0
    try:
        for position, sample in enumerate(samples, 1):
            sample_id = sample["sample_id"]
            if (
                args.resume
                and sample_id in rows_by_id
                and str(rows_by_id[sample_id].get("runner_returncode")) == "0"
            ):
                continue
            sample_dir = root / "hammerbench" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "visible_input.txt").write_text(
                sample["visible_input"], encoding="utf-8"
            )
            atomic_json(
                sample_dir / "label.json",
                {"name": sample["label_name"], "arguments": sample["label_parameters"]},
            )
            try:
                formatted_prompt = hammer.build_prompt("HAMMERBENCH", sample, {})
                runtime = runner.run(
                    task="HAMMERBENCH",
                    sample_id=sample_id,
                    formatted_prompt=formatted_prompt,
                    output_budget=None,
                    local_dir=sample_dir,
                    remote_namespace=remote_namespace,
                )
                completion = str(runtime.pop("completion"))
                row = {
                    "task": "HAMMERBENCH",
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    **runtime,
                    **hammer.score_output("HAMMERBENCH", sample, completion, {}),
                }
            except Exception as exc:
                row = {
                    "task": "HAMMERBENCH",
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "source_id": sample["source_id"],
                    "requested_budget_tokens": "none",
                    "runner_returncode": -1,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    **hammer.score_output("HAMMERBENCH", sample, "", {}),
                }
                if is_transport_error(exc):
                    hammer.append_journal(journal, row)
                    rows_by_id[sample_id] = row
                    print(
                        f"[{position}/{len(samples)}] {sample['hammer_type']} {sample_id} "
                        "ADB transport disconnected; stopping run",
                        flush=True,
                    )
                    raise RuntimeError(
                        f"ADB transport disconnected while running {sample_id}; "
                        "result checkpoint saved, reconnect and use --resume"
                    ) from exc
                if not args.continue_on_error:
                    hammer.append_journal(journal, row)
                    rows_by_id[sample_id] = row
                    raise
            hammer.append_journal(journal, row)
            rows_by_id[sample_id] = row
            processed += 1
            print(
                f"[{position}/{len(samples)}] {sample['hammer_type']} {sample_id} "
                f"rc={row.get('runner_returncode')} gen={row.get('generated_tokens')} "
                f"ttft={row.get('ttft_ms')}ms decode={row.get('decode_token_per_sec')} "
                f"correct={row.get('score_correct')}",
                flush=True,
            )
            if row.get("device_connection_status") == "offline" or is_transport_error(
                row.get("error")
            ):
                raise RuntimeError(
                    f"ADB transport disconnected while running {sample_id}; "
                    "result checkpoint saved, reconnect and use --resume"
                )
            if processed >= args.checkpoint_every:
                ordered = hammer.ordered_rows(rows_by_id, samples)
                hammer.write_outputs(root, ordered, samples_by_id, deadlines, full_selected_count)
                processed = 0
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        ordered = hammer.ordered_rows(rows_by_id, samples)
        if ordered:
            hammer.write_outputs(root, ordered, samples_by_id, deadlines, full_selected_count)
    print(f"Results: {root}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
