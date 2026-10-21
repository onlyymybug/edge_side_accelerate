#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from appendix_common import (
    CODE_ROOT,
    DEFAULT_CONFIG,
    PLAN_DIR,
    append_csv,
    exact_length_chatml,
    load_module,
    read_config,
    read_csv,
    read_jsonl,
)

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def dataset_map(cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    from math_benchmarks import load_benchmark_samples
    result = {}
    for task, path in cfg["datasets"].items():
        result[task] = {row["sample_id"]: row for row in load_benchmark_samples(Path(path), task)}
    return result


def select_plan(
    experiment: str,
    runtime: str | None,
    limit: int | None,
    batch_threads: int | None = None,
    decode_threads: int | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(PLAN_DIR / f"{experiment}.jsonl")
    if runtime:
        rows = [row for row in rows if row.get("runtime") == runtime]
    if batch_threads is not None:
        rows = [row for row in rows if row.get("batch_threads") == batch_threads]
    if decode_threads is not None:
        rows = [row for row in rows if row.get("decode_threads") == decode_threads]
    return rows[:limit] if limit else rows


def make_run_root(cfg: dict[str, Any], experiment: str, run_id: str | None) -> Path:
    stamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg["output_root"]) / experiment / stamp


def actual_prefill_calls(row: dict[str, Any], regex: str | None) -> str | int:
    if not regex:
        return ""
    log_path = row.get("log_path") or row.get("output_path")
    if not log_path or not Path(str(log_path)).is_file():
        return ""
    matches = re.findall(regex, Path(str(log_path)).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return ""
    value = matches[-1]
    return int(value[0] if isinstance(value, tuple) else value)


def missing_measurements(experiment: str, row: dict[str, Any]) -> list[str]:
    required = ["prompt_tokens", "energy_j"]
    if experiment == "graph_boundary":
        if row.get("ttft_ms", "") == "" and row.get("prefill_latency_ms", "") == "":
            required.append("ttft_ms/prefill_latency_ms")
    else:
        required.extend(["generated_tokens", "e2e_latency_ms"])
    return [key for key in required if key == "ttft_ms/prefill_latency_ms" or row.get(key, "") == ""]


class TmacAdapter:
    def __init__(self) -> None:
        self.module = load_module("appendix_c_tmac_runtime", CODE_ROOT / "T-MAC" / "runtime.py")
        self.base = self.module.TmacConfig.from_env()
        self.runners: dict[tuple[int, int], Any] = {}

    def runner(self, tb: int, threads: int):
        key = (tb, threads)
        if key not in self.runners:
            self.runners[key] = self.module.TmacPhoneRunner(replace(self.base, batch_threads=tb, threads=threads))
        return self.runners[key]

    def run(self, plan: dict[str, Any], sample: dict[str, Any], local_dir: Path, namespace: str) -> dict[str, Any]:
        from math_benchmarks import build_benchmark_prompt, score_benchmark_answer
        runner = self.runner(int(plan["batch_threads"]), int(plan["decode_threads"]))
        prompt = self.module.qwen_chatml(build_benchmark_prompt(plan["task"], sample["question"]))
        row = runner.run(task=plan["task"], sample_id=plan["request_id"], formatted_prompt=prompt,
                         output_budget=plan.get("output_budget"), local_dir=local_dir, remote_namespace=namespace)
        completion = Path(row["completion_path"]).read_text(encoding="utf-8", errors="replace")
        row.update(score_benchmark_answer(plan["task"], completion, sample["gold_answer"]))
        return row


class LlmNpuGraphAdapter:
    def __init__(self, mode: str) -> None:
        self.module = load_module("appendix_c_llmnpu_runtime", CODE_ROOT / "llm.npu" / "runtime.py")
        base = self.module.LlmNpuConfig.from_env()
        self.runner = self.module.LlmNpuPhoneRunner(replace(base, htp_performance=mode))

    def run(self, plan: dict[str, Any], local_dir: Path, namespace: str) -> dict[str, Any]:
        prompt = exact_length_chatml(self.runner.tokenizer, int(plan["prefix_tokens"]), self.module.qwen_chatml)
        measured = len(self.runner.tokenizer.encode(prompt, add_special_tokens=False).ids)
        if measured != plan["prefix_tokens"]:
            raise AssertionError((measured, plan["prefix_tokens"]))
        return self.runner.run(task="APPENDIX_C_GRAPH", sample_id=plan["request_id"], formatted_prompt=prompt,
                               output_budget=int(plan["output_budget"]), local_dir=local_dir, remote_namespace=namespace)


class QnnGraphAdapter:
    def __init__(self, mode: str) -> None:
        mode_number = {"burst": "1", "balanced": "8", "low_power_saver": "6"}[mode]
        os.environ["HTP_PERFORMANCE_MODE"] = mode_number
        self.module = load_module("appendix_c_qnn_runner", CODE_ROOT / "QNN" / "qwen" / "run_qnn_math_benchmarks.py")
        from tokenizers import Tokenizer
        tokenizer_path = self.module.MODEL_ROOT / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        state = self.module.adb("get-state", timeout=15)
        if state.returncode != 0 or state.stdout.strip() != "device":
            raise RuntimeError(f"Android device unavailable: {state.stdout.strip()}")

    def run(self, plan: dict[str, Any], local_dir: Path, namespace: str) -> dict[str, Any]:
        prompt = exact_length_chatml(self.tokenizer, int(plan["prefix_tokens"]), lambda value: value,
                                     add_special_tokens=True)
        measured = len(self.tokenizer.encode(prompt).ids)
        if measured != plan["prefix_tokens"]:
            raise AssertionError((measured, plan["prefix_tokens"]))
        sample = {"sample_id": plan["request_id"], "question": prompt, "gold_answer": ""}
        return self.module.run_sample(
            task="APPENDIX_C_GRAPH", sample=sample, cfg={}, budget=int(plan["output_budget"]),
            tokenizer=self.tokenizer, remote_root=f"{self.module.PHONE_DIR}/outputs/pace/{namespace}",
            local_root=local_dir.parent, telemetry_interval=0.5,
            prompt_builder=lambda task, item, cfg: item["question"], score_builder=lambda *args: {},
        )


def execute(args: argparse.Namespace) -> None:
    cfg = read_config(args.config)
    plans = select_plan(
        args.experiment,
        args.runtime,
        args.limit,
        args.batch_threads,
        args.decode_threads,
    )
    if not plans:
        raise SystemExit("no requests selected")
    if args.dry_run:
        print(json.dumps({"experiment": args.experiment, "runtime": args.runtime,
                          "requests": len(plans), "first": plans[0], "last": plans[-1]}, ensure_ascii=False, indent=2))
        return
    if not os.environ.get("ANDROID_SERIAL"):
        raise SystemExit("set ANDROID_SERIAL before a phone run")
    root = make_run_root(cfg, args.experiment, args.run_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"created_at": datetime.now().isoformat(),
        "host": socket.gethostname(), "experiment": args.experiment, "runtime": args.runtime,
        "config": cfg, "requests": len(plans)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = root / "summary.csv"
    completed = {r["request_id"] for r in read_csv(summary) if r.get("status") == "ok"} if summary.exists() else set()
    samples = dataset_map(cfg) if args.experiment != "graph_boundary" else {}
    if args.experiment == "graph_boundary":
        if args.runtime == "qnn":
            adapter = QnnGraphAdapter(plans[0]["mode"])
        elif args.runtime == "llm.npu":
            adapter = LlmNpuGraphAdapter(plans[0]["mode"])
        else:
            raise SystemExit("graph_boundary requires --runtime qnn or --runtime llm.npu")
    else:
        adapter = TmacAdapter()
    if not args.skip_warmup:
        warmup_root = root / "warmup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.experiment == "graph_boundary":
            print("warmup: graph runtime", flush=True)
            adapter.run(plans[0], warmup_root / plans[0]["request_id"], root.name + "_warmup")
        else:
            settings = {(int(row["batch_threads"]), int(row["decode_threads"])) for row in plans}
            for tb, threads in sorted(settings):
                warm_plan = next(row for row in plans if int(row["batch_threads"]) == tb and int(row["decode_threads"]) == threads)
                print(f"warmup: -tb {tb} -t {threads}", flush=True)
                adapter.run(warm_plan, samples[warm_plan["task"]][warm_plan["sample_id"]],
                            warmup_root / warm_plan["request_id"], root.name + "_warmup")
    for index, plan in enumerate(plans, 1):
        if plan["request_id"] in completed:
            print(f"[{index}/{len(plans)}] skip {plan['request_id']}")
            continue
        print(f"[{index}/{len(plans)}] run {plan['request_id']}", flush=True)
        local_dir = root / "artifacts" / plan["request_id"]
        started = datetime.now().isoformat()
        result: dict[str, Any] = {}
        try:
            if args.experiment == "graph_boundary":
                result = adapter.run(plan, local_dir, root.name)
            else:
                result = adapter.run(plan, samples[plan["task"]][plan["sample_id"]], local_dir, root.name)
            missing = missing_measurements(args.experiment, result)
            if missing:
                raise RuntimeError(
                    "invalid physical measurement; missing " + ", ".join(missing)
                    + ". Check telemetry.csv and wireless ADB, then resume with the same --run-id"
                )
            status, error = "ok", ""
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        row = {**plan, **result, "status": status, "error": error, "started_at": started,
               "finished_at": datetime.now().isoformat()}
        if args.experiment == "graph_boundary":
            row["actual_prefill_calls"] = actual_prefill_calls(row, args.prefill_call_regex)
        append_csv(summary, row)
        if status == "error" and not args.continue_on_error:
            raise SystemExit(f"stopped after error: {error}; rerun with same --run-id to resume")
        if args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
    print(f"complete: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Appendix C physical controls on the phone")
    parser.add_argument("experiment", choices=["graph_boundary", "tmac_threads", "budget_termination"])
    parser.add_argument("--runtime", choices=["qnn", "llm.npu"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="stable directory name; reuse to resume")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-threads", type=int, help="select plans with this T-MAC -tb value")
    parser.add_argument("--decode-threads", type=int, help="select plans with this T-MAC -t value")
    parser.add_argument("--cooldown-seconds", type=float, default=0)
    parser.add_argument("--prefill-call-regex", help="native log regex with one capture group for actual call count")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true", help="skip unmeasured runtime/config warmups")
    parser.add_argument("--dry-run", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
