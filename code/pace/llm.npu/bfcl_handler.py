"""BFCL model handler backed by the Android llm.npu Qwen runner."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_qnn_phone import (
    Qwen25QnnPhoneHandler,
)

from runtime import LlmNpuConfig, LlmNpuPhoneRunner


class Qwen25LlmNpuPhoneHandler(Qwen25QnnPhoneHandler):
    """Reuse BFCL's official Qwen FC prompting/evaluation with llm.npu inference."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._run_lock = threading.Lock()

    def _ensure_phone_ready(self) -> None:
        if self._phone_ready:
            return
        self.llm_npu_runner = LlmNpuPhoneRunner(LlmNpuConfig.from_env())
        config = self.llm_npu_runner.config
        self.android_serial = config.serial
        self.phone_dir = config.phone_dir
        self.max_context_length = config.max_seq_len
        self.tokenizer = self.llm_npu_runner.tokenizer
        self.model_path_or_id = self.model_name_huggingface
        project_root = Path(
            os.environ.get(
                "BFCL_PROJECT_ROOT", "/home/lyyyy/pace/results/llm.npu/redmi/bfcl"
            )
        ).resolve()
        self.artifact_root = Path(
            os.environ.get(
                "LLM_NPU_BFCL_ARTIFACT_DIR",
                str(project_root / "llm_npu_phone_artifacts" / self.registry_dir_name),
            )
        ).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.artifact_root / "summary_metrics.csv"
        print(
            "llm.npu phone backend ready: "
            f"serial={config.serial} phone_dir={config.phone_dir} "
            f"max_context={config.max_seq_len} ar_len={config.ar_len} "
            f"htp_performance={config.htp_performance}"
        )
        self._phone_ready = True

    def _write_metrics_row(self, row: dict[str, Any]) -> None:
        rows: list[dict[str, Any]] = []
        if self.metrics_path.is_file():
            with self.metrics_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        row_key = (str(row["test_id"]), str(row["step"]))
        rows = [
            existing
            for existing in rows
            if (str(existing.get("test_id")), str(existing.get("step"))) != row_key
        ]
        rows.append(row)
        columns: list[str] = []
        for existing in rows:
            for key in existing:
                if key not in columns:
                    columns.append(key)
        temporary = self.metrics_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.metrics_path)

    def _query_prompting(self, inference_data: dict[str, Any]):
        self._ensure_phone_ready()
        messages = inference_data["message"]
        functions = inference_data["function"]
        test_id = str(inference_data.get("test_id", "unknown"))
        formatted_prompt = self._format_prompt(messages, functions)
        inference_data["inference_input_log"] = {
            "formatted_prompt": formatted_prompt,
            "transport": "adb_mllm_qnn_aot_htp",
            "prompt_transport": "caller_rendered_chatml_via_no_template_stdin",
            "device_serial": self.android_serial,
        }
        prompt_tokens = self._count_prompt_tokens(formatted_prompt)
        if prompt_tokens + 2 > self.max_context_length:
            raise ValueError(
                f"{test_id}: official BFCL Qwen prompt has {prompt_tokens} tokens; "
                f"it does not fit max_context_len={self.max_context_length}"
            )

        with self._run_lock:
            step = self._next_step(test_id)
            local_dir = self.artifact_root / test_id / f"step_{step:02d}"
            started = time.monotonic()
            runtime = self.llm_npu_runner.run(
                task="BFCL",
                sample_id=f"{test_id}_step_{step:02d}",
                formatted_prompt=formatted_prompt,
                output_budget=None,
                local_dir=local_dir,
                remote_namespace=f"bfcl_{self.registry_dir_name}",
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            completion = str(runtime.pop("completion"))
            if int(runtime.get("runner_returncode", -1)) != 0:
                raise RuntimeError(
                    f"{test_id}: llm.npu mllm runner returned "
                    f"{runtime.get('runner_returncode')}; see {local_dir / 'runner.log'}"
                )
            output_tokens = int(runtime.get("generated_tokens", 0))
            metrics = {
                "test_id": test_id,
                "step": step,
                "device_serial": self.android_serial,
                "backend": "mllm QNN AOT HTP/NPU",
                "bfcl_call_latency_ms": latency_ms,
                **{
                    key: value
                    for key, value in runtime.items()
                    if key
                    not in {
                        "returncode",
                        "runner_reported_returncode",
                        "adb_pull_returncode",
                        "device_connection_status",
                        "error",
                    }
                },
            }
            (local_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_metrics_row(metrics)
            response = SimpleNamespace(
                choices=[SimpleNamespace(text=completion)],
                usage=SimpleNamespace(
                    prompt_tokens=int(runtime.get("prompt_tokens", prompt_tokens)),
                    completion_tokens=output_tokens,
                ),
            )
            return response, latency_ms / 1000.0
