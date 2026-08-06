import re
import json
from pathlib import Path
from typing import Any

from .metrics import fmt
from .constants import NA


_ASSISTANT_MARKER = "<|im_start|>assistant\n"
_TERMINAL_CONTROL_TOKENS = ("<|im_end|>", "<|endoftext|>")


def strip_terminal_control_tokens(text: str) -> str:
    result = text
    while True:
        without_trailing_whitespace = result.rstrip()
        matched_token = next(
            (
                token
                for token in _TERMINAL_CONTROL_TOKENS
                if without_trailing_whitespace.endswith(token)
            ),
            None,
        )
        if matched_token is None:
            return result
        result = without_trailing_whitespace[: -len(matched_token)]


def _extract_assistant_completion(text: str, prompt: str | None = None) -> str:
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :].lstrip()
    if _ASSISTANT_MARKER in text:
        text = text.split(_ASSISTANT_MARKER, 1)[1]
    return strip_terminal_control_tokens(text)


def _regex_int(text: str, pattern: str) -> Any:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else NA

def _regex_float(text: str, pattern: str) -> Any:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else NA

def _parse_observer_json(text: str) -> dict[str, Any]:
    m = re.search(r"PyTorchObserver\s+(\{.*?\})", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    
def read_generation_text(
    token_timestamps_path: Path,
    output_text_path: Path,
    *,
    prompt: str | None = None,
) -> tuple[str, str]:
    if token_timestamps_path.exists():
        pieces: list[str] = []
        with token_timestamps_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not bool(row.get("prompt_echo", False)):
                    pieces.append(str(row.get("piece", "")))
        if pieces:
            return _extract_assistant_completion(
                "".join(pieces), prompt
            ), "token_timestamps"

    if output_text_path.exists():
        text = output_text_path.read_text(encoding="utf-8", errors="replace")
        return _extract_assistant_completion(text, prompt), "outputs_txt"

    return "", "missing_output"


def parse_runner_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    observer = _parse_observer_json(text)
    metrics: dict[str, Any] = {}

    if observer:
        prompt_tokens = observer.get("prompt_tokens")
        generated_tokens = observer.get("generated_tokens")
        inference_start = observer.get("inference_start_ms")
        inference_end = observer.get("inference_end_ms")
        first_token = observer.get("first_token_ms")
        prompt_eval_end = observer.get("prompt_eval_end_ms")
        decode_tps = observer.get("decode_token_per_sec")

        metrics["prompt_tokens"] = prompt_tokens if prompt_tokens is not None else NA
        metrics["actual_output_tokens"] = generated_tokens if generated_tokens is not None else NA
        metrics["ttft_ms"] = (
            first_token - inference_start
            if inference_start is not None and first_token is not None
            else NA
        )
        metrics["e2e_latency_ms"] = (
            inference_end - inference_start
            if inference_start is not None and inference_end is not None
            else NA
        )
        metrics["decode_latency_ms"] = (
            inference_end - prompt_eval_end
            if prompt_eval_end is not None and inference_end is not None
            else NA
        )
        metrics["visible_tokens_per_second"] = decode_tps if decode_tps is not None else NA
        metrics["metrics_source"] = "PyTorchObserver"
    else:
        metrics["prompt_tokens"] = _regex_int(text, r"Prompt Tokens:\s*(\d+)")
        metrics["actual_output_tokens"] = _regex_int(text, r"Generated Tokens:\s*(\d+)")

        total_s = _regex_float(text, r"Total inference time:\s*([0-9.]+)\s*s")
        prompt_eval_s = _regex_float(text, r"Prompt evaluation:\s*([0-9.]+)\s*s")
        decode_rate = _regex_float(text, r"Token generation:\s*[0-9.]+\s*s,\s*Rate\s*([0-9.]+)")
        if decode_rate == NA:
            decode_rate = _regex_float(text, r"Total inference time:\s*[0-9.]+\s*s,\s*Rate\s*([0-9.]+)")

        metrics["e2e_latency_ms"] = total_s * 1000.0 if total_s != NA else NA
        if total_s != NA and prompt_eval_s != NA:
            metrics["decode_latency_ms"] = max(0.0, (total_s - prompt_eval_s) * 1000.0)
        else:
            metrics["decode_latency_ms"] = NA
        metrics["ttft_ms"] = NA
        metrics["visible_tokens_per_second"] = decode_rate
        metrics["metrics_source"] = "runner_text_summary"

    errors: list[str] = []
    if "Generation stopped at seq_len limit" in text:
        errors.extend(["seq_len_limit", "no_eos"])
    if "returncode = 0" not in text:
        errors.append("runtime_error_or_missing_returncode")
    if "Traceback (most recent call last)" in text:
        errors.append("python_traceback")
    if "Error" in text and "returncode = 0" not in text:
        errors.append("runtime_error_text")

    metrics["error_type"] = ";".join(dict.fromkeys(errors)) if errors else "none"
    return metrics

def estimate_tpot_ms(decode_latency_ms: Any, actual_output_tokens: Any) -> Any:
    try:
        if decode_latency_ms == NA or actual_output_tokens == NA:
            return NA
        n = float(actual_output_tokens)
        if n <= 0:
            return NA
        return fmt(float(decode_latency_ms) / n)
    except Exception:
        return NA
