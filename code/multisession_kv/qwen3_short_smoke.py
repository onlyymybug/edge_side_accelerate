#!/usr/bin/env python3
"""One short Qwen3 phone inference using the existing JSONL runner protocol."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = ROOT / "qnn_multisession_phone_bundle_ctx6144" / "host"
sys.path.insert(0, str(CLIENT_DIR))
from multisession_client import JsonlRunner, adb_command  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "short_smoke")
    parser.add_argument("--device-dir", default="/data/local/tmp/qwen3_short_smoke_512")
    parser.add_argument("--decoder-model-version", default="qwen3")
    args = parser.parse_args()

    device = args.bundle / "device"
    tokenizer = Tokenizer.from_file(str(device / "tokenizer.json"))
    prompt = "<|im_start|>user\nReply with exactly the word OK.<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(tokens) + 32 > 512:
        raise ValueError("short smoke prompt exceeds 512-token PTE")
    connection = Namespace(
        adb="adb", serial=args.serial, device_dir=args.device_dir,
        binary="qnn_llama_runner_multisession", model="hybrid_llama_qnn.pte",
        tokenizer=".", decoder_model_version=args.decoder_model_version,
        htp_performance_mode=8, max_resident_sessions=1,
    )
    runner = JsonlRunner(adb_command(connection), request_timeout=300)
    try:
        capabilities = runner.request({"command": "get_capabilities"})
        print(json.dumps({"capabilities": capabilities}, ensure_ascii=False), flush=True)
        if capabilities.get("status") != "ok":
            raise RuntimeError("get_capabilities failed")
        created = runner.request({"command": "create_session", "session_id": "smoke"})
        print(json.dumps({"create_session": created}, ensure_ascii=False), flush=True)
        if created.get("status") != "ok":
            raise RuntimeError("create_session failed")
        response = runner.request({
            "command": "run_request", "request_id": "smoke_01", "session_id": "smoke",
            "input_token_ids": tokens, "max_new_tokens": 32,
            "temperature": 0, "stop_on_eos": True,
        })
        print(json.dumps({"inference": response}, ensure_ascii=False), flush=True)
        if response.get("status") != "ok" or not response.get("generated_token_ids"):
            raise RuntimeError("inference failed or generated no tokens")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
