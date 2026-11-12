#!/usr/bin/env python3
"""Inject the mllm phone handler into the installed BFCL CLI."""

from __future__ import annotations

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

from bfcl_handler import Qwen25MllmPhoneHandler


MODEL_KEY = "Qwen/Qwen2.5-3B-Instruct-MLLM-V2-Native-FC"
MODEL_CONFIG_MAPPING[MODEL_KEY] = ModelConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    display_name="Qwen2.5-3B-Instruct mllm Android (FC)",
    url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    org="Qwen",
    license="apache-2.0",
    model_handler=Qwen25MllmPhoneHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)

from bfcl_eval.__main__ import cli  # noqa: E402


if __name__ == "__main__":
    cli()
