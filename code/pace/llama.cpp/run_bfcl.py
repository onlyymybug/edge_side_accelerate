#!/usr/bin/env python3
"""Inject the llama.cpp phone handler into the installed BFCL CLI."""

from __future__ import annotations

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

from bfcl_handler import Qwen25LlamaCppPhoneHandler


MODEL_KEY = "Qwen/Qwen2.5-3B-Instruct-LLAMACPP-FC"
MODEL_CONFIG_MAPPING[MODEL_KEY] = ModelConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    display_name="Qwen2.5-3B-Instruct llama.cpp Android (FC)",
    url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    org="Qwen",
    license="apache-2.0",
    model_handler=Qwen25LlamaCppPhoneHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)

from bfcl_eval.__main__ import cli  # noqa: E402


if __name__ == "__main__":
    cli()

