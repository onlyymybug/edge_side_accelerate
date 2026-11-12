# llm.npu Android benchmark harness

This directory runs the Qwen2.5-3B mllm QNN AOT bundle on the SM8750 Redmi
and OnePlus phones with the same PACE datasets, prompts, scorers, telemetry,
and result fields used by the existing QNN/T-MAC runners.

The deployment artifact is `/home/lyyyy/qwen25_3b_mllm_sm8750_bundle`:

- QAIRT 2.37 / HTP V79;
- LPBQ W4A16 G32, W8A8 KV cache;
- 1024-token context and fixed 32-token prefill graph;
- Qwen2.5 Q/K/V projection-bias fix.

## Prompt parity

Every benchmark sends a fully rendered prompt through `--no_template`, so the
phone runner never wraps it a second time.

- GSM8K/MATH-500 use `qwen2.5_default_system_user_v1`, byte-for-byte matching
  the existing QNN Qwen runner, including the official Qwen system prompt.
- HammerBench reuses the existing official prompt builder and scorer.
- BFCL inherits the existing `Qwen25QnnPhoneHandler` prompt formatter and
  evaluation flow.

Each sample stores `formatted_prompt.txt`, `completion.txt`, the combined
runner log, parsed metrics, and telemetry for direct comparison.

## Device and deployment

`DEVICE_LABEL=redmi` selects hardware serial `189f0fad`; `oneplus` selects
`3B15B800A0R00000`. An explicit `ANDROID_SERIAL` always wins. USB is preferred
when both USB and Wi-Fi transports are available.

Deploy to whichever device is connected:

```bash
DEVICE_LABEL=redmi ./code/llm.npu/deploy_bundle.sh
DEVICE_LABEL=oneplus ./code/llm.npu/deploy_bundle.sh
```

The default phone directory is `/data/local/tmp/mllm-qwen25-3b-sm8750`.
Deployment skips the 2.3 GB Context when a file of the expected size is already
present, but refreshes the runner and libraries.

## HTP modes

Set exactly one of:

```bash
HTP_PERFORMANCE=burst
HTP_PERFORMANCE=balanced
HTP_PERFORMANCE=low_power_saver
```

The harness validates that the backend reports the requested mode and loads
both `model.0.s32` and `model.0.s1`. The result path includes the HTP mode.

## Commands

GSM8K and MATH-500 accept a positive output budget or `no_budget`. No-budget
uses the remaining capacity in the fixed 1024-token context.

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst GSM8K_BUDGET=512 \
  ./code/llm.npu/run_gsm8k.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=balanced MATH500_BUDGET=256 \
  ./code/llm.npu/run_math500.sh

DEVICE_LABEL=oneplus HTP_PERFORMANCE=low_power_saver MATH500_BUDGET=no_budget \
  ./code/llm.npu/run_math500.sh
```

HammerBench and BFCL use all remaining context as their output ceiling:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst HAMMERBENCH_SPLIT=single-turn \
  ./code/llm.npu/run_hammerbench.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=burst HAMMERBENCH_SPLIT=multi-turn \
  ./code/llm.npu/run_hammerbench.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=burst ./code/llm.npu/run_bfcl.sh
```

Use `--limit` for smoke tests and `--resume` after an interruption:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst GSM8K_BUDGET=128 \
  ./code/llm.npu/run_gsm8k.sh --limit 3
```

Results are isolated by device, task, model, and HTP mode below
`results/llm.npu/`.

## Fixed AOT constraints

`MAX_SEQ_LEN=1024` and `LLM_NPU_AR_LEN=32` describe shapes baked into this
Context. The harness rejects other values. These are not llama.cpp batch or
ubatch parameters.
