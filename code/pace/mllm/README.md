# Native mllm v2 Android benchmark harness

This is the PACE harness for the native mllm v2 Qwen2.5-3B QNN AOT bundle.
It is deliberately separate from `code/llm.npu`: results go to
`results/mllm`, and the default result model name contains `mllm-v2-native`.

The default deployment artifact is
`/home/lyyyy/mllm-v2-qwen2.5-3b-sm8750-c138f9de/deploy`. It was built from
official mllm v2 commit `033d5cd4805383ea9a3d68fe3c162ee0e689887a`
with QAIRT 2.37.0.250724 and contains:

- HTP V79 / SM8750 QNN graphs;
- LPBQ W4A16 G32 weights and W8A8 KV cache;
- a 1024-token maximum context, 32-token prefill graph, and 1-token decode graph;
- the Qwen2.5-3B Q/K/V projection biases;
- Context SHA-256 `38a0544eb898a012011f4c01e07a36b1340cb2473247929d904ffbe920b261a7`.

## Prompt parity

Every benchmark renders the prompt on the host and sends it to the native
runner through stdin-to-EOF with `--no_template`; the runner does not wrap it
a second time.

- GSM8K/MATH-500 use `qwen2.5_default_system_user_v1`, byte-for-byte matching
  the existing QNN Qwen runner, including the official Qwen system prompt.
- HammerBench reuses the existing official prompt builder and scorer.
- BFCL inherits the existing `Qwen25QnnPhoneHandler` prompt formatter and
  evaluation flow.

Each sample stores `formatted_prompt.txt`, `raw_output.txt`,
`runner_stderr.log`, combined `runner.log`, `completion.txt`, parsed metrics,
and `telemetry.csv` for direct comparison.

## Device and deployment

`DEVICE_LABEL=redmi` selects hardware serial `189f0fad`; `oneplus` selects
`3B15B800A0R00000`. An explicit `ANDROID_SERIAL` always wins. USB is preferred
when both USB and Wi-Fi transports are available.

The harness also selects the platform-library order per ROM: OnePlus uses
`/system/lib64:/vendor/lib64` so `libbinder_ndk.so` resolves against the system
`libbinder.so`; Redmi retains `/vendor/lib64:/system/lib64`. Override
`MLLM_PLATFORM_LIBRARY_PATH` only for diagnostics.

Deploy to whichever device is connected:

```bash
DEVICE_LABEL=redmi ./code/mllm/deploy_bundle.sh
DEVICE_LABEL=oneplus ./code/mllm/deploy_bundle.sh
```

The default phone directory is `/data/local/tmp/mllm-qwen2.5-3b-sm8750`.
Deployment verifies local `SHA256SUMS`, refreshes the runner/libraries/config,
and skips the 2.3 GB Context only when both its remote size and SHA-256 match.

Defaults are device-specific so ordinary launch commands do not need bundle,
phone-directory, or result-model overrides. Redmi uses the verified
`c138f9de` bundle; OnePlus uses the allocator-fixed `a38e28d9` bundle and its
independent phone/result directories.

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
  ./code/mllm/run_gsm8k.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=balanced MATH500_BUDGET=256 \
  ./code/mllm/run_math500.sh

DEVICE_LABEL=oneplus HTP_PERFORMANCE=low_power_saver MATH500_BUDGET=no_budget \
  ./code/mllm/run_math500.sh
```

HammerBench and BFCL use all remaining context as their output ceiling:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst HAMMERBENCH_SPLIT=single-turn \
  ./code/mllm/run_hammerbench.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=burst HAMMERBENCH_SPLIT=multi-turn \
  ./code/mllm/run_hammerbench.sh

DEVICE_LABEL=redmi HTP_PERFORMANCE=burst ./code/mllm/run_bfcl.sh
```

Use `--limit` for smoke tests and `--resume` after an interruption:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst GSM8K_BUDGET=128 \
  ./code/mllm/run_gsm8k.sh --limit 3
```

Results are isolated by device, task, model, and HTP mode below
`results/mllm/`.

Runtime telemetry and CSV fields follow the same PACE policy as QNN, T-MAC,
`llama.cpp-NPU`, and `llm.npu`. Battery-derived energy is marked invalid while
USB/external power is connected; disconnect external power for publishable
energy values.

## Fixed AOT constraints

`MAX_SEQ_LEN=1024` and `MLLM_AR_LEN=32` describe shapes baked into this
Context. The harness rejects other values. These are not llama.cpp batch or
ubatch parameters.

## Validation status

The host-side scripts have syntax/parser/prompt parity checks, but this new
harness has intentionally not been launched against a phone yet. Run a small
`--limit` smoke test after the current phone experiment finishes, then start a
full dataset run only after checking its completion, HTP evidence, token
counts, and telemetry.
