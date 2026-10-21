# PACE experiment code

The active code is split by execution backend while sharing dataset prompts,
scoring, and telemetry policy:

- `QNN/`: existing ExecuTorch/QNN BitNet, Qwen, and Llama experiments.
- `T-MAC/`: Qwen2.5-3B Android CPU/T-MAC experiments.
- `llama.cpp-NPU/`: Qwen2.5-3B llama.cpp Qualcomm HTP/NPU experiments.
- `llm.npu/`: Qwen2.5-3B mllm QNN AOT HTP/NPU experiments.
- `mllm/`: native mllm v2 Qwen2.5-3B QNN AOT HTP/NPU experiments.
- `common/`: backend-neutral battery, temperature, memory, and energy sampling.
- `math_benchmarks.py`: shared GSM8K/MATH-500 prompt and grading policy.
- `vendor/`: vendored MATH-compatible PRM800K grader.
- `tests/`: prompt, parser, scorer, and HammerBench regression tests.

## QNN

Device/model/task launchers are under `QNN/scripts/`. Existing command names
and result layouts are unchanged apart from the new directory prefix. For
example:

```bash
./code/QNN/scripts/redmi_qwen_gsm8k.sh
./code/QNN/scripts/oneplus_bitnet_math500.sh
./code/QNN/scripts/oneplus_llama_gsm8k.sh
```

Lower-level QNN entry points live under `QNN/qwen/` and `QNN/bitnet/`.

## T-MAC

The T-MAC backend has one Python driver and one shell entry point per dataset:

```bash
ANDROID_SERIAL=10.29.46.37:5555 GSM8K_BUDGET=512 ./code/T-MAC/run_gsm8k.sh
ANDROID_SERIAL=10.29.46.37:5555 MATH500_BUDGET=512 ./code/T-MAC/run_math500.sh
ANDROID_SERIAL=10.29.46.37:5555 ./code/T-MAC/run_hammerbench.sh
ANDROID_SERIAL=10.29.46.37:5555 ./code/T-MAC/run_bfcl.sh
```

See `T-MAC/README.md` for backend constraints, budgets, and output artifacts.

## llama.cpp-NPU

The llama.cpp HTP/NPU backend mirrors the T-MAC dataset entry points while
writing into an independent result hierarchy:

```bash
DEVICE_LABEL=oneplus GSM8K_BUDGET=512 ./code/llama.cpp-NPU/run_gsm8k.sh
DEVICE_LABEL=oneplus MATH500_BUDGET=no_budget ./code/llama.cpp-NPU/run_math500.sh
DEVICE_LABEL=oneplus HAMMERBENCH_SPLIT=single-turn ./code/llama.cpp-NPU/run_hammerbench.sh
DEVICE_LABEL=oneplus ./code/llama.cpp-NPU/run_bfcl.sh
```

See `llama.cpp-NPU/README.md` for bundle paths, HTP verification, metrics, and
device selection.

## llm.npu

The mllm QNN AOT runner accepts the same task prompts while exposing the HTP
performance policy directly:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst GSM8K_BUDGET=512 ./code/llm.npu/run_gsm8k.sh
DEVICE_LABEL=oneplus HTP_PERFORMANCE=balanced MATH500_BUDGET=256 ./code/llm.npu/run_math500.sh
```

See `llm.npu/README.md` for deployment, supported HTP modes, and AOT shape
constraints.

## Native mllm v2

The native mllm v2 harness uses the newer nested `bin/lib/dsp/model` bundle
and writes to an independent result hierarchy:

```bash
DEVICE_LABEL=redmi HTP_PERFORMANCE=burst GSM8K_BUDGET=512 ./code/mllm/run_gsm8k.sh
DEVICE_LABEL=oneplus HTP_PERFORMANCE=balanced MATH500_BUDGET=512 ./code/mllm/run_math500.sh
```

See `mllm/README.md` for deployment, HammerBench/BFCL commands, provenance,
HTP modes, and fixed AOT constraints.
