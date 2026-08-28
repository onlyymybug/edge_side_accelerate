# PACE experiment code

The active code is split by execution backend while sharing dataset prompts,
scoring, and telemetry policy:

- `QNN/`: existing ExecuTorch/QNN BitNet, Qwen, and Llama experiments.
- `T-MAC/`: Qwen2.5-3B Android CPU/T-MAC experiments.
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
