# T-MAC Android experiments

This directory runs the Qwen2.5-3B GPTQ-INT4 W4A16G128 T-MAC bundle already
deployed on an Android device. It uses the same datasets, relaxed math graders,
HammerBench scoring, BFCL prompting, battery sampler, and result metric names as
the QNN experiments under `../QNN`.

The phone bundle must be available at
`/data/local/tmp/qwen25_3b_tmac_android_bundle`. The runner is always invoked
with `--no-mmap`; its T-MAC kernel configuration is selected through
`TMAC_KCFG_FILE`.

Select the target device label; `redmi` is already the default:

```bash
export DEVICE_LABEL=redmi
```

`ANDROID_SERIAL` normally does not need to be set.  The launchers identify the
selected phone by its stable `ro.serialno`, so the same command works over USB
or Wi-Fi ADB even if the Wi-Fi address changes; USB is preferred when both are
connected.  `redmi` is the default device label.  To override discovery, set
`ANDROID_SERIAL` explicitly.  For another phone, either use a known
`DEVICE_LABEL` or provide its stable serial through `TMAC_HW_SERIAL`.

Formal outputs are stored below `results/T-MAC/<device>/`; QNN experiments use
the separate `results/QNN/<device>/` hierarchy.

Math budgets accept a positive integer or `no_budget` (the latter is bounded by
`MAX_SEQ_LEN`, which defaults to 1024):

```bash
GSM8K_BUDGET=512 ./code/T-MAC/run_gsm8k.sh
MATH500_BUDGET=256 ./code/T-MAC/run_math500.sh
MATH500_BUDGET=no_budget ./code/T-MAC/run_math500.sh
```

HammerBench and BFCL intentionally have no fixed output budget; generation ends
at EOS or the remaining model context:

```bash
HAMMERBENCH_SPLIT=single-turn ./code/T-MAC/run_hammerbench.sh
HAMMERBENCH_SPLIT=multi-turn ./code/T-MAC/run_hammerbench.sh
./code/T-MAC/run_bfcl.sh
```

Every math/HammerBench sample contains the raw output, cleaned completion,
stderr, combined runner log, structured metrics JSON, prompt, command, and
telemetry CSV. Run roots contain a manifest and summary tables. BFCL retains its
official result/score files plus per-step T-MAC artifacts and metrics.

To replace only known-contaminated math samples in a completed run, combine
`--resume` with a comma-separated `--rerun-sample-ids` list. The selected rows
and their per-sample artifacts are replaced in place; all unlisted rows remain
unchanged. Do not combine this repair mode with `--limit`.

Latency output keeps both the QNN-compatible inference boundary and the actual
per-process cold-start boundary:

- `ttft_ms` and `e2e_latency_ms` exclude model loading;
- `model_load_ms` records loading separately;
- `cold_start_ttft_ms = model_load_ms + ttft_ms`;
- `cold_start_e2e_latency_ms = model_load_ms + inference_ms`.

Energy, J/token, and EDP continue to use the inference-only window so existing
QNN/T-MAC comparisons retain their original meaning.
