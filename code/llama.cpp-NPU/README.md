# llama.cpp-NPU Android experiments

This directory runs the Qwen2.5-3B-Instruct llama.cpp HTP/NPU bundle from
`/home/lyyyy/phone-bundle-qwen25-3b` with the same datasets, prompts, graders,
ADB transport handling, telemetry, timing fields, and result summaries as the
T-MAC/QNN harnesses.

The bundle contains an IQ4_NL + Q8_0 HMX/HVX GGUF, an ARM64 llama.cpp runner,
and a Qualcomm HTP op package. The known OnePlus deployment is:

```text
/data/local/tmp/llama-npu-qwen25-3b-20260827
```

Override `LLAMACPP_NPU_PHONE_DIR` when the bundle is deployed elsewhere. The
default model filename is
`qwen2.5-3b-instruct.iq4_nl+q8_0-hmx.gguf`.

## Device and HTP validation

`DEVICE_LABEL=oneplus` is the default. Set `DEVICE_LABEL=redmi` to select the
Redmi by its stable hardware serial. USB is preferred when USB and Wi-Fi
transports for the same phone are both online; an explicit `ANDROID_SERIAL`
always wins.

Every successful sample must contain all three forms of HTP evidence:

- `Initializing HTP backend`;
- a positive `RPCMEM model buffer size`;
- a positive `graph splits` count.

If they are missing, the harness changes the sample return code to `-2` and
does not silently accept it as an NPU result. This guard can be disabled only
for diagnostics with `LLAMACPP_NPU_REQUIRE_HTP=0`.

Formal runs require the mask-fixed DSP skel with SHA-256
`e808e9333a1efdc601bb87588e27ff6348d165b0f1b64bf17a8354a754051a9e`.
The phone preflight rejects any other `libhtp_ops_skel.so`. The default command
passes `-b 512 -ub 64`: the larger logical batch avoids the observed outer
batch-splitting failure, while the physical microbatch remains below 128.
`LLAMACPP_NPU_UBATCH` must remain below 128 because physical batch 128 selects a
known-bad HTP prefill path.

Before a run starts, the harness also checks for an existing `llama-cli`
process on the selected phone. It refuses to start while another T-MAC or
llama.cpp-NPU workload is active, preventing concurrent runs from contaminating
latency, energy, and thermal measurements. The diagnostic-only override is
`LLAMACPP_NPU_ALLOW_ACTIVE_RUNNER=1`.

The harness passes the Qwen ChatML prompt as raw text together with
`llama-cli --special`. This flag is required by this build to parse
`<|im_start|>`/`<|im_end|>` as special tokens; without it, the same prompt is
treated as ordinary text and produces invalid output.

## Commands

GSM8K and MATH-500 accept a positive output budget or `no_budget`. No-budget
means that the only output limit is the remaining `MAX_SEQ_LEN` context, which
defaults to 1024.

```bash
DEVICE_LABEL=oneplus LLAMACPP_NPU_THREADS=4 GSM8K_BUDGET=512 \
  ./code/llama.cpp-NPU/run_gsm8k.sh

DEVICE_LABEL=oneplus LLAMACPP_NPU_THREADS=4 MATH500_BUDGET=256 \
  ./code/llama.cpp-NPU/run_math500.sh

DEVICE_LABEL=redmi LLAMACPP_NPU_THREADS=4 MATH500_BUDGET=no_budget \
  ./code/llama.cpp-NPU/run_math500.sh
```

HammerBench and BFCL intentionally have no fixed output budget:

```bash
DEVICE_LABEL=oneplus HAMMERBENCH_SPLIT=single-turn \
  ./code/llama.cpp-NPU/run_hammerbench.sh

DEVICE_LABEL=oneplus HAMMERBENCH_SPLIT=multi-turn \
  ./code/llama.cpp-NPU/run_hammerbench.sh

DEVICE_LABEL=oneplus ./code/llama.cpp-NPU/run_bfcl.sh
```

Use `--limit` for a smoke run and `--resume` after a recoverable interruption:

```bash
DEVICE_LABEL=oneplus GSM8K_BUDGET=512 \
  ./code/llama.cpp-NPU/run_gsm8k.sh --limit 3

DEVICE_LABEL=oneplus GSM8K_BUDGET=512 \
  ./code/llama.cpp-NPU/run_gsm8k.sh --resume
```

Formal outputs are isolated below:

```text
results/llama.cpp-NPU/<device>/<dataset>/
  qwen2.5-3B-llamacpp-npu-iq4nl-q8/threads_<N>_b512_ub64/...
```

Each math/HammerBench sample includes the raw output, cleaned completion,
stderr, combined runner log, structured metrics, prompt, command, and telemetry
CSV. Run roots include manifests and summary/deadline tables. BFCL retains its
official generated/evaluated files plus per-step phone artifacts.

Latency remains aligned with the other PACE runtimes:

- `ttft_ms`: llama.cpp prompt-eval time, excluding model loading;
- `model_load_ms`: process start to backprojected inference start;
- `cold_start_ttft_ms = model_load_ms + ttft_ms`;
- `e2e_latency_ms`: inference-only llama.cpp total time;
- `cold_start_e2e_latency_ms = model_load_ms + inference_ms`.

Energy, J/token, and EDP use the inference-only window. Additional fields
`htp_backend_initialized`, `htp_rpcmem_model_mib`, `htp_graph_splits`, and
`htp_execution_verified` make the NPU execution evidence explicit.
