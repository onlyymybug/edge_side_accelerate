#!/usr/bin/env bash

set -euo pipefail

BITNET_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${BITNET_CODE_ROOT}/config.env"
set +a

export RUNNER_MODEL_PROFILE=bitnet
export BITNET_HAMMER_PROMPT_LAYOUT=native_raw
export QWEN_BUNDLE_ROOT="${RUNTIME_BUNDLE_ROOT}"
export QWEN_MODEL_ROOT="${BUNDLE_ROOT}"
export HAMMERBENCH_TOKENIZER_PATH="${TOKENIZER_PATH}"
export PHONE_DIR="${DEVICE_DIR}"
export REMOTE_MODEL_NAME
export MODEL_RESULT_NAME QUANTIZATION_RESULT_NAME PERFORMANCE_MODE_RESULT_NAME
export HTP_PERFORMANCE_MODE HTP_PERFORMANCE_MODE_NAME
export QNN_RUNNER_NAME QNN_RUNNER_LOCAL_PATH QNN_RUNNER_REMOTE_PATH BITNET_RAW_PROMPT
export RESULTS_ROOT="${PACE_ROOT}/results/QNN"
export MAX_SEQ_LEN RESULTS_ROOT DEVICE_LABEL ADB_BIN ANDROID_SERIAL
export PYTHONUNBUFFERED=1
export HAMMERBENCH_DATASET_ROOT="${HAMMERBENCH_DATASET_ROOT:-/home/lyyyy/HammerBench/full_dataset/data}"

exec /home/lyyyy/miniconda3/envs/pace/bin/python -u \
  "${BITNET_CODE_ROOT}/../qwen/run_hammerbench.py" "$@"
