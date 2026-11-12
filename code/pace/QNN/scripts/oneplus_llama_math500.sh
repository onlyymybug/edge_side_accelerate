#!/usr/bin/env bash
# oneplus · Llama-3.1-8B Hybrid-1024 · MATH500
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUN_ID="$(output_budget_dir "${MATH500_BUDGET}")"
OUTPUT_ROOT="${QNN_RESULTS_ROOT}/oneplus/math500/${LLAMA_RESULT_MODEL_DIR}/${LLAMA_PERFORMANCE_MODE}"

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${ONEPLUS_SERIAL}" \
  DEVICE_LABEL=oneplus \
  RUNNER_MODEL_PROFILE=llama \
  DECODER_MODEL_VERSION=llama3 \
  QWEN_MODEL_ROOT=/home/lyyyy/llama31_8b_qnn237_artifacts_hybrid_1024 \
  PHONE_DIR=/data/local/tmp/lyy/executorch/llama31_8b_qnn237_dynamic \
  QNN_RUNNER_REMOTE_PATH=/data/local/tmp/lyy/executorch/llama31_8b_qnn237_dynamic/qnn_llama_runner \
  MODEL_RESULT_NAME=llama3.1-8b \
  QUANTIZATION_RESULT_NAME=qnn237-hybrid-1024 \
  RESULT_MODEL_DIR_NAME="${LLAMA_RESULT_MODEL_DIR}" \
  PERFORMANCE_MODE_RESULT_NAME="${LLAMA_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE="${LLAMA_HTP_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE_NAME="${LLAMA_HTP_PERFORMANCE_MODE_NAME}" \
  "${QWEN_CODE}/run_benchmark.sh" \
    --tasks MATH500 \
    --math500-budget "${MATH500_BUDGET}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --continue-on-error \
    "$@"
