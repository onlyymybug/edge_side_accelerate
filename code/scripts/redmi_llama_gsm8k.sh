#!/usr/bin/env bash
# redmi · Llama-3.1-8B Hybrid-1024 · GSM8K
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUN_ID="$(output_budget_dir "${GSM8K_BUDGET}")"
OUTPUT_ROOT="${PACE_ROOT}/results/redmi/gsm8k/${LLAMA_RESULT_MODEL_DIR}/${LLAMA_PERFORMANCE_MODE}"

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${REDMI_SERIAL}" \
  DEVICE_LABEL=redmi \
  RUNNER_MODEL_PROFILE=llama \
  DECODER_MODEL_VERSION=llama3 \
  QWEN_MODEL_ROOT=/home/lyyyy/llama31_8b_qnn237_artifacts_hybrid_1024 \
  PHONE_DIR=/data/local/tmp/lyy/executorch/llama31_8b_qnn237_hybrid_1024 \
  QNN_RUNNER_REMOTE_PATH=/data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced/qnn_llama_runner_htp_mode \
  MODEL_RESULT_NAME=llama3.1-8b \
  QUANTIZATION_RESULT_NAME=qnn237-hybrid-1024 \
  RESULT_MODEL_DIR_NAME="${LLAMA_RESULT_MODEL_DIR}" \
  PERFORMANCE_MODE_RESULT_NAME="${LLAMA_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE="${LLAMA_HTP_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE_NAME="${LLAMA_HTP_PERFORMANCE_MODE_NAME}" \
  "${QWEN_CODE}/run_all.sh" \
    --tasks GSM8K \
    --gsm8k-budget "${GSM8K_BUDGET}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --continue-on-error \
    "$@"
