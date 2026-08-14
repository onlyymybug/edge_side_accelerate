#!/usr/bin/env bash
# redmi · Qwen2.5-3B (runtime-selectable HTP mode) · GSM8K
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUN_ID="$(output_budget_dir "${GSM8K_BUDGET}")"
OUTPUT_ROOT="${PACE_ROOT}/results/redmi/gsm8k/${QWEN_RESULT_MODEL_DIR}/${QWEN_PERFORMANCE_MODE}"

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${REDMI_SERIAL}" \
  DEVICE_LABEL=redmi \
  QWEN_MODEL_ROOT=/home/lyyyy/qwen25_3b_qnn_artifacts_hybrid_1024_balanced \
  PHONE_DIR=/data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced \
  PERFORMANCE_MODE_RESULT_NAME="${QWEN_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE="${HTP_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE_NAME="${HTP_PERFORMANCE_MODE_NAME}" \
  "${QWEN_CODE}/run_all.sh" \
    --tasks GSM8K \
    --gsm8k-budget "${GSM8K_BUDGET}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --continue-on-error \
    "$@"
