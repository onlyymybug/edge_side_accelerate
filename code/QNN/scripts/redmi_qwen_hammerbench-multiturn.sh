#!/usr/bin/env bash
# redmi · Qwen2.5-3B (runtime-selectable HTP mode) · HammerBench multi-turn
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${REDMI_SERIAL}" \
  DEVICE_LABEL=redmi \
  QWEN_MODEL_ROOT=/home/lyyyy/qwen25_3b_qnn_artifacts_hybrid_1024_balanced \
  PHONE_DIR=/data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced \
  PERFORMANCE_MODE_RESULT_NAME="${QWEN_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE="${HTP_PERFORMANCE_MODE}" \
  HTP_PERFORMANCE_MODE_NAME="${HTP_PERFORMANCE_MODE_NAME}" \
  "${QWEN_CODE}/run_hammerbench.sh" \
    --dataset-root "${HAMMERBENCH_DATASET_ROOT_EQ0}" \
    --language en \
    --split multi-turn \
    --output-root "${QNN_RESULTS_ROOT}/redmi/hammerbench-multi/${QWEN_RESULT_MODEL_DIR}" \
    --run-id "${HAMMERBENCH_RUN_ID:-${QWEN_PERFORMANCE_MODE}_official_prompt_v2}" \
    --continue-on-error \
    "$@"
