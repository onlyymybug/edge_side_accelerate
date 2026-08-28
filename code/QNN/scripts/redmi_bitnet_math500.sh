#!/usr/bin/env bash
# redmi · BitNet-2B (runtime-selectable HTP mode) · MATH500
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUN_ID="$(output_budget_dir "${MATH500_BUDGET}")_raw_v1"
OUTPUT_ROOT="${QNN_RESULTS_ROOT}/redmi/math500/${BITNET_RESULT_MODEL_DIR}/${BITNET_PERFORMANCE_MODE}"
BITNET_BUDGET_ARGS=()
BITNET_TARGET_OUTPUT_TOKENS="${MATH500_BUDGET}"
case "${MATH500_BUDGET}" in
  none|no_budget)
    BITNET_TARGET_OUTPUT_TOKENS=256
    BITNET_BUDGET_ARGS=(--model-max-seq-len)
    ;;
esac

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${REDMI_SERIAL}" \
  DEVICE_LABEL=redmi \
  BITNET_PERFORMANCE_MODE="${BITNET_PERFORMANCE_MODE}" \
  BITNET_RESULTS_ROOT="${OUTPUT_ROOT}" \
  MATH500_TARGET_OUTPUT_TOKENS="${BITNET_TARGET_OUTPUT_TOKENS}" \
  "${BITNET_SCRIPTS}/run_math500.sh" \
    --run-id "${RUN_ID}" \
    "${BITNET_BUDGET_ARGS[@]}" \
    --continue-on-error \
    "$@"
