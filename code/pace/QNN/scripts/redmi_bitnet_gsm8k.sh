#!/usr/bin/env bash
# redmi · BitNet-2B (runtime-selectable HTP mode) · GSM8K
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUN_ID="$(output_budget_dir "${GSM8K_BUDGET}")_raw_v1"
OUTPUT_ROOT="${QNN_RESULTS_ROOT}/redmi/gsm8k/${BITNET_RESULT_MODEL_DIR}/${BITNET_PERFORMANCE_MODE}"
BITNET_BUDGET_ARGS=()
BITNET_TARGET_OUTPUT_TOKENS="${GSM8K_BUDGET}"
case "${GSM8K_BUDGET}" in
  none|no_budget)
    BITNET_TARGET_OUTPUT_TOKENS=128
    BITNET_BUDGET_ARGS=(--model-max-seq-len)
    ;;
esac

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${REDMI_SERIAL}" \
  DEVICE_LABEL=redmi \
  BITNET_PERFORMANCE_MODE="${BITNET_PERFORMANCE_MODE}" \
  BITNET_RESULTS_ROOT="${OUTPUT_ROOT}" \
  GSM8K_TARGET_OUTPUT_TOKENS="${BITNET_TARGET_OUTPUT_TOKENS}" \
  "${BITNET_SCRIPTS}/run_gsm8k.sh" \
    --run-id "${RUN_ID}" \
    "${BITNET_BUDGET_ARGS[@]}" \
    --continue-on-error \
    "$@"
