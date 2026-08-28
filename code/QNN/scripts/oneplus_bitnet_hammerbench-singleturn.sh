#!/usr/bin/env bash
# oneplus · BitNet-2B (runtime-selectable HTP mode) · HammerBench single-turn (en_mod20_eq0)
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

env -u ADB_SERVER_SOCKET \
  ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
  ANDROID_SERIAL="${ONEPLUS_SERIAL}" \
  DEVICE_LABEL=oneplus \
  BITNET_PERFORMANCE_MODE="${BITNET_PERFORMANCE_MODE}" \
  "${BITNET_CODE}/run_hammerbench.sh" \
    --dataset-root "${HAMMERBENCH_DATASET_ROOT_EQ0}" \
    --language en \
    --split single-turn \
    --output-root "${QNN_RESULTS_ROOT}/oneplus/hammerbench-single/${BITNET_RESULT_MODEL_DIR}" \
    --run-id "${HAMMERBENCH_RUN_ID:-${BITNET_PERFORMANCE_MODE}_official_prompt_v5}" \
    --continue-on-error \
    "$@"
