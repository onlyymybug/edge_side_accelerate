#!/usr/bin/env bash
# oneplus · BitNet-2B (runtime-selectable HTP mode) · BFCL (simple_python)
# 用法: oneplus_bitnet_bfcl.sh            -> 先 generate 再 evaluate
#       oneplus_bitnet_bfcl.sh generate   -> 只跑 generate
#       oneplus_bitnet_bfcl.sh evaluate   -> 只跑 evaluate
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

STAGES=("generate" "evaluate")
if [[ $# -gt 0 && ( "$1" == "generate" || "$1" == "evaluate" ) ]]; then
  STAGES=("$1"); shift
fi

for stage in "${STAGES[@]}"; do
  echo ">>> BFCL ${stage} (oneplus/bitnet)"
  extra=()
  [[ "${stage}" == "generate" ]] && extra=(--temperature 0 --num-threads 1 --include-input-log)
  env -u ADB_SERVER_SOCKET \
    ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
    ANDROID_SERIAL="${ONEPLUS_SERIAL}" \
    DEVICE_LABEL=oneplus \
    BFCL_PROJECT_ROOT="${PACE_ROOT}/results/oneplus/bfcl/${BITNET_RESULT_MODEL_DIR}/${BITNET_PERFORMANCE_MODE}/output_budget_no_budget" \
    BITNET_PERFORMANCE_MODE="${BITNET_PERFORMANCE_MODE}" \
    "${BITNET_CODE}/run_bfcl.sh" "${stage}" \
      --model BitDistiller/BitNet-2B-4T-QNN-FC \
      --test-category simple_python \
      "${extra[@]}" \
      "$@"
done
