#!/usr/bin/env bash
# oneplus · Qwen2.5-3B (runtime-selectable HTP mode) · BFCL (simple_python)
# 用法: oneplus_qwen_bfcl.sh            -> 先 generate 再 evaluate
#       oneplus_qwen_bfcl.sh generate   -> 只跑 generate
#       oneplus_qwen_bfcl.sh evaluate   -> 只跑 evaluate
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

STAGES=("generate" "evaluate")
if [[ $# -gt 0 && ( "$1" == "generate" || "$1" == "evaluate" ) ]]; then
  STAGES=("$1"); shift
fi

for stage in "${STAGES[@]}"; do
  echo ">>> BFCL ${stage} (oneplus/qwen)"
  extra=()
  [[ "${stage}" == "generate" ]] && extra=(--temperature 0)
  env -u ADB_SERVER_SOCKET \
    ADB_BIN="${ADB_BIN}" ADB_SERVER_PORT="${ADB_SERVER_PORT}" \
    ANDROID_SERIAL="${ONEPLUS_SERIAL}" \
    DEVICE_LABEL=oneplus \
    BFCL_PROJECT_ROOT="${QNN_RESULTS_ROOT}/oneplus/bfcl/${QWEN_RESULT_MODEL_DIR}/${QWEN_PERFORMANCE_MODE}/output_budget_no_budget" \
    QWEN_MODEL_ROOT=/home/lyyyy/qwen25_3b_qnn_artifacts_hybrid_1024_balanced \
    PHONE_DIR=/data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced \
    PERFORMANCE_MODE_RESULT_NAME="${QWEN_PERFORMANCE_MODE}" \
    HTP_PERFORMANCE_MODE="${HTP_PERFORMANCE_MODE}" \
    HTP_PERFORMANCE_MODE_NAME="${HTP_PERFORMANCE_MODE_NAME}" \
    "${QWEN_CODE}/run_bfcl.sh" "${stage}" \
      --model Qwen/Qwen2.5-3B-Instruct-QNN-FC \
      --test-category simple_python \
      "${extra[@]}" \
      "$@"
done
