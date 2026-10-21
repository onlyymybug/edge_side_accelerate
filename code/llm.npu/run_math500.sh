#!/usr/bin/env bash
set -euo pipefail
LLM_NPU_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${LLM_NPU_CODE_ROOT}/config.env"
set +a

case "${MATH500_BUDGET}" in
  none|no_budget|no-budget) RUN_ID=output_budget_no_budget ;;
  *) RUN_ID="output_budget_${MATH500_BUDGET}" ;;
esac
OUTPUT_ROOT="${LLM_NPU_RESULTS_ROOT}/${DEVICE_LABEL}/math500/${RESULT_MODEL_DIR_NAME}/${LLM_NPU_CONFIGURATION_NAME}"

exec "${PYTHON_BIN}" "${LLM_NPU_CODE_ROOT}/run_math_benchmarks.py" \
  --tasks MATH500 \
  --math500-budget "${MATH500_BUDGET}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --continue-on-error \
  "$@"
