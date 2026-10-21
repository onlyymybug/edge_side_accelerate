#!/usr/bin/env bash
set -euo pipefail
MLLM_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${MLLM_CODE_ROOT}/config.env"
set +a

case "${GSM8K_BUDGET}" in
  none|no_budget|no-budget) RUN_ID=output_budget_no_budget ;;
  *) RUN_ID="output_budget_${GSM8K_BUDGET}" ;;
esac
OUTPUT_ROOT="${MLLM_RESULTS_ROOT}/${DEVICE_LABEL}/gsm8k/${RESULT_MODEL_DIR_NAME}/${MLLM_CONFIGURATION_NAME}"

exec "${PYTHON_BIN}" "${MLLM_CODE_ROOT}/run_math_benchmarks.py" \
  --tasks GSM8K \
  --gsm8k-budget "${GSM8K_BUDGET}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --continue-on-error \
  "$@"
