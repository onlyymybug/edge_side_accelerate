#!/usr/bin/env bash
set -euo pipefail
LLAMACPP_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${LLAMACPP_CODE_ROOT}/config.env"
set +a

case "${MATH500_BUDGET}" in
  none|no_budget) RUN_ID=output_budget_no_budget ;;
  *) RUN_ID="output_budget_${MATH500_BUDGET}" ;;
esac
OUTPUT_ROOT="${LLAMACPP_RESULTS_ROOT}/${DEVICE_LABEL}/math500/${RESULT_MODEL_DIR_NAME}/${LLAMACPP_CONFIGURATION_NAME}"

exec "${PYTHON_BIN}" "${LLAMACPP_CODE_ROOT}/run_math_benchmarks.py" \
  --tasks MATH500 \
  --math500-budget "${MATH500_BUDGET}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --continue-on-error \
  "$@"

