#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${BITNET_RESULTS_ROOT:-}" ]]; then
  export RESULTS_ROOT="${PACE_ROOT}/results/QNN/${DEVICE_LABEL}/math500/${RESULT_MODEL_DIR_NAME}/${PERFORMANCE_MODE_RESULT_NAME}"
fi
default_run_args=()
if [[ " $* " != *" --run-id "* ]]; then
  default_run_args=(--run-id "output_budget_${MATH500_TARGET_OUTPUT_TOKENS}_raw_v1")
fi

require_device
require_file "${MATH500_DATASET_PATH}"
require_file "${TOKENIZER_PATH}"
require_file "${PYTHON_BIN}"

require_phone_bundle

exec "${PYTHON_BIN}" "${BITNET_CODE_ROOT}/scripts/run_math_benchmarks.py" \
  --dataset-kind math500 "${default_run_args[@]}" "$@"
