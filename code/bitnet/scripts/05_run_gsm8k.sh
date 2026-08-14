#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${BITNET_RESULTS_ROOT:-}" ]]; then
  export RESULTS_ROOT="${PACE_ROOT}/results/${DEVICE_LABEL}/gsm8k/${RESULT_MODEL_DIR_NAME}/${PERFORMANCE_MODE_RESULT_NAME}"
fi
default_run_args=()
if [[ " $* " != *" --run-id "* ]]; then
  default_run_args=(--run-id "output_budget_${GSM8K_TARGET_OUTPUT_TOKENS}")
fi

require_device
require_file "${GSM8K_DATASET_PATH}"
require_file "${TOKENIZER_PATH}"
require_file "${PYTHON_BIN}"

if ! adb_run shell "test -f '${DEVICE_DIR}/${REMOTE_MODEL_NAME}'"; then
  echo "Hybrid model is not deployed: ${DEVICE_DIR}/${REMOTE_MODEL_NAME}" >&2
  echo "Run scripts/01_deploy_model.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${BITNET_CODE_ROOT}/scripts/run_w2.py" \
  --dataset-kind gsm8k "${default_run_args[@]}" "$@"
