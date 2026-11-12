#!/usr/bin/env bash
set -euo pipefail
TMAC_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${TMAC_CODE_ROOT}/config.env"
set +a

if [[ ! -x "${BFCL_PYTHON}" ]]; then
  echo "BFCL Python environment is missing: ${BFCL_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${BFCL_ROOT}/bfcl_eval" ]]; then
  echo "BFCL repository is missing: ${BFCL_ROOT}" >&2
  exit 1
fi

STAGES=(generate evaluate)
if [[ $# -gt 0 && ( "$1" == generate || "$1" == evaluate ) ]]; then
  STAGES=("$1")
  shift
fi
BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-${TMAC_RESULTS_ROOT}/${DEVICE_LABEL}/bfcl/${RESULT_MODEL_DIR_NAME}/${TMAC_CONFIGURATION_NAME}/output_budget_no_budget}"
mkdir -p "${BFCL_PROJECT_ROOT}"
export BFCL_PROJECT_ROOT BFCL_ROOT
export PYTHONPATH="${TMAC_CODE_ROOT}:${PACE_ROOT}/code:${BFCL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for stage in "${STAGES[@]}"; do
  extra=()
  [[ "${stage}" == generate ]] && extra=(--temperature 0)
  "${BFCL_PYTHON}" "${TMAC_CODE_ROOT}/run_bfcl.py" "${stage}" \
    --model Qwen/Qwen2.5-3B-Instruct-TMAC-FC \
    --test-category "${BFCL_TEST_CATEGORY:-simple_python}" \
    "${extra[@]}" \
    "$@"
done
