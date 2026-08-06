#!/usr/bin/env bash

set -euo pipefail

QWEN_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${QWEN_CODE_ROOT}/config.env"
set +a

BFCL_ROOT="${BFCL_ROOT:-/home/lyyyy/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/lyyyy/miniconda3/envs/bfcl-phone/bin/python}"
BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-${RESULTS_ROOT}/${DEVICE_LABEL}/bfcl_${MODEL_RESULT_NAME}_${QUANTIZATION_RESULT_NAME}_${PERFORMANCE_MODE_RESULT_NAME}}"
QNN_BFCL_TELEMETRY_INTERVAL_SECONDS="${QNN_BFCL_TELEMETRY_INTERVAL_SECONDS:-${TELEMETRY_INTERVAL_SECONDS}}"

if [[ ! -x "${BFCL_PYTHON}" ]]; then
  echo "BFCL Python environment is missing: ${BFCL_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${BFCL_ROOT}/bfcl_eval" ]]; then
  echo "BFCL repository is missing: ${BFCL_ROOT}" >&2
  exit 1
fi

mkdir -p "${BFCL_PROJECT_ROOT}"
export BFCL_PROJECT_ROOT BFCL_ROOT PACE_CODE_ROOT
export QNN_BFCL_TELEMETRY_INTERVAL_SECONDS
export PYTHONPATH="${PACE_CODE_ROOT}:${BFCL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${BFCL_PYTHON}" -m bfcl_eval "$@"
